"""Automated retraining pipeline triggered by detected feature drift.

Usage:
    python -m src.pipelines.retrain_pipeline \\
        --reference-data data/X_train.parquet \\
        --current-data data/X_val.parquet \\
        --auto-promote
"""

import argparse
import logging
import sys

import pandas as pd

from src.config import settings
from src.data.loader import DataLoader
from src.monitoring.drift import DriftDetector
from src.monitoring.retraining import RetrainingTrigger
from src.registry.model_registry import ModelRegistry
from src.training.trainer import Trainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MLForge automated retraining pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--reference-data",
        required=True,
        help="Path to Parquet file containing reference (training) features.",
    )
    parser.add_argument(
        "--current-data",
        required=True,
        help="Path to Parquet file containing current (serving) features.",
    )
    parser.add_argument(
        "--auto-promote",
        action="store_true",
        help="Automatically promote challenger to champion if it beats the current champion.",
    )
    parser.add_argument(
        "--data-dir",
        default=settings.data_dir,
        help="Directory containing train/val/test splits for retraining.",
    )
    return parser.parse_args(argv)


def run_retrain_pipeline(args: argparse.Namespace) -> None:
    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("Loading reference data from %s...", args.reference_data)
    reference_df = pd.read_parquet(args.reference_data)

    logger.info("Loading current data from %s...", args.current_data)
    current_df = pd.read_parquet(args.current_data)

    logger.info(
        "Reference: %d rows | Current: %d rows",
        len(reference_df),
        len(current_df),
    )

    # ── Drift detection ───────────────────────────────────────────────────────
    drift_detector = DriftDetector(
        psi_threshold=settings.psi_threshold,
        ks_pvalue_threshold=settings.ks_pvalue_threshold,
    )

    trainer = Trainer(
        experiment_name=settings.mlflow_experiment_name,
        tracking_uri=settings.mlflow_tracking_uri,
    )
    registry = ModelRegistry(
        tracking_uri=settings.mlflow_tracking_uri,
        model_name=settings.model_name,
    )

    trigger = RetrainingTrigger(
        drift_detector=drift_detector,
        trainer=trainer,
        registry=registry,
    )

    decision = trigger.check_and_trigger(reference_df, current_df)

    logger.info("Drift detection result: should_retrain=%s", decision.should_retrain)
    logger.info("Reason: %s", decision.reason)

    if decision.drift_report:
        dr = decision.drift_report
        logger.info("Overall drift score (mean PSI): %.4f", dr.overall_drift_score)
        if dr.drifted_features:
            logger.warning("Drifted features: %s", ", ".join(dr.drifted_features))

    if not decision.should_retrain:
        logger.info("No retraining required. Exiting.")
        return

    # ── Retrain ───────────────────────────────────────────────────────────────
    logger.info("Loading train/val splits for retraining from %s...", args.data_dir)
    loader = DataLoader()
    X_train, X_val, X_test, y_train, y_val, y_test = loader.load_splits(args.data_dir)

    run_id = trigger.trigger_retraining(X_train, y_train, X_val, y_val)
    logger.info("Retraining complete. Run ID: %s", run_id)

    # ── Compare champion vs challenger ────────────────────────────────────────
    logger.info("Comparing champion vs challenger on test set...")
    try:
        report = registry.compare_champion_challenger(X_test=X_test, y_test=y_test)

        logger.info(
            "Champion v%s AUC: %.4f | Challenger v%s AUC: %.4f | Delta: %.4f",
            report.champion_version,
            report.champion_metrics["roc_auc"],
            report.challenger_version,
            report.challenger_metrics["roc_auc"],
            report.delta_roc_auc,
        )
        logger.info("Recommendation: %s", report.recommendation)

        if args.auto_promote and report.recommendation == "promote":
            registry.promote_to_production(version=report.challenger_version)
            logger.info(
                "Auto-promoted challenger v%s to Production.", report.challenger_version
            )
        elif report.recommendation == "keep":
            logger.info(
                "Challenger did not beat champion by required margin (%.3f). Keeping champion.",
                0.005,
            )
    except RuntimeError as exc:
        logger.warning("Could not run champion/challenger comparison: %s", exc)

    logger.info("Retrain pipeline complete.")


def main(argv=None) -> None:
    args = parse_args(argv)
    run_retrain_pipeline(args)


if __name__ == "__main__":
    main()
