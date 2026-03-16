"""End-to-end training pipeline CLI.

Usage:
    python -m src.pipelines.train_pipeline \\
        --experiment adult-income \\
        --data-dir data \\
        --output-dir data \\
        --register \\
        --promote-staging
"""

import argparse
import logging
import sys
from pathlib import Path

from src.config import settings
from src.data.loader import DataLoader
from src.data.validation import DataValidator, get_adult_schema
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
        description="MLForge training pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--experiment",
        default=settings.mlflow_experiment_name,
        help="MLflow experiment name.",
    )
    parser.add_argument(
        "--data-dir",
        default=settings.data_dir,
        help="Directory to load/save data splits.",
    )
    parser.add_argument(
        "--output-dir",
        default=settings.data_dir,
        help="Directory to write data splits (if fetching fresh data).",
    )
    parser.add_argument(
        "--register",
        action="store_true",
        help="Register the best model to the MLflow Model Registry.",
    )
    parser.add_argument(
        "--promote-staging",
        action="store_true",
        help="Promote the newly registered model to Staging.",
    )
    parser.add_argument(
        "--load-splits",
        action="store_true",
        help="Load pre-saved splits from --data-dir instead of fetching from OpenML.",
    )
    return parser.parse_args(argv)


def _print_metrics_table(metrics: dict) -> None:
    """Pretty-print a metrics dict as a formatted table."""
    print("\n" + "=" * 54)
    print(f"  {'Metric':<28} {'Value':>10}")
    print("=" * 54)
    for k, v in sorted(metrics.items()):
        if isinstance(v, float):
            print(f"  {k:<28} {v:>10.4f}")
        else:
            print(f"  {k:<28} {str(v):>10}")
    print("=" * 54 + "\n")


def run_pipeline(args: argparse.Namespace) -> None:
    loader = DataLoader()
    validator = DataValidator()
    schema = get_adult_schema()

    # ── Data ──────────────────────────────────────────────────────────────────
    if args.load_splits and Path(args.data_dir).exists():
        logger.info("Loading pre-saved splits from %s...", args.data_dir)
        X_train, X_val, X_test, y_train, y_val, y_test = loader.load_splits(args.data_dir)
    else:
        logger.info("Fetching UCI Adult dataset from OpenML...")
        X, y = loader.load()

        logger.info("Validating dataset schema...")
        report = validator.validate(X, schema)
        if not report.passed:
            logger.error("Data validation failed:\n%s", "\n".join(report.errors))
            sys.exit(1)
        if report.warnings:
            for w in report.warnings:
                logger.warning("Validation warning: %s", w)
        logger.info(
            "Data validation passed. %d rows, %d errors, %d warnings.",
            report.row_count,
            len(report.errors),
            len(report.warnings),
        )

        X_train, X_val, X_test, y_train, y_val, y_test = loader.split(X, y)
        loader.save_splits(X_train, X_val, X_test, y_train, y_val, y_test, args.output_dir)

    # ── Training ──────────────────────────────────────────────────────────────
    trainer = Trainer(
        experiment_name=args.experiment,
        tracking_uri=settings.mlflow_tracking_uri,
    )

    logger.info("Starting training run...")
    result = trainer.train(X_train, y_train, X_val, y_val)

    logger.info("Training complete. Run ID: %s", result.run_id)
    _print_metrics_table(result.metrics)

    # ── Registry ──────────────────────────────────────────────────────────────
    if args.register:
        registry = ModelRegistry(
            tracking_uri=settings.mlflow_tracking_uri,
            model_name=settings.model_name,
        )
        version = registry.register_model(run_id=result.run_id)
        logger.info("Model registered as version %s.", version.version)

        if args.promote_staging:
            registry.promote_to_staging(version=version.version)
            logger.info("Version %s promoted to Staging.", version.version)

    logger.info("Pipeline complete.")


def main(argv=None) -> None:
    args = parse_args(argv)
    run_pipeline(args)


if __name__ == "__main__":
    main()
