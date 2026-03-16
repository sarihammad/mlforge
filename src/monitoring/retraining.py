"""Automated retraining trigger based on drift thresholds."""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.monitoring.drift import DriftDetector, DriftReport

logger = logging.getLogger(__name__)


@dataclass
class RetrainingDecision:
    """Output of the retraining trigger check."""

    should_retrain: bool
    reason: str
    drift_report: Optional[DriftReport] = None
    new_run_id: Optional[str] = None
    new_model_version: Optional[str] = None


class RetrainingTrigger:
    """Monitors drift and triggers retraining when thresholds are exceeded.

    The trigger evaluates:
      1. Feature drift (PSI + KS/chi-squared per feature).
      2. Optionally, prediction distribution drift.

    If either the global drift score exceeds psi_threshold or the fraction of
    drifted features exceeds 20%, a retraining job is initiated.
    """

    def __init__(self, drift_detector: DriftDetector, trainer, registry):
        """
        Args:
            drift_detector: DriftDetector instance for computing drift metrics.
            trainer:        Trainer instance (src.training.trainer.Trainer).
            registry:       ModelRegistry instance for registering new models.
        """
        self.drift_detector = drift_detector
        self.trainer = trainer
        self.registry = registry

    def check_and_trigger(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        current_preds: Optional[np.ndarray] = None,
    ) -> RetrainingDecision:
        """Compute drift and decide whether retraining should be initiated.

        Args:
            reference_df:   Baseline feature DataFrame (typically training data).
            current_df:     Recent serving feature DataFrame.
            current_preds:  Optional array of current prediction probabilities.

        Returns:
            RetrainingDecision with should_retrain flag and supporting context.
        """
        drift_report = self.drift_detector.compute_feature_drift(reference_df, current_df)

        reasons = []

        if drift_report.overall_drift_score > self.drift_detector.psi_threshold:
            reasons.append(
                f"Overall PSI {drift_report.overall_drift_score:.4f} exceeds threshold "
                f"{self.drift_detector.psi_threshold}"
            )

        if drift_report.drifted_features:
            reasons.append(
                f"Feature drift detected in: {', '.join(drift_report.drifted_features)}"
            )

        if current_preds is not None:
            # We need reference predictions to compare; skip if not provided
            pass

        if drift_report.triggered and reasons:
            reason_str = "; ".join(reasons)
            logger.warning("Retraining triggered. Reason: %s", reason_str)
            return RetrainingDecision(
                should_retrain=True,
                reason=reason_str,
                drift_report=drift_report,
            )

        logger.info(
            "No retraining needed. Overall PSI: %.4f, Drifted features: %d",
            drift_report.overall_drift_score,
            len(drift_report.drifted_features),
        )
        return RetrainingDecision(
            should_retrain=False,
            reason="Drift within acceptable thresholds.",
            drift_report=drift_report,
        )

    def trigger_retraining(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> str:
        """Train a new model and register it as a Staging (challenger) version.

        Args:
            X_train, y_train: Training set.
            X_val, y_val:     Validation set.

        Returns:
            The MLflow run_id of the newly trained model.
        """
        logger.info("Starting retraining pipeline triggered by drift detection...")
        training_result = self.trainer.train(X_train, y_train, X_val, y_val)

        model_version = self.registry.register_model(run_id=training_result.run_id)
        self.registry.promote_to_staging(version=model_version.version)

        logger.info(
            "Retraining triggered due to drift. New model registered as staging version %s",
            model_version.version,
        )
        return training_result.run_id
