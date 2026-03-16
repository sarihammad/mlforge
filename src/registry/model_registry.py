"""MLflow Model Registry: registration, promotion, and champion/challenger comparison."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import mlflow
import mlflow.pyfunc
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.entities.model_registry import ModelVersion
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import settings

logger = logging.getLogger(__name__)

# AUC improvement required before recommending a promotion
_PROMOTION_DELTA = 0.005


@dataclass
class ComparisonReport:
    """Side-by-side performance report for champion vs. challenger."""

    champion_version: str
    challenger_version: str
    champion_metrics: Dict[str, float]
    challenger_metrics: Dict[str, float]
    recommendation: str  # 'promote' or 'keep'
    delta_roc_auc: float


class ModelRegistry:
    """Thin wrapper around MlflowClient for lifecycle management."""

    def __init__(
        self,
        tracking_uri: str = settings.mlflow_tracking_uri,
        model_name: str = settings.model_name,
    ):
        self.model_name = model_name
        mlflow.set_tracking_uri(tracking_uri)
        self.client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
        self._ensure_registered_model()

    def _ensure_registered_model(self) -> None:
        """Create the registered model entry if it does not yet exist."""
        try:
            self.client.get_registered_model(self.model_name)
        except mlflow.exceptions.MlflowException:
            self.client.create_registered_model(self.model_name)
            logger.info("Created registered model '%s'.", self.model_name)

    def register_model(self, run_id: str, model_name: Optional[str] = None) -> ModelVersion:
        """Register the model artifact from a completed run."""
        name = model_name or self.model_name
        model_uri = f"runs:/{run_id}/model"
        version = mlflow.register_model(model_uri=model_uri, name=name)
        logger.info(
            "Registered model '%s' version %s from run %s.", name, version.version, run_id
        )
        return version

    def promote_to_staging(self, model_name: Optional[str] = None, version: str = "") -> None:
        """Transition a model version to Staging and tag it as challenger."""
        name = model_name or self.model_name
        self.client.transition_model_version_stage(
            name=name,
            version=version,
            stage="Staging",
            archive_existing_versions=False,
        )
        self.client.set_registered_model_alias(name, settings.challenger_alias, version)
        logger.info("Version %s of '%s' promoted to Staging (alias: %s).", version, name, settings.challenger_alias)

    def promote_to_production(self, model_name: Optional[str] = None, version: str = "") -> None:
        """Transition a model version to Production and archive the previous champion."""
        name = model_name or self.model_name

        # Archive previous production versions
        try:
            prod_versions = self.client.get_latest_versions(name, stages=["Production"])
            for prev in prod_versions:
                if prev.version != version:
                    self.client.transition_model_version_stage(
                        name=name,
                        version=prev.version,
                        stage="Archived",
                        archive_existing_versions=False,
                    )
                    logger.info("Archived previous production version %s of '%s'.", prev.version, name)
        except mlflow.exceptions.MlflowException as exc:
            logger.warning("Could not archive previous production versions: %s", exc)

        self.client.transition_model_version_stage(
            name=name,
            version=version,
            stage="Production",
            archive_existing_versions=False,
        )
        self.client.set_registered_model_alias(name, settings.champion_alias, version)
        logger.info(
            "Version %s of '%s' promoted to Production (alias: %s).", version, name, settings.champion_alias
        )

    def get_production_model(
        self, model_name: Optional[str] = None
    ) -> Tuple[mlflow.pyfunc.PyFuncModel, ModelVersion]:
        """Load the current Production (champion) model."""
        name = model_name or self.model_name
        prod_versions = self.client.get_latest_versions(name, stages=["Production"])
        if not prod_versions:
            raise RuntimeError(f"No Production version found for model '{name}'.")

        version = prod_versions[0]
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{name}/{version.version}")
        logger.info("Loaded Production model '%s' version %s.", name, version.version)
        return model, version

    def get_staging_model(
        self, model_name: Optional[str] = None
    ) -> Optional[Tuple[mlflow.pyfunc.PyFuncModel, ModelVersion]]:
        """Load the current Staging (challenger) model, returning None if absent."""
        name = model_name or self.model_name
        staging_versions = self.client.get_latest_versions(name, stages=["Staging"])
        if not staging_versions:
            logger.info("No Staging version found for model '%s'.", name)
            return None

        version = staging_versions[0]
        model = mlflow.pyfunc.load_model(model_uri=f"models:/{name}/{version.version}")
        logger.info("Loaded Staging model '%s' version %s.", name, version.version)
        return model, version

    def load_model_by_version(
        self, model_name: Optional[str] = None, version: str = ""
    ) -> mlflow.pyfunc.PyFuncModel:
        """Load an arbitrary version of a registered model."""
        name = model_name or self.model_name
        return mlflow.pyfunc.load_model(model_uri=f"models:/{name}/{version}")

    def compare_champion_challenger(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_name: Optional[str] = None,
    ) -> ComparisonReport:
        """Evaluate champion (Production) and challenger (Staging) on the test set.

        Recommends 'promote' if challenger AUC-ROC > champion AUC-ROC + _PROMOTION_DELTA.
        """
        name = model_name or self.model_name

        champion_model, champion_version = self.get_production_model(name)
        staging_result = self.get_staging_model(name)

        if staging_result is None:
            raise RuntimeError(f"No Staging model found for '{name}'. Cannot compare.")

        challenger_model, challenger_version = staging_result

        champion_metrics = self._evaluate(champion_model, X_test, y_test)
        challenger_metrics = self._evaluate(challenger_model, X_test, y_test)

        delta = challenger_metrics["roc_auc"] - champion_metrics["roc_auc"]
        recommendation = "promote" if delta > _PROMOTION_DELTA else "keep"

        logger.info(
            "Champion AUC: %.4f | Challenger AUC: %.4f | Delta: %.4f | Recommendation: %s",
            champion_metrics["roc_auc"],
            challenger_metrics["roc_auc"],
            delta,
            recommendation,
        )

        return ComparisonReport(
            champion_version=champion_version.version,
            challenger_version=challenger_version.version,
            champion_metrics=champion_metrics,
            challenger_metrics=challenger_metrics,
            recommendation=recommendation,
            delta_roc_auc=delta,
        )

    @staticmethod
    def _evaluate(model: mlflow.pyfunc.PyFuncModel, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Run predictions and return a dict of standard metrics."""
        preds = model.predict(X)
        # pyfunc models may return a DataFrame or ndarray
        if hasattr(preds, "values"):
            preds = preds.values.ravel()
        else:
            preds = np.asarray(preds).ravel()

        # If the model returns floats in (0,1), threshold at 0.5 for binary labels
        if preds.dtype in (np.float32, np.float64) and preds.max() <= 1.0:
            probs = preds
            labels = (preds >= 0.5).astype(int)
        else:
            labels = preds.astype(int)
            # Probabilities not directly available from pyfunc; use labels as proxy
            probs = labels.astype(float)

        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        return {
            "accuracy": float(accuracy_score(y, labels)),
            "roc_auc": float(roc_auc_score(y, probs)),
            "f1": float(f1_score(y, labels, zero_division=0)),
            "precision": float(precision_score(y, labels, zero_division=0)),
            "recall": float(recall_score(y, labels, zero_division=0)),
            "average_precision": float(average_precision_score(y, probs)),
        }
