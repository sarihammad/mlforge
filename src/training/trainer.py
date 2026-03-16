"""Training orchestrator with full MLflow tracking."""

import json
import logging
import tempfile
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.config import settings
from src.features.pipeline import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    build_full_pipeline,
)
from src.training.experiment import ExperimentManager

logger = logging.getLogger(__name__)

# Default XGBoost hyperparameters
DEFAULT_XGBOOST_PARAMS: Dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "auc",
    "use_label_encoder": False,
    "random_state": settings.random_state,
}


@dataclass
class TrainingResult:
    run_id: str
    metrics: Dict[str, float]
    model_uri: str
    feature_importances: Dict[str, float]


@dataclass
class CVTrainingResult:
    run_id: str
    mean_metrics: Dict[str, float]
    std_metrics: Dict[str, float]
    model_uri: str


def _build_model(model_params: Optional[Dict[str, Any]] = None):
    """Instantiate an XGBClassifier, falling back to GradientBoostingClassifier."""
    params = {**DEFAULT_XGBOOST_PARAMS, **(model_params or {})}

    try:
        from xgboost import XGBClassifier  # type: ignore

        # XGBClassifier does not accept use_label_encoder in recent versions
        params.pop("use_label_encoder", None)
        params.pop("eval_metric", None)
        model = XGBClassifier(**params)
        logger.info("Using XGBClassifier.")
    except ImportError:
        from sklearn.ensemble import GradientBoostingClassifier

        gb_params = {
            k: v
            for k, v in params.items()
            if k in {"n_estimators", "max_depth", "learning_rate", "subsample", "random_state"}
        }
        model = GradientBoostingClassifier(**gb_params)
        logger.warning("xgboost not installed; falling back to GradientBoostingClassifier.")

    return model


def _compute_metrics(y_true, y_pred, y_prob) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, y_prob)),
    }


class Trainer:
    """Orchestrates model training with full MLflow experiment tracking."""

    def __init__(
        self,
        experiment_name: str = settings.mlflow_experiment_name,
        tracking_uri: str = settings.mlflow_tracking_uri,
    ):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self._experiment_manager = ExperimentManager(tracking_uri=tracking_uri)
        self._experiment_id = self._experiment_manager.get_or_create_experiment(experiment_name)

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        model_params: Optional[Dict[str, Any]] = None,
    ) -> TrainingResult:
        """Train a model, log everything to MLflow, and return a TrainingResult."""
        model = _build_model(model_params)
        pipeline = build_full_pipeline(model)

        all_params = {**DEFAULT_XGBOOST_PARAMS, **(model_params or {})}

        with mlflow.start_run(experiment_id=self._experiment_id) as run:
            run_id = run.info.run_id

            # Log hyperparameters and dataset sizes
            mlflow.log_params(all_params)
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("val_size", len(X_val))
            mlflow.log_param("features", len(NUMERICAL_FEATURES) + len(CATEGORICAL_FEATURES))

            logger.info("Fitting pipeline on %d training samples...", len(X_train))
            pipeline.fit(X_train, y_train)

            # Validation metrics
            y_val_pred = pipeline.predict(X_val)
            y_val_prob = pipeline.predict_proba(X_val)[:, 1]
            val_metrics = {f"val_{k}": v for k, v in _compute_metrics(y_val, y_val_pred, y_val_prob).items()}
            mlflow.log_metrics(val_metrics)

            # Training metrics (check for overfitting)
            y_train_pred = pipeline.predict(X_train)
            y_train_prob = pipeline.predict_proba(X_train)[:, 1]
            train_metrics = {f"train_{k}": v for k, v in _compute_metrics(y_train, y_train_pred, y_train_prob).items()}
            mlflow.log_metrics(train_metrics)

            logger.info(
                "Validation — AUC: %.4f  F1: %.4f  Accuracy: %.4f",
                val_metrics["val_roc_auc"],
                val_metrics["val_f1"],
                val_metrics["val_accuracy"],
            )

            # Feature importances (from underlying model)
            feature_importances = self._extract_feature_importances(pipeline)
            with tempfile.TemporaryDirectory() as tmpdir:
                fi_path = os.path.join(tmpdir, "feature_importances.json")
                with open(fi_path, "w") as fh:
                    json.dump(feature_importances, fh, indent=2)
                mlflow.log_artifact(fi_path)

                # Confusion matrix
                cm = confusion_matrix(y_val, y_val_pred)
                cm_data = {
                    "labels": ["<=50K", ">50K"],
                    "matrix": cm.tolist(),
                }
                cm_path = os.path.join(tmpdir, "confusion_matrix.json")
                with open(cm_path, "w") as fh:
                    json.dump(cm_data, fh, indent=2)
                mlflow.log_artifact(cm_path)

            # Log the model
            model_uri = f"runs:/{run_id}/model"
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                registered_model_name=None,  # explicit registration done via registry module
            )

            logger.info("Run %s complete. Model URI: %s", run_id, model_uri)

            return TrainingResult(
                run_id=run_id,
                metrics={**val_metrics, **train_metrics},
                model_uri=model_uri,
                feature_importances=feature_importances,
            )

    def train_with_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        model_params: Optional[Dict[str, Any]] = None,
    ) -> CVTrainingResult:
        """Stratified K-fold cross-validation with MLflow tracking."""
        model = _build_model(model_params)
        pipeline = build_full_pipeline(model)

        all_params = {**DEFAULT_XGBOOST_PARAMS, **(model_params or {})}

        scoring = {
            "accuracy": "accuracy",
            "roc_auc": "roc_auc",
            "f1": "f1",
            "precision": "precision",
            "recall": "recall",
            "average_precision": "average_precision",
        }

        logger.info("Running %d-fold stratified CV on %d samples...", cv, len(X))
        cv_results = cross_validate(
            pipeline,
            X,
            y,
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=settings.random_state),
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1,
        )

        mean_metrics: Dict[str, float] = {}
        std_metrics: Dict[str, float] = {}

        for key, values in cv_results.items():
            if key.startswith("test_") or key.startswith("train_"):
                clean_key = key.replace("test_", "val_").replace("train_", "train_")
                mean_metrics[f"cv_mean_{clean_key}"] = float(np.mean(values))
                std_metrics[f"cv_std_{clean_key}"] = float(np.std(values))

        with mlflow.start_run(experiment_id=self._experiment_id) as run:
            run_id = run.info.run_id
            mlflow.log_params({**all_params, "cv_folds": cv, "train_size": len(X)})
            mlflow.log_metrics(mean_metrics)
            mlflow.log_metrics(std_metrics)

            # Fit on full dataset to produce a deployable model
            pipeline.fit(X, y)
            mlflow.sklearn.log_model(pipeline, artifact_path="model")
            model_uri = f"runs:/{run_id}/model"

        logger.info(
            "CV complete. Mean val AUC: %.4f ± %.4f",
            mean_metrics.get("cv_mean_val_roc_auc", float("nan")),
            std_metrics.get("cv_std_val_roc_auc", float("nan")),
        )

        return CVTrainingResult(
            run_id=run_id,
            mean_metrics=mean_metrics,
            std_metrics=std_metrics,
            model_uri=model_uri,
        )

    @staticmethod
    def _extract_feature_importances(pipeline) -> Dict[str, float]:
        """Extract feature importances from the fitted pipeline."""
        feature_names = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
        model_step = pipeline.named_steps.get("model")

        if model_step is None:
            return {}

        importances: Optional[np.ndarray] = None
        if hasattr(model_step, "feature_importances_"):
            importances = model_step.feature_importances_

        if importances is None or len(importances) != len(feature_names):
            return {name: 0.0 for name in feature_names}

        total = importances.sum()
        if total == 0:
            return {name: 0.0 for name in feature_names}

        return {
            name: float(imp / total)
            for name, imp in zip(feature_names, importances)
        }
