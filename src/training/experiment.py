"""MLflow experiment management utilities."""

import logging
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.entities
import pandas as pd

from src.config import settings

logger = logging.getLogger(__name__)


class ExperimentManager:
    """Thin wrapper around the MLflow tracking client for experiment management."""

    def __init__(self, tracking_uri: str = settings.mlflow_tracking_uri):
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self.client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)

    def get_or_create_experiment(self, name: str) -> str:
        """Return the experiment_id, creating the experiment if it does not exist."""
        experiment = self.client.get_experiment_by_name(name)
        if experiment is not None:
            logger.debug("Found existing experiment '%s' (id=%s).", name, experiment.experiment_id)
            return experiment.experiment_id

        experiment_id = self.client.create_experiment(name)
        logger.info("Created new experiment '%s' (id=%s).", name, experiment_id)
        return experiment_id

    def get_best_run(
        self,
        experiment_name: str,
        metric: str = "val_roc_auc",
        higher_is_better: bool = True,
    ) -> Optional[mlflow.entities.Run]:
        """Return the run with the best value for the given metric."""
        experiment = self.client.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.warning("Experiment '%s' not found.", experiment_name)
            return None

        order = "DESC" if higher_is_better else "ASC"
        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="",
            order_by=[f"metrics.{metric} {order}"],
            max_results=1,
        )

        if not runs:
            logger.warning("No runs found in experiment '%s'.", experiment_name)
            return None

        best_run = runs[0]
        logger.info(
            "Best run for metric '%s': run_id=%s, value=%.4f",
            metric,
            best_run.info.run_id,
            best_run.data.metrics.get(metric, float("nan")),
        )
        return best_run

    def compare_runs(self, run_ids: List[str]) -> pd.DataFrame:
        """Build a comparison DataFrame of metrics for the given run IDs."""
        rows = []
        for run_id in run_ids:
            try:
                run = self.client.get_run(run_id)
                row = {"run_id": run_id, **run.data.metrics, **run.data.params}
                rows.append(row)
            except mlflow.exceptions.MlflowException as exc:
                logger.warning("Could not fetch run %s: %s", run_id, exc)

        return pd.DataFrame(rows)

    def get_run_metrics(self, run_id: str) -> Dict[str, float]:
        """Return all logged metrics for a run."""
        run = self.client.get_run(run_id)
        return dict(run.data.metrics)

    def list_runs(self, experiment_name: str, n: int = 10) -> pd.DataFrame:
        """Return the n most-recent runs for an experiment as a DataFrame."""
        experiment = self.client.get_experiment_by_name(experiment_name)
        if experiment is None:
            return pd.DataFrame()

        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=n,
        )

        rows = []
        for run in runs:
            row: Dict[str, Any] = {
                "run_id": run.info.run_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
            }
            row.update(run.data.metrics)
            rows.append(row)

        return pd.DataFrame(rows)
