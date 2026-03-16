"""Canary router: deterministic champion/challenger traffic splitting."""

import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Unified prediction output regardless of which model served the request."""

    prediction: int
    probability: float
    model_version: str
    routed_to: str  # 'champion' or 'challenger'


class CanaryRouter:
    """Routes inference requests between a champion and an optional challenger model.

    Routing is deterministic: hash(request_id) % 100 < split * 100 sends the request
    to the challenger. This guarantees the same request_id always hits the same model,
    which makes canary experiments reproducible and easier to debug.
    """

    def __init__(
        self,
        champion_model,
        challenger_model=None,
        split: float = 0.1,
        champion_version: str = "unknown",
        challenger_version: str = "unknown",
    ):
        self.champion_model = champion_model
        self.challenger_model = challenger_model
        self.split = split
        self.champion_version = champion_version
        self.challenger_version = challenger_version
        self._canary_enabled = challenger_model is not None

    def route(self, request_id: str) -> str:
        """Deterministically route a request to 'champion' or 'challenger'.

        Args:
            request_id: A stable identifier for the request (e.g. UUID).

        Returns:
            'challenger' for the canary slice, 'champion' otherwise.
        """
        if not self._canary_enabled:
            return "champion"

        digest = int(hashlib.sha256(request_id.encode()).hexdigest(), 16)
        bucket = digest % 100
        return "challenger" if bucket < int(self.split * 100) else "champion"

    def predict(self, features: Dict[str, Any], request_id: str) -> PredictionResult:
        """Route the request and return a PredictionResult with model metadata.

        Args:
            features: Dict mapping feature names to raw values (pre-pipeline).
            request_id: Stable request identifier used for canary routing.

        Returns:
            PredictionResult with prediction, probability, model version, and route.
        """
        import pandas as pd

        routed_to = self.route(request_id)
        model = self.champion_model if routed_to == "champion" else self.challenger_model
        version = self.champion_version if routed_to == "champion" else self.challenger_version

        df = pd.DataFrame([features])
        raw_output = model.predict(df)

        # Handle sklearn pipelines (return ndarray) and mlflow pyfunc models (may return DataFrame)
        if hasattr(raw_output, "values"):
            raw_output = raw_output.values.ravel()

        import numpy as np

        raw_output = np.asarray(raw_output).ravel()

        # Attempt to get probabilities
        try:
            proba_output = model.predict_proba(df)
            if hasattr(proba_output, "values"):
                proba_output = proba_output.values
            proba_output = np.asarray(proba_output)
            probability = float(proba_output[0, 1])
            prediction = int(raw_output[0])
        except AttributeError:
            # pyfunc model without predict_proba: infer from raw output
            val = float(raw_output[0])
            if 0.0 <= val <= 1.0:
                probability = val
                prediction = int(val >= 0.5)
            else:
                prediction = int(val)
                probability = float(prediction)

        return PredictionResult(
            prediction=prediction,
            probability=probability,
            model_version=version,
            routed_to=routed_to,
        )

    def update_challenger(self, model, version: str = "unknown") -> None:
        """Swap in a new challenger model."""
        self.challenger_model = model
        self.challenger_version = version
        self._canary_enabled = True
        logger.info("Challenger updated to version %s.", version)

    def disable_canary(self) -> None:
        """Route 100% of traffic to the champion."""
        self.challenger_model = None
        self._canary_enabled = False
        logger.info("Canary disabled. All traffic routed to champion.")
