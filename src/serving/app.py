"""FastAPI serving application with canary routing and Prometheus metrics."""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import mlflow
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config import settings
from src.monitoring.drift import DriftDetector, DriftReport
from src.registry.model_registry import ModelRegistry
from src.serving.canary import CanaryRouter
from src.serving.middleware import (
    CANARY_TRAFFIC_SPLIT,
    MODELS_LOADED,
    PREDICTION_COUNT,
    PREDICTION_PROBABILITY,
    get_metrics_response,
    prometheus_middleware,
)

logger = logging.getLogger(__name__)

# Module-level state initialised in the lifespan handler
_router: Optional[CanaryRouter] = None
_registry: Optional[ModelRegistry] = None
_drift_detector: Optional[DriftDetector] = None
_latest_drift_report: Optional[DriftReport] = None
_startup_time: float = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models from the MLflow registry at startup."""
    global _router, _registry, _drift_detector, _startup_time

    _startup_time = time.time()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    _registry = ModelRegistry(
        tracking_uri=settings.mlflow_tracking_uri,
        model_name=settings.model_name,
    )
    _drift_detector = DriftDetector(
        psi_threshold=settings.psi_threshold,
        ks_pvalue_threshold=settings.ks_pvalue_threshold,
    )

    champion_model = None
    champion_version = "none"
    challenger_model = None
    challenger_version = "none"

    try:
        champion_model, champion_mv = _registry.get_production_model()
        champion_version = champion_mv.version
        logger.info("Champion (Production) model loaded: version %s", champion_version)
    except RuntimeError as exc:
        logger.warning("No Production model available at startup: %s", exc)

    try:
        staging_result = _registry.get_staging_model()
        if staging_result is not None:
            challenger_model, challenger_mv = staging_result
            challenger_version = challenger_mv.version
            logger.info("Challenger (Staging) model loaded: version %s", challenger_version)
    except Exception as exc:
        logger.warning("Could not load Staging model: %s", exc)

    if champion_model is None:
        logger.warning(
            "No champion model loaded. /predict will return 503 until a Production model exists."
        )

    _router = CanaryRouter(
        champion_model=champion_model,
        challenger_model=challenger_model,
        split=settings.canary_traffic_split,
        champion_version=champion_version,
        challenger_version=challenger_version,
    )

    MODELS_LOADED.set(sum([champion_model is not None, challenger_model is not None]))
    CANARY_TRAFFIC_SPLIT.set(settings.canary_traffic_split if challenger_model else 0.0)

    yield

    logger.info("MLForge serving application shutting down.")


app = FastAPI(
    title="MLForge",
    description="Production ML serving platform with canary deployment and drift monitoring.",
    version="1.0.0",
    lifespan=lifespan,
)

app.middleware("http")(prometheus_middleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request / Response models ───────────────────────────────────────────────


class PredictRequest(BaseModel):
    features: Dict[str, Any] = Field(..., description="Feature dict matching the Adult schema.")
    request_id: Optional[str] = Field(
        default=None, description="Optional stable ID for canary routing."
    )


class PredictResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str
    routed_to: str
    request_id: str


class BatchPredictRequest(BaseModel):
    instances: List[Dict[str, Any]]
    request_ids: Optional[List[str]] = None


class BatchPredictResponse(BaseModel):
    predictions: List[PredictResponse]


class PromoteRequest(BaseModel):
    version: str


# ─── Endpoints ───────────────────────────────────────────────────────────────


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(body: PredictRequest):
    """Route a single prediction request through the canary router."""
    if _router is None or _router.champion_model is None:
        raise HTTPException(status_code=503, detail="No model is loaded. Train and register a model first.")

    request_id = body.request_id or str(uuid.uuid4())

    try:
        result = _router.predict(features=body.features, request_id=request_id)
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc

    PREDICTION_COUNT.labels(
        model_version=result.model_version,
        routed_to=result.routed_to,
        prediction=str(result.prediction),
    ).inc()
    PREDICTION_PROBABILITY.labels(
        model_version=result.model_version,
        routed_to=result.routed_to,
    ).observe(result.probability)

    return PredictResponse(
        prediction=result.prediction,
        probability=result.probability,
        model_version=result.model_version,
        routed_to=result.routed_to,
        request_id=request_id,
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Inference"])
async def predict_batch(body: BatchPredictRequest):
    """Run predictions for a list of feature dicts."""
    if _router is None or _router.champion_model is None:
        raise HTTPException(status_code=503, detail="No model is loaded.")

    request_ids = body.request_ids or [str(uuid.uuid4()) for _ in body.instances]
    if len(request_ids) != len(body.instances):
        raise HTTPException(
            status_code=422,
            detail="Length of request_ids must match length of instances.",
        )

    results = []
    for features, rid in zip(body.instances, request_ids):
        try:
            result = _router.predict(features=features, request_id=rid)
        except Exception as exc:
            logger.exception("Batch prediction failed for request_id=%s: %s", rid, exc)
            raise HTTPException(status_code=500, detail=f"Prediction error: {exc}") from exc

        PREDICTION_COUNT.labels(
            model_version=result.model_version,
            routed_to=result.routed_to,
            prediction=str(result.prediction),
        ).inc()
        results.append(
            PredictResponse(
                prediction=result.prediction,
                probability=result.probability,
                model_version=result.model_version,
                routed_to=result.routed_to,
                request_id=rid,
            )
        )

    return BatchPredictResponse(predictions=results)


@app.get("/health", tags=["Operations"])
async def health():
    """Return service health and model metadata."""
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - _startup_time, 1),
        "champion_version": _router.champion_version if _router else "none",
        "challenger_version": _router.challenger_version if _router else "none",
        "canary_enabled": _router._canary_enabled if _router else False,
        "canary_split": settings.canary_traffic_split,
    }


@app.get("/metrics", tags=["Operations"])
async def metrics():
    """Expose Prometheus metrics for scraping."""
    return get_metrics_response()


@app.get("/model/info", tags=["Operations"])
async def model_info():
    """Return information about loaded models and canary configuration."""
    if _router is None:
        raise HTTPException(status_code=503, detail="Router not initialised.")

    return {
        "champion_version": _router.champion_version,
        "challenger_version": _router.challenger_version if _router._canary_enabled else None,
        "canary_split": settings.canary_traffic_split,
        "canary_enabled": _router._canary_enabled,
        "model_name": settings.model_name,
        "mlflow_tracking_uri": settings.mlflow_tracking_uri,
    }


@app.post("/model/promote", tags=["Operations"])
async def promote_model(body: PromoteRequest):
    """Promote a Staging model version to Production."""
    if _registry is None:
        raise HTTPException(status_code=503, detail="Registry not initialised.")

    try:
        _registry.promote_to_production(version=body.version)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"status": "promoted", "version": body.version}


@app.get("/drift/report", tags=["Monitoring"])
async def drift_report():
    """Return the most recent drift detection report."""
    if _latest_drift_report is None:
        return {"status": "no_report", "detail": "No drift report has been computed yet."}

    return {
        "triggered": _latest_drift_report.triggered,
        "overall_drift_score": _latest_drift_report.overall_drift_score,
        "timestamp": _latest_drift_report.timestamp,
        "feature_drift": {
            feat: {
                "psi": r.psi,
                "ks_statistic": r.ks_statistic,
                "ks_p_value": r.ks_p_value,
                "drifted": r.drifted,
            }
            for feat, r in _latest_drift_report.feature_drift.items()
        },
    }
