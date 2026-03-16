"""Prometheus metrics middleware for the FastAPI serving application."""

import time
from typing import Callable

from fastapi import Request, Response
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

REQUEST_COUNT = Counter(
    "mlforge_http_requests_total",
    "Total HTTP request count",
    ["method", "endpoint", "status_code"],
)

REQUEST_LATENCY = Histogram(
    "mlforge_http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

PREDICTION_COUNT = Counter(
    "mlforge_predictions_total",
    "Total prediction count",
    ["model_version", "routed_to", "prediction"],
)

PREDICTION_PROBABILITY = Histogram(
    "mlforge_prediction_probability",
    "Distribution of predicted probabilities",
    ["model_version", "routed_to"],
    buckets=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0],
)

MODELS_LOADED = Gauge(
    "mlforge_models_loaded",
    "Number of models currently loaded",
)

DRIFT_SCORE = Gauge(
    "mlforge_drift_score",
    "Latest overall drift score (PSI)",
    ["feature"],
)

CANARY_TRAFFIC_SPLIT = Gauge(
    "mlforge_canary_traffic_split",
    "Fraction of traffic routed to challenger",
)


async def prometheus_middleware(request: Request, call_next: Callable) -> Response:
    """Record request count and latency for every HTTP call."""
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start

    endpoint = request.url.path
    method = request.method
    status = str(response.status_code)

    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status_code=status).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)

    return response


def get_metrics_response() -> Response:
    """Return a Prometheus-format metrics scrape response."""
    data = generate_latest()
    return Response(content=data, media_type=CONTENT_TYPE_LATEST)
