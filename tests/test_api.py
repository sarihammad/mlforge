"""Tests for the FastAPI serving application (src/serving/app.py).

MLflow model loading is mocked so these tests do not require a running MLflow server.
"""

import uuid
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


# ── Fixtures and helpers ─────────────────────────────────────────────────


def _make_mock_model(prediction: int = 1, probability: float = 0.82):
    """Build a mock sklearn-style model that supports predict and predict_proba."""
    model = MagicMock()
    model.predict.return_value = np.array([prediction])
    model.predict_proba.return_value = np.array([[1 - probability, probability]])
    return model


def _adult_features() -> Dict[str, Any]:
    return {
        "age": 39,
        "workclass": "State-gov",
        "fnlwgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States",
    }


@pytest.fixture
def client():
    """Provide a TestClient with mocked MLflow registry."""
    champion_mock = _make_mock_model(prediction=1, probability=0.82)
    challenger_mock = _make_mock_model(prediction=0, probability=0.31)

    mock_registry = MagicMock()
    mock_registry.get_production_model.return_value = (champion_mock, MagicMock(version="3"))
    mock_registry.get_staging_model.return_value = (challenger_mock, MagicMock(version="4"))

    with (
        patch("src.serving.app.ModelRegistry", return_value=mock_registry),
        patch("src.serving.app.mlflow.set_tracking_uri"),
    ):
        from src.serving.app import app

        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


# ── /health ────────────────────────────────────────────────────────────────


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200


def test_health_response_contains_required_keys(client):
    body = client.get("/health").json()
    assert "status" in body
    assert "uptime_seconds" in body
    assert "champion_version" in body


def test_health_status_is_ok(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"


# ── /predict ──────────────────────────────────────────────────────────────


def test_predict_returns_200(client):
    response = client.post("/predict", json={"features": _adult_features()})
    assert response.status_code == 200


def test_predict_response_schema(client):
    response = client.post("/predict", json={"features": _adult_features()})
    body = response.json()
    assert "prediction" in body
    assert "probability" in body
    assert "model_version" in body
    assert "routed_to" in body
    assert "request_id" in body


def test_predict_prediction_is_binary(client):
    body = client.post("/predict", json={"features": _adult_features()}).json()
    assert body["prediction"] in (0, 1)


def test_predict_probability_is_in_range(client):
    body = client.post("/predict", json={"features": _adult_features()}).json()
    assert 0.0 <= body["probability"] <= 1.0


def test_predict_uses_provided_request_id(client):
    rid = str(uuid.uuid4())
    body = client.post(
        "/predict", json={"features": _adult_features(), "request_id": rid}
    ).json()
    assert body["request_id"] == rid


def test_predict_generates_request_id_when_absent(client):
    body = client.post("/predict", json={"features": _adult_features()}).json()
    # Should be a non-empty string (UUID)
    assert isinstance(body["request_id"], str)
    assert len(body["request_id"]) > 0


# ── /predict/batch ────────────────────────────────────────────────────────


def test_batch_predict_processes_multiple_instances(client):
    instances = [_adult_features() for _ in range(5)]
    response = client.post("/predict/batch", json={"instances": instances})
    assert response.status_code == 200
    body = response.json()
    assert len(body["predictions"]) == 5


def test_batch_predict_all_responses_have_schema(client):
    instances = [_adult_features(), _adult_features()]
    body = client.post("/predict/batch", json={"instances": instances}).json()
    for pred in body["predictions"]:
        assert "prediction" in pred
        assert "probability" in pred
        assert "routed_to" in pred


def test_batch_predict_with_explicit_request_ids(client):
    rids = [str(uuid.uuid4()), str(uuid.uuid4())]
    payload = {"instances": [_adult_features(), _adult_features()], "request_ids": rids}
    body = client.post("/predict/batch", json=payload).json()
    returned_ids = [p["request_id"] for p in body["predictions"]]
    assert returned_ids == rids


def test_batch_predict_mismatched_request_ids_returns_422(client):
    payload = {
        "instances": [_adult_features(), _adult_features()],
        "request_ids": ["only-one-id"],
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 422


# ── /model/info ───────────────────────────────────────────────────────────


def test_model_info_returns_200(client):
    response = client.get("/model/info")
    assert response.status_code == 200


def test_model_info_contains_champion_version(client):
    body = client.get("/model/info").json()
    assert "champion_version" in body


# ── /drift/report ─────────────────────────────────────────────────────────


def test_drift_report_when_no_report_available(client):
    response = client.get("/drift/report")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "no_report"
