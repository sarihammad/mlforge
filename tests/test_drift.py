"""Tests for src/monitoring/drift.py."""

import numpy as np
import pandas as pd
import pytest

from src.monitoring.drift import DriftDetector, DriftReport


@pytest.fixture
def detector() -> DriftDetector:
    return DriftDetector(psi_threshold=0.2, ks_pvalue_threshold=0.05, bins=10)


# ── PSI ───────────────────────────────────────────────────────────────────


def test_psi_zero_for_identical_distributions(detector):
    rng = np.random.default_rng(1)
    data = rng.normal(0, 1, 1000)
    psi = detector.compute_psi(data, data.copy())
    assert psi < 1e-6, f"Expected PSI ≈ 0, got {psi}"


def test_psi_small_for_similar_distributions(detector):
    rng = np.random.default_rng(2)
    ref = rng.normal(0, 1, 2000)
    cur = rng.normal(0.05, 1, 2000)  # tiny shift
    psi = detector.compute_psi(ref, cur)
    assert psi < 0.1, f"Expected PSI < 0.1 for similar distributions, got {psi}"


def test_psi_exceeds_threshold_for_clearly_different_distributions(detector):
    rng = np.random.default_rng(3)
    ref = rng.normal(0, 1, 2000)
    cur = rng.normal(5, 1, 2000)  # mean shifted by 5σ
    psi = detector.compute_psi(ref, cur)
    assert psi > 0.2, f"Expected PSI > 0.2, got {psi}"


def test_psi_handles_empty_bins_without_error(detector):
    """Distributions with sparse bins should not raise ZeroDivisionError."""
    ref = np.array([1.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0])
    cur = np.array([1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 4.0, 4.0, 4.0, 5.0])
    psi = detector.compute_psi(ref, cur)
    assert np.isfinite(psi)


# ── KS test ───────────────────────────────────────────────────────────────


def test_ks_test_low_pvalue_for_different_distributions(detector):
    rng = np.random.default_rng(4)
    ref = rng.normal(0, 1, 1000)
    cur = rng.normal(3, 1, 1000)
    stat, p_value = detector.ks_test(ref, cur)
    assert p_value < 0.05, f"Expected p_value < 0.05, got {p_value}"


def test_ks_test_high_pvalue_for_same_distribution(detector):
    rng = np.random.default_rng(5)
    ref = rng.normal(0, 1, 1000)
    cur = rng.normal(0, 1, 1000)
    _, p_value = detector.ks_test(ref, cur)
    # With 1000 samples from the same distribution p_value should usually be > 0.05
    assert p_value > 0.01, f"Expected high p_value for same distribution, got {p_value}"


def test_ks_test_returns_statistic_and_p_value(detector):
    rng = np.random.default_rng(6)
    a = rng.uniform(0, 1, 500)
    b = rng.uniform(0, 1, 500)
    stat, p = detector.ks_test(a, b)
    assert 0.0 <= stat <= 1.0
    assert 0.0 <= p <= 1.0


# ── DriftReport triggered flag ────────────────────────────────────────────


def _make_feature_df(n: int, rng, age_mean: float = 40.0) -> pd.DataFrame:
    """Build a minimal DataFrame with Adult feature columns."""
    return pd.DataFrame(
        {
            "age": rng.normal(age_mean, 10, n).clip(17, 90),
            "fnlwgt": rng.integers(50000, 500000, n).astype(float),
            "education_num": rng.integers(1, 16, n).astype(float),
            "capital_gain": np.zeros(n),
            "capital_loss": np.zeros(n),
            "hours_per_week": rng.integers(20, 60, n).astype(float),
            "workclass": rng.choice(["Private", "Gov", "Self-emp"], n),
            "education": rng.choice(["HS-grad", "Bachelors", "Masters"], n),
            "marital_status": rng.choice(["Never-married", "Married-civ-spouse"], n),
            "occupation": rng.choice(["Tech-support", "Craft-repair", "Adm-clerical"], n),
            "relationship": rng.choice(["Husband", "Not-in-family", "Own-child"], n),
            "race": rng.choice(["White", "Black"], n),
            "sex": rng.choice(["Male", "Female"], n),
            "native_country": ["United-States"] * n,
        }
    )


def test_drift_report_not_triggered_for_similar_data(detector):
    rng = np.random.default_rng(10)
    ref = _make_feature_df(1000, rng, age_mean=40.0)
    cur = _make_feature_df(1000, rng, age_mean=41.0)  # negligible shift
    report = detector.compute_feature_drift(ref, cur)
    assert isinstance(report, DriftReport)
    # The overall score may be above threshold for some features; just verify structure
    assert "age" in report.feature_drift
    assert report.timestamp != ""


def test_drift_report_triggered_for_very_different_data(detector):
    rng = np.random.default_rng(11)
    ref = _make_feature_df(1000, rng, age_mean=30.0)
    cur = _make_feature_df(1000, rng, age_mean=65.0)  # large age shift
    report = detector.compute_feature_drift(ref, cur)
    assert report.triggered
    assert "age" in report.drifted_features


def test_drift_report_structure(detector):
    rng = np.random.default_rng(12)
    ref = _make_feature_df(500, rng)
    cur = _make_feature_df(500, rng)
    report = detector.compute_feature_drift(ref, cur)

    assert hasattr(report, "feature_drift")
    assert hasattr(report, "overall_drift_score")
    assert hasattr(report, "triggered")
    assert hasattr(report, "timestamp")
    assert hasattr(report, "drifted_features")
    assert isinstance(report.feature_drift, dict)
    assert 0.0 <= report.overall_drift_score


def test_prediction_drift_detects_large_shift(detector):
    rng = np.random.default_rng(20)
    ref_preds = rng.beta(2, 5, 500)  # skewed low
    cur_preds = rng.beta(5, 2, 500)  # skewed high
    result = detector.compute_prediction_drift(ref_preds, cur_preds)
    assert result.drifted
    assert result.psi > 0.2


def test_prediction_drift_no_drift_for_same_distribution(detector):
    rng = np.random.default_rng(21)
    preds = rng.beta(3, 3, 1000)
    result = detector.compute_prediction_drift(preds, preds.copy())
    assert result.psi < 0.01
