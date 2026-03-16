"""Tests for src/data/validation.py."""

import numpy as np
import pandas as pd
import pytest

from src.data.validation import DataValidator, SchemaDefinition, get_adult_schema


@pytest.fixture
def minimal_schema() -> SchemaDefinition:
    return SchemaDefinition(
        required_columns=["age", "workclass", "hours_per_week"],
        dtypes={"age": "numeric", "workclass": "object", "hours_per_week": "numeric"},
        value_ranges={"age": (17.0, 90.0), "hours_per_week": (1.0, 99.0)},
        allowed_values={"workclass": ["Private", "Self-emp", "Gov"]},
        max_null_fraction={"age": 0.01, "workclass": 0.05, "hours_per_week": 0.01},
    )


@pytest.fixture
def valid_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [25, 40, 55],
            "workclass": ["Private", "Gov", "Self-emp"],
            "hours_per_week": [40, 35, 50],
        }
    )


@pytest.fixture
def validator() -> DataValidator:
    return DataValidator(p_value_threshold=0.05)


# ── Schema: missing columns ────────────────────────────────────────────────


def test_missing_required_columns_fails(validator, minimal_schema):
    df = pd.DataFrame({"age": [30], "workclass": ["Private"]})  # missing hours_per_week
    report = validator.validate(df, minimal_schema)
    assert not report.passed
    assert any("hours_per_week" in e for e in report.errors)


def test_all_required_columns_present_passes(validator, minimal_schema, valid_df):
    report = validator.validate(valid_df, minimal_schema)
    assert report.passed
    assert len(report.errors) == 0


# ── Out-of-range values ────────────────────────────────────────────────────


def test_out_of_range_value_fails(validator, minimal_schema):
    df = pd.DataFrame(
        {
            "age": [16, 40, 55],   # 16 is below min=17
            "workclass": ["Private", "Gov", "Self-emp"],
            "hours_per_week": [40, 35, 50],
        }
    )
    report = validator.validate(df, minimal_schema)
    assert not report.passed
    assert any("age" in e and "minimum" in e for e in report.errors)


def test_above_max_range_fails(validator, minimal_schema):
    df = pd.DataFrame(
        {
            "age": [25, 40, 92],   # 92 > max=90
            "workclass": ["Private", "Gov", "Self-emp"],
            "hours_per_week": [40, 35, 50],
        }
    )
    report = validator.validate(df, minimal_schema)
    assert not report.passed
    assert any("age" in e and "maximum" in e for e in report.errors)


def test_in_range_values_pass(validator, minimal_schema, valid_df):
    report = validator.validate(valid_df, minimal_schema)
    assert report.passed


# ── Null fraction threshold ────────────────────────────────────────────────


def test_null_fraction_below_threshold_passes(validator, minimal_schema):
    df = pd.DataFrame(
        {
            "age": [25, None, 55],       # 1/3 ≈ 0.33 — above threshold 0.01
            "workclass": ["Private", "Gov", "Self-emp"],
            "hours_per_week": [40, 35, 50],
        }
    )
    report = validator.validate(df, minimal_schema)
    assert not report.passed
    assert any("age" in e and "null fraction" in e for e in report.errors)


def test_no_nulls_passes_null_threshold(validator, minimal_schema, valid_df):
    report = validator.validate(valid_df, minimal_schema)
    assert report.passed
    assert report.null_counts["age"] == 0


def test_null_fraction_exactly_at_threshold_passes(validator):
    """Null fraction equal to the threshold must pass (not strictly greater than)."""
    schema = SchemaDefinition(
        required_columns=["age"],
        dtypes={"age": "numeric"},
        value_ranges={},
        allowed_values={},
        max_null_fraction={"age": 0.5},
    )
    df = pd.DataFrame({"age": [25, None]})  # exactly 50% null
    report = validator.validate(df, schema)
    assert report.passed


# ── Skew detection ─────────────────────────────────────────────────────────


def test_skew_detection_returns_per_feature_results(validator):
    rng = np.random.default_rng(42)
    train_df = pd.DataFrame(
        {
            "age": rng.normal(40, 10, 500).clip(17, 90),
            "sex": rng.choice(["Male", "Female"], size=500),
        }
    )
    # Serving data has a shifted age distribution
    serving_df = pd.DataFrame(
        {
            "age": rng.normal(55, 10, 500).clip(17, 90),
            "sex": rng.choice(["Male", "Female"], size=500),
        }
    )
    skew_report = validator.detect_training_serving_skew(
        train_df, serving_df, columns=["age", "sex"]
    )

    assert "age" in skew_report.feature_results
    assert "sex" in skew_report.feature_results
    # Large shift in age should be detected
    assert skew_report.feature_results["age"].drifted


def test_skew_detection_identical_data_no_drift(validator):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"age": rng.normal(40, 5, 300).clip(17, 90)})
    skew_report = validator.detect_training_serving_skew(df, df.copy(), columns=["age"])
    # Identical data → no drift
    assert not skew_report.feature_results["age"].drifted


def test_adult_schema_has_required_keys():
    schema = get_adult_schema()
    assert "age" in schema.required_columns
    assert "education_num" in schema.required_columns
    assert "age" in schema.value_ranges
    assert len(schema.required_columns) == 14
