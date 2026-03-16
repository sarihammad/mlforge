"""Tests for src/features/pipeline.py."""

import pickle

import numpy as np
import pandas as pd
import pytest

from src.features.pipeline import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    build_full_pipeline,
    build_preprocessing_pipeline,
    get_feature_names,
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Small representative DataFrame matching the Adult dataset schema."""
    return pd.DataFrame(
        {
            "age": [25, 38, 52, 44, 30],
            "fnlwgt": [226802, 89814, 336951, 160323, 77516],
            "education_num": [11, 9, 9, 14, 9],
            "capital_gain": [0, 0, 0, 7298, 0],
            "capital_loss": [0, 0, 0, 0, 0],
            "hours_per_week": [40, 50, 40, 40, 40],
            "workclass": ["Private", "Self-emp-not-inc", "Private", "Private", "Private"],
            "education": [
                "11th",
                "HS-grad",
                "HS-grad",
                "Bachelors",
                "HS-grad",
            ],
            "marital_status": [
                "Never-married",
                "Married-civ-spouse",
                "Married-civ-spouse",
                "Married-civ-spouse",
                "Never-married",
            ],
            "occupation": [
                "Machine-op-inspct",
                "Farming-fishing",
                "Transport-moving",
                "Machine-op-inspct",
                "Adm-clerical",
            ],
            "relationship": [
                "Own-child",
                "Husband",
                "Husband",
                "Husband",
                "Not-in-family",
            ],
            "race": ["Black", "White", "White", "Black", "White"],
            "sex": ["Male", "Male", "Male", "Male", "Male"],
            "native_country": [
                "United-States",
                "United-States",
                "United-States",
                "United-States",
                "United-States",
            ],
        }
    )


# ── Column selection ───────────────────────────────────────────────────────


def test_feature_names_returns_correct_order():
    names = get_feature_names()
    assert names[: len(NUMERICAL_FEATURES)] == NUMERICAL_FEATURES
    assert names[len(NUMERICAL_FEATURES) :] == CATEGORICAL_FEATURES


def test_preprocessing_pipeline_produces_correct_column_count(sample_df):
    preprocessor = build_preprocessing_pipeline()
    transformed = preprocessor.fit_transform(sample_df)
    expected_cols = len(NUMERICAL_FEATURES) + len(CATEGORICAL_FEATURES)
    assert transformed.shape[1] == expected_cols


# ── No NaNs in output ─────────────────────────────────────────────────────


def test_no_nans_in_output_with_nulls_in_input(sample_df):
    """The imputation steps must eliminate all NaNs."""
    df_with_nulls = sample_df.copy()
    df_with_nulls.loc[0, "age"] = np.nan
    df_with_nulls.loc[1, "workclass"] = None

    preprocessor = build_preprocessing_pipeline()
    output = preprocessor.fit_transform(df_with_nulls)
    assert not np.any(np.isnan(output))


def test_no_nans_in_clean_input(sample_df):
    preprocessor = build_preprocessing_pipeline()
    output = preprocessor.fit_transform(sample_df)
    assert not np.any(np.isnan(output))


# ── Output shape ──────────────────────────────────────────────────────────


def test_output_shape_matches_input_rows(sample_df):
    preprocessor = build_preprocessing_pipeline()
    output = preprocessor.fit_transform(sample_df)
    assert output.shape[0] == len(sample_df)


def test_full_pipeline_output_shape(sample_df):
    from sklearn.linear_model import LogisticRegression

    y = pd.Series([0, 1, 1, 1, 0])
    pipeline = build_full_pipeline(LogisticRegression(max_iter=200))
    pipeline.fit(sample_df, y)
    preds = pipeline.predict(sample_df)
    assert preds.shape == (len(sample_df),)


# ── Serialisation ─────────────────────────────────────────────────────────


def test_preprocessing_pipeline_is_picklable(sample_df):
    preprocessor = build_preprocessing_pipeline()
    preprocessor.fit(sample_df)
    blob = pickle.dumps(preprocessor)
    restored = pickle.loads(blob)
    original_output = preprocessor.transform(sample_df)
    restored_output = restored.transform(sample_df)
    np.testing.assert_array_almost_equal(original_output, restored_output)


def test_full_pipeline_is_picklable(sample_df):
    from sklearn.linear_model import LogisticRegression

    y = pd.Series([0, 1, 1, 1, 0])
    pipeline = build_full_pipeline(LogisticRegression(max_iter=200))
    pipeline.fit(sample_df, y)

    blob = pickle.dumps(pipeline)
    restored = pickle.loads(blob)
    np.testing.assert_array_equal(pipeline.predict(sample_df), restored.predict(sample_df))


# ── Unknown categories at inference time ─────────────────────────────────


def test_unknown_categories_handled_without_error(sample_df):
    """OrdinalEncoder with handle_unknown='use_encoded_value' should not raise."""
    preprocessor = build_preprocessing_pipeline()
    preprocessor.fit(sample_df)

    test_df = sample_df.copy()
    test_df.loc[0, "workclass"] = "UNSEEN_CATEGORY"
    output = preprocessor.transform(test_df)
    assert output is not None
    assert not np.any(np.isnan(output))
