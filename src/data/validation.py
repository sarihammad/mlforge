"""Schema validation and training-serving skew detection for the Adult dataset."""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

logger = logging.getLogger(__name__)


@dataclass
class SchemaDefinition:
    """Declarative schema for a DataFrame."""

    required_columns: List[str]
    dtypes: Dict[str, str]  # column → expected pandas dtype category ('numeric', 'object')
    value_ranges: Dict[str, Tuple[float, float]]  # column → (min, max)
    allowed_values: Dict[str, List[Any]]  # column → list of accepted values
    max_null_fraction: Dict[str, float]  # column → max fraction of nulls (0–1)


@dataclass
class ValidationReport:
    """Result of validating a DataFrame against a SchemaDefinition."""

    passed: bool
    errors: List[str]
    warnings: List[str]
    row_count: int
    null_counts: Dict[str, int]
    stats: Dict[str, Any]


@dataclass
class FeatureSkewResult:
    """Skew test result for a single feature."""

    feature: str
    test: str  # 'ks' or 'chi2'
    statistic: float
    p_value: float
    drifted: bool  # True if p_value < threshold


@dataclass
class SkewReport:
    """Aggregated training-serving skew report."""

    feature_results: Dict[str, FeatureSkewResult]
    global_drift_detected: bool
    drifted_features: List[str]


class DataValidator:
    """Validates DataFrames against a schema and detects training-serving skew."""

    def __init__(self, p_value_threshold: float = 0.05):
        self.p_value_threshold = p_value_threshold

    def validate(self, df: pd.DataFrame, schema: SchemaDefinition) -> ValidationReport:
        """Run all schema checks and return a ValidationReport."""
        errors: List[str] = []
        warnings: List[str] = []
        null_counts: Dict[str, int] = {}
        stats: Dict[str, Any] = {}

        # 1. Required columns
        missing_cols = [c for c in schema.required_columns if c not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")

        present_cols = [c for c in schema.required_columns if c in df.columns]

        # 2. Null fraction check
        for col in present_cols:
            null_count = int(df[col].isnull().sum())
            null_counts[col] = null_count
            null_frac = null_count / max(len(df), 1)

            max_frac = schema.max_null_fraction.get(col, 1.0)
            if null_frac > max_frac:
                errors.append(
                    f"Column '{col}' null fraction {null_frac:.4f} exceeds threshold {max_frac:.4f}"
                )

        # 3. Numeric range checks
        for col, (lo, hi) in schema.value_ranges.items():
            if col not in df.columns:
                continue
            numeric_col = pd.to_numeric(df[col], errors="coerce")
            below = int((numeric_col < lo).sum())
            above = int((numeric_col > hi).sum())
            if below > 0:
                errors.append(
                    f"Column '{col}' has {below} value(s) below minimum {lo}"
                )
            if above > 0:
                errors.append(
                    f"Column '{col}' has {above} value(s) above maximum {hi}"
                )
            stats[col] = {
                "min": float(numeric_col.min()),
                "max": float(numeric_col.max()),
                "mean": float(numeric_col.mean()),
                "std": float(numeric_col.std()),
            }

        # 4. Categorical allowed-value checks
        for col, allowed in schema.allowed_values.items():
            if col not in df.columns:
                continue
            actual_values = set(df[col].dropna().unique())
            unexpected = actual_values - set(allowed)
            if unexpected:
                warnings.append(
                    f"Column '{col}' contains unexpected values: {unexpected}"
                )

        # 5. dtype sanity (numeric vs object)
        for col, expected_dtype in schema.dtypes.items():
            if col not in df.columns:
                continue
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            if expected_dtype == "numeric" and not is_numeric:
                warnings.append(f"Column '{col}' expected numeric dtype but got {df[col].dtype}")
            elif expected_dtype == "object" and is_numeric:
                warnings.append(
                    f"Column '{col}' expected object/categorical dtype but got {df[col].dtype}"
                )

        passed = len(errors) == 0
        return ValidationReport(
            passed=passed,
            errors=errors,
            warnings=warnings,
            row_count=len(df),
            null_counts=null_counts,
            stats=stats,
        )

    def detect_training_serving_skew(
        self,
        train_df: pd.DataFrame,
        serving_df: pd.DataFrame,
        columns: List[str],
    ) -> SkewReport:
        """Run KS test (numerical) or chi-squared test (categorical) per feature.

        Returns a SkewReport with per-feature p-values and a global drift flag.
        """
        feature_results: Dict[str, FeatureSkewResult] = {}

        for col in columns:
            if col not in train_df.columns or col not in serving_df.columns:
                logger.warning("Column '%s' not found in both DataFrames; skipping.", col)
                continue

            train_col = train_df[col].dropna()
            serving_col = serving_df[col].dropna()

            if pd.api.types.is_numeric_dtype(train_col):
                stat, p_value = ks_2samp(
                    train_col.astype(float).values,
                    serving_col.astype(float).values,
                )
                test_name = "ks"
            else:
                # Align categories
                all_cats = list(set(train_col.unique()) | set(serving_col.unique()))
                train_counts = train_col.value_counts().reindex(all_cats, fill_value=0)
                serving_counts = serving_col.value_counts().reindex(all_cats, fill_value=0)
                contingency = np.array([train_counts.values, serving_counts.values])
                # Avoid chi2 on zero-only columns
                if contingency.sum() == 0:
                    stat, p_value = 0.0, 1.0
                else:
                    stat, p_value, _, _ = chi2_contingency(contingency)
                test_name = "chi2"

            drifted = bool(p_value < self.p_value_threshold)
            feature_results[col] = FeatureSkewResult(
                feature=col,
                test=test_name,
                statistic=float(stat),
                p_value=float(p_value),
                drifted=drifted,
            )

        drifted_features = [f for f, r in feature_results.items() if r.drifted]
        global_drift = len(drifted_features) > 0

        return SkewReport(
            feature_results=feature_results,
            global_drift_detected=global_drift,
            drifted_features=drifted_features,
        )


def get_adult_schema() -> SchemaDefinition:
    """Return the hardcoded schema for the UCI Adult dataset."""
    return SchemaDefinition(
        required_columns=[
            "age",
            "workclass",
            "fnlwgt",
            "education",
            "education_num",
            "marital_status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital_gain",
            "capital_loss",
            "hours_per_week",
            "native_country",
        ],
        dtypes={
            "age": "numeric",
            "fnlwgt": "numeric",
            "education_num": "numeric",
            "capital_gain": "numeric",
            "capital_loss": "numeric",
            "hours_per_week": "numeric",
            "workclass": "object",
            "education": "object",
            "marital_status": "object",
            "occupation": "object",
            "relationship": "object",
            "race": "object",
            "sex": "object",
            "native_country": "object",
        },
        value_ranges={
            "age": (17.0, 90.0),
            "fnlwgt": (10000.0, 1500000.0),
            "education_num": (1.0, 16.0),
            "capital_gain": (0.0, 99999.0),
            "capital_loss": (0.0, 4356.0),
            "hours_per_week": (1.0, 99.0),
        },
        allowed_values={
            "sex": ["Male", "Female", "male", "female"],
            "race": [
                "White",
                "Black",
                "Asian-Pac-Islander",
                "Amer-Indian-Eskimo",
                "Other",
            ],
        },
        max_null_fraction={
            "age": 0.01,
            "fnlwgt": 0.01,
            "education_num": 0.01,
            "capital_gain": 0.01,
            "capital_loss": 0.01,
            "hours_per_week": 0.01,
            "workclass": 0.10,
            "education": 0.01,
            "marital_status": 0.01,
            "occupation": 0.10,
            "relationship": 0.01,
            "race": 0.01,
            "sex": 0.01,
            "native_country": 0.05,
        },
    )
