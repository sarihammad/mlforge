"""sklearn ColumnTransformer feature engineering pipeline for the Adult dataset."""

from typing import Any, List

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

NUMERICAL_FEATURES: List[str] = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]

CATEGORICAL_FEATURES: List[str] = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]


def build_preprocessing_pipeline() -> ColumnTransformer:
    """Build a ColumnTransformer that handles numerical and categorical features.

    Numerical pipeline:
        1. SimpleImputer(strategy='median') — robust to outliers
        2. StandardScaler — zero mean, unit variance

    Categorical pipeline:
        1. SimpleImputer(strategy='most_frequent') — fill with mode
        2. OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1) —
           safe at inference time when unseen categories appear
    """
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, NUMERICAL_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def build_full_pipeline(model: Any) -> Pipeline:
    """Combine the preprocessing ColumnTransformer with a downstream model.

    Args:
        model: Any sklearn-compatible estimator (e.g. XGBClassifier).

    Returns:
        A Pipeline with steps [('preprocessor', ...), ('model', ...)].
    """
    preprocessor = build_preprocessing_pipeline()
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def get_feature_names() -> List[str]:
    """Return the ordered list of feature names output by the preprocessor.

    The ColumnTransformer preserves insertion order: numerical features first,
    then categorical features.
    """
    return NUMERICAL_FEATURES + CATEGORICAL_FEATURES
