"""Load the UCI Adult dataset and provide train/val/test splits."""

import logging
import os
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from src.config import settings

logger = logging.getLogger(__name__)


class DataLoader:
    """Handles loading, cleaning, and splitting the UCI Adult dataset."""

    def load(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Fetch the Adult dataset from OpenML and return (X, y).

        Returns:
            X: Feature DataFrame with clean column names.
            y: Binary integer Series (1 if income >50K, else 0).
        """
        logger.info("Fetching UCI Adult dataset from OpenML (version=2)...")
        dataset = fetch_openml(name="adult", version=2, as_frame=True)

        X: pd.DataFrame = dataset.frame.drop(columns=["class"]).copy()
        raw_target = dataset.frame["class"]

        # Normalise column names: lowercase, spaces/hyphens → underscores
        X.columns = [
            col.strip().lower().replace("-", "_").replace(" ", "_") for col in X.columns
        ]

        # Convert target to binary integer labels
        y = (raw_target.str.strip().str.lower() == ">50k").astype(int)
        y.name = "income_gt50k"

        logger.info(
            "Dataset loaded: %d rows, %d features. Positive rate: %.3f",
            len(X),
            X.shape[1],
            y.mean(),
        )
        return X, y

    def split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Stratified train / validation / test split.

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # First carve out the test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X,
            y,
            test_size=settings.test_size,
            random_state=settings.random_state,
            stratify=y,
        )

        # Then split the remainder into train and val.
        # val_size is a fraction of the *original* dataset, so we need to rescale.
        val_fraction_of_temp = settings.val_size / (1.0 - settings.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_fraction_of_temp,
            random_state=settings.random_state,
            stratify=y_temp,
        )

        logger.info(
            "Split sizes — train: %d, val: %d, test: %d",
            len(X_train),
            len(X_val),
            len(X_test),
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def save_splits(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_val: pd.Series,
        y_test: pd.Series,
        output_dir: str,
    ) -> None:
        """Persist all splits as Parquet files."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        splits = {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train.to_frame(),
            "y_val": y_val.to_frame(),
            "y_test": y_test.to_frame(),
        }

        for name, df in splits.items():
            path = os.path.join(output_dir, f"{name}.parquet")
            df.to_parquet(path, index=True)
            logger.info("Saved %s → %s", name, path)

    def load_splits(
        self, data_dir: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """Load previously saved splits from Parquet files."""
        X_train = pd.read_parquet(os.path.join(data_dir, "X_train.parquet"))
        X_val = pd.read_parquet(os.path.join(data_dir, "X_val.parquet"))
        X_test = pd.read_parquet(os.path.join(data_dir, "X_test.parquet"))
        y_train = pd.read_parquet(os.path.join(data_dir, "y_train.parquet")).squeeze()
        y_val = pd.read_parquet(os.path.join(data_dir, "y_val.parquet")).squeeze()
        y_test = pd.read_parquet(os.path.join(data_dir, "y_test.parquet")).squeeze()

        logger.info("Loaded splits from %s", data_dir)
        return X_train, X_val, X_test, y_train, y_val, y_test
