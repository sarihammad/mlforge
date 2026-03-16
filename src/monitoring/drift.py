"""PSI + KS drift detection for feature distributions and prediction shifts."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

from src.config import settings
from src.features.pipeline import CATEGORICAL_FEATURES, NUMERICAL_FEATURES

logger = logging.getLogger(__name__)

_EPSILON = 1e-8  # prevents log(0) in PSI formula


@dataclass
class FeatureDriftResult:
    """Drift test results for a single feature."""

    feature: str
    psi: float
    ks_statistic: Optional[float]
    ks_p_value: Optional[float]
    chi2_statistic: Optional[float]
    chi2_p_value: Optional[float]
    drifted: bool
    test_used: str  # 'psi_ks' for numerical, 'psi_chi2' for categorical


@dataclass
class DriftReport:
    """Aggregated drift detection results across all features."""

    feature_drift: Dict[str, FeatureDriftResult]
    overall_drift_score: float  # mean PSI across all features
    triggered: bool
    timestamp: str
    drifted_features: List[str]


@dataclass
class DriftResult:
    """Drift result for a single array (e.g. prediction distribution)."""

    ks_statistic: float
    ks_p_value: float
    psi: float
    drifted: bool


class DriftDetector:
    """Detects data drift using Population Stability Index and statistical tests.

    PSI interpretation:
        < 0.1  — no significant drift
        0.1–0.2 — moderate drift, worth investigating
        > 0.2  — significant drift, trigger retraining
    """

    def __init__(
        self,
        psi_threshold: float = settings.psi_threshold,
        ks_pvalue_threshold: float = settings.ks_pvalue_threshold,
        bins: int = 10,
    ):
        self.psi_threshold = psi_threshold
        self.ks_pvalue_threshold = ks_pvalue_threshold
        self.bins = bins

    # ─── Core statistics ─────────────────────────────────────────────────────

    def compute_psi(
        self,
        expected: np.ndarray,
        actual: np.ndarray,
        bins: Optional[int] = None,
    ) -> float:
        """Compute the Population Stability Index between two distributions.

        PSI = Σ (actual_% - expected_%) × ln(actual_% / expected_%)

        Bin edges are derived from the expected distribution so both arrays
        share the same discretisation.

        Args:
            expected: Reference/training distribution.
            actual:   Current/serving distribution.
            bins:     Number of histogram bins (defaults to self.bins).

        Returns:
            PSI value (float).
        """
        n_bins = bins or self.bins
        expected = np.asarray(expected, dtype=float)
        actual = np.asarray(actual, dtype=float)

        # Derive bin edges from the expected distribution
        breakpoints = np.linspace(expected.min(), expected.max(), n_bins + 1)
        breakpoints[0] -= 1e-6  # ensure min value is included
        breakpoints[-1] += 1e-6

        expected_counts, _ = np.histogram(expected, bins=breakpoints)
        actual_counts, _ = np.histogram(actual, bins=breakpoints)

        expected_pct = expected_counts / (expected_counts.sum() + _EPSILON)
        actual_pct = actual_counts / (actual_counts.sum() + _EPSILON)

        # Replace zeros with epsilon to avoid log(0)
        expected_pct = np.where(expected_pct == 0, _EPSILON, expected_pct)
        actual_pct = np.where(actual_pct == 0, _EPSILON, actual_pct)

        psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
        return float(psi_values.sum())

    def ks_test(
        self, reference: np.ndarray, current: np.ndarray
    ) -> Tuple[float, float]:
        """Two-sample Kolmogorov-Smirnov test.

        Returns:
            (statistic, p_value)
        """
        stat, p_value = ks_2samp(
            np.asarray(reference, dtype=float),
            np.asarray(current, dtype=float),
        )
        return float(stat), float(p_value)

    def chi2_test(
        self, reference: pd.Series, current: pd.Series
    ) -> Tuple[float, float]:
        """Chi-squared test for categorical distributions.

        Returns:
            (statistic, p_value)
        """
        all_cats = list(set(reference.dropna().unique()) | set(current.dropna().unique()))
        ref_counts = reference.value_counts().reindex(all_cats, fill_value=0)
        cur_counts = current.value_counts().reindex(all_cats, fill_value=0)

        contingency = np.array([ref_counts.values, cur_counts.values])
        if contingency.sum() == 0:
            return 0.0, 1.0

        stat, p_value, _, _ = chi2_contingency(contingency)
        return float(stat), float(p_value)

    # ─── High-level API ───────────────────────────────────────────────────────

    def compute_feature_drift(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
    ) -> DriftReport:
        """Compute PSI and statistical tests for every feature.

        Numerical features: PSI + KS test.
        Categorical features: PSI (on ordinal-encoded distribution) + chi-squared.

        Args:
            reference_df: Training/baseline DataFrame.
            current_df:   Current serving DataFrame.

        Returns:
            DriftReport with per-feature results and a global trigger flag.
        """
        feature_drift: Dict[str, FeatureDriftResult] = {}
        all_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

        for feat in all_features:
            if feat not in reference_df.columns or feat not in current_df.columns:
                logger.warning("Feature '%s' missing from one or both DataFrames; skipping.", feat)
                continue

            ref_col = reference_df[feat].dropna()
            cur_col = current_df[feat].dropna()

            if len(ref_col) == 0 or len(cur_col) == 0:
                logger.warning("Feature '%s' is entirely null in one partition; skipping.", feat)
                continue

            if feat in NUMERICAL_FEATURES:
                psi = self.compute_psi(ref_col.values, cur_col.values)
                ks_stat, ks_p = self.ks_test(ref_col.values, cur_col.values)
                drifted = psi > self.psi_threshold or ks_p < self.ks_pvalue_threshold

                feature_drift[feat] = FeatureDriftResult(
                    feature=feat,
                    psi=psi,
                    ks_statistic=ks_stat,
                    ks_p_value=ks_p,
                    chi2_statistic=None,
                    chi2_p_value=None,
                    drifted=drifted,
                    test_used="psi_ks",
                )
            else:
                # Encode categories as integer codes for PSI
                all_cats = list(set(ref_col.unique()) | set(cur_col.unique()))
                cat_map = {c: i for i, c in enumerate(sorted(all_cats))}
                ref_encoded = ref_col.map(cat_map).dropna().values
                cur_encoded = cur_col.map(cat_map).dropna().values

                psi = self.compute_psi(ref_encoded, cur_encoded)
                chi2_stat, chi2_p = self.chi2_test(ref_col, cur_col)
                drifted = psi > self.psi_threshold or chi2_p < self.ks_pvalue_threshold

                feature_drift[feat] = FeatureDriftResult(
                    feature=feat,
                    psi=psi,
                    ks_statistic=None,
                    ks_p_value=None,
                    chi2_statistic=chi2_stat,
                    chi2_p_value=chi2_p,
                    drifted=drifted,
                    test_used="psi_chi2",
                )

        drifted_features = [f for f, r in feature_drift.items() if r.drifted]
        all_psi = [r.psi for r in feature_drift.values()]
        overall_score = float(np.mean(all_psi)) if all_psi else 0.0
        triggered = overall_score > self.psi_threshold or len(drifted_features) > 0

        logger.info(
            "Drift detection complete. Overall PSI: %.4f | Drifted features: %s | Triggered: %s",
            overall_score,
            drifted_features,
            triggered,
        )

        return DriftReport(
            feature_drift=feature_drift,
            overall_drift_score=overall_score,
            triggered=triggered,
            timestamp=datetime.now(timezone.utc).isoformat(),
            drifted_features=drifted_features,
        )

    def compute_prediction_drift(
        self,
        reference_preds: np.ndarray,
        current_preds: np.ndarray,
    ) -> DriftResult:
        """Detect drift in the prediction distribution using KS test and PSI.

        Args:
            reference_preds: Prediction probabilities from the baseline period.
            current_preds:   Prediction probabilities from the current period.

        Returns:
            DriftResult with ks_statistic, ks_p_value, psi, and drifted flag.
        """
        reference_preds = np.asarray(reference_preds, dtype=float)
        current_preds = np.asarray(current_preds, dtype=float)

        ks_stat, ks_p = self.ks_test(reference_preds, current_preds)
        psi = self.compute_psi(reference_preds, current_preds)
        drifted = psi > self.psi_threshold or ks_p < self.ks_pvalue_threshold

        logger.info(
            "Prediction drift — PSI: %.4f | KS p-value: %.4f | Drifted: %s",
            psi,
            ks_p,
            drifted,
        )

        return DriftResult(ks_statistic=ks_stat, ks_p_value=ks_p, psi=psi, drifted=drifted)
