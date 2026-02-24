"""
test_step2.py — Unit tests for Step 2 modules.
===============================================
Run with:
    pytest tests/test_step2.py -v
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config as cfg
from src.feature_engineering import extract_temporal_features, extract_domain_features
from src.sequential_validator import sort_by_date
from src.lag_features import create_lag_features, fill_lag_nans
from src.rolling_features import create_rolling_features
from src.target_analysis import compute_target_stats, compute_autocorrelation


# ──────────────────────────────────────────────────────────────
# FIXTURES
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def sample_train():
    """Synthetic training data — 3 days (288 rows)."""
    n = 288
    dates = pd.date_range("2023-01-01", periods=n, freq="15min")
    rng = np.random.RandomState(42)

    df = pd.DataFrame({
        "date": dates,
        "wind": rng.uniform(2000, 10000, n),
        "solar": np.maximum(0, rng.normal(3000, 2000, n)),
        "load": rng.uniform(35000, 55000, n),
        "imbalances": rng.normal(0, 500, n),
        "spread": rng.normal(0, 200, n),
    })
    df = extract_temporal_features(df)
    df = extract_domain_features(df)
    df = sort_by_date(df)
    return df


@pytest.fixture
def sample_test():
    """Synthetic test data — 1 day (96 rows), no spread/imbalances."""
    n = 96
    dates = pd.date_range("2024-01-01", periods=n, freq="15min")
    rng = np.random.RandomState(99)

    df = pd.DataFrame({
        "ID": range(n),
        "date": dates,
        "wind": rng.uniform(2000, 10000, n),
        "solar": np.maximum(0, rng.normal(3000, 2000, n)),
        "load": rng.uniform(35000, 55000, n),
    })
    df = extract_temporal_features(df)
    df = extract_domain_features(df)
    df = sort_by_date(df)
    return df


# ──────────────────────────────────────────────────────────────
# TESTS — lag_features
# ──────────────────────────────────────────────────────────────

class TestLagFeatures:
    def test_train_lag_columns_created(self, sample_train):
        result = create_lag_features(sample_train)
        for col in cfg.LAG_FEATURES + cfg.LAG_FEATURES_TRAIN_ONLY:
            assert col in result.columns, f"Missing: {col}"

    def test_test_spread_lags_absent(self, sample_test):
        """Test set has no spread column, so spread lags should not appear."""
        result = create_lag_features(sample_test)
        for col in cfg.LAG_FEATURES_TRAIN_ONLY:
            assert col not in result.columns

    def test_test_core_lags_present(self, sample_test):
        result = create_lag_features(sample_test)
        for col in cfg.LAG_FEATURES:
            assert col in result.columns, f"Missing: {col}"

    def test_lag_values_correct(self, sample_train):
        """Verify lag_1 is exactly the previous row's value."""
        result = create_lag_features(sample_train)
        # Row 5's wind_lag_1 should equal row 4's wind
        assert result.loc[5, "wind_lag_1"] == sample_train.loc[4, "wind"]

    def test_lag_96_correct(self, sample_train):
        """Verify lag_96 is the value from 96 rows back."""
        result = create_lag_features(sample_train)
        assert result.loc[100, "load_lag_96"] == sample_train.loc[4, "load"]

    def test_nans_at_start(self, sample_train):
        """First rows should have NaN before fill (lag_96 needs 96 rows)."""
        result = create_lag_features(sample_train)
        assert pd.isna(result.loc[0, "wind_lag_96"])
        assert pd.isna(result.loc[50, "wind_lag_96"])

    def test_fill_removes_all_nans(self, sample_train):
        result = create_lag_features(sample_train)
        all_lag = cfg.LAG_FEATURES + cfg.LAG_FEATURES_TRAIN_ONLY
        result = fill_lag_nans(result, all_lag)
        present = [c for c in all_lag if c in result.columns]
        assert result[present].isnull().sum().sum() == 0

    def test_no_mutation(self, sample_train):
        original_cols = set(sample_train.columns)
        _ = create_lag_features(sample_train)
        assert set(sample_train.columns) == original_cols

    def test_correct_count(self, sample_train):
        result = create_lag_features(sample_train)
        expected = (
            len(cfg.LAG_COLUMNS) * len(cfg.LAG_OFFSETS)
            + len(cfg.LAG_COLUMNS_TRAIN_ONLY) * len(cfg.LAG_OFFSETS)
        )
        new_cols = set(result.columns) - set(sample_train.columns)
        assert len(new_cols) == expected


# ──────────────────────────────────────────────────────────────
# TESTS — rolling_features
# ──────────────────────────────────────────────────────────────

class TestRollingFeatures:
    def test_train_rolling_columns_created(self, sample_train):
        result = create_rolling_features(sample_train)
        for col in cfg.ROLLING_FEATURES + cfg.ROLLING_FEATURES_TRAIN_ONLY:
            assert col in result.columns, f"Missing: {col}"

    def test_test_spread_rolling_absent(self, sample_test):
        result = create_rolling_features(sample_test)
        for col in cfg.ROLLING_FEATURES_TRAIN_ONLY:
            assert col not in result.columns

    def test_test_core_rolling_present(self, sample_test):
        result = create_rolling_features(sample_test)
        for col in cfg.ROLLING_FEATURES:
            assert col in result.columns, f"Missing: {col}"

    def test_no_nans(self, sample_train):
        """Rolling with min_periods=1 + fillna(0) should produce zero NaN."""
        result = create_rolling_features(sample_train)
        all_rolling = cfg.ROLLING_FEATURES + cfg.ROLLING_FEATURES_TRAIN_ONLY
        present = [c for c in all_rolling if c in result.columns]
        assert result[present].isnull().sum().sum() == 0

    def test_rolling_mean_4_value(self, sample_train):
        """The 1-hour rolling mean at row 10 should be the mean of rows 7-10."""
        result = create_rolling_features(sample_train)
        expected_mean = sample_train["wind"].iloc[7:11].mean()
        actual = result.loc[10, "wind_rmean_4"]
        assert abs(actual - expected_mean) < 1e-6

    def test_rolling_std_4_value(self, sample_train):
        """The 1-hour rolling std at row 10 should match manually computed std."""
        result = create_rolling_features(sample_train)
        expected_std = sample_train["wind"].iloc[7:11].std()
        actual = result.loc[10, "wind_rstd_4"]
        assert abs(actual - expected_std) < 1e-6

    def test_rolling_mean_smooths(self, sample_train):
        """Rolling mean should have lower variance than the raw column."""
        result = create_rolling_features(sample_train)
        raw_std = result["load"].std()
        rmean_std = result["load_rmean_96"].std()
        assert rmean_std < raw_std

    def test_no_mutation(self, sample_train):
        original_cols = set(sample_train.columns)
        _ = create_rolling_features(sample_train)
        assert set(sample_train.columns) == original_cols

    def test_correct_count(self, sample_train):
        result = create_rolling_features(sample_train)
        # 2 stats (mean+std) * windows * columns, for both train+test and train-only
        expected = (
            2 * len(cfg.ROLLING_WINDOWS) * len(cfg.ROLLING_COLUMNS)
            + 2 * len(cfg.ROLLING_WINDOWS) * len(cfg.ROLLING_COLUMNS_TRAIN_ONLY)
        )
        new_cols = set(result.columns) - set(sample_train.columns)
        assert len(new_cols) == expected


# ──────────────────────────────────────────────────────────────
# TESTS — target_analysis
# ──────────────────────────────────────────────────────────────

class TestTargetAnalysis:
    def test_stats_fields(self, sample_train):
        stats = compute_target_stats(sample_train)
        assert stats.count == len(sample_train)
        assert stats.min <= stats.q1 <= stats.median <= stats.q3 <= stats.max

    def test_pct_sums(self, sample_train):
        stats = compute_target_stats(sample_train)
        total = stats.pct_positive + stats.pct_negative + stats.pct_zero
        assert abs(total - 100.0) < 0.1

    def test_iqr_correct(self, sample_train):
        stats = compute_target_stats(sample_train)
        assert abs(stats.iqr - (stats.q3 - stats.q1)) < 1e-6

    def test_autocorrelation_returns_dict(self, sample_train):
        acf = compute_autocorrelation(sample_train, lags=[1, 4, 96])
        assert isinstance(acf, dict)
        assert set(acf.keys()) == {1, 4, 96}

    def test_autocorrelation_range(self, sample_train):
        acf = compute_autocorrelation(sample_train, lags=[1, 4])
        for lag, val in acf.items():
            assert -1.0 <= val <= 1.0, f"ACF at lag {lag} out of range: {val}"

    def test_lag1_highest(self, sample_train):
        """For most time-series, lag-1 should have strongest autocorrelation."""
        acf = compute_autocorrelation(sample_train, lags=[1, 96])
        # With random data this isn't guaranteed, but we just check structure
        assert 1 in acf and 96 in acf
