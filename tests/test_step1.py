"""
test_step1.py — Unit tests for every Step 1 module.
====================================================
Run with:
    pytest tests/test_step1.py -v
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

# Make sure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config as cfg
from src.feature_engineering import extract_temporal_features, extract_domain_features
from src.sequential_validator import sort_by_date, validate_intervals
from src.scaler import fit_scalers, apply_scaling
from src.clustering import run_kmeans_elbow, fit_kmeans, assign_regimes


# ──────────────────────────────────────────────────────────────
# FIXTURES
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def sample_train():
    """Create a small synthetic training DataFrame (2 days = 192 rows)."""
    n = 192  # 96 intervals/day * 2 days
    dates = pd.date_range("2023-01-01", periods=n, freq="15min")
    rng = np.random.RandomState(42)

    return pd.DataFrame({
        "date": dates,
        "wind": rng.uniform(2000, 10000, n),
        "solar": np.maximum(0, rng.normal(3000, 2000, n)),
        "load": rng.uniform(35000, 55000, n),
        "imbalances": rng.normal(0, 500, n),
        "spread": rng.normal(0, 200, n),
    })


@pytest.fixture
def sample_test():
    """Create a small synthetic test DataFrame (1 day = 96 rows)."""
    n = 96
    dates = pd.date_range("2024-01-01", periods=n, freq="15min")
    rng = np.random.RandomState(99)

    return pd.DataFrame({
        "ID": range(n),
        "date": dates,
        "wind": rng.uniform(2000, 10000, n),
        "solar": np.maximum(0, rng.normal(3000, 2000, n)),
        "load": rng.uniform(35000, 55000, n),
    })


# ──────────────────────────────────────────────────────────────
# TESTS — feature_engineering
# ──────────────────────────────────────────────────────────────

class TestTemporalFeatures:
    def test_columns_created(self, sample_train):
        result = extract_temporal_features(sample_train)
        for col in cfg.TEMPORAL_FEATURES:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_mutation(self, sample_train):
        original_cols = set(sample_train.columns)
        _ = extract_temporal_features(sample_train)
        assert set(sample_train.columns) == original_cols

    def test_cyclical_range(self, sample_train):
        result = extract_temporal_features(sample_train)
        for col in ["hour_sin", "hour_cos", "month_sin", "month_cos",
                     "dow_sin", "dow_cos", "interval_sin", "interval_cos"]:
            assert result[col].min() >= -1.0 - 1e-9
            assert result[col].max() <= 1.0 + 1e-9

    def test_is_weekend_binary(self, sample_train):
        result = extract_temporal_features(sample_train)
        assert set(result["is_weekend"].unique()).issubset({0, 1})

    def test_interval_of_day_range(self, sample_train):
        result = extract_temporal_features(sample_train)
        assert result["interval_of_day"].min() >= 0
        assert result["interval_of_day"].max() <= 95


class TestDomainFeatures:
    def test_columns_created(self, sample_train):
        result = extract_domain_features(sample_train)
        for col in cfg.DOMAIN_FEATURES:
            assert col in result.columns, f"Missing column: {col}"

    def test_net_load_formula(self, sample_train):
        result = extract_domain_features(sample_train)
        expected = sample_train["load"] - sample_train["wind"] - sample_train["solar"]
        pd.testing.assert_series_equal(result["net_load"], expected, check_names=False)

    def test_total_renewable_formula(self, sample_train):
        result = extract_domain_features(sample_train)
        expected = sample_train["wind"] + sample_train["solar"]
        pd.testing.assert_series_equal(
            result["total_renewable"], expected, check_names=False
        )

    def test_no_nans(self, sample_train):
        result = extract_domain_features(sample_train)
        for col in cfg.DOMAIN_FEATURES:
            assert not result[col].isna().any(), f"NaN in {col}"


# ──────────────────────────────────────────────────────────────
# TESTS — sequential_validator
# ──────────────────────────────────────────────────────────────

class TestSequentialValidator:
    def test_sort_preserves_rows(self, sample_train):
        shuffled = sample_train.sample(frac=1, random_state=0)
        sorted_df = sort_by_date(shuffled)
        assert len(sorted_df) == len(sample_train)

    def test_sorted_order(self, sample_train):
        shuffled = sample_train.sample(frac=1, random_state=0)
        sorted_df = sort_by_date(shuffled)
        assert sorted_df["date"].is_monotonic_increasing

    def test_perfect_intervals(self, sample_train):
        sorted_df = sort_by_date(sample_train)
        report = validate_intervals(sorted_df, label="test")
        assert report.is_perfectly_sequential
        assert report.gap_count == 0

    def test_detects_gaps(self, sample_train):
        # Remove a row to create a gap
        df = sample_train.drop(index=5).reset_index(drop=True)
        df = sort_by_date(df)
        report = validate_intervals(df, label="test")
        assert report.gap_count >= 1
        assert not report.is_perfectly_sequential


# ──────────────────────────────────────────────────────────────
# TESTS — scaler
# ──────────────────────────────────────────────────────────────

class TestScaler:
    def _prepare(self, sample_train):
        df = extract_temporal_features(sample_train)
        df = extract_domain_features(df)
        return df

    def test_fit_returns_bundle(self, sample_train):
        df = self._prepare(sample_train)
        bundle = fit_scalers(df)
        assert bundle.feature_scaler is not None
        assert bundle.target_scaler is not None
        assert len(bundle.feature_columns) == len(cfg.FEATURES_TO_SCALE)

    def test_train_zero_mean(self, sample_train):
        df = self._prepare(sample_train)
        bundle = fit_scalers(df)
        scaled = apply_scaling(df, bundle, is_train=True)
        for col in cfg.FEATURES_TO_SCALE:
            assert abs(scaled[col].mean()) < 0.01, f"{col} mean != 0"

    def test_train_unit_std(self, sample_train):
        df = self._prepare(sample_train)
        bundle = fit_scalers(df)
        scaled = apply_scaling(df, bundle, is_train=True)
        for col in cfg.FEATURES_TO_SCALE:
            assert abs(scaled[col].std() - 1.0) < 0.05, f"{col} std != 1"

    def test_spread_scaled_exists(self, sample_train):
        df = self._prepare(sample_train)
        bundle = fit_scalers(df)
        scaled = apply_scaling(df, bundle, is_train=True)
        assert "spread_scaled" in scaled.columns

    def test_test_missing_imbalances(self, sample_test):
        """Test set has no imbalances column — scaling should still work."""
        train_stub = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=10, freq="15min"),
            "wind": np.random.rand(10) * 5000,
            "solar": np.random.rand(10) * 3000,
            "load": np.random.rand(10) * 50000,
            "imbalances": np.random.rand(10) * 200,
            "spread": np.random.randn(10) * 100,
            "net_load": np.random.rand(10) * 40000,
            "renewable_ratio": np.random.rand(10),
            "wind_solar_ratio": np.random.rand(10) * 10,
            "total_renewable": np.random.rand(10) * 8000,
        })
        test_df = extract_temporal_features(sample_test)
        test_df = extract_domain_features(test_df)

        bundle = fit_scalers(train_stub)
        scaled = apply_scaling(test_df, bundle, is_train=False)
        assert "spread_scaled" not in scaled.columns
        assert "wind" in scaled.columns


# ──────────────────────────────────────────────────────────────
# TESTS — clustering
# ──────────────────────────────────────────────────────────────

class TestClustering:
    def _prepare_scaled(self, sample_train):
        df = extract_temporal_features(sample_train)
        df = extract_domain_features(df)
        bundle = fit_scalers(df)
        return apply_scaling(df, bundle, is_train=True)

    def test_elbow_result_length(self, sample_train):
        scaled = self._prepare_scaled(sample_train)
        elbow = run_kmeans_elbow(scaled, k_range=range(2, 6))
        assert len(elbow.k_values) == 4
        assert len(elbow.inertias) == 4

    def test_inertia_decreasing(self, sample_train):
        scaled = self._prepare_scaled(sample_train)
        elbow = run_kmeans_elbow(scaled, k_range=range(2, 6))
        # Inertia should generally decrease as k increases
        assert elbow.inertias[0] >= elbow.inertias[-1]

    def test_fit_returns_model(self, sample_train):
        scaled = self._prepare_scaled(sample_train)
        result = fit_kmeans(scaled, n_clusters=3)
        assert result.n_clusters == 3
        assert len(result.regime_columns) == 3

    def test_assign_creates_columns(self, sample_train):
        scaled = self._prepare_scaled(sample_train)
        result = fit_kmeans(scaled, n_clusters=3)
        out = assign_regimes(scaled, result)
        assert "market_regime" in out.columns
        for i in range(3):
            assert f"regime_{i}" in out.columns

    def test_regime_labels_valid(self, sample_train):
        scaled = self._prepare_scaled(sample_train)
        result = fit_kmeans(scaled, n_clusters=3)
        out = assign_regimes(scaled, result)
        assert set(out["market_regime"].unique()).issubset({0, 1, 2})

    def test_one_hot_sums_to_one(self, sample_train):
        scaled = self._prepare_scaled(sample_train)
        result = fit_kmeans(scaled, n_clusters=3)
        out = assign_regimes(scaled, result)
        regime_cols = [f"regime_{i}" for i in range(3)]
        row_sums = out[regime_cols].sum(axis=1)
        assert (row_sums == 1).all()
