"""
tests/test_probability.py — Tests for the probability conversion module.
=========================================================================
Validates SpreadCalibrator (logistic calibration) and the simple
sigmoid fallback function.
"""

import numpy as np
import pytest

from src.probability import SpreadCalibrator, spread_to_proba_simple


# ──────────────────────────────────────────────────────────────
# FIXTURES
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_val_data():
    """
    Generate synthetic validation data where positive spread → positive sign.
    Adds noise so calibration has something meaningful to learn.
    """
    rng = np.random.RandomState(42)
    n = 500
    actual = rng.randn(n) * 200   # spread in EUR
    # Predictions are noisy versions of actual
    preds = actual + rng.randn(n) * 50
    return preds, actual


@pytest.fixture
def fitted_calibrator(synthetic_val_data):
    """Return a SpreadCalibrator fitted on synthetic data."""
    preds, actual = synthetic_val_data
    cal = SpreadCalibrator()
    cal.fit(preds, actual)
    return cal


# ──────────────────────────────────────────────────────────────
# SPREAD CALIBRATOR TESTS
# ──────────────────────────────────────────────────────────────

class TestSpreadCalibrator:
    """Tests for the SpreadCalibrator class."""

    def test_fit_returns_self(self, synthetic_val_data):
        preds, actual = synthetic_val_data
        cal = SpreadCalibrator()
        result = cal.fit(preds, actual)
        assert result is cal

    def test_is_fitted_flag(self, synthetic_val_data):
        preds, actual = synthetic_val_data
        cal = SpreadCalibrator()
        assert not cal.is_fitted
        cal.fit(preds, actual)
        assert cal.is_fitted

    def test_predict_before_fit_raises(self):
        cal = SpreadCalibrator()
        with pytest.raises(RuntimeError, match="fit.*must be called"):
            cal.predict_proba(np.array([1.0, 2.0]))

    def test_output_shape(self, fitted_calibrator, synthetic_val_data):
        preds, _ = synthetic_val_data
        probs = fitted_calibrator.predict_proba(preds)
        assert probs.shape == preds.shape

    def test_output_in_zero_one(self, fitted_calibrator, synthetic_val_data):
        preds, _ = synthetic_val_data
        probs = fitted_calibrator.predict_proba(preds)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_positive_spread_gives_higher_prob(self, fitted_calibrator):
        """Large positive spread should give P > 0.5; large negative < 0.5."""
        test = np.array([500.0, -500.0])
        probs = fitted_calibrator.predict_proba(test)
        assert probs[0] > 0.5, "Positive spread should give P > 0.5"
        assert probs[1] < 0.5, "Negative spread should give P < 0.5"

    def test_monotonically_increasing(self, fitted_calibrator):
        """Increasing spread should give non-decreasing probability."""
        spread_range = np.linspace(-1000, 1000, 100)
        probs = fitted_calibrator.predict_proba(spread_range)
        # Allow tiny numerical tolerance
        assert np.all(np.diff(probs) >= -1e-10)

    def test_zero_spread_near_half(self, fitted_calibrator):
        """Spread of 0 should produce probability near 0.5."""
        probs = fitted_calibrator.predict_proba(np.array([0.0]))
        assert 0.3 < probs[0] < 0.7

    def test_coef_and_intercept_stored(self, fitted_calibrator):
        assert fitted_calibrator.coef_ is not None
        assert fitted_calibrator.intercept_ is not None
        assert isinstance(fitted_calibrator.coef_, float)
        assert isinstance(fitted_calibrator.intercept_, float)

    def test_directional_accuracy_stored(self, fitted_calibrator):
        assert fitted_calibrator.val_directional_acc is not None
        assert 0.0 <= fitted_calibrator.val_directional_acc <= 1.0

    def test_directional_accuracy_reasonable(self, fitted_calibrator):
        """With correlated predictions, accuracy should exceed random (0.5)."""
        assert fitted_calibrator.val_directional_acc > 0.5

    def test_single_prediction(self, fitted_calibrator):
        """Should work with a single value."""
        probs = fitted_calibrator.predict_proba(np.array([100.0]))
        assert probs.shape == (1,)
        assert 0.0 <= probs[0] <= 1.0


# ──────────────────────────────────────────────────────────────
# SIGMOID FALLBACK TESTS
# ──────────────────────────────────────────────────────────────

class TestSigmoidFallback:
    """Tests for spread_to_proba_simple()."""

    def test_output_shape(self):
        arr = np.array([10.0, -10.0, 0.0, 50.0])
        probs = spread_to_proba_simple(arr)
        assert probs.shape == arr.shape

    def test_output_in_zero_one(self):
        arr = np.linspace(-1000, 1000, 200)
        probs = spread_to_proba_simple(arr)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_zero_gives_half(self):
        probs = spread_to_proba_simple(np.array([0.0]))
        assert abs(probs[0] - 0.5) < 1e-10

    def test_positive_above_half(self):
        probs = spread_to_proba_simple(np.array([100.0]), temperature=50)
        assert probs[0] > 0.5

    def test_negative_below_half(self):
        probs = spread_to_proba_simple(np.array([-100.0]), temperature=50)
        assert probs[0] < 0.5

    def test_custom_temperature(self):
        """Lower temperature → sharper sigmoid (more extreme probabilities)."""
        arr = np.array([50.0])
        prob_low_t = spread_to_proba_simple(arr, temperature=10)
        prob_high_t = spread_to_proba_simple(arr, temperature=100)
        # Lower temp should give higher probability for positive spread
        assert prob_low_t[0] > prob_high_t[0]

    def test_auto_temperature(self):
        """When temperature is None, uses std of input."""
        arr = np.array([100.0, -100.0, 50.0, -50.0])
        probs = spread_to_proba_simple(arr)
        # Should produce reasonable probabilities (not all 0 or 1)
        assert np.all(probs > 0.01)
        assert np.all(probs < 0.99)

    def test_symmetric(self):
        """P(+x) + P(-x) should equal 1.0 (sigmoid symmetry)."""
        arr = np.array([100.0, -100.0])
        probs = spread_to_proba_simple(arr, temperature=50)
        assert abs(probs[0] + probs[1] - 1.0) < 1e-10
