"""
probability.py — Convert raw spread predictions to P(spread > 0).
==================================================================
The Kaggle competition expects *probability* forecasts, not raw EUR
values.  This module provides two conversion strategies:

1. **Logistic calibration** (default, recommended) — fits a
   ``LogisticRegression`` on (predicted_spread_eur → actual_sign) using
   the validation set, then applies it to test predictions.  This is the
   same approach used by the ARIMA baseline model.

2. **Sigmoid with temperature** — a model-free fallback that applies
   ``sigma(spread / T)`` where ``T`` is either given or estimated from
   the validation standard deviation.

Design notes:
    * The calibrator is always *fitted on validation data* to avoid
      leaking test information into the probability mapping.
    * Both ``fit`` and ``predict_proba`` accept 1-D numpy arrays.
    * A convenience function ``spread_to_proba_simple`` is provided
      for quick one-shot conversion without fitting.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


# ──────────────────────────────────────────────────────────────
# LOGISTIC CALIBRATION (recommended)
# ──────────────────────────────────────────────────────────────

class SpreadCalibrator:
    """
    Calibrates raw spread (EUR) predictions to P(spread > 0).

    Uses sklearn ``LogisticRegression`` fitted on validation
    predictions vs. actual binary sign.

    Parameters
    ----------
    C : float
        Regularisation strength (inverse). Higher = less regularisation.
        Default ``1e6`` gives an essentially unregularised fit.

    Examples
    --------
    >>> cal = SpreadCalibrator()
    >>> cal.fit(val_preds_eur, val_targets_eur)
    >>> test_probs = cal.predict_proba(test_preds_eur)
    """

    def __init__(self, C: float = 1e6):
        self.C = C
        self.model = LogisticRegression(C=C, solver="lbfgs", max_iter=1000)
        self._fitted = False
        self.coef_: float | None = None
        self.intercept_: float | None = None
        self.val_directional_acc: float | None = None

    def fit(
        self,
        spread_predictions_eur: np.ndarray,
        actual_spread_eur: np.ndarray,
    ) -> "SpreadCalibrator":
        """
        Fit the calibrator on validation data.

        Parameters
        ----------
        spread_predictions_eur : np.ndarray
            Model's raw spread predictions in EUR, shape ``(N,)``.
        actual_spread_eur : np.ndarray
            True spread values in EUR, shape ``(N,)``.

        Returns
        -------
        self
        """
        X = np.asarray(spread_predictions_eur).reshape(-1, 1)
        y = (np.asarray(actual_spread_eur) > 0).astype(int)

        self.model.fit(X, y)
        self._fitted = True

        self.coef_ = float(self.model.coef_[0][0])
        self.intercept_ = float(self.model.intercept_[0])

        # Directional accuracy on validation set
        pred_sign = (spread_predictions_eur > 0).astype(int)
        self.val_directional_acc = float(np.mean(pred_sign == y))

        return self

    def predict_proba(self, spread_predictions_eur: np.ndarray) -> np.ndarray:
        """
        Convert raw spread (EUR) to calibrated P(spread > 0).

        Parameters
        ----------
        spread_predictions_eur : np.ndarray
            Shape ``(N,)``.

        Returns
        -------
        np.ndarray
            Probabilities in [0, 1], shape ``(N,)``.
        """
        if not self._fitted:
            raise RuntimeError(
                "SpreadCalibrator.fit() must be called before predict_proba()."
            )
        X = np.asarray(spread_predictions_eur).reshape(-1, 1)
        return self.model.predict_proba(X)[:, 1]

    @property
    def is_fitted(self) -> bool:
        return self._fitted


# ──────────────────────────────────────────────────────────────
# SIMPLE SIGMOID FALLBACK
# ──────────────────────────────────────────────────────────────

def spread_to_proba_simple(
    spread_eur: np.ndarray,
    temperature: float | None = None,
) -> np.ndarray:
    """
    Model-free conversion: ``P = sigmoid(spread / temperature)``.

    Parameters
    ----------
    spread_eur : np.ndarray
        Raw spread predictions in EUR.
    temperature : float, optional
        Scaling factor.  If ``None``, uses ``std(spread_eur)`` so that
        ±1 std maps to roughly [0.27, 0.73].

    Returns
    -------
    np.ndarray
        Probabilities in [0, 1].
    """
    arr = np.asarray(spread_eur, dtype=np.float64)
    if temperature is None:
        temperature = max(float(np.std(arr)), 1e-8)
    scaled = arr / temperature
    return 1.0 / (1.0 + np.exp(-scaled))


# ──────────────────────────────────────────────────────────────
# PRINTING HELPERS
# ──────────────────────────────────────────────────────────────

def print_calibration_summary(
    calibrator: SpreadCalibrator,
    probs: np.ndarray,
    label: str = "Test",
) -> None:
    """Print a concise summary of calibration results."""
    print(f"  Calibration coefficient:   {calibrator.coef_:.4f}")
    print(f"  Calibration intercept:     {calibrator.intercept_:.4f}")
    print(f"  Val directional accuracy:  {calibrator.val_directional_acc:.4f}")
    print(f"  {label} P(spread>0) range:  "
          f"[{probs.min():.4f}, {probs.max():.4f}]")
    print(f"  {label} P(spread>0) mean:   {probs.mean():.4f}")
