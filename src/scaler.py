"""
scaler.py — Normalise and scale numerical features.
=====================================================
Responsibilities:
    1. Fit a ``StandardScaler`` on the **training data only** to prevent
       data leakage from future (test) observations.
    2. Transform both training and test sets using the same fitted
       parameters (mean, std).
    3. Provide a separate scaler for the target variable (``spread``)
       so that predictions can be inverse-transformed back to the
       original euro scale.

Design notes:
    * The test set does not contain ``imbalances``.  The scaler is
      fitted on all ``FEATURES_TO_SCALE`` (which includes imbalances),
      but at transform time only the columns present in the test
      DataFrame are scaled — each one using its own training-derived
      mean and std.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import config as cfg


# ──────────────────────────────────────────────────────────────
# DATA CLASSES
# ──────────────────────────────────────────────────────────────

@dataclass
class ScalerBundle:
    """Container for all fitted scalers and their metadata."""
    feature_scaler: StandardScaler
    target_scaler: StandardScaler
    feature_columns: list[str]


# ──────────────────────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────────────────────

def fit_scalers(
    train: pd.DataFrame,
    features_to_scale: list[str] | None = None,
    target_col: str | None = None,
) -> ScalerBundle:
    """
    Fit StandardScalers on the training data.

    Parameters
    ----------
    train : pd.DataFrame
        Training set **before** scaling (must contain all feature
        columns and the target).
    features_to_scale : list[str], optional
        Columns to normalise. Defaults to ``config.FEATURES_TO_SCALE``.
    target_col : str, optional
        Target column name. Defaults to ``config.TARGET``.

    Returns
    -------
    ScalerBundle
        Contains the fitted feature scaler, target scaler, and the
        list of column names.
    """
    features_to_scale = features_to_scale or cfg.FEATURES_TO_SCALE
    target_col = target_col or cfg.TARGET

    # Feature scaler
    feature_scaler = StandardScaler()
    feature_scaler.fit(train[features_to_scale])

    # Target scaler
    target_scaler = StandardScaler()
    target_scaler.fit(train[[target_col]])

    return ScalerBundle(
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        feature_columns=features_to_scale,
    )


def apply_scaling(
    df: pd.DataFrame,
    bundle: ScalerBundle,
    is_train: bool = True,
) -> pd.DataFrame:
    """
    Scale a DataFrame using a previously fitted ScalerBundle.

    For the **training set** (``is_train=True``):
        * All ``bundle.feature_columns`` are transformed.
        * A ``spread_scaled`` column is added from the target scaler.

    For the **test set** (``is_train=False``):
        * Only the feature columns that actually exist in *df* are
          transformed, each using its own training-derived (mean, std).
        * No target column is created (test has no ground truth).

    Parameters
    ----------
    df : pd.DataFrame
        Raw (unscaled) DataFrame.
    bundle : ScalerBundle
        Fitted scalers from ``fit_scalers()``.
    is_train : bool
        Whether this is the training set.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with scaled values in-place for the relevant
        columns (and ``spread_scaled`` if training).
    """
    out = df.copy()
    scaler = bundle.feature_scaler
    all_cols = bundle.feature_columns

    if is_train:
        # Transform all features in one go
        out[all_cols] = scaler.transform(df[all_cols])
        # Scale the target
        out["spread_scaled"] = bundle.target_scaler.transform(
            df[[cfg.TARGET]]
        )
    else:
        # Test may be missing some columns (e.g. imbalances)
        available = [c for c in all_cols if c in df.columns]
        for col in available:
            idx = all_cols.index(col)
            out[col] = (df[col] - scaler.mean_[idx]) / scaler.scale_[idx]

    return out


def print_scaling_summary(
    train_scaled: pd.DataFrame,
    train_raw: pd.DataFrame,
    bundle: ScalerBundle,
) -> None:
    """Print descriptive statistics to verify zero-mean / unit-variance."""
    cols = bundle.feature_columns
    stats = train_scaled[cols].describe().loc[["mean", "std"]].round(4)

    print(f"  Scaler fitted on {len(cols)} features from training data")
    print(f"  Scaled features: {cols}")
    print(f"  Train scaled stats (should be ~0 mean, ~1 std):")
    print(f"{stats.to_string()}")
    print()
    print(
        f"  Target (spread) — original range: "
        f"[{train_raw[cfg.TARGET].min():.2f}, "
        f"{train_raw[cfg.TARGET].max():.2f}]"
    )
    print(
        f"  Target (spread) — scaled range:   "
        f"[{train_scaled['spread_scaled'].min():.4f}, "
        f"{train_scaled['spread_scaled'].max():.4f}]"
    )
