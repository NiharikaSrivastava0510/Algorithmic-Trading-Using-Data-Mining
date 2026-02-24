"""
lag_features.py — Create lagged variables for short-term momentum.
==================================================================
Responsibilities:
    1. Generate lagged copies of historical forecast and target columns
       at offsets of 1 interval (15 min), 4 intervals (1 hour), and
       96 intervals (24 hours).
    2. Handle NaN values introduced at the start of the series where
       lag history is unavailable.

Why lags matter:
    Energy markets exhibit strong autocorrelation — the spread at
    time *t* is heavily influenced by what happened at *t-15min*,
    *t-1h*, and *t-24h* (same time yesterday).  Providing these as
    explicit features lets the neural network capture short-term
    momentum and recurring daily patterns without relying solely on
    its internal memory.

Design notes:
    * Lag features are created on **raw (unscaled)** data so that the
      subsequent StandardScaler can normalise them together with the
      original columns.
    * The DataFrame must be **sorted by date** before calling these
      functions — ``sequential_validator.sort_by_date()`` should have
      run first.
    * NaN rows (from the first 96 rows of training / first rows of
      test) are forward-filled, then any remaining leading NaNs are
      back-filled.  This is safe because the NaN count is small
      relative to the 140 k training samples.
"""

import pandas as pd

import config as cfg


# ──────────────────────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────────────────────

def create_lag_features(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    lag_offsets: dict[int, str] | None = None,
) -> pd.DataFrame:
    """
    Add lagged columns to a DataFrame.

    For each ``(col, k)`` pair, a new column ``{col}_lag_{k}`` is
    created by shifting ``col`` down by ``k`` rows.

    Parameters
    ----------
    df : pd.DataFrame
        Must be sorted chronologically by ``date``.
    columns : list[str], optional
        Columns to lag.  Defaults to ``config.LAG_COLUMNS``.
        Only columns that actually exist in *df* are processed
        (gracefully skips e.g. ``spread`` in the test set).
    lag_offsets : dict[int, str], optional
        Mapping of ``{offset_rows: label}``.
        Defaults to ``config.LAG_OFFSETS``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with the new lag columns appended.
    """
    columns = columns or (cfg.LAG_COLUMNS + cfg.LAG_COLUMNS_TRAIN_ONLY)
    lag_offsets = lag_offsets or cfg.LAG_OFFSETS

    out = df.copy()

    for col in columns:
        if col not in out.columns:
            continue
        for k in lag_offsets:
            lag_name = f"{col}_lag_{k}"
            out[lag_name] = out[col].shift(k)

    return out


def fill_lag_nans(df: pd.DataFrame, lag_columns: list[str]) -> pd.DataFrame:
    """
    Fill NaN values in lag columns using forward-fill then back-fill.

    Forward-fill propagates the most recent valid value into NaN
    positions.  Back-fill handles the very first rows where no
    history exists at all.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing lag columns with potential NaNs.
    lag_columns : list[str]
        Names of the lag columns to fill.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with NaN-free lag columns.
    """
    out = df.copy()
    present = [c for c in lag_columns if c in out.columns]
    out[present] = out[present].ffill().bfill()
    return out


def print_lag_summary(df: pd.DataFrame, is_train: bool = True) -> None:
    """Print a summary of the created lag features."""
    if is_train:
        all_lag = cfg.LAG_FEATURES + cfg.LAG_FEATURES_TRAIN_ONLY
    else:
        all_lag = cfg.LAG_FEATURES

    present = [c for c in all_lag if c in df.columns]
    missing_vals = df[present].isnull().sum().sum() if present else 0

    print(f"  Lag features created: {len(present)}")
    print(f"  Lag offsets: {cfg.LAG_OFFSETS}")
    print(f"  Columns lagged: {cfg.LAG_COLUMNS}", end="")
    if is_train:
        print(f" + {cfg.LAG_COLUMNS_TRAIN_ONLY} (train-only)")
    else:
        print(f"  (spread lags excluded — not available at test time)")
    print(f"  Remaining NaN values after fill: {missing_vals}")
