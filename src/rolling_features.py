"""
rolling_features.py — Rolling averages and standard deviations.
================================================================
Responsibilities:
    1. Compute rolling mean and rolling standard deviation over
       1-hour (4-interval) and 24-hour (96-interval) windows for
       load and weather forecast columns.
    2. Handle NaN values at the start of the rolling window.

Why rolling statistics matter:
    * **Rolling means** smooth out noise and let the model see the
      underlying trend over the past hour or day.
    * **Rolling standard deviations** quantify recent volatility —
      a sudden spike in wind variability, for example, signals an
      uncertain market condition where the spread may swing.

Design notes:
    * Rolling stats are computed with ``min_periods=1`` so that the
      first rows produce valid (if noisy) values rather than NaN.
    * The DataFrame must be **sorted by date** before calling these
      functions.
    * All computations use raw (unscaled) data; the scaler normalises
      them afterwards.
"""

import pandas as pd

import config as cfg


# ──────────────────────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────────────────────

def create_rolling_features(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    windows: dict[int, str] | None = None,
) -> pd.DataFrame:
    """
    Add rolling mean and rolling std columns to a DataFrame.

    For each ``(col, w)`` pair, two columns are created:
        * ``{col}_rmean_{w}``  — rolling mean over the past *w* rows.
        * ``{col}_rstd_{w}``   — rolling std over the past *w* rows.

    Parameters
    ----------
    df : pd.DataFrame
        Must be sorted chronologically by ``date``.
    columns : list[str], optional
        Columns to compute rolling stats for.
        Defaults to ``config.ROLLING_COLUMNS + config.ROLLING_COLUMNS_TRAIN_ONLY``.
        Only columns present in *df* are processed.
    windows : dict[int, str], optional
        Mapping of ``{window_size: label}``.
        Defaults to ``config.ROLLING_WINDOWS``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with the new rolling columns appended.
    """
    columns = columns or (cfg.ROLLING_COLUMNS + cfg.ROLLING_COLUMNS_TRAIN_ONLY)
    windows = windows or cfg.ROLLING_WINDOWS

    out = df.copy()

    for col in columns:
        if col not in out.columns:
            continue
        for w in windows:
            roller = out[col].rolling(window=w, min_periods=1)
            out[f"{col}_rmean_{w}"] = roller.mean()
            out[f"{col}_rstd_{w}"] = roller.std()

    # Rolling std with min_periods=1 produces NaN for the very first
    # row (std of a single value is undefined).  Fill with 0 — zero
    # volatility is the safest assumption when we have no history.
    rolling_std_cols = [
        c for c in out.columns if c.endswith(tuple(f"_rstd_{w}" for w in windows))
    ]
    out[rolling_std_cols] = out[rolling_std_cols].fillna(0)

    return out


def print_rolling_summary(df: pd.DataFrame, is_train: bool = True) -> None:
    """Print a summary of the created rolling features."""
    if is_train:
        all_rolling = cfg.ROLLING_FEATURES + cfg.ROLLING_FEATURES_TRAIN_ONLY
    else:
        all_rolling = cfg.ROLLING_FEATURES

    present = [c for c in all_rolling if c in df.columns]
    missing_vals = df[present].isnull().sum().sum() if present else 0

    # Split into mean and std for clearer reporting
    rmean = [c for c in present if "_rmean_" in c]
    rstd = [c for c in present if "_rstd_" in c]

    print(f"  Rolling features created: {len(present)}")
    print(f"    - Rolling means:  {len(rmean)}  {rmean}")
    print(f"    - Rolling stds:   {len(rstd)}  {rstd}")
    print(f"  Windows: {cfg.ROLLING_WINDOWS}")
    print(f"  Columns: {cfg.ROLLING_COLUMNS}", end="")
    if is_train:
        print(f" + {cfg.ROLLING_COLUMNS_TRAIN_ONLY} (train-only)")
    else:
        print(f"  (spread rolling excluded — not available at test time)")
    print(f"  Remaining NaN values after fill: {missing_vals}")
