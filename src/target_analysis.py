"""
target_analysis.py — Target variable definition and analysis.
==============================================================
Responsibilities:
    1. Formally define the target variable: the electricity market
       *spread*, which is the difference between the imbalance price
       and the day-ahead price for each 15-minute interval.
    2. Compute and print summary statistics (mean, std, skew,
       kurtosis, percentiles) to characterise the target distribution.
    3. Analyse autocorrelation of the spread to justify the lag
       feature offsets chosen in Step 2(b).

The target ``spread`` is already provided in ``train.csv``.  This
module analyses its properties so we can make informed decisions
about loss functions, outlier handling, and evaluation metrics.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config as cfg


# ──────────────────────────────────────────────────────────────
# DATA CLASSES
# ──────────────────────────────────────────────────────────────

@dataclass
class TargetStats:
    """Comprehensive summary statistics for the target variable."""
    count: int
    mean: float
    std: float
    min: float
    q1: float
    median: float
    q3: float
    max: float
    skewness: float
    kurtosis: float
    iqr: float
    pct_positive: float
    pct_negative: float
    pct_zero: float


# ──────────────────────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────────────────────

def compute_target_stats(train: pd.DataFrame) -> TargetStats:
    """
    Compute summary statistics for the spread column.

    Parameters
    ----------
    train : pd.DataFrame
        Training data with a ``spread`` column.

    Returns
    -------
    TargetStats
        Structured statistics.
    """
    s = train[cfg.TARGET]
    q1, median, q3 = s.quantile([0.25, 0.50, 0.75])
    n = len(s)

    return TargetStats(
        count=n,
        mean=float(s.mean()),
        std=float(s.std()),
        min=float(s.min()),
        q1=float(q1),
        median=float(median),
        q3=float(q3),
        max=float(s.max()),
        skewness=float(s.skew()),
        kurtosis=float(s.kurtosis()),
        iqr=float(q3 - q1),
        pct_positive=float((s > 0).sum() / n * 100),
        pct_negative=float((s < 0).sum() / n * 100),
        pct_zero=float((s == 0).sum() / n * 100),
    )


def compute_autocorrelation(
    train: pd.DataFrame,
    lags: list[int] | None = None,
) -> dict[int, float]:
    """
    Compute the autocorrelation of the spread at specified lag offsets.

    Parameters
    ----------
    train : pd.DataFrame
        Training data with a ``spread`` column, sorted by date.
    lags : list[int], optional
        Lag offsets (in number of 15-min intervals).
        Defaults to the keys of ``config.LAG_OFFSETS`` plus a few extras.

    Returns
    -------
    dict[int, float]
        ``{lag: autocorrelation_coefficient}``
    """
    if lags is None:
        lags = sorted(set(list(cfg.LAG_OFFSETS.keys()) + [2, 8, 48, 192]))

    s = train[cfg.TARGET]
    return {lag: float(s.autocorr(lag=lag)) for lag in lags}


def print_target_summary(stats: TargetStats, acf: dict[int, float]) -> None:
    """Pretty-print the target analysis to stdout."""
    print("  Target variable: spread (imbalance price − day-ahead price)")
    print()
    print("  Distribution statistics:")
    print(f"    Count:     {stats.count:>10,}")
    print(f"    Mean:      {stats.mean:>10.2f} EUR")
    print(f"    Std:       {stats.std:>10.2f} EUR")
    print(f"    Min:       {stats.min:>10.2f} EUR")
    print(f"    Q1 (25%):  {stats.q1:>10.2f} EUR")
    print(f"    Median:    {stats.median:>10.2f} EUR")
    print(f"    Q3 (75%):  {stats.q3:>10.2f} EUR")
    print(f"    Max:       {stats.max:>10.2f} EUR")
    print(f"    IQR:       {stats.iqr:>10.2f} EUR")
    print(f"    Skewness:  {stats.skewness:>10.4f}")
    print(f"    Kurtosis:  {stats.kurtosis:>10.4f}")
    print()
    print(f"    Positive:  {stats.pct_positive:>5.1f}% of intervals")
    print(f"    Negative:  {stats.pct_negative:>5.1f}% of intervals")
    print(f"    Zero:      {stats.pct_zero:>5.1f}% of intervals")
    print()
    print("  Autocorrelation (justifies lag feature selection):")
    for lag, r in sorted(acf.items()):
        label = cfg.LAG_OFFSETS.get(lag, "")
        bar = "+" * int(abs(r) * 40)
        sign = "+" if r >= 0 else "-"
        tag = f"  <- {label}" if label else ""
        print(f"    lag {lag:>3}: {sign}{abs(r):.4f}  |{bar}{tag}")
