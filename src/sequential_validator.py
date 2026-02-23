"""
sequential_validator.py — Maintain 15-minute interval integrity.
================================================================
Responsibilities:
    1. Sort DataFrames strictly by the ``date`` column.
    2. Verify that every consecutive pair of timestamps is exactly
       15 minutes apart.
    3. Report any gaps (e.g. DST transitions, year boundaries) so the
       downstream pipeline can decide how to handle them.

Why this matters:
    The neural network will later receive windowed sequences of
    consecutive rows.  If the data is not strictly ordered or
    contains hidden gaps, the model would silently learn from
    discontinuous intervals — a subtle but damaging data-quality bug.
"""

from dataclasses import dataclass

import pandas as pd

import config as cfg


# ──────────────────────────────────────────────────────────────
# DATA CLASSES
# ──────────────────────────────────────────────────────────────

@dataclass
class IntervalReport:
    """Result of a 15-minute interval validation check."""
    total_intervals: int
    correct_intervals: int
    gap_count: int
    gap_locations: pd.Series          # datetime values where gaps occur
    is_perfectly_sequential: bool


# ──────────────────────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────────────────────

def sort_by_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort a DataFrame by the ``date`` column and reset the index.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``date`` column.

    Returns
    -------
    pd.DataFrame
        Sorted copy with a clean integer index starting from 0.
    """
    return df.sort_values("date").reset_index(drop=True)


def validate_intervals(df: pd.DataFrame, label: str = "data") -> IntervalReport:
    """
    Check that every consecutive timestamp is exactly 15 minutes apart.

    Parameters
    ----------
    df : pd.DataFrame
        Must be **already sorted** by ``date``.
    label : str
        Human-readable name used in log messages (e.g. ``"train"``).

    Returns
    -------
    IntervalReport
        Structured result with counts and gap locations.
    """
    expected = pd.Timedelta(minutes=cfg.EXPECTED_INTERVAL_MINUTES)
    diffs = df["date"].diff().dropna()

    correct = int((diffs == expected).sum())
    total = len(diffs)
    gap_count = total - correct

    gap_mask = diffs != expected
    gap_locs = df.loc[gap_mask.index[gap_mask], "date"]

    report = IntervalReport(
        total_intervals=total,
        correct_intervals=correct,
        gap_count=gap_count,
        gap_locations=gap_locs,
        is_perfectly_sequential=(gap_count == 0),
    )

    return report


def print_interval_report(report: IntervalReport, label: str = "data") -> None:
    """Pretty-print the validation results to stdout."""
    print(
        f"  {label}: {report.correct_intervals}/{report.total_intervals} "
        f"intervals are exactly {cfg.EXPECTED_INTERVAL_MINUTES} min"
    )
    if report.gap_count > 0:
        print(f"  WARNING: {report.gap_count} gap(s) detected!")
        print(f"  Gap locations (up to 10):")
        for dt in report.gap_locations.head(10):
            print(f"    - {dt}")
    else:
        print("  No gaps — sequential integrity fully preserved.")
