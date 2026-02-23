"""
feature_engineering.py — Temporal and domain feature extraction.
================================================================
Responsibilities:
    1. Extract cyclical time features from the ``date`` column so
       the neural network sees smooth periodic signals rather than
       raw integers with artificial discontinuities.
    2. Derive energy-market domain features (net load, renewable
       ratios, etc.) that encode known physical drivers of the spread.

Design notes:
    * Every function receives a DataFrame and returns a **copy** — the
      caller's original data is never mutated.
    * Feature names are defined centrally in ``config.py``; functions
      here create those exact column names.
"""

import numpy as np
import pandas as pd

import config as cfg


# ──────────────────────────────────────────────────────────────
# TEMPORAL FEATURES
# ──────────────────────────────────────────────────────────────

def extract_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create time-based features from the ``date`` column.

    Features created
    ----------------
    +-----------------+----------------------------------------------+
    | Column          | Description                                  |
    +-----------------+----------------------------------------------+
    | hour            | Hour of the day (0-23)                       |
    | minute          | Minute of the hour (0, 15, 30, 45)           |
    | day_of_week     | Monday=0 … Sunday=6                          |
    | month           | Month of the year (1-12)                     |
    | day_of_year     | Day of the year (1-366)                      |
    | quarter         | Calendar quarter (1-4)                       |
    | interval_of_day | 15-min slot index (0-95)                     |
    | is_weekend      | 1 if Saturday/Sunday, else 0                 |
    | hour_sin/cos    | Cyclical encoding of hour                    |
    | month_sin/cos   | Cyclical encoding of month                   |
    | dow_sin/cos     | Cyclical encoding of day of week             |
    | interval_sin/cos| Cyclical encoding of 15-min interval         |
    +-----------------+----------------------------------------------+

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``date`` column of dtype ``datetime64``.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with the new columns appended.
    """
    df = df.copy()

    # ── raw integer components ──
    df["hour"] = df["date"].dt.hour
    df["minute"] = df["date"].dt.minute
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["quarter"] = df["date"].dt.quarter

    # 15-minute slot within the day (0–95)
    df["interval_of_day"] = df["hour"] * 4 + df["minute"] // 15

    # Binary weekend flag
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # ── cyclical sin / cos encoding ──
    # Mapping each periodic variable onto the unit circle avoids the
    # discontinuity a neural network would otherwise see (e.g. the
    # jump from hour 23 to hour 0).
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)

    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    df["interval_sin"] = np.sin(
        2 * np.pi * df["interval_of_day"] / cfg.INTERVALS_PER_DAY
    )
    df["interval_cos"] = np.cos(
        2 * np.pi * df["interval_of_day"] / cfg.INTERVALS_PER_DAY
    )

    return df


# ──────────────────────────────────────────────────────────────
# DOMAIN FEATURES
# ──────────────────────────────────────────────────────────────

def extract_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive energy-market features from the core forecast columns.

    Features created
    ----------------
    +-------------------+--------------------------------------------------+
    | Column            | Description                                      |
    +-------------------+--------------------------------------------------+
    | net_load          | load - wind - solar (residual demand after        |
    |                   | subtracting renewables; primary spread driver)    |
    | renewable_ratio   | (wind + solar) / load (share of demand met by    |
    |                   | renewables)                                       |
    | wind_solar_ratio  | wind / (solar + 1)  (day vs night generation     |
    |                   | balance; +1 prevents division-by-zero)            |
    | total_renewable   | wind + solar (total green generation)             |
    +-------------------+--------------------------------------------------+

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``wind``, ``solar``, and ``load`` columns.

    Returns
    -------
    pd.DataFrame
        Copy of *df* with the new columns appended.
    """
    df = df.copy()

    # Net load — the demand that must be met by conventional generation
    df["net_load"] = df["load"] - df["wind"] - df["solar"]

    # Renewable share of total load
    df["renewable_ratio"] = (
        (df["wind"] + df["solar"])
        / df["load"].replace(0, np.nan)
    )
    df["renewable_ratio"] = df["renewable_ratio"].fillna(0)

    # Wind-to-solar ratio (captures day/night generation mix)
    df["wind_solar_ratio"] = df["wind"] / (df["solar"] + 1)

    # Total renewable generation
    df["total_renewable"] = df["wind"] + df["solar"]

    return df
