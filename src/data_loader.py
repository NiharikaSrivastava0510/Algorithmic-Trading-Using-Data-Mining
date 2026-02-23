"""
data_loader.py — Load and merge the ENSIMAG IF 2025 raw CSV files.
===================================================================
Responsibilities:
    1. Read train.csv, imbalances.csv, test.csv, and sample.csv.
    2. Parse the *date* column as datetime objects.
    3. Left-join the imbalances table into the training set.
    4. Run basic sanity checks (shape, dtypes, missing values).
"""

import os
import pandas as pd

import config as cfg


# ──────────────────────────────────────────────────────────────
# PUBLIC API
# ──────────────────────────────────────────────────────────────

def load_raw_data(data_dir: str | None = None) -> dict[str, pd.DataFrame]:
    """
    Load all four raw CSV files and return them in a dictionary.

    Parameters
    ----------
    data_dir : str, optional
        Path to the folder containing the CSVs.
        Defaults to ``config.DATA_DIR``.

    Returns
    -------
    dict
        Keys: ``"train"``, ``"imbalances"``, ``"test"``, ``"sample"``
    """
    data_dir = data_dir or cfg.DATA_DIR

    train = pd.read_csv(
        os.path.join(data_dir, cfg.TRAIN_FILE), parse_dates=["date"]
    )
    imbalances = pd.read_csv(
        os.path.join(data_dir, cfg.IMBALANCES_FILE), parse_dates=["date"]
    )
    test = pd.read_csv(
        os.path.join(data_dir, cfg.TEST_FILE), parse_dates=["date"]
    )
    sample = pd.read_csv(os.path.join(data_dir, cfg.SAMPLE_FILE))

    return {
        "train": train,
        "imbalances": imbalances,
        "test": test,
        "sample": sample,
    }


def merge_imbalances(
    train: pd.DataFrame,
    imbalances: pd.DataFrame,
) -> pd.DataFrame:
    """
    Left-join the *imbalances* table into *train* on the ``date`` column.

    Parameters
    ----------
    train : pd.DataFrame
        Training data with columns [date, wind, solar, load, spread].
    imbalances : pd.DataFrame
        Imbalance prices with columns [date, imbalances].

    Returns
    -------
    pd.DataFrame
        Training data with the additional ``imbalances`` column.
    """
    merged = train.merge(imbalances, on="date", how="left")
    return merged


def print_data_summary(datasets: dict[str, pd.DataFrame]) -> None:
    """Print shapes, date ranges, and missing-value counts."""
    train = datasets["train"]
    test = datasets["test"]

    print(f"  Train shape:        {train.shape}")
    print(f"  Test shape:         {test.shape}")
    print(f"  Date range (train): {train['date'].min()} -> {train['date'].max()}")
    print(f"  Date range (test):  {test['date'].min()} -> {test['date'].max()}")

    missing = train.isnull().sum()
    if missing.sum() == 0:
        print("  Missing values:     None")
    else:
        print(f"  Missing values:\n{missing[missing > 0]}")
