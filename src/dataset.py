"""
dataset.py — Windowed sequence Dataset and temporal train/val split.
=====================================================================
Responsibilities:
    1. Load the prepared CSV artefacts from Step 2.
    2. Extract the feature matrix and target vector.
    3. Build a sliding-window PyTorch Dataset so that each sample is
       a (sequence_length × num_features) tensor paired with the
       spread target at the last timestep of that window.
    4. Split the data **chronologically** into training and validation
       subsets (no shuffling — future data never leaks into training).

Design notes:
    * The LSTM receives windows of ``SEQUENCE_LENGTH`` (96 = 1 day)
      consecutive 15-minute intervals.  For each window the model
      predicts the spread at the final interval.
    * NaN-free guarantee: Step 2 already filled all lag/rolling NaNs.
    * Sequences are built over the full contiguous array; the rare
      DST gaps (8 out of 140 k rows) have negligible impact.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import config as cfg


# ──────────────────────────────────────────────────────────────
# DATA CLASSES
# ──────────────────────────────────────────────────────────────

@dataclass
class PreparedArrays:
    """Numpy arrays ready for the PyTorch Dataset."""
    features: np.ndarray          # (N, F) float32
    targets: np.ndarray           # (N,)   float32  — spread_scaled
    dates: np.ndarray             # (N,)   datetime64
    feature_names: list[str]


@dataclass
class SplitArrays:
    """Chronological train / validation split."""
    train: PreparedArrays
    val: PreparedArrays
    split_index: int              # first row of the val set


# ──────────────────────────────────────────────────────────────
# LOADING
# ──────────────────────────────────────────────────────────────

def load_prepared_data(
    artefact_dir: str | None = None,
    feature_cols: list[str] | None = None,
    target_col: str = "spread_scaled",
) -> PreparedArrays:
    """
    Load the prepared training CSV and extract feature / target arrays.

    Parameters
    ----------
    artefact_dir : str, optional
        Directory containing ``train_prepared.csv``.
        Defaults to ``config.ARTEFACT_DIR``.
    feature_cols : list[str], optional
        Columns to use as features.  Defaults to ``config.FINAL_FEATURES``.
    target_col : str
        Name of the scaled target column.

    Returns
    -------
    PreparedArrays
    """
    artefact_dir = artefact_dir or cfg.ARTEFACT_DIR
    feature_cols = feature_cols or cfg.FINAL_FEATURES

    df = pd.read_csv(
        os.path.join(artefact_dir, "train_prepared.csv"),
        parse_dates=["date"],
    )

    features = df[feature_cols].values.astype(np.float32)
    targets = df[target_col].values.astype(np.float32)
    dates = df["date"].values

    return PreparedArrays(
        features=features,
        targets=targets,
        dates=dates,
        feature_names=feature_cols,
    )


def load_test_data(
    artefact_dir: str | None = None,
    feature_cols: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the prepared test CSV.

    Returns
    -------
    (features, ids, dates)
        features: (N, F) float32
        ids: (N,) int  — submission IDs
        dates: (N,) datetime64
    """
    artefact_dir = artefact_dir or cfg.ARTEFACT_DIR
    feature_cols = feature_cols or cfg.FINAL_FEATURES

    df = pd.read_csv(
        os.path.join(artefact_dir, "test_prepared.csv"),
        parse_dates=["date"],
    )

    features = df[feature_cols].values.astype(np.float32)
    ids = df["ID"].values
    dates = df["date"].values

    return features, ids, dates


# ──────────────────────────────────────────────────────────────
# TEMPORAL SPLIT
# ──────────────────────────────────────────────────────────────

def temporal_train_val_split(
    data: PreparedArrays,
    val_fraction: float | None = None,
) -> SplitArrays:
    """
    Split the data chronologically — the last *val_fraction* of rows
    become the validation set.

    This is critical for time-series: a random split would allow the
    model to see future data during training.

    Parameters
    ----------
    data : PreparedArrays
    val_fraction : float, optional
        Fraction of rows for validation.  Defaults to
        ``config.VALIDATION_FRACTION``.

    Returns
    -------
    SplitArrays
    """
    val_fraction = val_fraction or cfg.VALIDATION_FRACTION
    n = len(data.targets)
    split = int(n * (1 - val_fraction))

    train = PreparedArrays(
        features=data.features[:split],
        targets=data.targets[:split],
        dates=data.dates[:split],
        feature_names=data.feature_names,
    )
    val = PreparedArrays(
        features=data.features[split:],
        targets=data.targets[split:],
        dates=data.dates[split:],
        feature_names=data.feature_names,
    )

    return SplitArrays(train=train, val=val, split_index=split)


# ──────────────────────────────────────────────────────────────
# PYTORCH DATASET
# ──────────────────────────────────────────────────────────────

class SpreadSequenceDataset(Dataset):
    """
    Sliding-window dataset for LSTM training.

    Each sample consists of:
        X : (sequence_length, num_features)  — the feature window
        y : (1,)                             — spread_scaled at the
                                               last timestep of the window

    Parameters
    ----------
    features : np.ndarray
        Shape (N, F), float32.
    targets : np.ndarray
        Shape (N,), float32.
    sequence_length : int
        Number of consecutive intervals per window.
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        sequence_length: int | None = None,
    ):
        self.features = features
        self.targets = targets
        self.seq_len = sequence_length or cfg.SEQUENCE_LENGTH

        if len(self.features) < self.seq_len:
            raise ValueError(
                f"Data has {len(self.features)} rows but sequence_length "
                f"is {self.seq_len}."
            )

    def __len__(self) -> int:
        return len(self.features) - self.seq_len + 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        X = self.features[idx : idx + self.seq_len]     # (seq_len, F)
        y = self.targets[idx + self.seq_len - 1]        # scalar
        return torch.from_numpy(X), torch.tensor([y], dtype=torch.float32)


# ──────────────────────────────────────────────────────────────
# DATA LOADERS
# ──────────────────────────────────────────────────────────────

def build_dataloaders(
    split: SplitArrays,
    batch_size: int | None = None,
    sequence_length: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """
    Build PyTorch DataLoaders for training and validation.

    Training loader shuffles; validation loader does not.

    Returns
    -------
    (train_loader, val_loader)
    """
    batch_size = batch_size or cfg.BATCH_SIZE
    sequence_length = sequence_length or cfg.SEQUENCE_LENGTH

    train_ds = SpreadSequenceDataset(
        split.train.features, split.train.targets, sequence_length
    )
    val_ds = SpreadSequenceDataset(
        split.val.features, split.val.targets, sequence_length
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    return train_loader, val_loader


def print_dataset_summary(split: SplitArrays, seq_len: int | None = None) -> None:
    """Print shapes and sample counts."""
    seq_len = seq_len or cfg.SEQUENCE_LENGTH

    n_train = len(split.train.targets) - seq_len + 1
    n_val = len(split.val.targets) - seq_len + 1
    n_feat = split.train.features.shape[1]

    print(f"  Feature columns:     {n_feat}")
    print(f"  Sequence length:     {seq_len} intervals (= {seq_len * 15 / 60:.0f} hours)")
    print(f"  Training rows:       {len(split.train.targets):,}  ->  {n_train:,} sequences")
    print(f"  Validation rows:     {len(split.val.targets):,}  ->  {n_val:,} sequences")
    print(f"  Train period:        {split.train.dates[0]} -> {split.train.dates[-1]}")
    print(f"  Val   period:        {split.val.dates[0]} -> {split.val.dates[-1]}")
