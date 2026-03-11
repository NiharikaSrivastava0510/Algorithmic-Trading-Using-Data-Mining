"""
walk_forward_cv.py — Expanding-window time-series cross-validation.
====================================================================
Responsibilities:
    1. Split the prepared data into chronological folds where the
       training window **expands** forward in time (never shuffled).
    2. For each fold, train a fresh model with full regularisation
       and early stopping, then evaluate on the held-out validation
       window.
    3. Aggregate per-fold metrics to give a robust, leak-free
       estimate of generalisation performance.

The walk-forward approach satisfies the strict time-series CV
requirement: future data is **never** used to predict the past.

Example with 4 folds and 6-month validation windows
(on data spanning 2020-01-01 to 2023-12-31):

    Fold 1: Train [2020-01 .. 2021-12] → Val [2022-01 .. 2022-06]
    Fold 2: Train [2020-01 .. 2022-06] → Val [2022-07 .. 2022-12]
    Fold 3: Train [2020-01 .. 2022-12] → Val [2023-01 .. 2023-06]
    Fold 4: Train [2020-01 .. 2023-06] → Val [2023-07 .. 2023-12]
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import config as cfg
from src.dataset import PreparedArrays, SpreadSequenceDataset, SplitArrays
from src.model import SpreadLSTM, get_device
from src.trainer import Trainer, TrainingResult


# ──────────────────────────────────────────────────────────────
# DATA CLASSES
# ──────────────────────────────────────────────────────────────

@dataclass
class FoldMetrics:
    """Metrics for a single walk-forward fold."""
    fold_id: int
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    train_rows: int
    val_rows: int
    best_epoch: int
    stopped_early: bool
    train_loss: float
    val_loss: float
    val_mae: float
    val_rmse: float
    val_r2: float
    overfit_gap: float
    training_time_sec: float


@dataclass
class WalkForwardResult:
    """Aggregated results across all walk-forward folds."""
    folds: list[FoldMetrics] = field(default_factory=list)
    mean_mae: float = 0.0
    std_mae: float = 0.0
    mean_rmse: float = 0.0
    std_rmse: float = 0.0
    mean_r2: float = 0.0
    std_r2: float = 0.0
    total_time_sec: float = 0.0


# ──────────────────────────────────────────────────────────────
# FOLD GENERATION
# ──────────────────────────────────────────────────────────────

def generate_walk_forward_folds(
    data: PreparedArrays,
    n_folds: int | None = None,
    val_months: int | None = None,
) -> list[SplitArrays]:
    """
    Generate expanding-window chronological folds.

    The validation window is fixed at ``val_months`` months.
    Training data grows with each fold (expanding window).
    Folds are constructed backwards from the end of the data.

    Parameters
    ----------
    data : PreparedArrays
        Full prepared dataset (features, targets, dates).
    n_folds : int
        Number of folds (default: ``config.CV_N_FOLDS``).
    val_months : int
        Months per validation window (default: ``config.CV_VAL_MONTHS``).

    Returns
    -------
    list[SplitArrays]
        One SplitArrays per fold, chronologically ordered.
    """
    n_folds = n_folds or cfg.CV_N_FOLDS
    val_months = val_months or cfg.CV_VAL_MONTHS

    dates = pd.DatetimeIndex(data.dates)
    data_end = dates.max()

    # Pre-compute fold boundaries to guarantee non-overlapping.
    # Work backwards: the last fold ends at data_end; each fold's
    # val_start becomes the previous fold's val_end.
    boundaries: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = pd.Timestamp(data_end)
    for _ in range(n_folds):
        val_end = cursor
        val_start = val_end - pd.DateOffset(months=val_months)
        boundaries.append((val_start, val_end))
        cursor = val_start  # next fold ends where this one started
    boundaries.reverse()  # chronological order

    folds: list[SplitArrays] = []

    for i, (val_start, val_end) in enumerate(boundaries):
        is_last = (i == len(boundaries) - 1)

        train_mask = dates < val_start
        if is_last:
            val_mask = (dates >= val_start) & (dates <= val_end)
        else:
            val_mask = (dates >= val_start) & (dates < val_end)

        if train_mask.sum() < cfg.SEQUENCE_LENGTH or val_mask.sum() < cfg.SEQUENCE_LENGTH:
            continue

        train = PreparedArrays(
            features=data.features[train_mask],
            targets=data.targets[train_mask],
            dates=data.dates[train_mask],
            feature_names=data.feature_names,
        )
        val = PreparedArrays(
            features=data.features[val_mask],
            targets=data.targets[val_mask],
            dates=data.dates[val_mask],
            feature_names=data.feature_names,
        )

        split_index = int(train_mask.sum())
        folds.append(SplitArrays(train=train, val=val, split_index=split_index))

    return folds


# ──────────────────────────────────────────────────────────────
# FOLD EVALUATION HELPER
# ──────────────────────────────────────────────────────────────

def _compute_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> tuple[float, float, float]:
    """Compute MAE, RMSE, R² from raw arrays."""
    residuals = predictions - targets
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((targets - targets.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return mae, rmse, r2


# ──────────────────────────────────────────────────────────────
# WALK-FORWARD CV
# ──────────────────────────────────────────────────────────────

def run_walk_forward_cv(
    data: PreparedArrays,
    target_scaler=None,
    n_folds: int | None = None,
    val_months: int | None = None,
) -> WalkForwardResult:
    """
    Run the full walk-forward cross-validation.

    For each fold:
        1. Build a fresh SpreadLSTM model
        2. Train with early stopping and all regularisation
        3. Evaluate on the validation window
        4. Record per-fold metrics

    Parameters
    ----------
    data : PreparedArrays
        Full prepared dataset.
    target_scaler : optional
        Fitted StandardScaler to inverse-transform predictions to EUR.
        If None, metrics are reported in scaled space.
    n_folds : int
    val_months : int

    Returns
    -------
    WalkForwardResult
    """
    t0 = time.time()
    device = get_device()

    folds_splits = generate_walk_forward_folds(data, n_folds, val_months)
    result = WalkForwardResult()

    for i, split in enumerate(folds_splits):
        fold_id = i + 1
        print(f"\n  ── Fold {fold_id}/{len(folds_splits)} ──")
        print(f"     Train: {str(split.train.dates[0])[:10]} -> "
              f"{str(split.train.dates[-1])[:10]}  "
              f"({len(split.train.targets):,} rows)")
        print(f"     Val:   {str(split.val.dates[0])[:10]} -> "
              f"{str(split.val.dates[-1])[:10]}  "
              f"({len(split.val.targets):,} rows)")

        # Build data loaders
        train_ds = SpreadSequenceDataset(
            split.train.features, split.train.targets, cfg.SEQUENCE_LENGTH
        )
        val_ds = SpreadSequenceDataset(
            split.val.features, split.val.targets, cfg.SEQUENCE_LENGTH
        )
        train_loader = DataLoader(
            train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=0
        )
        val_loader = DataLoader(
            val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=0
        )

        # Fresh model for each fold
        model = SpreadLSTM()

        trainer = Trainer(
            model=model,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            quiet=True,
        )

        train_result = trainer.fit()

        # Predictions
        val_preds_scaled = trainer.predict(val_loader)
        seq_len = cfg.SEQUENCE_LENGTH
        val_targets_scaled = split.val.targets[seq_len - 1:]

        # Inverse-transform if scaler provided
        if target_scaler is not None:
            val_preds = target_scaler.inverse_transform(
                val_preds_scaled.reshape(-1, 1)
            ).flatten()
            val_targets = target_scaler.inverse_transform(
                val_targets_scaled.reshape(-1, 1)
            ).flatten()
        else:
            val_preds = val_preds_scaled
            val_targets = val_targets_scaled

        mae, rmse, r2 = _compute_metrics(val_targets, val_preds)

        final = train_result.history[-1] if train_result.history else None

        fold_metrics = FoldMetrics(
            fold_id=fold_id,
            train_start=str(split.train.dates[0])[:10],
            train_end=str(split.train.dates[-1])[:10],
            val_start=str(split.val.dates[0])[:10],
            val_end=str(split.val.dates[-1])[:10],
            train_rows=len(split.train.targets),
            val_rows=len(split.val.targets),
            best_epoch=train_result.best_epoch,
            stopped_early=train_result.stopped_early,
            train_loss=final.train_loss if final else 0.0,
            val_loss=final.val_loss if final else 0.0,
            val_mae=mae,
            val_rmse=rmse,
            val_r2=r2,
            overfit_gap=train_result.overfit_gap_at_best,
            training_time_sec=train_result.total_time_sec,
        )
        result.folds.append(fold_metrics)

        print(f"     Best epoch: {fold_metrics.best_epoch} | "
              f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | "
              f"R²: {r2:.4f} | "
              f"Time: {fold_metrics.training_time_sec:.1f}s")

    # Aggregate
    if result.folds:
        maes = [f.val_mae for f in result.folds]
        rmses = [f.val_rmse for f in result.folds]
        r2s = [f.val_r2 for f in result.folds]

        result.mean_mae = float(np.mean(maes))
        result.std_mae = float(np.std(maes))
        result.mean_rmse = float(np.mean(rmses))
        result.std_rmse = float(np.std(rmses))
        result.mean_r2 = float(np.mean(r2s))
        result.std_r2 = float(np.std(r2s))

    result.total_time_sec = time.time() - t0
    return result


# ──────────────────────────────────────────────────────────────
# PRINTING
# ──────────────────────────────────────────────────────────────

def print_cv_summary(result: WalkForwardResult) -> None:
    """Print aggregated walk-forward CV results."""
    print(f"\n  Walk-Forward Cross-Validation Summary ({len(result.folds)} folds):")
    print(f"  {'─' * 55}")
    print(f"  MAE:   {result.mean_mae:>8.2f} ± {result.std_mae:.2f} EUR")
    print(f"  RMSE:  {result.mean_rmse:>8.2f} ± {result.std_rmse:.2f} EUR")
    print(f"  R²:    {result.mean_r2:>8.4f} ± {result.std_r2:.4f}")
    print(f"  Total CV time: {result.total_time_sec:.1f}s")
