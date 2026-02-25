#!/usr/bin/env python3
"""
run_step3.py — Neural network training pipeline (Step 3).
==========================================================
Loads the prepared data from Step 2, builds an LSTM model, trains it
with early stopping, evaluates on the held-out validation set, and
generates test-set predictions.

Prerequisites:
    run_step2.py must have been executed first so that
    ``outputs/artefacts/train_prepared.csv`` exists.

Usage
-----
    python run_step3.py
"""

import argparse
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config as cfg
from src.dataset import (
    load_prepared_data,
    load_test_data,
    temporal_train_val_split,
    build_dataloaders,
    SpreadSequenceDataset,
    print_dataset_summary,
)
from src.model import SpreadLSTM, get_device, print_model_summary
from src.trainer import Trainer, print_training_summary
from src.visualisation import (
    plot_training_curves,
    plot_predictions_vs_actual,
    plot_val_timeseries,
)


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────

def _header(title: str) -> None:
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print(f"{'=' * 65}")


def _set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ──────────────────────────────────────────────────────────────
# PIPELINE
# ──────────────────────────────────────────────────────────────

def run_pipeline() -> None:
    t0 = time.time()
    _set_seed(cfg.RANDOM_STATE)

    # ═══════════════════════════════════════════════════════════
    # 3(a)  LOAD PREPARED DATA & BUILD SEQUENCES
    # ═══════════════════════════════════════════════════════════
    _header("STEP 3: Loading Prepared Data from Step 2")

    data = load_prepared_data()
    split = temporal_train_val_split(data)
    print_dataset_summary(split)

    train_loader, val_loader = build_dataloaders(split)
    print(f"\n  Batch size:          {cfg.BATCH_SIZE}")
    print(f"  Train batches/epoch: {len(train_loader)}")
    print(f"  Val batches/epoch:   {len(val_loader)}")

    # ═══════════════════════════════════════════════════════════
    # 3(b)  BUILD LSTM MODEL
    # ═══════════════════════════════════════════════════════════
    _header("STEP 3: Designing LSTM Architecture")

    device = get_device()
    model = SpreadLSTM()
    print_model_summary(model, device)

    # ═══════════════════════════════════════════════════════════
    # 3(c)  TRAIN
    # ═══════════════════════════════════════════════════════════
    _header("STEP 3: Training with Early Stopping")

    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    result = trainer.fit()

    print()
    print_training_summary(result)

    # ═══════════════════════════════════════════════════════════
    # VALIDATION EVALUATION
    # ═══════════════════════════════════════════════════════════
    _header("Evaluating on Validation Set")

    # Get predictions (in scaled space)
    val_preds_scaled = trainer.predict(val_loader)

    # Inverse-transform to EUR
    target_scaler = joblib.load(
        os.path.join(cfg.ARTEFACT_DIR, "target_scaler.pkl")
    )
    val_preds_eur = target_scaler.inverse_transform(
        val_preds_scaled.reshape(-1, 1)
    ).flatten()

    # Get actual values (also need inverse-transform)
    # The val targets correspond to indices [seq_len-1 .. end] of val split
    seq_len = cfg.SEQUENCE_LENGTH
    val_targets_scaled = split.val.targets[seq_len - 1:]
    val_targets_eur = target_scaler.inverse_transform(
        val_targets_scaled.reshape(-1, 1)
    ).flatten()

    val_dates = split.val.dates[seq_len - 1:]

    # Metrics
    residuals = val_preds_eur - val_targets_eur
    mae = np.mean(np.abs(residuals))
    rmse = np.sqrt(np.mean(residuals ** 2))
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((val_targets_eur - val_targets_eur.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    print(f"  Validation samples:  {len(val_targets_eur):,}")
    print(f"  MAE:                 {mae:.2f} EUR")
    print(f"  RMSE:                {rmse:.2f} EUR")
    print(f"  R² score:            {r2:.4f}")

    # ═══════════════════════════════════════════════════════════
    # VISUALISATIONS
    # ═══════════════════════════════════════════════════════════
    _header("Generating Step 3 Visualisations")

    p1 = plot_training_curves(result.history)
    print(f"  Saved: {p1}")

    p2 = plot_predictions_vs_actual(val_targets_eur, val_preds_eur, "Validation")
    print(f"  Saved: {p2}")

    p3 = plot_val_timeseries(val_dates, val_targets_eur, val_preds_eur, n_days=7)
    print(f"  Saved: {p3}")

    # ═══════════════════════════════════════════════════════════
    # SAVE MODEL & ARTEFACTS
    # ═══════════════════════════════════════════════════════════
    _header("Saving Model and Artefacts")

    # Model state dict
    model_path = os.path.join(cfg.ARTEFACT_DIR, "spread_lstm.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_size": model.input_size,
            "hidden_size": model.hidden_size,
            "num_layers": model.num_layers,
            "lstm_dropout": model.lstm_dropout,
            "dense_hidden": model.dense_hidden,
            "dense_dropout": model.dense_dropout,
        },
        "training_result": {
            "best_epoch": result.best_epoch,
            "best_val_loss": result.best_val_loss,
            "stopped_early": result.stopped_early,
            "total_epochs": len(result.history),
        },
        "val_metrics": {
            "mae_eur": float(mae),
            "rmse_eur": float(rmse),
            "r2": float(r2),
        },
    }, model_path)
    print(f"  spread_lstm.pt      — model weights + config")

    # Training history
    history_df = pd.DataFrame([
        {
            "epoch": m.epoch,
            "train_loss": m.train_loss,
            "val_loss": m.val_loss,
            "learning_rate": m.learning_rate,
        }
        for m in result.history
    ])
    hist_path = os.path.join(cfg.ARTEFACT_DIR, "training_history.csv")
    history_df.to_csv(hist_path, index=False)
    print(f"  training_history.csv — per-epoch metrics")

    # ═══════════════════════════════════════════════════════════
    # TEST SET PREDICTIONS
    # ═══════════════════════════════════════════════════════════
    _header("Generating Test Set Predictions")

    test_features, test_ids, test_dates = load_test_data()

    # Build sequences from test data
    test_ds = SpreadSequenceDataset(
        features=test_features,
        targets=np.zeros(len(test_features), dtype=np.float32),  # dummy
        sequence_length=cfg.SEQUENCE_LENGTH,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False
    )

    test_preds_scaled = trainer.predict(test_loader)
    test_preds_eur = target_scaler.inverse_transform(
        test_preds_scaled.reshape(-1, 1)
    ).flatten()

    # The predictions correspond to test rows [seq_len-1 .. end]
    # Align with submission IDs
    pred_ids = test_ids[seq_len - 1:]

    # For the first (seq_len - 1) IDs we don't have full sequences;
    # use the first available prediction as a backfill
    all_preds = np.full(len(test_ids), fill_value=test_preds_eur[0])
    all_preds[seq_len - 1:] = test_preds_eur

    submission = pd.DataFrame({
        "ID": test_ids,
        "forecast": all_preds,
    })

    sub_path = os.path.join(cfg.ARTEFACT_DIR, "submission.csv")
    submission.to_csv(sub_path, index=False)
    print(f"  submission.csv       — {len(submission)} predictions")
    print(f"  Prediction range:    [{all_preds.min():.2f}, {all_preds.max():.2f}] EUR")
    print(f"  Prediction mean:     {all_preds.mean():.2f} EUR")

    elapsed = time.time() - t0
    _header(f"STEP 3 COMPLETE  ({elapsed:.1f}s)")


# ──────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 3: LSTM neural network training for "
                    "electricity market spread prediction.",
    )
    args = parser.parse_args()
    run_pipeline()


if __name__ == "__main__":
    main()
