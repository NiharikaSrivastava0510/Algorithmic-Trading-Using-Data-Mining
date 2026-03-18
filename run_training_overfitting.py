#!/usr/bin/env python3
"""
run_training_overfitting.py — Training & Mitigating Overfitting.
=================================================================
1. Walk-forward expanding-window cross-validation (4c)
2. Enhanced regularisation summary (4a — input dropout, batch norm)
3. Enhanced early stopping with min_delta (4b)
4. Final holdout training with full regularisation
5. Test-set submission generation

Prerequisites:
    run_step2.py must have been executed first so that
    ``outputs/artefacts/train_prepared.csv`` exists.

Usage
-----
    python run_step4.py
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
from src.probability import SpreadCalibrator, print_calibration_summary
from src.trainer import Trainer, print_training_summary
from src.walk_forward_cv import (
    run_walk_forward_cv,
    print_cv_summary,
)
from src.visualisation import (
    plot_training_curves,
    plot_predictions_vs_actual,
    plot_val_timeseries,
    plot_cv_fold_metrics,
    plot_overfitting_analysis,
    plot_regularisation_comparison,
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
    # 4(a)  LOAD DATA & DISPLAY ENHANCED REGULARISATION
    # ═══════════════════════════════════════════════════════════
    _header("STEP 4(a): Enhanced Regularisation")

    data = load_prepared_data()

    print("\n  Regularisation techniques applied:")
    print(f"    1. Input dropout:           {cfg.INPUT_DROPOUT} (randomly zeroes features)")
    print(f"    2. LSTM inter-layer dropout: {cfg.LSTM_DROPOUT} (between stacked layers)")
    print(f"    3. Batch normalisation:      BatchNorm1d({cfg.DENSE_HIDDEN_SIZE})")
    print(f"    4. Dense head dropout:       {cfg.DENSE_DROPOUT}")
    print(f"    5. L2 regularisation:        weight_decay={cfg.WEIGHT_DECAY}")
    print(f"    6. Gradient clipping:        max_norm={cfg.GRADIENT_CLIP_NORM}")

    # Load target scaler for EUR conversion
    target_scaler = joblib.load(
        os.path.join(cfg.ARTEFACT_DIR, "target_scaler.pkl")
    )

    # ═══════════════════════════════════════════════════════════
    # 4(c)  WALK-FORWARD CROSS-VALIDATION
    # ═══════════════════════════════════════════════════════════
    _header("STEP 4(c): Walk-Forward Cross-Validation")

    print(f"\n  Strategy: Expanding-window time-series CV")
    print(f"  Folds:    {cfg.CV_N_FOLDS}")
    print(f"  Val window: {cfg.CV_VAL_MONTHS} months per fold")
    print(f"  Guarantee: Future data NEVER used to predict the past")

    cv_result = run_walk_forward_cv(
        data=data,
        target_scaler=target_scaler,
    )
    print_cv_summary(cv_result)

    # ═══════════════════════════════════════════════════════════
    # 4(b)  FINAL HOLDOUT TRAINING (enhanced early stopping)
    # ═══════════════════════════════════════════════════════════
    _header("STEP 4(b): Final Training with Enhanced Early Stopping")

    split = temporal_train_val_split(data)
    print_dataset_summary(split)

    train_loader, val_loader = build_dataloaders(split)
    print(f"\n  Batch size:          {cfg.BATCH_SIZE}")
    print(f"  Train batches/epoch: {len(train_loader)}")
    print(f"  Val batches/epoch:   {len(val_loader)}")
    print(f"\n  Early stopping config:")
    print(f"    Patience:          {cfg.EARLY_STOPPING_PATIENCE} epochs")
    print(f"    Min delta:         {cfg.EARLY_STOPPING_MIN_DELTA} (noise filter)")

    device = get_device()
    model = SpreadLSTM()
    print_model_summary(model, device)

    _set_seed(cfg.RANDOM_STATE)
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

    val_preds_scaled = trainer.predict(val_loader)

    val_preds_eur = target_scaler.inverse_transform(
        val_preds_scaled.reshape(-1, 1)
    ).flatten()

    seq_len = cfg.SEQUENCE_LENGTH
    val_targets_scaled = split.val.targets[seq_len - 1:]
    val_targets_eur = target_scaler.inverse_transform(
        val_targets_scaled.reshape(-1, 1)
    ).flatten()

    val_dates = split.val.dates[seq_len - 1:]

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
    _header("Generating Step 4 Visualisations")

    p1 = plot_cv_fold_metrics(cv_result.folds)
    print(f"  Saved: {p1}")

    p2 = plot_overfitting_analysis(result.history)
    print(f"  Saved: {p2}")

    p3 = plot_regularisation_comparison(result.history, cv_result.folds)
    print(f"  Saved: {p3}")

    p4 = plot_training_curves(result.history)
    print(f"  Saved: {p4}")

    p5 = plot_predictions_vs_actual(val_targets_eur, val_preds_eur, "Step4 Validation")
    print(f"  Saved: {p5}")

    p6 = plot_val_timeseries(val_dates, val_targets_eur, val_preds_eur, n_days=7)
    print(f"  Saved: {p6}")

    # ═══════════════════════════════════════════════════════════
    # SAVE MODEL & ARTEFACTS
    # ═══════════════════════════════════════════════════════════
    _header("Saving Model and Artefacts")

    # Calibrator
    calib_path = os.path.join(cfg.ARTEFACT_DIR, "spread_calibrator_step4.pkl")
    joblib.dump(calibrator, calib_path)
    print(f"  spread_calibrator_step4.pkl — logistic calibration model")

    model_path = os.path.join(cfg.ARTEFACT_DIR, "spread_lstm_step4.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {
            "input_size": model.input_size,
            "hidden_size": model.hidden_size,
            "num_layers": model.num_layers,
            "lstm_dropout": model.lstm_dropout,
            "dense_hidden": model.dense_hidden,
            "dense_dropout": model.dense_dropout,
            "input_dropout": model.input_dropout_rate,
        },
        "training_result": {
            "best_epoch": result.best_epoch,
            "best_val_loss": result.best_val_loss,
            "stopped_early": result.stopped_early,
            "total_epochs": len(result.history),
            "overfit_gap_at_best": result.overfit_gap_at_best,
        },
        "val_metrics": {
            "mae_eur": float(mae),
            "rmse_eur": float(rmse),
            "r2": float(r2),
        },
        "cv_metrics": {
            "mean_mae": cv_result.mean_mae,
            "std_mae": cv_result.std_mae,
            "mean_rmse": cv_result.mean_rmse,
            "std_rmse": cv_result.std_rmse,
            "mean_r2": cv_result.mean_r2,
            "std_r2": cv_result.std_r2,
            "n_folds": len(cv_result.folds),
        },
    }, model_path)
    print(f"  spread_lstm_step4.pt — model + CV metrics")

    # CV results as CSV
    cv_df = pd.DataFrame([
        {
            "fold": f.fold_id,
            "train_start": f.train_start,
            "train_end": f.train_end,
            "val_start": f.val_start,
            "val_end": f.val_end,
            "train_rows": f.train_rows,
            "val_rows": f.val_rows,
            "best_epoch": f.best_epoch,
            "val_mae": f.val_mae,
            "val_rmse": f.val_rmse,
            "val_r2": f.val_r2,
            "overfit_gap": f.overfit_gap,
        }
        for f in cv_result.folds
    ])
    cv_path = os.path.join(cfg.ARTEFACT_DIR, "cv_results.csv")
    cv_df.to_csv(cv_path, index=False)
    print(f"  cv_results.csv       — per-fold CV metrics")

    # Training history
    history_df = pd.DataFrame([
        {
            "epoch": m.epoch,
            "train_loss": m.train_loss,
            "val_loss": m.val_loss,
            "overfit_gap": m.overfit_gap,
            "learning_rate": m.learning_rate,
        }
        for m in result.history
    ])
    hist_path = os.path.join(cfg.ARTEFACT_DIR, "training_history.csv")
    history_df.to_csv(hist_path, index=False)
    print(f"  training_history.csv — per-epoch metrics (updated)")

    # ═══════════════════════════════════════════════════════════
    # LOGISTIC CALIBRATION — P(spread > 0)
    # ═══════════════════════════════════════════════════════════
    _header("Logistic Calibration: spread → P(spread > 0)")

    print("  The Kaggle competition expects probability forecasts.")
    print("  Fitting logistic calibration on validation predictions...\n")

    calibrator = SpreadCalibrator()
    calibrator.fit(val_preds_eur, val_targets_eur)

    val_probs = calibrator.predict_proba(val_preds_eur)
    print_calibration_summary(calibrator, val_probs, label="Val")

    # ═══════════════════════════════════════════════════════════
    # TEST SET PREDICTIONS
    # ═══════════════════════════════════════════════════════════
    _header("Generating Test Set Predictions")

    test_features, test_ids, test_dates = load_test_data()

    test_ds = SpreadSequenceDataset(
        features=test_features,
        targets=np.zeros(len(test_features), dtype=np.float32),
        sequence_length=cfg.SEQUENCE_LENGTH,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False
    )

    test_preds_scaled = trainer.predict(test_loader)
    test_preds_eur = target_scaler.inverse_transform(
        test_preds_scaled.reshape(-1, 1)
    ).flatten()

    all_preds_eur = np.full(len(test_ids), fill_value=test_preds_eur[0])
    all_preds_eur[seq_len - 1:] = test_preds_eur

    # Convert EUR → P(spread > 0) using fitted calibrator
    all_probs = calibrator.predict_proba(all_preds_eur)

    submission = pd.DataFrame({
        "ID": test_ids,
        "forecast": all_probs,
    })

    sub_path = os.path.join(cfg.ARTEFACT_DIR, "submission.csv")
    submission.to_csv(sub_path, index=False)
    print(f"  submission.csv       — {len(submission)} probability forecasts")
    print(f"  P(spread>0) range:   [{all_probs.min():.4f}, {all_probs.max():.4f}]")
    print(f"  P(spread>0) mean:    {all_probs.mean():.4f}")
    print(f"  Raw EUR range:       [{all_preds_eur.min():.2f}, {all_preds_eur.max():.2f}]")

    elapsed = time.time() - t0
    _header(f"STEP 4 COMPLETE  ({elapsed:.1f}s)")


# ──────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 4: Training & mitigating overfitting — "
                    "walk-forward CV, enhanced regularisation, "
                    "early stopping with min_delta.",
    )
    args = parser.parse_args()
    run_pipeline()


if __name__ == "__main__":
    main()
