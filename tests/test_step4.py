"""
test_step4.py — Unit tests for Step 4 modules (enhanced model, trainer, walk-forward CV).
==========================================================================================
Run with:
    pytest tests/test_step4.py -v
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import config as cfg
from src.dataset import (
    PreparedArrays,
    SpreadSequenceDataset,
    SplitArrays,
    temporal_train_val_split,
    build_dataloaders,
)
from src.model import SpreadLSTM, get_device, count_parameters
from src.trainer import Trainer, EpochMetrics, TrainingResult
from src.walk_forward_cv import (
    generate_walk_forward_folds,
    run_walk_forward_cv,
    WalkForwardResult,
    FoldMetrics,
)
from src.visualisation import (
    plot_cv_fold_metrics,
    plot_overfitting_analysis,
    plot_regularisation_comparison,
)


# ──────────────────────────────────────────────────────────────
# FIXTURES
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_data():
    """Synthetic prepared arrays — 2000 rows, 45 features over ~20 days."""
    n = 2000
    n_feat = len(cfg.FINAL_FEATURES)
    rng = np.random.RandomState(42)

    return PreparedArrays(
        features=rng.randn(n, n_feat).astype(np.float32),
        targets=rng.randn(n).astype(np.float32),
        dates=pd.date_range("2022-01-01", periods=n, freq="15min").values,
        feature_names=cfg.FINAL_FEATURES,
    )


@pytest.fixture
def long_synthetic_data():
    """Synthetic data spanning 3 years for walk-forward CV.

    Uses hourly frequency to cover a long date range without
    needing 100k+ rows.  The walk-forward fold generator only
    cares about dates for splitting — row count just needs to
    exceed SEQUENCE_LENGTH per fold.
    """
    dates = pd.date_range("2020-01-01", "2023-12-31", freq="h")
    n = len(dates)
    n_feat = len(cfg.FINAL_FEATURES)
    rng = np.random.RandomState(42)

    return PreparedArrays(
        features=rng.randn(n, n_feat).astype(np.float32),
        targets=rng.randn(n).astype(np.float32),
        dates=dates.values,
        feature_names=cfg.FINAL_FEATURES,
    )


@pytest.fixture
def small_seq_len():
    return 8


@pytest.fixture
def small_model():
    """Enhanced model with input dropout and batch norm for fast tests."""
    return SpreadLSTM(
        input_size=len(cfg.FINAL_FEATURES),
        hidden_size=16,
        num_layers=1,
        lstm_dropout=0.0,
        dense_hidden=8,
        dense_dropout=0.0,
        input_dropout=0.1,
    )


# ──────────────────────────────────────────────────────────────
# TESTS — Enhanced SpreadLSTM (4a: regularisation)
# ──────────────────────────────────────────────────────────────

class TestEnhancedModel:
    def test_input_dropout_exists(self, small_model):
        """Model should have an input_drop layer."""
        assert hasattr(small_model, "input_drop")
        assert isinstance(small_model.input_drop, torch.nn.Dropout)

    def test_input_dropout_rate(self, small_model):
        """Input dropout rate should match config."""
        assert small_model.input_dropout_rate == 0.1

    def test_batch_norm_in_head(self, small_model):
        """Dense head should contain BatchNorm1d."""
        head_modules = list(small_model.head.children())
        bn_layers = [m for m in head_modules if isinstance(m, torch.nn.BatchNorm1d)]
        assert len(bn_layers) == 1, "Head should have exactly one BatchNorm1d"

    def test_output_shape_unchanged(self, small_model, small_seq_len):
        """Output shape must still be (batch, 1)."""
        x = torch.randn(4, small_seq_len, len(cfg.FINAL_FEATURES))
        out = small_model(x)
        assert out.shape == (4, 1)

    def test_input_dropout_effect_in_training(self, small_model, small_seq_len):
        """In training mode, input dropout should produce different outputs
        for the same input (stochastic)."""
        small_model.train()
        x = torch.randn(8, small_seq_len, len(cfg.FINAL_FEATURES))
        out1 = small_model(x).detach()
        out2 = small_model(x).detach()
        # With dropout active, outputs should differ (very high probability)
        assert not torch.allclose(out1, out2, atol=1e-6), (
            "Dropout should make outputs stochastic during training"
        )

    def test_input_dropout_disabled_in_eval(self, small_model, small_seq_len):
        """In eval mode, input dropout should be disabled (deterministic)."""
        small_model.eval()
        x = torch.randn(8, small_seq_len, len(cfg.FINAL_FEATURES))
        out1 = small_model(x).detach()
        out2 = small_model(x).detach()
        assert torch.allclose(out1, out2, atol=1e-6), (
            "Eval mode should be deterministic (dropout disabled)"
        )

    def test_backward_compatible_defaults(self):
        """Default model should use config values."""
        model = SpreadLSTM()
        assert model.input_dropout_rate == cfg.INPUT_DROPOUT
        assert model.lstm_dropout == cfg.LSTM_DROPOUT
        assert model.dense_dropout == cfg.DENSE_DROPOUT

    def test_no_activation_on_output(self, small_model):
        """Final layer must still be nn.Linear (linear activation)."""
        head_modules = list(small_model.head.children())
        assert isinstance(head_modules[-1], torch.nn.Linear)


# ──────────────────────────────────────────────────────────────
# TESTS — Enhanced Trainer (4b: early stopping with min_delta)
# ──────────────────────────────────────────────────────────────

class TestEnhancedTrainer:
    def _make_trainer(self, synthetic_data, small_model, small_seq_len,
                      min_delta=0.0, patience=10):
        split = temporal_train_val_split(synthetic_data, val_fraction=0.3)
        train_loader, val_loader = build_dataloaders(
            split, batch_size=32, sequence_length=small_seq_len
        )
        trainer = Trainer(
            model=small_model,
            device=torch.device("cpu"),
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=3,
            patience=patience,
            min_delta=min_delta,
            lr=1e-3,
        )
        return trainer, val_loader

    def test_min_delta_stored(self, synthetic_data, small_model, small_seq_len):
        """Trainer should store the min_delta value."""
        trainer, _ = self._make_trainer(
            synthetic_data, small_model, small_seq_len, min_delta=0.05
        )
        assert trainer.min_delta == 0.05

    def test_overfit_gap_tracked(self, synthetic_data, small_model, small_seq_len):
        """Each epoch should record an overfit_gap value."""
        trainer, _ = self._make_trainer(
            synthetic_data, small_model, small_seq_len
        )
        result = trainer.fit()
        for m in result.history:
            assert hasattr(m, "overfit_gap")
            # gap = val_loss - train_loss (should be a real number)
            assert isinstance(m.overfit_gap, float)

    def test_overfit_gap_formula(self, synthetic_data, small_model, small_seq_len):
        """overfit_gap should equal val_loss - train_loss."""
        trainer, _ = self._make_trainer(
            synthetic_data, small_model, small_seq_len
        )
        result = trainer.fit()
        for m in result.history:
            expected_gap = m.val_loss - m.train_loss
            assert abs(m.overfit_gap - expected_gap) < 1e-8

    def test_result_has_overfit_gap_at_best(self, synthetic_data, small_model,
                                             small_seq_len):
        """TrainingResult should record gap at best epoch."""
        trainer, _ = self._make_trainer(
            synthetic_data, small_model, small_seq_len
        )
        result = trainer.fit()
        assert hasattr(result, "overfit_gap_at_best")
        assert isinstance(result.overfit_gap_at_best, float)

    def test_min_delta_prevents_noise_resets(self, synthetic_data, small_seq_len):
        """With a very large min_delta, early stopping should trigger sooner
        because tiny improvements won't reset the counter."""
        model = SpreadLSTM(
            input_size=len(cfg.FINAL_FEATURES),
            hidden_size=8, num_layers=1,
            lstm_dropout=0.0, dense_hidden=4, dense_dropout=0.0,
            input_dropout=0.0,
        )
        split = temporal_train_val_split(synthetic_data, val_fraction=0.3)
        train_loader, val_loader = build_dataloaders(
            split, batch_size=32, sequence_length=small_seq_len
        )
        trainer = Trainer(
            model=model, device=torch.device("cpu"),
            train_loader=train_loader, val_loader=val_loader,
            max_epochs=50, patience=2, min_delta=100.0, lr=1e-3,
        )
        result = trainer.fit()
        # With min_delta=100, almost no improvement will count,
        # so it should stop within patience+1 epochs
        assert len(result.history) <= 4

    def test_quiet_mode(self, synthetic_data, small_model, small_seq_len, capsys):
        """Quiet mode should suppress training output."""
        split = temporal_train_val_split(synthetic_data, val_fraction=0.3)
        train_loader, val_loader = build_dataloaders(
            split, batch_size=32, sequence_length=small_seq_len
        )
        trainer = Trainer(
            model=small_model, device=torch.device("cpu"),
            train_loader=train_loader, val_loader=val_loader,
            max_epochs=2, patience=10, lr=1e-3, quiet=True,
        )
        trainer.fit()
        captured = capsys.readouterr()
        assert "Epoch" not in captured.out


# ──────────────────────────────────────────────────────────────
# TESTS — Walk-Forward Cross-Validation (4c)
# ──────────────────────────────────────────────────────────────

class TestWalkForwardCV:
    def test_fold_generation_count(self, long_synthetic_data):
        """Should generate the requested number of folds (or fewer if data
        is too short for all folds)."""
        folds = generate_walk_forward_folds(
            long_synthetic_data, n_folds=3, val_months=6
        )
        assert len(folds) <= 3
        assert len(folds) >= 1

    def test_expanding_training_window(self, long_synthetic_data):
        """Each fold's training set should be >= the previous fold's."""
        folds = generate_walk_forward_folds(
            long_synthetic_data, n_folds=3, val_months=6
        )
        if len(folds) >= 2:
            for i in range(1, len(folds)):
                assert len(folds[i].train.targets) >= len(folds[i - 1].train.targets)

    def test_no_future_leakage(self, long_synthetic_data):
        """Train dates must always be before val dates within each fold."""
        folds = generate_walk_forward_folds(
            long_synthetic_data, n_folds=3, val_months=6
        )
        for fold in folds:
            assert fold.train.dates[-1] < fold.val.dates[0], (
                "Training data must not contain future validation data"
            )

    def test_val_periods_non_overlapping(self, long_synthetic_data):
        """Validation periods across folds should not overlap."""
        folds = generate_walk_forward_folds(
            long_synthetic_data, n_folds=3, val_months=6
        )
        if len(folds) >= 2:
            for i in range(1, len(folds)):
                prev_val_end = folds[i - 1].val.dates[-1]
                curr_val_start = folds[i].val.dates[0]
                # Allow exact boundary (end of one = start of next)
                assert prev_val_end <= curr_val_start, (
                    f"Val periods overlap: fold {i} starts before fold {i - 1} ends"
                )

    def test_folds_chronological(self, long_synthetic_data):
        """Folds should be in chronological order."""
        folds = generate_walk_forward_folds(
            long_synthetic_data, n_folds=3, val_months=6
        )
        if len(folds) >= 2:
            for i in range(1, len(folds)):
                assert folds[i].val.dates[0] >= folds[i - 1].val.dates[0]

    def test_minimum_fold_size(self, long_synthetic_data):
        """Each fold must have at least SEQUENCE_LENGTH rows for both
        train and val."""
        folds = generate_walk_forward_folds(
            long_synthetic_data, n_folds=3, val_months=6
        )
        for fold in folds:
            assert len(fold.train.targets) >= cfg.SEQUENCE_LENGTH
            assert len(fold.val.targets) >= cfg.SEQUENCE_LENGTH


# ──────────────────────────────────────────────────────────────
# TESTS — Step 4 Visualisations
# ──────────────────────────────────────────────────────────────

class TestVisualisations:
    def test_plot_cv_fold_metrics(self, tmp_path):
        """plot_cv_fold_metrics should produce a PNG file."""
        folds = [
            FoldMetrics(
                fold_id=1, train_start="2020-01-01", train_end="2021-12-31",
                val_start="2022-01-01", val_end="2022-06-30",
                train_rows=1000, val_rows=200, best_epoch=5,
                stopped_early=True, train_loss=0.5, val_loss=0.8,
                val_mae=50.0, val_rmse=100.0, val_r2=0.1,
                overfit_gap=0.3, training_time_sec=10.0,
            ),
            FoldMetrics(
                fold_id=2, train_start="2020-01-01", train_end="2022-06-30",
                val_start="2022-07-01", val_end="2022-12-31",
                train_rows=1200, val_rows=200, best_epoch=8,
                stopped_early=True, train_loss=0.4, val_loss=0.7,
                val_mae=45.0, val_rmse=95.0, val_r2=0.15,
                overfit_gap=0.3, training_time_sec=12.0,
            ),
        ]
        path = plot_cv_fold_metrics(folds, save_dir=str(tmp_path))
        assert os.path.exists(path)
        assert path.endswith(".png")

    def test_plot_overfitting_analysis(self, tmp_path):
        """plot_overfitting_analysis should produce a PNG file."""
        history = [
            EpochMetrics(epoch=i, train_loss=1.0 / i, val_loss=1.0 / i + 0.1,
                         learning_rate=1e-3, elapsed_sec=1.0,
                         overfit_gap=0.1)
            for i in range(1, 6)
        ]
        path = plot_overfitting_analysis(history, save_dir=str(tmp_path))
        assert os.path.exists(path)
        assert path.endswith(".png")
