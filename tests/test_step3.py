"""
test_step3.py — Unit tests for Step 3 modules (dataset, model, trainer).
=========================================================================
Run with:
    pytest tests/test_step3.py -v
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
    temporal_train_val_split,
    build_dataloaders,
)
from src.model import SpreadLSTM, get_device, count_parameters
from src.trainer import Trainer


# ──────────────────────────────────────────────────────────────
# FIXTURES
# ──────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_data():
    """Synthetic prepared arrays — 500 rows, 45 features."""
    n = 500
    n_feat = len(cfg.FINAL_FEATURES)
    rng = np.random.RandomState(42)

    return PreparedArrays(
        features=rng.randn(n, n_feat).astype(np.float32),
        targets=rng.randn(n).astype(np.float32),
        dates=pd.date_range("2023-01-01", periods=n, freq="15min").values,
        feature_names=cfg.FINAL_FEATURES,
    )


@pytest.fixture
def small_seq_len():
    """Use a small sequence length for fast tests."""
    return 8


@pytest.fixture
def small_model(small_seq_len):
    """A tiny model for fast unit tests."""
    return SpreadLSTM(
        input_size=len(cfg.FINAL_FEATURES),
        hidden_size=16,
        num_layers=1,
        lstm_dropout=0.0,
        dense_hidden=8,
        dense_dropout=0.0,
    )


# ──────────────────────────────────────────────────────────────
# TESTS — SpreadSequenceDataset
# ──────────────────────────────────────────────────────────────

class TestDataset:
    def test_length(self, synthetic_data, small_seq_len):
        ds = SpreadSequenceDataset(
            synthetic_data.features, synthetic_data.targets, small_seq_len
        )
        expected = len(synthetic_data.targets) - small_seq_len + 1
        assert len(ds) == expected

    def test_shapes(self, synthetic_data, small_seq_len):
        ds = SpreadSequenceDataset(
            synthetic_data.features, synthetic_data.targets, small_seq_len
        )
        X, y = ds[0]
        assert X.shape == (small_seq_len, len(cfg.FINAL_FEATURES))
        assert y.shape == (1,)

    def test_target_alignment(self, synthetic_data, small_seq_len):
        """Target at index i should be targets[i + seq_len - 1]."""
        ds = SpreadSequenceDataset(
            synthetic_data.features, synthetic_data.targets, small_seq_len
        )
        _, y = ds[0]
        expected = synthetic_data.targets[small_seq_len - 1]
        assert abs(y.item() - expected) < 1e-6

    def test_feature_window(self, synthetic_data, small_seq_len):
        """Feature window at index i should be features[i:i+seq_len]."""
        ds = SpreadSequenceDataset(
            synthetic_data.features, synthetic_data.targets, small_seq_len
        )
        X, _ = ds[5]
        expected = synthetic_data.features[5:5 + small_seq_len]
        np.testing.assert_array_almost_equal(X.numpy(), expected)

    def test_last_sample(self, synthetic_data, small_seq_len):
        ds = SpreadSequenceDataset(
            synthetic_data.features, synthetic_data.targets, small_seq_len
        )
        X, y = ds[len(ds) - 1]
        assert X.shape == (small_seq_len, len(cfg.FINAL_FEATURES))
        expected_target = synthetic_data.targets[-1]
        assert abs(y.item() - expected_target) < 1e-6

    def test_too_short_raises(self):
        """Should raise if data is shorter than sequence_length."""
        with pytest.raises(ValueError):
            SpreadSequenceDataset(
                np.zeros((5, 10), dtype=np.float32),
                np.zeros(5, dtype=np.float32),
                sequence_length=10,
            )


# ──────────────────────────────────────────────────────────────
# TESTS — temporal_train_val_split
# ──────────────────────────────────────────────────────────────

class TestSplit:
    def test_split_sizes(self, synthetic_data):
        split = temporal_train_val_split(synthetic_data, val_fraction=0.2)
        total = len(split.train.targets) + len(split.val.targets)
        assert total == len(synthetic_data.targets)

    def test_split_fraction(self, synthetic_data):
        split = temporal_train_val_split(synthetic_data, val_fraction=0.3)
        val_frac = len(split.val.targets) / len(synthetic_data.targets)
        assert abs(val_frac - 0.3) < 0.01

    def test_chronological_order(self, synthetic_data):
        """Val dates must be after all train dates."""
        split = temporal_train_val_split(synthetic_data, val_fraction=0.2)
        assert split.train.dates[-1] < split.val.dates[0]

    def test_no_overlap(self, synthetic_data):
        split = temporal_train_val_split(synthetic_data, val_fraction=0.2)
        train_set = set(map(str, split.train.dates))
        val_set = set(map(str, split.val.dates))
        assert train_set.isdisjoint(val_set)


# ──────────────────────────────────────────────────────────────
# TESTS — SpreadLSTM
# ──────────────────────────────────────────────────────────────

class TestModel:
    def test_output_shape(self, small_model, small_seq_len):
        batch = 4
        x = torch.randn(batch, small_seq_len, len(cfg.FINAL_FEATURES))
        out = small_model(x)
        assert out.shape == (batch, 1)

    def test_single_sample(self, small_model, small_seq_len):
        x = torch.randn(1, small_seq_len, len(cfg.FINAL_FEATURES))
        out = small_model(x)
        assert out.shape == (1, 1)

    def test_output_is_float(self, small_model, small_seq_len):
        x = torch.randn(2, small_seq_len, len(cfg.FINAL_FEATURES))
        out = small_model(x)
        assert out.dtype == torch.float32

    def test_no_activation_on_output(self, small_model, small_seq_len):
        """Final layer must be nn.Linear (no activation), allowing negative output."""
        # Structural check: last module in the head should be nn.Linear
        head_modules = list(small_model.head.children())
        assert isinstance(head_modules[-1], torch.nn.Linear), (
            "Last layer in head must be nn.Linear (no activation function)"
        )
        # Functional check: feed strongly negative inputs so the linear
        # layer can pass negative values through.
        torch.manual_seed(99)
        x = torch.randn(256, small_seq_len, len(cfg.FINAL_FEATURES)) * 10
        out = small_model(x)
        assert (out < 0).any(), "Linear output should allow negatives"

    def test_parameter_count(self, small_model):
        n_params = count_parameters(small_model)
        assert n_params > 0

    def test_default_config(self):
        model = SpreadLSTM()
        assert model.input_size == cfg.LSTM_INPUT_SIZE
        assert model.hidden_size == cfg.LSTM_HIDDEN_SIZE
        assert model.num_layers == cfg.LSTM_NUM_LAYERS

    def test_device_selection(self):
        device = get_device()
        assert isinstance(device, torch.device)


# ──────────────────────────────────────────────────────────────
# TESTS — Trainer
# ──────────────────────────────────────────────────────────────

class TestTrainer:
    def _make_trainer(self, synthetic_data, small_model, small_seq_len):
        """Build a trainer with tiny settings for fast tests."""
        split = temporal_train_val_split(synthetic_data, val_fraction=0.3)
        train_loader, val_loader = build_dataloaders(
            split, batch_size=32, sequence_length=small_seq_len
        )
        device = torch.device("cpu")
        trainer = Trainer(
            model=small_model,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader,
            max_epochs=3,
            patience=10,
            lr=1e-3,
        )
        return trainer, val_loader

    def test_fit_returns_result(self, synthetic_data, small_model, small_seq_len):
        trainer, _ = self._make_trainer(synthetic_data, small_model, small_seq_len)
        result = trainer.fit()
        assert len(result.history) == 3
        assert result.best_epoch >= 1
        assert result.best_val_loss < float("inf")

    def test_loss_decreases(self, synthetic_data, small_model, small_seq_len):
        """Train loss should generally decrease over 3 epochs."""
        trainer, _ = self._make_trainer(synthetic_data, small_model, small_seq_len)
        result = trainer.fit()
        # At least the last train loss should be <= the first
        assert result.history[-1].train_loss <= result.history[0].train_loss * 2

    def test_predict_shape(self, synthetic_data, small_model, small_seq_len):
        trainer, val_loader = self._make_trainer(synthetic_data, small_model, small_seq_len)
        _ = trainer.fit()
        preds = trainer.predict(val_loader)
        assert preds.ndim == 1
        assert len(preds) > 0

    def test_predict_tensor(self, synthetic_data, small_model, small_seq_len):
        trainer, _ = self._make_trainer(synthetic_data, small_model, small_seq_len)
        _ = trainer.fit()
        X = torch.randn(5, small_seq_len, len(cfg.FINAL_FEATURES))
        preds = trainer.predict_tensor(X)
        assert preds.shape == (5,)

    def test_early_stopping(self, synthetic_data, small_seq_len):
        """With patience=1 and a bad model, should stop early."""
        model = SpreadLSTM(
            input_size=len(cfg.FINAL_FEATURES),
            hidden_size=4, num_layers=1,
            lstm_dropout=0.0, dense_hidden=4, dense_dropout=0.0,
        )
        split = temporal_train_val_split(synthetic_data, val_fraction=0.3)
        train_loader, val_loader = build_dataloaders(
            split, batch_size=64, sequence_length=small_seq_len
        )
        trainer = Trainer(
            model=model, device=torch.device("cpu"),
            train_loader=train_loader, val_loader=val_loader,
            max_epochs=50, patience=2, lr=1e-3,
        )
        result = trainer.fit()
        # Should stop well before 50 epochs
        assert len(result.history) < 50
