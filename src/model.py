"""
model.py — SpreadLSTM neural network architecture.
====================================================
Responsibilities:
    1. Define the LSTM-based network that maps a window of
       (sequence_length × num_features) observations to a single
       continuous spread prediction.
    2. Provide helper functions for device selection, model summary,
       and parameter counting.

Architecture
------------
::

    Input  (batch, seq_len, 45)
      │
      ▼
    ┌──────────────────────┐
    │  LSTM (2 layers)     │   hidden_size = 128
    │  dropout = 0.2       │   captures temporal dependencies
    └──────────┬───────────┘
               │  take last hidden state → (batch, 128)
               ▼
    ┌──────────────────────┐
    │  Linear(128 → 64)    │
    │  ReLU                │
    │  Dropout(0.3)        │
    └──────────┬───────────┘
               ▼
    ┌──────────────────────┐
    │  Linear(64 → 1)      │   single node, **linear** activation
    └──────────┬───────────┘   (continuous regression output)
               ▼
    Output  (batch, 1)

Design notes:
    * Two stacked LSTM layers with inter-layer dropout mitigate
      vanishing gradients while regularising to avoid overfitting.
    * The dense head reduces dimensionality gradually (128 → 64 → 1)
      with a ReLU non-linearity and dropout.
    * The output layer has **no activation** (linear), because we are
      predicting a continuous value (the spread in scaled EUR space).
    * After training, predictions are inverse-transformed via the
      target scaler back to real EUR.
"""

from __future__ import annotations

import torch
import torch.nn as nn

import config as cfg


# ──────────────────────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────────────────────

class SpreadLSTM(nn.Module):
    """
    LSTM network for electricity market spread prediction.

    Parameters
    ----------
    input_size : int
        Number of features per timestep (default: 45).
    hidden_size : int
        LSTM hidden state dimensionality.
    num_layers : int
        Number of stacked LSTM layers.
    lstm_dropout : float
        Dropout probability between LSTM layers (>= 2 layers required).
    dense_hidden : int
        Width of the intermediate dense layer.
    dense_dropout : float
        Dropout probability in the dense head.
    """

    def __init__(
        self,
        input_size: int | None = None,
        hidden_size: int | None = None,
        num_layers: int | None = None,
        lstm_dropout: float | None = None,
        dense_hidden: int | None = None,
        dense_dropout: float | None = None,
    ):
        super().__init__()

        self.input_size = input_size or cfg.LSTM_INPUT_SIZE
        self.hidden_size = hidden_size or cfg.LSTM_HIDDEN_SIZE
        self.num_layers = num_layers or cfg.LSTM_NUM_LAYERS
        self.lstm_dropout = lstm_dropout if lstm_dropout is not None else cfg.LSTM_DROPOUT
        self.dense_hidden = dense_hidden or cfg.DENSE_HIDDEN_SIZE
        self.dense_dropout = dense_dropout if dense_dropout is not None else cfg.DENSE_DROPOUT

        # ── LSTM encoder ──
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.lstm_dropout if self.num_layers > 1 else 0.0,
        )

        # ── Dense regression head ──
        self.head = nn.Sequential(
            nn.Linear(self.hidden_size, self.dense_hidden),
            nn.ReLU(),
            nn.Dropout(self.dense_dropout),
            nn.Linear(self.dense_hidden, 1),  # single output, linear activation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(batch, sequence_length, input_size)``.

        Returns
        -------
        torch.Tensor
            Shape ``(batch, 1)`` — predicted spread (scaled).
        """
        # lstm_out: (batch, seq_len, hidden_size)
        lstm_out, _ = self.lstm(x)

        # Take the output at the LAST timestep
        last_hidden = lstm_out[:, -1, :]          # (batch, hidden_size)

        return self.head(last_hidden)             # (batch, 1)


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Select the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: SpreadLSTM, device: torch.device) -> None:
    """Print a concise architecture summary."""
    total_params = count_parameters(model)

    print(f"  Architecture: SpreadLSTM")
    print(f"  Device:       {device}")
    print()
    print(f"  LSTM Encoder:")
    print(f"    Input size:    {model.input_size} features")
    print(f"    Hidden size:   {model.hidden_size}")
    print(f"    Num layers:    {model.num_layers}")
    print(f"    LSTM dropout:  {model.lstm_dropout}")
    print()
    print(f"  Dense Head:")
    print(f"    {model.hidden_size} -> {model.dense_hidden} (ReLU, dropout={model.dense_dropout})")
    print(f"    {model.dense_hidden} -> 1 (linear activation)")
    print()
    print(f"  Total trainable parameters: {total_params:,}")
    print()
    print(f"  Anti-overfitting measures:")
    print(f"    - LSTM inter-layer dropout:  {model.lstm_dropout}")
    print(f"    - Dense head dropout:        {model.dense_dropout}")
    print(f"    - L2 regularisation:         weight_decay={cfg.WEIGHT_DECAY}")
    print(f"    - Early stopping:            patience={cfg.EARLY_STOPPING_PATIENCE}")
    print(f"    - Gradient clipping:         max_norm={cfg.GRADIENT_CLIP_NORM}")
    print(f"    - LR scheduling:             ReduceLROnPlateau(factor={cfg.SCHEDULER_FACTOR})")
