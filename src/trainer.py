"""
trainer.py — Training loop with early stopping and LR scheduling.
==================================================================
Responsibilities:
    1. Run the training loop over ``MAX_EPOCHS``, computing MSE loss
       on both training and validation sets each epoch.
    2. Implement **early stopping** — halt training when the
       validation loss stops improving, preventing overfitting.
    3. Apply a **ReduceLROnPlateau** scheduler that halves the
       learning rate when validation loss plateaus.
    4. Clip gradients to ``GRADIENT_CLIP_NORM`` to guard against
       exploding gradients, a known LSTM issue.
    5. Record per-epoch metrics so training curves can be plotted.

Step 4 enhancements:
    * **min_delta** — val loss must improve by at least this threshold
      to count as genuine improvement (prevents noise-driven resets).
    * **Overfitting gap tracking** — each epoch records the gap
      (val_loss − train_loss) so the divergence can be visualised.

Design notes:
    * The trainer checkpoints the best model (lowest validation loss)
      and restores it at the end if early stopping triggered.
    * All hyperparameters default to ``config.py`` but can be
      overridden for experimentation.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config as cfg


# ──────────────────────────────────────────────────────────────
# DATA CLASSES
# ──────────────────────────────────────────────────────────────

@dataclass
class EpochMetrics:
    """Metrics recorded after a single epoch."""
    epoch: int
    train_loss: float
    val_loss: float
    learning_rate: float
    elapsed_sec: float
    overfit_gap: float = 0.0       # (Step 4) val_loss − train_loss


@dataclass
class TrainingResult:
    """Full training outcome."""
    history: list[EpochMetrics] = field(default_factory=list)
    best_epoch: int = 0
    best_val_loss: float = float("inf")
    stopped_early: bool = False
    total_time_sec: float = 0.0
    overfit_gap_at_best: float = 0.0   # (Step 4) gap at best epoch


# ──────────────────────────────────────────────────────────────
# TRAINER
# ──────────────────────────────────────────────────────────────

class Trainer:
    """
    Encapsulates the full training procedure.

    Parameters
    ----------
    model : nn.Module
    device : torch.device
    train_loader : DataLoader
    val_loader : DataLoader
    max_epochs : int
    lr : float
    weight_decay : float
    patience : int
        Early stopping patience (epochs).
    min_delta : float
        Minimum improvement in val loss to reset patience (Step 4).
    clip_norm : float
        Gradient clipping max norm.
    scheduler_factor : float
    scheduler_patience : int
    quiet : bool
        If True, suppress per-epoch print output.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int | None = None,
        lr: float | None = None,
        weight_decay: float | None = None,
        patience: int | None = None,
        min_delta: float | None = None,
        clip_norm: float | None = None,
        scheduler_factor: float | None = None,
        scheduler_patience: int | None = None,
        quiet: bool = False,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.quiet = quiet

        self.max_epochs = max_epochs or cfg.MAX_EPOCHS
        self.lr = lr or cfg.LEARNING_RATE
        self.weight_decay = weight_decay or cfg.WEIGHT_DECAY
        self.patience = patience or cfg.EARLY_STOPPING_PATIENCE
        self.min_delta = min_delta if min_delta is not None else cfg.EARLY_STOPPING_MIN_DELTA
        self.clip_norm = clip_norm or cfg.GRADIENT_CLIP_NORM
        self.sched_factor = scheduler_factor or cfg.SCHEDULER_FACTOR
        self.sched_patience = scheduler_patience or cfg.SCHEDULER_PATIENCE

        # Loss function — MSE for continuous regression
        self.criterion = nn.MSELoss()

        # Optimiser — AdamW includes L2 regularisation via weight_decay
        self.optimiser = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # LR scheduler — reduce learning rate when val loss plateaus
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimiser,
            mode="min",
            factor=self.sched_factor,
            patience=self.sched_patience,
        )

        # Early stopping state
        self._best_val_loss = float("inf")
        self._best_state = None
        self._epochs_without_improvement = 0

    # ── public API ───────────────────────────────────────────

    def fit(self) -> TrainingResult:
        """
        Run the full training loop.

        Returns
        -------
        TrainingResult
            Contains per-epoch history, best metrics, and timing.
        """
        result = TrainingResult()
        t0 = time.time()

        for epoch in range(1, self.max_epochs + 1):
            t_epoch = time.time()

            train_loss = self._train_one_epoch()
            val_loss = self._evaluate()

            current_lr = self.optimiser.param_groups[0]["lr"]

            # Step 4: compute overfitting gap
            overfit_gap = val_loss - train_loss

            metrics = EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                learning_rate=current_lr,
                elapsed_sec=time.time() - t_epoch,
                overfit_gap=overfit_gap,
            )
            result.history.append(metrics)

            # LR scheduling
            self.scheduler.step(val_loss)

            # Early stopping check (Step 4: min_delta threshold)
            if val_loss < self._best_val_loss - self.min_delta:
                self._best_val_loss = val_loss
                self._best_state = copy.deepcopy(self.model.state_dict())
                self._epochs_without_improvement = 0
                result.best_epoch = epoch
                result.best_val_loss = val_loss
                result.overfit_gap_at_best = overfit_gap
                marker = " *"
            else:
                self._epochs_without_improvement += 1
                marker = ""

            # Logging
            if not self.quiet and (epoch <= 5 or epoch % 5 == 0 or marker):
                print(
                    f"  Epoch {epoch:>3}/{self.max_epochs} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f} | "
                    f"Gap: {overfit_gap:+.4f} | "
                    f"LR: {current_lr:.2e} | "
                    f"{metrics.elapsed_sec:.1f}s{marker}"
                )

            if self._epochs_without_improvement >= self.patience:
                if not self.quiet:
                    print(
                        f"\n  Early stopping at epoch {epoch} "
                        f"(no improvement > {self.min_delta} for "
                        f"{self.patience} epochs)"
                    )
                result.stopped_early = True
                break

        # Restore best model
        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)

        result.total_time_sec = time.time() - t0
        return result

    def predict(self, loader: DataLoader) -> np.ndarray:
        """
        Generate predictions for an entire DataLoader.

        Returns
        -------
        np.ndarray
            Shape ``(N,)`` — predicted spread_scaled values.
        """
        self.model.eval()
        preds = []

        with torch.no_grad():
            for X_batch, _ in loader:
                X_batch = X_batch.to(self.device)
                y_hat = self.model(X_batch)        # (batch, 1)
                preds.append(y_hat.cpu().numpy())

        return np.concatenate(preds, axis=0).flatten()

    def predict_tensor(self, X: torch.Tensor) -> np.ndarray:
        """
        Predict from a raw feature tensor (no DataLoader needed).

        Parameters
        ----------
        X : torch.Tensor
            Shape ``(N, seq_len, features)``.

        Returns
        -------
        np.ndarray
            Shape ``(N,)``
        """
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            return self.model(X).cpu().numpy().flatten()

    # ── internal ─────────────────────────────────────────────

    def _train_one_epoch(self) -> float:
        """Run one training epoch; return average loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in self.train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimiser.zero_grad()
            y_hat = self.model(X_batch)
            loss = self.criterion(y_hat, y_batch)
            loss.backward()

            # Gradient clipping — prevents exploding gradients in LSTM
            nn.utils.clip_grad_norm_(
                self.model.parameters(), self.clip_norm
            )

            self.optimiser.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    def _evaluate(self) -> float:
        """Run validation; return average loss."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                y_hat = self.model(X_batch)
                loss = self.criterion(y_hat, y_batch)

                total_loss += loss.item()
                n_batches += 1

        return total_loss / max(n_batches, 1)


# ──────────────────────────────────────────────────────────────
# PRINTING
# ──────────────────────────────────────────────────────────────

def print_training_summary(result: TrainingResult) -> None:
    """Print a concise summary after training completes."""
    final = result.history[-1] if result.history else None
    print(f"  Total epochs run:    {len(result.history)}")
    print(f"  Best epoch:          {result.best_epoch}")
    print(f"  Best val loss (MSE): {result.best_val_loss:.6f}")
    if final:
        print(f"  Final train loss:    {final.train_loss:.6f}")
        print(f"  Final val loss:      {final.val_loss:.6f}")
        print(f"  Final overfit gap:   {final.overfit_gap:+.4f}")
        print(f"  Final LR:            {final.learning_rate:.2e}")
    print(f"  Gap at best epoch:   {result.overfit_gap_at_best:+.4f}")
    print(f"  Early stopped:       {result.stopped_early}")
    print(f"  Total training time: {result.total_time_sec:.1f}s")
