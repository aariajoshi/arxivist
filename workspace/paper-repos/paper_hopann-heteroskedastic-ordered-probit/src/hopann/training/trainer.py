"""
hopann.training.trainer — Training loop, early stopping, and hyperparameter search.

Implements:
    EarlyStopping         — monitors a metric and stops training when it stagnates
    HyperparameterSearcher — grid search over hidden_dim (Q) and learning rate
    Trainer               — main training loop with checkpointing and logging

ASSUMED hyperparameters (all marked per SIR):
    batch_size = 64          (conf 0.52)
    lr         = 1e-3        (conf 0.52)
    patience   = 15          (conf 0.52)
    Q          = 16          (conf 0.52; searched over grid)
"""

from __future__ import annotations

import copy
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from hopann.training.losses import OrderedProbitNLL
from hopann.evaluation.metrics import OrdinalMetrics

logger = logging.getLogger(__name__)


# =============================================================================
# Early Stopping
# =============================================================================

class EarlyStopping:
    """
    Monitors a validation metric and signals early stopping when it plateaus.

    Stops training when the monitored metric fails to improve by at least
    `min_delta` for `patience` consecutive epochs.

    Args:
        patience (int):   Number of epochs to wait without improvement.
                          ASSUMED: 15 (conf 0.52).
        min_delta (float): Minimum change to qualify as improvement.
        mode (str):        'min' (e.g. loss) or 'max' (e.g. F1-macro).

    Usage:
        stopper = EarlyStopping(patience=15)
        for epoch in range(max_epochs):
            val_loss = ...
            if stopper.step(val_loss):
                break  # stop training
        best_epoch = stopper.best_epoch
    """

    def __init__(
        self,
        patience: int = 15,     # ASSUMED: conf 0.52
        min_delta: float = 1e-6,
        mode: str = "min",
    ) -> None:
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got '{mode}'")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self._counter = 0
        self._best_value = float("inf") if mode == "min" else float("-inf")
        self.best_epoch = 0
        self._epoch = 0

    def step(self, value: float) -> bool:
        """
        Update the stopper with a new metric value.

        Args:
            value: Current epoch's monitored metric.

        Returns:
            True if training should stop, False otherwise.
        """
        self._epoch += 1
        if self._is_improvement(value):
            self._best_value = value
            self._counter = 0
            self.best_epoch = self._epoch
        else:
            self._counter += 1

        if self._counter >= self.patience:
            logger.info(
                "EarlyStopping triggered at epoch %d "
                "(best=%.6f at epoch %d, patience=%d).",
                self._epoch, self._best_value, self.best_epoch, self.patience,
            )
            return True
        return False

    def _is_improvement(self, value: float) -> bool:
        if self.mode == "min":
            return value < self._best_value - self.min_delta
        return value > self._best_value + self.min_delta

    def reset(self) -> None:
        """Reset the stopper state."""
        self._counter = 0
        self._best_value = float("inf") if self.mode == "min" else float("-inf")
        self.best_epoch = 0
        self._epoch = 0

    def __repr__(self) -> str:
        return (
            f"EarlyStopping(patience={self.patience}, mode='{self.mode}', "
            f"counter={self._counter})"
        )


# =============================================================================
# Hyperparameter Searcher
# =============================================================================

class HyperparameterSearcher:
    """
    Grid search over hidden_dim (Q) and learning rate for OPANN/HOPANN.

    The paper searches over Q (hidden nodes) but does not specify the exact grid.
    ASSUMED: grid = Q ∈ {8, 16, 32, 64} and lr ∈ {1e-2, 1e-3, 1e-4} (conf 0.52).

    Trains each configuration for `max_epochs` and returns the best one by
    validation NLL.

    Args:
        model_class:       OPANN or HOPANN class.
        model_kwargs:      Fixed kwargs for the model (input_dim, num_classes, etc.)
                           excluding hidden_dim (searched).
        hidden_dims (list): Q values to search. ASSUMED: [8, 16, 32, 64].
        learning_rates (list): lr values to search. ASSUMED: [1e-2, 1e-3, 1e-4].
        max_epochs (int):  Epochs per configuration. ASSUMED: 50 (quick eval).
        patience (int):    EarlyStopping patience during search.
        batch_size (int):  Training batch size. ASSUMED: 64 (conf 0.52).
        device (str):      Device to train on.
    """

    def __init__(
        self,
        model_class: type,
        model_kwargs: dict[str, Any],
        hidden_dims: list[int] | None = None,        # ASSUMED grid
        learning_rates: list[float] | None = None,   # ASSUMED grid
        max_epochs: int = 50,
        patience: int = 10,
        batch_size: int = 64,     # ASSUMED: conf 0.52
        device: str = "cpu",
    ) -> None:
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.hidden_dims = hidden_dims or [8, 16, 32, 64]    # ASSUMED grid
        self.learning_rates = learning_rates or [1e-2, 1e-3, 1e-4]  # ASSUMED grid
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.device = device

    def search(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> dict[str, Any]:
        """
        Run grid search and return the best hyperparameter configuration.

        Args:
            train_loader: DataLoader for training data.
            val_loader:   DataLoader for validation data.

        Returns:
            Dict with keys: 'hidden_dim', 'lr', 'val_nll', 'model_state_dict'.
        """
        loss_fn = OrderedProbitNLL()
        best_result: dict[str, Any] = {"val_nll": float("inf")}

        total_configs = len(self.hidden_dims) * len(self.learning_rates)
        logger.info(
            "HyperparameterSearcher: evaluating %d configurations...", total_configs
        )

        for q in self.hidden_dims:
            for lr in self.learning_rates:
                kwargs = {**self.model_kwargs, "hidden_dim": q}
                model = self.model_class(**kwargs).to(self.device)
                optimizer = optim.Adam(model.parameters(), lr=lr)
                stopper = EarlyStopping(patience=self.patience, mode="min")

                best_val_nll = float("inf")
                best_state = None

                for epoch in range(self.max_epochs):
                    model.train()
                    for xb, yb in train_loader:
                        xb, yb = xb.to(self.device), yb.to(self.device)
                        optimizer.zero_grad()
                        probs = model(xb)
                        loss = loss_fn(probs, yb)
                        loss.backward()
                        optimizer.step()

                    # Validation
                    model.eval()
                    val_nll_vals = []
                    with torch.no_grad():
                        for xv, yv in val_loader:
                            xv, yv = xv.to(self.device), yv.to(self.device)
                            probs_v = model(xv)
                            val_nll_vals.append(loss_fn(probs_v, yv).item())
                    val_nll = float(np.mean(val_nll_vals))

                    if val_nll < best_val_nll:
                        best_val_nll = val_nll
                        best_state = copy.deepcopy(model.state_dict())

                    if stopper.step(val_nll):
                        break

                logger.info(
                    "  Q=%d, lr=%g → val_nll=%.4f", q, lr, best_val_nll
                )
                if best_val_nll < best_result["val_nll"]:
                    best_result = {
                        "hidden_dim": q,
                        "lr": lr,
                        "val_nll": best_val_nll,
                        "model_state_dict": best_state,
                    }

        logger.info(
            "Best config: Q=%d, lr=%g, val_nll=%.4f",
            best_result["hidden_dim"], best_result["lr"], best_result["val_nll"],
        )
        return best_result

    def __repr__(self) -> str:
        return (
            f"HyperparameterSearcher(model={self.model_class.__name__}, "
            f"hidden_dims={self.hidden_dims}, lrs={self.learning_rates})"
        )


# =============================================================================
# Trainer
# =============================================================================

class Trainer:
    """
    Main training loop for OPANN and HOPANN.

    Supports:
        - Early stopping (monitored on validation NLL)
        - Checkpoint saving (best model and periodic)
        - Training summary at start (param count, dataset size, steps/epoch)
        - Metric logging every N steps
        - Resuming from checkpoint

    Args:
        model (nn.Module):      OPANN or HOPANN instance.
        optimizer (Optimizer):  PyTorch optimiser.
        loss_fn (nn.Module):    OrderedProbitNLL instance.
        device (str):           'cpu', 'cuda', or 'mps'.
        output_dir (str):       Directory to save checkpoints.
        max_epochs (int):       Maximum training epochs.
        patience (int):         EarlyStopping patience.
                                ASSUMED: 15 (conf 0.52).
        log_every_n_steps (int): Log training metrics every N batches.
        save_every_n_epochs (int): Save periodic checkpoint every N epochs.
        monitor_metric (str):   Metric to monitor for early stopping.
                                Default: 'val_nll'.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module,
        device: str,
        output_dir: str,
        max_epochs: int = 200,     # ASSUMED: reasonable upper bound
        patience: int = 15,        # ASSUMED: conf 0.52
        log_every_n_steps: int = 50,
        save_every_n_epochs: int = 10,
        monitor_metric: str = "val_nll",
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.output_dir = Path(output_dir)
        self.max_epochs = max_epochs
        self.patience = patience
        self.log_every_n_steps = log_every_n_steps
        self.save_every_n_epochs = save_every_n_epochs
        self.monitor_metric = monitor_metric

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._early_stopping = EarlyStopping(patience=patience, mode="min")
        self._best_val_nll = float("inf")
        self._best_epoch = 0
        self._global_step = 0
        self._history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> dict[str, Any]:
        """
        Train the model with the configured settings.

        Args:
            train_loader: DataLoader for training data.
            val_loader:   DataLoader for validation data.

        Returns:
            Training history dict with per-epoch metrics.
        """
        self._print_training_summary(train_loader, val_loader)
        self.model.to(self.device)

        for epoch in range(1, self.max_epochs + 1):
            t0 = time.time()
            train_nll = self._train_epoch(train_loader, epoch)
            val_nll, val_metrics = self._validate(val_loader)
            elapsed = time.time() - t0

            log_entry = {
                "epoch": epoch,
                "train_nll": train_nll,
                "val_nll": val_nll,
                **{f"val_{k}": v for k, v in val_metrics.items()},
                "elapsed_s": elapsed,
            }
            self._history.append(log_entry)

            logger.info(
                "Epoch %3d/%d | train_nll=%.4f | val_nll=%.4f | "
                "val_f1=%.4f | %.1fs",
                epoch, self.max_epochs, train_nll, val_nll,
                val_metrics.get("f1_macro", float("nan")), elapsed,
            )

            # Save periodic checkpoint
            if epoch % self.save_every_n_epochs == 0:
                self._save_checkpoint(epoch, suffix=f"epoch_{epoch}")

            # Save best checkpoint
            if val_nll < self._best_val_nll:
                self._best_val_nll = val_nll
                self._best_epoch = epoch
                self._save_checkpoint(epoch, suffix="best")
                logger.debug("New best model at epoch %d (val_nll=%.4f)", epoch, val_nll)

            # Early stopping
            if self._early_stopping.step(val_nll):
                logger.info(
                    "Early stopping triggered. Best epoch: %d", self._best_epoch
                )
                break

        self._save_history()
        logger.info(
            "Training complete. Best epoch=%d, best val_nll=%.4f",
            self._best_epoch, self._best_val_nll,
        )
        return {"history": self._history, "best_epoch": self._best_epoch}

    def load_best(self) -> None:
        """Load the best checkpoint back into the model."""
        ckpt_path = self.output_dir / "checkpoint_best.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Best checkpoint not found at {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Loaded best checkpoint from epoch %d.", ckpt.get("epoch", "?"))

    def resume(self, checkpoint_path: str) -> int:
        """
        Resume training from a checkpoint file.

        Args:
            checkpoint_path: Path to the .pt checkpoint.

        Returns:
            Epoch to resume from (next epoch index).
        """
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        resumed_epoch = ckpt.get("epoch", 0)
        self._global_step = ckpt.get("global_step", 0)
        logger.info("Resumed from checkpoint at epoch %d.", resumed_epoch)
        return resumed_epoch

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _train_epoch(self, loader: DataLoader, epoch: int) -> float:
        """Run one training epoch and return mean NLL."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch_idx, (xb, yb) in enumerate(loader):
            xb, yb = xb.to(self.device), yb.to(self.device)

            self.optimizer.zero_grad()
            probs = self.model(xb)
            loss = self.loss_fn(probs, yb)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            self._global_step += 1

            if self._global_step % self.log_every_n_steps == 0:
                logger.debug(
                    "Step %d | epoch %d | batch %d | loss=%.4f",
                    self._global_step, epoch, batch_idx, loss.item(),
                )

        return total_loss / max(n_batches, 1)

    def _validate(
        self, loader: DataLoader
    ) -> tuple[float, dict[str, float]]:
        """
        Run validation pass and compute NLL + OrdinalMetrics.

        Returns:
            (val_nll, metrics_dict)
        """
        self.model.eval()
        all_probs = []
        all_labels = []
        total_nll = 0.0
        n_batches = 0

        with torch.no_grad():
            for xv, yv in loader:
                xv, yv = xv.to(self.device), yv.to(self.device)
                probs = self.model(xv)
                loss = self.loss_fn(probs, yv)
                total_nll += loss.item()
                n_batches += 1
                all_probs.append(probs.cpu().numpy())
                all_labels.append(yv.cpu().numpy())

        all_probs_np = np.concatenate(all_probs, axis=0)   # (N, J)
        all_labels_np = np.concatenate(all_labels, axis=0)  # (N,)

        metrics = OrdinalMetrics.compute(
            y_true=all_labels_np,
            probs=all_probs_np,
        )
        val_nll = total_nll / max(n_batches, 1)
        return val_nll, metrics

    def _save_checkpoint(self, epoch: int, suffix: str) -> None:
        """Save model + optimizer state to disk."""
        ckpt = {
            "epoch": epoch,
            "global_step": self._global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_nll": self._best_val_nll,
        }
        path = self.output_dir / f"checkpoint_{suffix}.pt"
        torch.save(ckpt, path)
        logger.debug("Saved checkpoint: %s", path)

    def _save_history(self) -> None:
        """Persist training history to JSON."""
        path = self.output_dir / "training_history.json"
        with open(path, "w") as f:
            json.dump(self._history, f, indent=2, default=str)
        logger.info("Training history saved to %s", path)

    def _print_training_summary(
        self, train_loader: DataLoader, val_loader: DataLoader
    ) -> None:
        """Print model/data statistics at the start of training."""
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_train = len(train_loader.dataset)  # type: ignore[arg-type]
        n_val = len(val_loader.dataset)      # type: ignore[arg-type]
        n_steps_per_epoch = len(train_loader)

        logger.info("=" * 60)
        logger.info("Training Summary")
        logger.info("=" * 60)
        logger.info("Model:             %s", repr(self.model))
        logger.info("Trainable params:  %d", n_params)
        logger.info("Train samples:     %d", n_train)
        logger.info("Val samples:       %d", n_val)
        logger.info("Steps per epoch:   %d", n_steps_per_epoch)
        logger.info("Max epochs:        %d", self.max_epochs)
        logger.info("Early stop patience: %d", self.patience)
        logger.info("Device:            %s", self.device)
        logger.info("Output dir:        %s", self.output_dir)
        logger.info("=" * 60)

    def __repr__(self) -> str:
        return (
            f"Trainer(model={self.model.__class__.__name__}, "
            f"max_epochs={self.max_epochs}, patience={self.patience})"
        )
