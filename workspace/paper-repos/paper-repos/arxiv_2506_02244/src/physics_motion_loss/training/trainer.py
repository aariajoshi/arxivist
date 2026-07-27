"""
training/trainer.py — Physics-guided fine-tuning loop for video diffusion models.

Implements the training setup from Section 4.1 of arXiv 2506.02244v2:
  - 4 epochs, cosine LR from 2e-5, 4× A100
  - Physics loss computed at every diffusion timestep on reconstructed x̂₀
  - Physics loss added to backbone denoising objective with weight=0.1
  - BF16 for backbone, FP32 for spectral/solver blocks

ASSUMED: AdamW optimizer — not stated in paper (SIR conf 0.65).
ASSUMED: Physics loss weight is constant throughout training (conf 0.78).
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.optim import AdamW  # ASSUMED — see module docstring
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from ..losses.physics_motion_loss import PhysicsMotionLoss
from ..utils.config import config_summary, set_seed

logger = logging.getLogger(__name__)


class PhysicsGuidedTrainer:
    """Fine-tunes a video diffusion backbone with the physics motion loss.

    Paper reference: Section 4.1.

    Args:
        cfg:          OmegaConf DictConfig loaded from configs/config.yaml.
        model:        Pretrained backbone (Open-Sora / MVDIT / Hunyuan).
        train_loader: DataLoader yielding {'video': Tensor, 'prompt': str, ...}.
        val_loader:   Optional validation DataLoader.
        output_dir:   Directory for checkpoints and logs.
    """

    def __init__(
        self,
        cfg: DictConfig,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        output_dir: str = "outputs/",
    ) -> None:
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Device
        self.device = torch.device(
            cfg.hardware.device if torch.cuda.is_available() else "cpu"
        )
        self.model = self.model.to(self.device)

        # Physics loss module (FP32 enforced internally)
        self.physics_loss = PhysicsMotionLoss.from_config(cfg).to(self.device)

        # Optimizer — ASSUMED: AdamW (conf 0.65, not stated in paper)
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.training.learning_rate,
            betas=tuple(cfg.training.optimizer_betas),
            weight_decay=cfg.training.weight_decay,
        )

        # LR scheduler: cosine annealing (Sec 4.1)
        total_steps = len(train_loader) * cfg.training.epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=total_steps, eta_min=0.0
        )

        # Mixed precision scaler for backbone (BF16 if requested)
        self.use_amp = cfg.training.mixed_precision_backbone == "bf16"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Metrics tracking
        self.global_step = 0
        self.best_val_metric = float("inf")

    def train(self) -> None:
        """Run the full training loop."""
        cfg = self.cfg
        set_seed(cfg.training.seed, cfg.training.deterministic)

        logger.info(config_summary(cfg))
        logger.info(self._training_summary())

        for epoch in range(cfg.training.epochs):
            self._run_epoch(epoch)

            if self.val_loader is not None:
                val_loss = self._run_validation(epoch)
                if val_loss < self.best_val_metric:
                    self.best_val_metric = val_loss
                    self._save_checkpoint(epoch, tag="best")

            self._save_checkpoint(epoch, tag=f"epoch_{epoch:03d}")

        logger.info("Training complete.")

    def _run_epoch(self, epoch: int) -> None:
        """Run one training epoch."""
        self.model.train()
        cfg = self.cfg
        t0 = time.time()

        for step, batch in enumerate(self.train_loader):
            metrics = self._training_step(batch)
            self.global_step += 1

            if self.global_step % cfg.training.log_every_n_steps == 0:
                elapsed = time.time() - t0
                logger.info(
                    f"[Epoch {epoch+1}/{cfg.training.epochs} | "
                    f"Step {self.global_step}] "
                    f"loss={metrics['total_loss']:.4f} | "
                    f"denoising={metrics['denoising_loss']:.4f} | "
                    f"L_motion={metrics['L_motion']:.4f} | "
                    f"L_trans={metrics['L_trans']:.4f} | "
                    f"L_rot={metrics['L_rot']:.4f} | "
                    f"L_scale={metrics['L_scale']:.4f} | "
                    f"w=({metrics['w_trans']:.2f},{metrics['w_rot']:.2f},{metrics['w_scale']:.2f}) | "
                    f"lr={self.scheduler.get_last_lr()[0]:.2e} | "
                    f"elapsed={elapsed:.1f}s"
                )
                t0 = time.time()

            if self.global_step % cfg.training.save_every_n_steps == 0:
                self._save_checkpoint(epoch, tag=f"step_{self.global_step:07d}")

    def _training_step(self, batch: Dict) -> Dict[str, float]:
        """Single forward + backward + optimiser step.

        Paper reference: Section 4.1 — "At every diffusion step t we reconstruct
        x̂₀, evaluate the physics-informed frequency loss on x̂₀, and add it to
        the standard denoising objective."
        """
        cfg = self.cfg
        self.optimizer.zero_grad()

        video = batch["video"].to(self.device)   # [B, C, T, H, W]

        with torch.cuda.amp.autocast(
            enabled=self.use_amp, dtype=torch.bfloat16
        ):
            # --- Backbone forward: compute denoising loss and x̂₀ ---
            # The backbone returns (denoising_loss, x0_hat) — exact API depends
            # on the specific backbone. See src/physics_motion_loss/data/backbones/
            model_out = self.model(video, prompts=batch.get("prompt", None))
            denoising_loss = model_out["loss"]
            x0_hat = model_out["x0_hat"]        # [B, C, T, H, W], float32

        # --- Physics loss (always FP32 — FP32Context enforced inside) ---
        phys_out = self.physics_loss(x0_hat.float())
        L_motion = phys_out["loss"]

        # --- Total loss (Sec 4.1, Table 5: weight=0.1) ---
        # ASSUMED: constant weight throughout training (conf 0.78)
        total_loss = denoising_loss + cfg.losses.physics_loss_weight * L_motion

        # --- Backward ---
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()

        return {
            "total_loss":     total_loss.item(),
            "denoising_loss": denoising_loss.item(),
            "L_motion":       L_motion.item(),
            "L_trans":        phys_out["L_trans"].item(),
            "L_rot":          phys_out["L_rot"].item(),
            "L_scale":        phys_out["L_scale"].item(),
            "w_trans":        phys_out["w_trans"].item(),
            "w_rot":          phys_out["w_rot"].item(),
            "w_scale":        phys_out["w_scale"].item(),
        }

    @torch.no_grad()
    def _run_validation(self, epoch: int) -> float:
        """Run validation and return mean total loss."""
        self.model.eval()
        total = 0.0
        n = 0
        for batch in self.val_loader:
            video = batch["video"].to(self.device)
            with torch.cuda.amp.autocast(enabled=self.use_amp, dtype=torch.bfloat16):
                model_out = self.model(video, prompts=batch.get("prompt", None))
                denoising_loss = model_out["loss"]
                x0_hat = model_out["x0_hat"]

            phys_out = self.physics_loss(x0_hat.float())
            loss = (denoising_loss + self.cfg.losses.physics_loss_weight * phys_out["loss"])
            total += loss.item()
            n += 1

        val_loss = total / max(n, 1)
        logger.info(f"[Epoch {epoch+1}] Val loss: {val_loss:.4f}")
        return val_loss

    def _save_checkpoint(self, epoch: int, tag: str) -> None:
        """Save model checkpoint and optimiser state."""
        ckpt_path = self.output_dir / f"checkpoint_{tag}.pt"
        torch.save(
            {
                "epoch": epoch,
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "best_val_metric": self.best_val_metric,
                "cfg": self.cfg,
            },
            ckpt_path,
        )
        logger.info(f"Checkpoint saved: {ckpt_path}")

    def load_checkpoint(self, path: str) -> None:
        """Resume training from a saved checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.global_step = ckpt.get("global_step", 0)
        self.best_val_metric = ckpt.get("best_val_metric", float("inf"))
        logger.info(
            f"Resumed from {path} at step {self.global_step}, "
            f"epoch {ckpt['epoch']}"
        )

    def _training_summary(self) -> str:
        """Produce a human-readable training summary for logging."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        total_steps = len(self.train_loader) * self.cfg.training.epochs
        return (
            "\n" + "=" * 60 + "\n"
            f"  Training Summary\n"
            "=" * 60 + "\n"
            f"  Model params (total):     {total_params:,}\n"
            f"  Model params (trainable): {trainable_params:,}\n"
            f"  Dataset size:             {len(self.train_loader.dataset):,}\n"
            f"  Batch size:               {self.train_loader.batch_size}\n"
            f"  Steps per epoch:          {len(self.train_loader)}\n"
            f"  Total steps:              {total_steps:,}\n"
            f"  Physics loss weight:      {self.cfg.losses.physics_loss_weight}\n"
            + "=" * 60
        )
