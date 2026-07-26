"""
utils/config.py — Configuration loading, validation, and seed utilities.

Implements OmegaConf-based config management for the Physics-Guided Motion Loss
training pipeline (arXiv 2506.02244v2).
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


# ---------------------------------------------------------------------------
# Seed utility
# ---------------------------------------------------------------------------

def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch for reproducibility.

    Args:
        seed: Integer seed value.
        deterministic: If True, enables torch.use_deterministic_algorithms.
            Note: this may significantly slow CUDA operations.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Config loading & validation
# ---------------------------------------------------------------------------

def load_config(path: str, overrides: Optional[list] = None) -> DictConfig:
    """Load and validate a config YAML file.

    Args:
        path: Path to config YAML.
        overrides: Optional list of dot-notation overrides, e.g. ["training.lr=1e-4"].

    Returns:
        Validated OmegaConf DictConfig.

    Raises:
        FileNotFoundError: If config path does not exist.
        ValueError: If required fields are missing or values are out of range.
    """
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg: DictConfig = OmegaConf.load(cfg_path)

    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)

    _validate_config(cfg)
    return cfg


def _validate_config(cfg: DictConfig) -> None:
    """Raise ValueError on invalid config values."""
    # Spectral
    rho = cfg.spectral.low_pass_rho
    if not (0.0 < rho <= 1.0):
        raise ValueError(f"spectral.low_pass_rho must be in (0, 1], got {rho}")

    if cfg.spectral.Nr_rings < 4:
        raise ValueError(f"spectral.Nr_rings must be >= 4, got {cfg.spectral.Nr_rings}")

    if cfg.spectral.delta_tolerance < 1:
        raise ValueError("spectral.delta_tolerance must be >= 1 (one temporal-freq bin)")

    # Losses
    tau = cfg.losses.softmax_temperature
    if tau <= 0:
        raise ValueError(f"losses.softmax_temperature must be > 0, got {tau}")

    w = cfg.losses.physics_loss_weight
    if not (0.0 < w <= 10.0):
        raise ValueError(f"losses.physics_loss_weight must be in (0, 10], got {w}")

    # Training
    if cfg.training.epochs < 1:
        raise ValueError(f"training.epochs must be >= 1, got {cfg.training.epochs}")

    if cfg.training.learning_rate <= 0:
        raise ValueError(f"training.learning_rate must be > 0")

    backbone_choices = {"open_sora", "mvdit", "hunyuan"}
    if cfg.training.backbone not in backbone_choices:
        raise ValueError(
            f"training.backbone must be one of {backbone_choices}, "
            f"got '{cfg.training.backbone}'"
        )

    # LoRA
    if cfg.lora.enabled:
        if cfg.lora.rank < 1:
            raise ValueError(f"lora.rank must be >= 1, got {cfg.lora.rank}")
        if not cfg.lora.target_modules:
            raise ValueError("lora.target_modules must not be empty when lora.enabled=True")


def config_summary(cfg: DictConfig) -> str:
    """Return a human-readable summary string for logging at training start."""
    lines = [
        "=" * 60,
        "Physics-Guided Motion Loss — Config Summary",
        "=" * 60,
        f"  Backbone         : {cfg.training.backbone}",
        f"  LoRA enabled     : {cfg.lora.enabled}",
        f"  Physics weight   : {cfg.losses.physics_loss_weight}",
        f"  Softmax tau      : {cfg.losses.softmax_temperature}",
        f"  Low-pass rho     : {cfg.spectral.low_pass_rho} ({cfg.spectral.low_pass_rho**3*100:.1f}% coefficients)",
        f"  Nr rings / M bins: {cfg.spectral.Nr_rings} / {cfg.spectral.M_angular_bins}",
        f"  Temporal window  : {cfg.data.T} frames ({cfg.spectral.temporal_window})",
        f"  LR / epochs      : {cfg.training.learning_rate} / {cfg.training.epochs}",
        f"  Precision        : backbone={cfg.training.mixed_precision_backbone}, spectral={cfg.training.mixed_precision_spectral}",
        f"  Seed             : {cfg.training.seed}",
        "=" * 60,
    ]
    return "\n".join(lines)
