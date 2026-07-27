"""
hopann.utils.reproducibility — Reproducibility utilities for the HOPANN pipeline.

Seeds Python's random module, NumPy, PyTorch (CPU and CUDA), and optionally
enables deterministic CUDA operations.

Usage:
    rm = ReproducibilityManager(seed=42, deterministic=False)
    rm.seed_everything()

Note on deterministic mode (from PyTorch docs):
    Setting torch.use_deterministic_algorithms(True) may cause a performance
    overhead and may raise RuntimeError for operations without a deterministic
    implementation. Use only when exact reproducibility is required.
"""

from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ReproducibilityManager:
    """
    Sets all random seeds and configures deterministic behaviour.

    Seeds: Python random, NumPy, PyTorch CPU, PyTorch CUDA (all GPUs).
    Also sets the PYTHONHASHSEED environment variable.

    Args:
        seed (int):          Random seed. Default: 42.
        deterministic (bool): If True, enable torch.use_deterministic_algorithms.
                             Note: may slow training and raise errors for some ops.
        cudnn_benchmark (bool): Controls torch.backends.cudnn.benchmark.
                             Set False when deterministic=True for full reproducibility.

    Example:
        rm = ReproducibilityManager(seed=42, deterministic=False)
        rm.seed_everything()
    """

    def __init__(
        self,
        seed: int = 42,
        deterministic: bool = False,
        cudnn_benchmark: bool = True,
    ) -> None:
        self.seed = seed
        self.deterministic = deterministic
        self.cudnn_benchmark = cudnn_benchmark

    def seed_everything(self) -> None:
        """
        Apply all seed settings.

        Call this once at the start of every entrypoint script (train.py,
        evaluate.py, inference.py, run_experiments.py).
        """
        # Python built-in random
        random.seed(self.seed)

        # NumPy
        np.random.seed(self.seed)

        # Environment hash seed (affects Python dict ordering etc.)
        os.environ["PYTHONHASHSEED"] = str(self.seed)

        # PyTorch CPU
        torch.manual_seed(self.seed)

        # PyTorch CUDA (all GPUs)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

        # CuDNN settings
        torch.backends.cudnn.benchmark = self.cudnn_benchmark
        if self.deterministic:
            # Note: deterministic mode may slow training
            torch.backends.cudnn.deterministic = True
            try:
                torch.use_deterministic_algorithms(True)
            except AttributeError:
                # PyTorch < 1.8 fallback
                torch.backends.cudnn.deterministic = True
            logger.info(
                "ReproducibilityManager: deterministic mode ENABLED (may slow training)."
            )
        else:
            torch.backends.cudnn.deterministic = False

        logger.info(
            "ReproducibilityManager: seeded with seed=%d, deterministic=%s.",
            self.seed, self.deterministic,
        )

    @classmethod
    def from_config(cls, cfg: "Config") -> "ReproducibilityManager":  # type: ignore[name-defined]
        """
        Construct from a Config object.

        Args:
            cfg: Config instance with reproducibility sub-config.

        Returns:
            Configured ReproducibilityManager.
        """
        return cls(
            seed=cfg.reproducibility.seed,
            deterministic=cfg.reproducibility.deterministic,
            cudnn_benchmark=cfg.reproducibility.cudnn_benchmark,
        )

    def __repr__(self) -> str:
        return (
            f"ReproducibilityManager(seed={self.seed}, "
            f"deterministic={self.deterministic}, "
            f"cudnn_benchmark={self.cudnn_benchmark})"
        )
