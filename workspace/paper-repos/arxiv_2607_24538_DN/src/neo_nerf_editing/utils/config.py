"""
utils/config.py

YAML config loading and typed access, plus the global reproducibility (seeding) utility.
Implements no specific paper equation; supports the "Reproducibility requirements" of the
ArXivist Stage 4 code-generation contract.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import yaml


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python's `random`, NumPy, and PyTorch (CPU + CUDA) for reproducibility.

    Args:
        seed: integer seed applied to all RNGs.
        deterministic: if True, also requests deterministic (slower) CUDA/CuDNN algorithms.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:  # pragma: no cover - torch is a hard runtime dependency, but keep import-safe
        pass


def resolve_device(device_cfg: str) -> str:
    """Resolve the 'auto' device setting to a concrete torch device string."""
    if device_cfg != "auto":
        return device_cfg
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover
        return "cpu"


@dataclass
class NEOConfig:
    """Typed, validated view over the YAML config tree used across all entrypoints.

    Args:
        raw: the parsed YAML dict, kept for round-tripping / passthrough of fields not
            promoted to typed attributes below.
    """

    raw: Dict[str, Any] = field(default_factory=dict)

    seed: int = 0
    deterministic: bool = False

    model: Dict[str, Any] = field(default_factory=dict)
    training: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    evaluation: Dict[str, Any] = field(default_factory=dict)
    hardware: Dict[str, Any] = field(default_factory=dict)
    logging: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str) -> "NEOConfig":
        """Load a NEOConfig from a YAML file path. Raises ValueError on missing required keys."""
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        required_top_level = ["model", "training", "data", "evaluation", "hardware"]
        missing = [k for k in required_top_level if k not in raw]
        if missing:
            raise ValueError(
                f"Config at {path} is missing required top-level section(s): {missing}"
            )

        cfg = cls(
            raw=raw,
            seed=raw.get("seed", 0),
            deterministic=raw.get("deterministic", False),
            model=raw["model"],
            training=raw["training"],
            data=raw["data"],
            evaluation=raw["evaluation"],
            hardware=raw["hardware"],
            logging=raw.get("logging", {"log_every_n_steps": 50, "checkpoint_every_n_steps": 200}),
        )
        cfg._validate()
        return cfg

    def _validate(self) -> None:
        if self.training.get("n_samples_uniform", 0) <= 0:
            raise ValueError("training.n_samples_uniform must be positive")
        if self.training.get("n_samples_hierarchical", 0) <= 0:
            raise ValueError("training.n_samples_hierarchical must be positive")
        if self.model.get("d_l", 0) <= 0:
            raise ValueError("model.d_l (language feature dim) must be positive")
        if self.data.get("near", 0) >= self.data.get("far", 0):
            raise ValueError("data.near must be < data.far")

    def device(self) -> str:
        return resolve_device(self.hardware.get("device", "auto"))

    def __repr__(self) -> str:  # pragma: no cover - convenience only
        return f"NEOConfig(seed={self.seed}, device={self.hardware.get('device')}, model={self.model})"
