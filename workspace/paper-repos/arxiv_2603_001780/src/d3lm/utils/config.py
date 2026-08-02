"""Config loading, seeding, device resolution for D3LM."""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
import torch
import yaml


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("device='cuda' requested but CUDA is not available.")
    return torch.device(device)


@dataclass
class Config:
    model: Dict[str, Any] = field(default_factory=dict)
    generation: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    evaluation: Dict[str, Any] = field(default_factory=dict)
    hardware: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:  # noqa: D105
        return f"Config(model={self.model.get('variant')}, dataset={self.data.get('dataset')})"


def load_config(path: str) -> Config:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    cfg = Config(
        model=raw.get("model", {}),
        generation=raw.get("generation", {}),
        data=raw.get("data", {}),
        evaluation=raw.get("evaluation", {}),
        hardware=raw.get("hardware", {}),
    )
    variant = cfg.model.get("variant", "D3LM-R")
    if variant not in ("D3LM", "D3LM-R"):
        raise ValueError(f"model.variant must be 'D3LM' or 'D3LM-R', got {variant!r}")
    if cfg.generation.get("temperature", 1.1) <= 0:
        raise ValueError("generation.temperature must be > 0")
    return cfg
