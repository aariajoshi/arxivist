"""YAML config loading and validation, plus RNG seeding utility.

No stochastic components exist in the numerical schemes themselves (everything is a
deterministic banded linear solve given b and gamma); the seeding utility is kept for
ArXivist repo-template consistency and for any future Monte-Carlo extension of the
benchmarks.
"""
from __future__ import annotations

import random
from typing import Any

import numpy as np
import yaml


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    _validate(cfg)
    return cfg


def _validate(cfg: dict) -> None:
    required_top = ["model", "training", "data", "evaluation", "hardware"]
    for key in required_top:
        if key not in cfg:
            raise ValueError(f"config.yaml missing required top-level section: '{key}'")
    valid_schemes = {"fcdf_a", "fcdf_b", "fcdf_dc", "active_set", "monotone_core", "unlimited", "chang_cooper"}
    scheme = cfg["model"].get("scheme")
    if scheme not in valid_schemes:
        raise ValueError(f"config.model.scheme='{scheme}' not in {valid_schemes}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
