"""
Config loading utility.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file.

    Args:
        path: path to a config.yaml file (see configs/config.yaml).

    Returns:
        The parsed config as a nested dict.
    """
    with Path(path).open("r") as f:
        return yaml.safe_load(f)
