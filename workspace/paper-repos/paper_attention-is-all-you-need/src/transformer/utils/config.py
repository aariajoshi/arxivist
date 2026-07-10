"""
Configuration loading and seeding utilities.
Implements reproducibility requirements for random seeding.
"""

import os
import random
import yaml
import torch
import numpy as np

def set_seed(seed: int = 42) -> None:
    """
    Set seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Note: deterministic operations may slow down training
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_config(config_path: str) -> dict:
    """
    Load YAML configuration file.
    """
    if not os.path.exists(config_path):
        raise ValueError(f"Config file not found: {config_path}")
        
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Basic validation
    if 'model' not in config:
        raise ValueError("Config missing 'model' section")
    if 'training' not in config:
        raise ValueError("Config missing 'training' section")
        
    return config
