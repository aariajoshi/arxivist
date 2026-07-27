"""
Configuration loading and reproducibility utilities for Mamba.
Implements configuration management and random seed utilities.
"""
import random
import yaml
import torch
import numpy as np

def load_config(config_path: str) -> dict:
    """
    Load YAML configuration file.
    Args:
        config_path: Path to the yaml config file
    Returns:
        Dictionary containing configuration
    Raises:
        ValueError: If config is invalid or missing required keys
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        
    # Basic validation
    required_keys = ["model", "training", "data", "hardware"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config section: {key}")
            
    return config

def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """
    Set random seed for reproducibility across Python, NumPy, and PyTorch.
    Args:
        seed: Random seed value
        deterministic: Whether to enforce deterministic operations in PyTorch (may slow down training)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
