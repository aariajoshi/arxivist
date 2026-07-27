"""
Configuration and reproducibility utilities.
Paper: Improving Random Forests by Smoothing
"""
import os
import random
import numpy as np
import torch
from omegaconf import OmegaConf, DictConfig

def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """
    Set random seed for Python, NumPy, and PyTorch for reproducibility.
    
    Args:
        seed: Random seed value.
        deterministic: Whether to force deterministic algorithms in PyTorch.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Note: torch.use_deterministic_algorithms(True) might be too strict for some ops

def load_config(config_path: str) -> DictConfig:
    """
    Load YAML configuration file.
    
    Args:
        config_path: Path to the config file.
        
    Returns:
        DictConfig: Loaded configuration object.
    """
    if not os.path.exists(config_path):
        raise ValueError(f"Config file not found at {config_path}")
    
    config = OmegaConf.load(config_path)
    return config
