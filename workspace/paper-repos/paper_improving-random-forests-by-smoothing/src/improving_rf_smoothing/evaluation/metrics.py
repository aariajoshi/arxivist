"""
Evaluation metrics.
"""
import torch
import numpy as np

def compute_mse(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return torch.mean((preds - targets) ** 2).item()

def compute_pimse(preds: torch.Tensor, base_preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Percentage Improvement in MSE"""
    mse_smooth = compute_mse(preds, targets)
    mse_base = compute_mse(base_preds, targets)
    if mse_base == 0:
        return 0.0
    return 100 * (mse_base - mse_smooth) / mse_base

def compute_mad(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Median Absolute Deviation"""
    return torch.median(torch.abs(preds - targets)).item()

def compute_max_absolute_error(preds: torch.Tensor, targets: torch.Tensor) -> float:
    return torch.max(torch.abs(preds - targets)).item()
