"""
Data transformations.
"""
import torch

class StandardScaler:
    """
    Standardize features by removing the mean and scaling to unit variance.
    """
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, x: torch.Tensor) -> None:
        self.mean = x.mean(dim=0, keepdim=True)
        self.std = x.std(dim=0, keepdim=True) + 1e-8
        
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.mean is None or self.std is None:
            raise ValueError("StandardScaler has not been fitted yet.")
        return (x - self.mean) / self.std
