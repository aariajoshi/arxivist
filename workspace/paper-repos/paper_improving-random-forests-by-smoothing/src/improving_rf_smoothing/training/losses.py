"""
Out-of-bag loss implementation.
"""
import torch
import torch.nn as nn

class OOBLoss(nn.Module):
    """
    Out-of-bag loss for optimizing smoothing parameters.
    Paper Reference: Section 3.2
    Eq: L_oob = \sum_{t=1}^T \sum_{i \in O_t} (\beta_1 \tilde{f}_t(x_i | \Sigma_t) + \beta_0 - y_i)^2
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction='sum')
        
    def forward(self, smoothed_preds_oob: torch.Tensor, targets_oob: torch.Tensor) -> torch.Tensor:
        """
        Compute OOB loss.
        
        Args:
            smoothed_preds_oob: Smoothed predictions for OOB samples [N]
            targets_oob: Target values for OOB samples [N]
            
        Returns:
            Scalar loss value
        """
        assert smoothed_preds_oob.shape == targets_oob.shape
        return self.mse(smoothed_preds_oob, targets_oob)
