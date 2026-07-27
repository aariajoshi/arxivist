"""
Loss functions for training Mamba.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLossWithZLoss(nn.Module):
    """
    Cross Entropy Loss for language modeling.
    """
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss.
        Args:
            logits: Shape [B, L, vocab_size]
            targets: Shape [B, L]
        Returns:
            Scalar loss
        """
        # Reshape for CrossEntropyLoss which expects [N, C] and [N]
        logits = logits.view(-1, self.vocab_size)
        targets = targets.view(-1)
        return self.loss_fn(logits, targets)
