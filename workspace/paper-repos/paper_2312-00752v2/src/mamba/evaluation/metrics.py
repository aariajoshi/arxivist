"""
Metrics for evaluation.
Computes Perplexity and Accuracy.
"""
import torch

def compute_perplexity(loss: float) -> float:
    """
    Compute Perplexity from CrossEntropy loss.
    """
    return torch.exp(torch.tensor(loss)).item()

def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute token prediction accuracy.
    Args:
        logits: Shape [B, L, vocab_size]
        targets: Shape [B, L]
    Returns:
        Accuracy as a float
    """
    preds = torch.argmax(logits, dim=-1)
    correct = (preds == targets).float().sum()
    total = targets.numel()
    return (correct / total).item()
