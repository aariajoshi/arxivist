"""
hopann.training.losses — Ordered probit negative log-likelihood loss.

Implements the NLL loss for the ordered probit model:

    ln LL = sum_i sum_j I(y_i = j) * ln P(y_i = j | x_i)           [Eq. NLL-1]

    In practice, for a single sample i with label y_i:
        nll_i = -ln P(y_i | x_i)

    Batch NLL (implemented here):
        NLL = -mean_i [ ln clamp(P(y_i | x_i), min=1e-7) ]         [Eq. NLL-2]

The probability P(y_i | x_i) is extracted from the full probability tensor
(B, J) output by OPANN/HOPANN by indexing with the true label y.

Clamping with min=1e-7 prevents log(0) = -inf from destabilising training.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class OrderedProbitNLL(nn.Module):
    """
    Negative log-likelihood loss for ordered probit models (OPANN / HOPANN).

    Given the full class probability distribution P(y=j|x) ∈ ℝ^J for each
    sample, this loss computes the NLL of the true labels.

    Equation (NLL-2):
        NLL = -mean_i [ ln clamp(P(y_i=y_i | x_i), min=eps) ]

    This is mathematically equivalent to the cross-entropy loss when the
    probabilities are produced by a softmax; here they come from the ordered
    probit CDF differences.

    Args:
        eps (float): Minimum probability for clamping before log.
                     Default: 1e-7. This prevents log(0) = -inf.

    Shape:
        probs: (B, J) — class probabilities from OPANN/HOPANN forward()
        labels: (B,) — integer class labels (0-indexed)
        output: scalar loss tensor
    """

    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, probs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute ordered probit NLL.

        Args:
            probs:  Probability tensor of shape (B, J). Values should be in (0, 1]
                    and sum to ~1 along dim=1. Produced by OPANN/HOPANN forward().
            labels: Integer class labels of shape (B,), values in {0, 1, ..., J-1}.

        Returns:
            Scalar NLL loss (mean over batch).

        Raises:
            AssertionError: If tensor shapes are inconsistent.
        """
        assert probs.dim() == 2, (
            f"Expected probs of shape [B, J], got {probs.shape}"
        )
        assert labels.dim() == 1, (
            f"Expected labels of shape [B], got {labels.shape}"
        )
        B, J = probs.shape
        assert labels.shape[0] == B, (
            f"Batch size mismatch: probs has {B} rows, labels has {labels.shape[0]}"
        )
        assert labels.max() < J, (
            f"Label value {labels.max().item()} >= num_classes {J}"
        )

        # Extract P(y_i | x_i) for each sample: index probs with true label
        # Shape: (B,) — probability of the correct class for each sample
        true_probs = probs[torch.arange(B, device=probs.device), labels]  # (B,)

        # Eq. NLL-2: NLL = -mean_i [ ln clamp(P(y_i | x_i), min=eps) ]
        # Clamp prevents log(0) = -inf
        log_probs = torch.log(true_probs.clamp(min=self.eps))  # (B,)
        nll = -log_probs.mean()  # scalar

        return nll

    def __repr__(self) -> str:
        return f"OrderedProbitNLL(eps={self.eps})"
