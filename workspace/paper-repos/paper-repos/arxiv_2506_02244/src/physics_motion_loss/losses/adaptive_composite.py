"""
losses/adaptive_composite.py — Adaptive weighting and composite motion loss.

Implements Section 3.7 of arXiv 2506.02244v2.

The composite loss weights the three motion-type losses via a softmax over
their negated values, with temperature τ:

    w_i = exp(−L_i / τ) / Σ_j exp(−L_j / τ)       (Sec 3.7)
    L_motion = Σ_i w_i · L_i                         (Eq. 3.4)

This gives higher weight to whichever motion type is already best represented
in the current clip (lower loss → higher weight). The mechanism is grounded
in the maximum-entropy principle (Sec 3.7).

Temperature behaviour:
  τ → 0: winner-takes-all (sharpest focus on dominant motion)
  τ → ∞: uniform weighting (equal mix of all three losses)

WARNING: whether stop-gradient is applied to the weights is unspecified in
the paper (SIR confidence 0.70). Default: fully differentiable (no stop-grad).
Set stop_grad_weights=True to test the alternative.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AdaptiveMotionLoss(nn.Module):
    """Softmax-weighted composite of translation, rotation, and scaling losses.

    Paper reference: Section 3.7, Equation 3.4.

    Args:
        tau: Softmax temperature (default 0.1, Table 5, conf 0.97).
        stop_grad_weights: If True, detach weights before weighting losses.
            ASSUMED False (fully differentiable) — not stated in paper (conf 0.70).
    """

    def __init__(self, tau: float = 0.1, stop_grad_weights: bool = False) -> None:
        super().__init__()
        if tau <= 0:
            raise ValueError(f"tau must be > 0, got {tau}")
        self.tau = tau
        self.stop_grad_weights = stop_grad_weights

    def forward(
        self,
        L_trans: Tensor,
        L_rot: Tensor,
        L_scale: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Compute adaptive composite loss and return weights for logging.

        Args:
            L_trans: Scalar translation loss.
            L_rot:   Scalar rotation loss.
            L_scale: Scalar scaling loss.

        Returns:
            Tuple of:
              - L_motion: Scalar composite loss (Eq. 3.4).
              - weights:  Tensor [3] of (w_trans, w_rot, w_scale) for logging.
        """
        losses = torch.stack([L_trans, L_rot, L_scale])  # [3]

        # Sec 3.7: w_i = exp(−L_i/τ) / Σ_j exp(−L_j/τ)
        weights = F.softmax(-losses / self.tau, dim=0)    # [3]

        if self.stop_grad_weights:
            # WARNING: low-confidence implementation — see module docstring
            # TODO: verify whether paper applies stop-gradient on weights (conf 0.70)
            weights = weights.detach()

        # Eq. 3.4: L_motion = Σ_i w_i · L_i
        L_motion = (weights * losses).sum()

        return L_motion, weights

    def __repr__(self) -> str:
        return (
            f"AdaptiveMotionLoss(tau={self.tau}, "
            f"stop_grad_weights={self.stop_grad_weights})"
        )
