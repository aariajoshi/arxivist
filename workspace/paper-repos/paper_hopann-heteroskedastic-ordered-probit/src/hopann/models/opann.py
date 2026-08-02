"""
hopann.models.opann — Ordered Probit Artificial Neural Network (OPANN).

Implements the homoskedastic OPANN model from Section 3.1 of the paper.

Model equations (Section 3.1):
    Latent index (via MeanNetwork):
        h_q = sigma(sum_k w_qk * x_ik + b_q),  q = 1 .. Q          [Eq. OPANN-1]
        f(x, theta) = sum_q v_q * h_q + c                           [Eq. OPANN-2]

    Class probabilities (homoskedastic, sigma_i = 1):
        P(y=j | x) = Phi(c_j - f(x)) - Phi(c_{j-1} - f(x))        [Eq. OPANN-3]

    where Phi is the standard normal CDF, and the boundary conditions are:
        c_0 = -inf,  c_J = +inf

OPANN is the homoskedastic base model; HOPANN extends it with a per-sample
variance network (see hopann.models.hopann).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.distributions as dist

from hopann.models.mean_network import MeanNetwork
from hopann.models.cutting_points import CuttingPoints

# Standard normal distribution for Phi(·)
_STANDARD_NORMAL = dist.Normal(0.0, 1.0)


def normal_cdf(x: torch.Tensor) -> torch.Tensor:
    """Standard normal CDF Phi(x) = P(Z <= x), Z ~ N(0,1)."""
    return _STANDARD_NORMAL.cdf(x)


class OPANN(nn.Module):
    """
    Ordered Probit Artificial Neural Network (homoskedastic).

    Combines a MeanNetwork (latent index f(x)) with learnable CuttingPoints
    (c_1, ..., c_{J-1}) to produce class probabilities via the ordered probit
    link function.

    Paper reference: Section 3.1.

    Args:
        input_dim (int):   K — number of input features (inferred from data).
        num_classes (int): J — number of ordinal classes.
        hidden_dim (int):  Q — hidden nodes in MeanNetwork.
                           ASSUMED: Q=16 (conf 0.52; searched in paper).
        dropout_rate (float): Dropout on hidden layer. ASSUMED: 0.0.

    Shape:
        Input:  x of shape (B, K)
        Output: probs of shape (B, J) — class probability distribution
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 16,       # ASSUMED: Q=16 (conf 0.52)
        dropout_rate: float = 0.0,  # ASSUMED: no dropout in paper
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        # f(x, theta): latent index network
        self.mean_network = MeanNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout_rate=dropout_rate,
        )
        # c_1, ..., c_{J-1}: monotone cutting points
        self.cutting_points = CuttingPoints(num_classes=num_classes)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute P(y=j | x) for all classes j=1..J.

        Args:
            x: Feature tensor of shape (B, K).

        Returns:
            Class probability tensor of shape (B, J).
            Each row sums to 1.
        """
        assert x.dim() == 2, f"Expected x of shape [B, K], got {x.shape}"
        B = x.size(0)

        # Eq. OPANN-2: latent index f(x, theta), shape (B, 1)
        f = self.mean_network(x)   # (B, 1)

        # Cutting points c_1 < ... < c_{J-1}, shape (J-1,)
        c = self.cutting_points()  # (J-1,)

        # Compute ordered probit probabilities [Eq. OPANN-3]
        probs = self._compute_probs(f, c, sigma=None)  # (B, J)

        assert probs.shape == (B, self.num_classes), (
            f"OPANN probs shape mismatch: expected ({B}, {self.num_classes}), got {probs.shape}"
        )
        return probs

    def _compute_probs(
        self,
        f: torch.Tensor,        # (B, 1) latent index
        c: torch.Tensor,        # (J-1,) cutting points
        sigma: torch.Tensor | None,  # (B, 1) per-sample std, or None for OPANN
    ) -> torch.Tensor:
        """
        Compute P(y=j|x) using the ordered probit CDF differences.

        Eq. OPANN-3 (homoskedastic, sigma=1):
            P(y=j|x) = Phi(c_j - f(x)) - Phi(c_{j-1} - f(x))

        Eq. HOPANN-2 (heteroskedastic, sigma=sigma_i):
            P(y=j|x,z) = Phi((c_j - f(x))/sigma_i) - Phi((c_{j-1} - f(x))/sigma_i)

        Boundary conditions:
            c_0 = -inf → Phi(-inf) = 0
            c_J = +inf → Phi(+inf) = 1

        Args:
            f:     Latent index (B, 1).
            c:     Cutting points (J-1,).
            sigma: Per-sample std (B, 1), or None for homoskedastic OPANN.

        Returns:
            probs: Class probabilities (B, J).
        """
        B = f.size(0)
        J = self.num_classes

        # Expand c to broadcast with f: (1, J-1)
        c_expanded = c.unsqueeze(0)  # (1, J-1)
        f_expanded = f               # (B, 1)

        if sigma is not None:
            # Eq. HOPANN-2: scale by per-sample std
            # sigma: (B, 1) — broadcast with c_expanded (1, J-1) → (B, J-1)
            scaled = (c_expanded - f_expanded) / sigma  # (B, J-1)
        else:
            # Eq. OPANN-3: standard normal, sigma=1 implicit
            scaled = c_expanded - f_expanded  # (B, J-1)

        # Phi(scaled): CDF at each cutting point — shape (B, J-1)
        cdf_vals = normal_cdf(scaled)  # (B, J-1)

        # Add boundary conditions: Phi(-inf)=0 at left, Phi(+inf)=1 at right
        zeros = torch.zeros(B, 1, device=f.device, dtype=f.dtype)
        ones = torch.ones(B, 1, device=f.device, dtype=f.dtype)
        cdf_with_bounds = torch.cat([zeros, cdf_vals, ones], dim=1)  # (B, J+1)

        # P(y=j) = Phi(c_j) - Phi(c_{j-1})
        probs = cdf_with_bounds[:, 1:] - cdf_with_bounds[:, :-1]  # (B, J)

        # Clamp to avoid numerical zeros (needed for stable log in NLL loss)
        probs = probs.clamp(min=1e-7)

        return probs

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the most likely class index (0-indexed) for each sample.

        Args:
            x: Feature tensor of shape (B, K).

        Returns:
            Predicted class indices of shape (B,).
        """
        probs = self.forward(x)
        return probs.argmax(dim=-1)

    def param_count(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"OPANN(input_dim={self.input_dim}, "
            f"num_classes={self.num_classes}, "
            f"hidden_dim={self.hidden_dim}, "
            f"params={self.param_count()})"
        )
