"""
hopann.models.mean_network — Latent index network f(x, theta) for OPANN/HOPANN.

Implements the mean (latent index) sub-network described in Section 3.1 of the paper.

Architecture (OPANN Section 3.1):
    Single hidden layer feedforward network:
        h_q = sigma(sum_k w_qk * x_ik + b_q),  q = 1 .. Q          [Eq. OPANN-1]
        f(x, theta) = sum_q v_q * h_q + c                           [Eq. OPANN-2]

    where sigma is the logistic sigmoid, Q is the number of hidden nodes,
    and c is a scalar bias on the output.

Note: The number of input features K is inferred dynamically from data (never
hardcoded) because the Amazon review dataset feature count varies by experiment
(K ≈ 12-19 according to SIR).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MeanNetwork(nn.Module):
    """
    Single-hidden-layer ANN that produces the latent index f(x, theta).

    This is the mean sub-network shared by OPANN and HOPANN. It maps a feature
    vector x_i of dimension K to a scalar latent index f(x_i) ∈ ℝ.

    Paper reference: Section 3.1, Equations OPANN-1 and OPANN-2.

    Args:
        input_dim (int):   K — number of input features. Inferred from data.
        hidden_dim (int):  Q — number of hidden nodes.
                           ASSUMED: Q=16 (conf 0.52, searched over a grid per paper).
        dropout_rate (float): Dropout probability applied after hidden activation.
                              ASSUMED: 0.0 (paper does not mention dropout explicitly).

    Shape:
        Input:  (B, K) — batch of B feature vectors
        Output: (B, 1) — latent index per sample
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 16,  # ASSUMED: Q=16 (conf 0.52); searched over grid
        dropout_rate: float = 0.0,  # ASSUMED: no dropout mentioned in paper
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        # Hidden layer: w_qk, b_q  [Eq. OPANN-1]
        self.hidden = nn.Linear(input_dim, hidden_dim)
        # Sigmoid activation sigma(·) as used in ordered probit ANN literature
        self.activation = nn.Sigmoid()
        # Output projection: v_q, c  [Eq. OPANN-2]
        # bias=True captures the scalar bias c in Eq. OPANN-2
        self.output = nn.Linear(hidden_dim, 1, bias=True)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(p=dropout_rate)
        else:
            self.dropout = nn.Identity()

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Xavier uniform initialisation for hidden weights; zero biases."""
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.zeros_(self.hidden.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute latent index f(x, theta).

        Args:
            x: Feature tensor of shape (B, K).

        Returns:
            Latent index tensor of shape (B, 1).
        """
        assert x.dim() == 2, f"Expected x of shape [B, K], got {x.shape}"
        assert x.size(1) == self.input_dim, (
            f"Expected input_dim={self.input_dim}, got x.size(1)={x.size(1)}"
        )

        # Eq. OPANN-1: h_q = sigma(sum_k w_qk * x_ik + b_q)
        h = self.activation(self.hidden(x))  # (B, Q)
        h = self.dropout(h)

        # Eq. OPANN-2: f(x, theta) = sum_q v_q * h_q + c
        f = self.output(h)  # (B, 1)

        assert f.shape == (x.size(0), 1), (
            f"MeanNetwork output shape mismatch: expected ({x.size(0)}, 1), got {f.shape}"
        )
        return f

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"MeanNetwork(input_dim={self.input_dim}, "
            f"hidden_dim={self.hidden_dim}, "
            f"dropout_rate={self.dropout_rate})"
        )
