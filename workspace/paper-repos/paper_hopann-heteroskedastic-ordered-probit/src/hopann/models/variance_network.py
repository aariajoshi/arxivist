"""
hopann.models.variance_network — Per-sample dispersion networks for HOPANN.

Implements the variance sub-network that produces sigma_i for the heteroskedastic
ordered probit model described in Section 3.2 of the paper.

HOPANN variance parameterisation (Section 3.2):
    sigma_i = exp(variance_ANN(z_i))                                 [Eq. HOPANN-1]

where z_i is an auxiliary conditioning variable.
    TODO (low confidence, conf 0.55): The paper does not clearly specify what z_i
    is. We ASSUME z_i = x_i (same features as mean network). Mark with WARNING.

Three variants are provided:
    VarianceNetworkANN    — ANN-based variance (ASSUMED default, conf 0.62)
    VarianceNetworkLinear — linear log-variance model (alternative / ablation)

Both inherit from VarianceNetworkBase (ABC).
"""

from __future__ import annotations

import abc
import torch
import torch.nn as nn


class VarianceNetworkBase(nn.Module, abc.ABC):
    """
    Abstract base class for HOPANN variance networks.

    All variance networks must implement `forward(z)` and return a strictly
    positive tensor sigma_i of shape (B, 1) via exp(·).

    Paper reference: Section 3.2, Equation HOPANN-1.
    """

    @abc.abstractmethod
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample standard deviation sigma_i.

        Args:
            z: Conditioning features of shape (B, D_z).
               # WARNING: low-confidence implementation
               # TODO: Paper does not specify z_i explicitly.
               # ASSUMED: z_i = x_i (same input as mean network, conf 0.55).

        Returns:
            sigma_i: Positive std tensor of shape (B, 1).
        """
        ...


class VarianceNetworkANN(VarianceNetworkBase):
    """
    ANN-based heteroskedastic variance network for HOPANN.

    Architecture:
        Single hidden layer feedforward → scalar log-variance → exp(·)

        log_sigma_i = ANN(z_i; phi)                                  [Eq. HOPANN-1a]
        sigma_i     = exp(log_sigma_i)                               [Eq. HOPANN-1b]

    # WARNING: low-confidence implementation (conf 0.62)
    # The specific architecture of the variance ANN is not fully specified.
    # ASSUMED: same hidden_dim as MeanNetwork; sigmoid activation.

    Paper reference: Section 3.2.

    Args:
        input_dim (int):  D_z — dimension of conditioning variable z_i.
                          ASSUMED: D_z = K (same as mean network input, conf 0.55).
        hidden_dim (int): Number of hidden units. ASSUMED: 16 (conf 0.52).

    Shape:
        Input:  (B, D_z)
        Output: (B, 1) — positive sigma_i values
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 16,  # ASSUMED: same Q as MeanNetwork (conf 0.52)
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # ASSUMED: single hidden layer with sigmoid, matching MeanNetwork structure
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.Sigmoid()
        # Output: scalar log-variance (unconstrained)
        self.output = nn.Linear(hidden_dim, 1, bias=True)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier uniform init; output bias initialised to 0 → sigma_i starts at 1."""
        nn.init.xavier_uniform_(self.hidden.weight)
        nn.init.zeros_(self.hidden.bias)
        nn.init.xavier_uniform_(self.output.weight)
        nn.init.zeros_(self.output.bias)  # exp(0) = 1 → unit std at init

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute sigma_i = exp(ANN(z_i)).

        Args:
            z: Conditioning tensor of shape (B, D_z).
               # WARNING: low-confidence implementation
               # TODO: z_i = x_i is ASSUMED (conf 0.55); replace if paper clarifies.

        Returns:
            sigma_i of shape (B, 1), strictly positive.
        """
        assert z.dim() == 2, f"Expected z of shape [B, D_z], got {z.shape}"
        assert z.size(1) == self.input_dim, (
            f"Expected input_dim={self.input_dim}, got z.size(1)={z.size(1)}"
        )

        # Eq. HOPANN-1a: log_sigma_i = ANN(z_i; phi)
        h = self.activation(self.hidden(z))  # (B, Q)
        log_sigma = self.output(h)           # (B, 1)

        # Eq. HOPANN-1b: sigma_i = exp(log_sigma_i) — guarantees positivity
        sigma = torch.exp(log_sigma)         # (B, 1)

        assert sigma.shape == (z.size(0), 1), (
            f"VarianceNetworkANN output shape mismatch: expected ({z.size(0)}, 1), got {sigma.shape}"
        )
        return sigma

    def __repr__(self) -> str:
        return (
            f"VarianceNetworkANN(input_dim={self.input_dim}, "
            f"hidden_dim={self.hidden_dim})"
        )


class VarianceNetworkLinear(VarianceNetworkBase):
    """
    Linear log-variance model for ablation / simpler heteroskedastic baseline.

    Architecture:
        log_sigma_i = w^T z_i + b                                    [linear ablation]
        sigma_i     = exp(log_sigma_i)

    This is not the primary HOPANN model; it serves as a simpler alternative
    to verify that heteroskedasticity gains over homoskedastic OPANN are not
    purely an ANN capacity effect.

    Args:
        input_dim (int): Dimension of conditioning variable z_i.

    Shape:
        Input:  (B, D_z)
        Output: (B, 1) — positive sigma_i values
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, 1, bias=True)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)  # sigma_i = 1 at init

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute sigma_i = exp(w^T z_i + b).

        Args:
            z: Conditioning tensor of shape (B, D_z).

        Returns:
            sigma_i of shape (B, 1), strictly positive.
        """
        assert z.dim() == 2, f"Expected z of shape [B, D_z], got {z.shape}"
        assert z.size(1) == self.input_dim, (
            f"Expected input_dim={self.input_dim}, got z.size(1)={z.size(1)}"
        )
        log_sigma = self.linear(z)     # (B, 1)
        sigma = torch.exp(log_sigma)   # (B, 1)
        return sigma

    def __repr__(self) -> str:
        return f"VarianceNetworkLinear(input_dim={self.input_dim})"


# Registry for selecting variance network type by name
VARIANCE_NETWORK_REGISTRY: dict[str, type[VarianceNetworkBase]] = {
    "ann": VarianceNetworkANN,
    "linear": VarianceNetworkLinear,
}


def build_variance_network(
    variance_type: str,
    input_dim: int,
    hidden_dim: int = 16,
) -> VarianceNetworkBase:
    """
    Factory function to instantiate a variance network by name.

    Args:
        variance_type: One of 'ann' (ASSUMED default, conf 0.62) or 'linear'.
        input_dim:     Dimension of conditioning variable z_i.
        hidden_dim:    Hidden layer size (ignored for 'linear' variant).

    Returns:
        Instantiated VarianceNetworkBase subclass.

    Raises:
        ValueError: If variance_type is not registered.
    """
    if variance_type not in VARIANCE_NETWORK_REGISTRY:
        raise ValueError(
            f"Unknown variance_type '{variance_type}'. "
            f"Choose from: {list(VARIANCE_NETWORK_REGISTRY.keys())}"
        )
    cls = VARIANCE_NETWORK_REGISTRY[variance_type]
    if variance_type == "ann":
        return cls(input_dim=input_dim, hidden_dim=hidden_dim)  # type: ignore[call-arg]
    return cls(input_dim=input_dim)  # type: ignore[call-arg]
