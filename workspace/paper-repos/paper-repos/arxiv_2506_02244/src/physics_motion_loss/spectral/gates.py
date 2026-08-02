"""
spectral/gates.py — Energy and observability gates for WLS weighting.

Implements the gating functions from Appendix A.6 of arXiv 2506.02244v2:
  - EnergyGate:        g_E = σ(f · (E/E_max − τ_E))
  - ObservabilityGate: g_obs(m) = m² / (m² + λ)

These gates down-weight low-energy or unobservable spectral samples
before they enter the WLS regression, improving numerical stability.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class EnergyGate(nn.Module):
    """Soft sigmoid gate that suppresses low-energy spectral samples.

    Paper reference: Appendix A.6 — "g_E = σ(f(E/E_max − τ_E))"

    Args:
        tau_E: Energy threshold in [0.1, 0.2]. Samples with E/E_max < tau_E
               are suppressed. (conf 0.90, Table 5 pins tau_E=0.10)
        f: Sigmoid sharpness in [6, 10]. Higher = harder threshold.
           (conf 0.90, Table 5 pins f=10)
    """

    def __init__(self, tau_E: float = 0.10, f: float = 10.0) -> None:
        super().__init__()
        self.tau_E = tau_E
        self.f = f

    def forward(self, E: Tensor, E_max: Tensor) -> Tensor:
        """Compute energy gate weights.

        Args:
            E: Energy values, any shape, float32.
            E_max: Maximum energy (scalar or broadcastable).

        Returns:
            Gate weights in (0, 1), same shape as E.
        """
        # App A.6: g_E = σ(f · (E/E_max − τ_E))
        ratio = E / (E_max + 1e-12)
        return torch.sigmoid(self.f * (ratio - self.tau_E))

    def __repr__(self) -> str:
        return f"EnergyGate(tau_E={self.tau_E}, f={self.f})"


class ObservabilityGate(nn.Module):
    """Smooth gate that down-weights angular/log-radial harmonics near zero.

    Paper reference: Appendix A.6 — "g_obs(m) = m²/(m²+λ)"

    The m=0 (DC) angular harmonic carries no rotation information; this gate
    smoothly suppresses it and near-zero harmonics where rotation/scaling
    is unobservable. m=0 is also explicitly excluded from all sums (Sec 3.5).

    Args:
        lam: Smoothing parameter λ = 1.0 (App A.6, conf 0.90).
    """

    def __init__(self, lam: float = 1.0) -> None:
        super().__init__()
        self.lam = lam

    def forward(self, harmonic_indices: Tensor) -> Tensor:
        """Compute observability gate for given harmonic indices.

        Args:
            harmonic_indices: Integer tensor of m or ν values, any shape.

        Returns:
            Gate weights in [0, 1), same shape as input.
        """
        # App A.6: g_obs(m) = m² / (m² + λ)
        m2 = harmonic_indices.float() ** 2
        return m2 / (m2 + self.lam)

    def __repr__(self) -> str:
        return f"ObservabilityGate(lam={self.lam})"
