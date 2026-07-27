"""
losses/scaling_loss.py — Scaling motion loss via radial energy flow analysis.

Implements Section 3.6 of arXiv 2506.02244v2.

Theory: uniform spatial scaling V(x,y,t) = V0(x/s(t), y/s(t)) with s(t)=e^{σ(t)}
shifts spectral energy radially over time. After a log-radius change ξ=log ρ,
scaling becomes translation: energy concentrates on ωt + α·ν = 0, α=σ̇.

Rather than fitting this line directly, two robust proxies are used (Sec 3.6):
  C_flow  = |Σ ∇̂ρE · ∇̂tE|     (radial-temporal gradient alignment)
  S_trend = |corr(ρc, t)|        (Pearson correlation of radial spectral centroid)

  L_scale = 1 − (C_flow + S_trend) / 2

Edge case: if T_lp < 3, both proxies default to 0.5 (Sec 3.6, conf 0.99).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class ScalingMotionLoss(nn.Module):
    """Radial energy flow alignment + spectral centroid trend for zoom/scale motion.

    Paper reference: Section 3.6, Appendix A.4–A.5.
    SIM(2) scaling slice: ωt + α·ν = 0  (Eq. A.3)

    Args:
        epsilon_stab: Stability constant for centroid denominator (Sec 3.6, conf 0.97).
        epsilon:      General numeric stability constant.
        T_min:        Minimum frames; below this both proxies default to 0.5 (conf 0.99).
    """

    def __init__(
        self,
        epsilon_stab: float = 1e-8,
        epsilon: float = 1e-8,
        T_min: int = 3,
    ) -> None:
        super().__init__()
        self.epsilon_stab = epsilon_stab
        self.epsilon = epsilon
        self.T_min = T_min

    def forward(self, ring_energies: Tensor) -> Tensor:
        """Compute scaling motion loss.

        Args:
            ring_energies: Spectral ring energies Ek(t), shape [Nr, T_lp], float32.

        Returns:
            Scalar L_scale ∈ [0, 1].  Zero iff motion is perfectly uniform scaling.
        """
        assert ring_energies.dim() == 2, (
            f"Expected [Nr, T_lp], got {ring_energies.shape}"
        )
        Nr, T_lp = ring_energies.shape

        # Edge case: too few frames for reliable gradient / correlation estimates
        # Sec 3.6: "For very short windows (T<3) we default both proxies to 0.5"
        if T_lp < self.T_min:
            return torch.tensor(0.5, device=ring_energies.device, dtype=ring_energies.dtype)

        C_flow  = self._radial_flow_alignment(ring_energies)  # scalar
        S_trend = self._centroid_trend(ring_energies)          # scalar

        # Sec 3.6: L_scale = 1 − (C_flow + S_trend) / 2
        L_scale = 1.0 - 0.5 * (C_flow + S_trend)

        return L_scale.clamp(0.0, 1.0)

    def _radial_flow_alignment(self, ring_energies: Tensor) -> Tensor:
        """Absolute dot product of normalised radial and temporal energy gradients.

        Paper reference: Section 3.6:
            ∇ρ Ek,t = Ek+1,t − Ek,t
            ∇t Ek,t = Ek,t+1 − Ek,t
            C_flow = |Σ_{k,t} ∇̂ρ · ∇̂t|

        For pure scaling at rate α, ∇t E = −α · ∇ρ E, so C_flow = 1.

        Args:
            ring_energies: [Nr, T_lp], float32.

        Returns:
            Scalar C_flow ∈ [0, 1].
        """
        Nr, T_lp = ring_energies.shape

        # Finite differences — shapes [Nr-1, T_lp] and [Nr, T_lp-1]
        # Clip to common support [Nr-1, T_lp-1] by truncation
        grad_rho = ring_energies[1:, :] - ring_energies[:-1, :]   # [Nr-1, T_lp]
        grad_t   = ring_energies[:, 1:] - ring_energies[:, :-1]   # [Nr, T_lp-1]

        # Align to common grid [Nr-1, T_lp-1]
        grad_rho = grad_rho[:, :-1]   # [Nr-1, T_lp-1]
        grad_t   = grad_t[:-1, :]     # [Nr-1, T_lp-1]

        # Normalise each field independently (Sec 3.6: "unit fields")
        norm_rho = torch.sqrt((grad_rho ** 2).sum() + self.epsilon)
        norm_t   = torch.sqrt((grad_t   ** 2).sum() + self.epsilon)

        grad_rho_hat = grad_rho / norm_rho
        grad_t_hat   = grad_t   / norm_t

        # C_flow = |⟨∇̂ρE, ∇̂tE⟩|  (direction-agnostic, Sec 3.6)
        C_flow = (grad_rho_hat * grad_t_hat).sum().abs()

        return C_flow.clamp(0.0, 1.0)

    def _centroid_trend(self, ring_energies: Tensor) -> Tensor:
        """Pearson correlation of radial spectral centroid over time.

        Paper reference: Section 3.6:
            ρc(t) = Σk k·Ek(t) / (Σk Ek(t) + ε_stab)
            S_trend = |corr(ρc, t)|

        For zoom-in (increasing s), centroid shifts to lower ρ (lower freq).
        For zoom-out, centroid shifts to higher ρ. Both yield |corr| → 1.

        Args:
            ring_energies: [Nr, T_lp], float32.

        Returns:
            Scalar S_trend ∈ [0, 1].
        """
        Nr, T_lp = ring_energies.shape
        device = ring_energies.device

        # Ring index axis: k ∈ {0, 1, ..., Nr-1}
        k = torch.arange(Nr, device=device, dtype=ring_energies.dtype)  # [Nr]

        # Sec 3.6: ρc(t) = Σk k·Ek(t) / (Σk Ek(t) + ε_stab)
        numerator   = (k.unsqueeze(1) * ring_energies).sum(dim=0)          # [T_lp]
        denominator = ring_energies.sum(dim=0) + self.epsilon_stab          # [T_lp]
        rho_c = numerator / denominator                                      # [T_lp]

        # Time index
        t = torch.arange(T_lp, device=device, dtype=ring_energies.dtype)   # [T_lp]

        # Pearson correlation: corr(ρc, t) = cov(ρc,t) / sqrt(var(ρc)·var(t))
        rho_c_mean = rho_c.mean()
        t_mean     = t.mean()

        cov   = ((rho_c - rho_c_mean) * (t - t_mean)).mean()
        var_rho = ((rho_c - rho_c_mean) ** 2).mean()
        var_t   = ((t    - t_mean)     ** 2).mean()

        denom = torch.sqrt(var_rho * var_t).clamp(min=self.epsilon)
        S_trend = (cov / denom).abs()

        return S_trend.clamp(0.0, 1.0)

    def __repr__(self) -> str:
        return (
            f"ScalingMotionLoss(epsilon_stab={self.epsilon_stab}, "
            f"T_min={self.T_min})"
        )
