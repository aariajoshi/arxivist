"""
losses/rotation_loss.py — Rotational motion loss.

Implements Section 3.5 and Appendix D.2 of arXiv 2506.02244v2.

Theory: in-plane rotation V(r,θ,t) = V0(r, θ−Ωt) concentrates spectral energy on:
  (i)  Annular rings in spatial frequency (Bessel-type radial response)
  (ii) Tilted lines ωt + m·Ω = 0 in the (m, ωt) plane

The loss combines both signatures:
    L_rot = 1 − (C_ring + C_rot) / 2           (Section 3.5)

where:
    C_ring = 1 − H̄_ring / log(Nr)              (annular concentration via entropy)
    C_rot  = E_line / E_all                     (tilted-line energy ratio)
    Ω*     = −Σ|C̃m|²·ωt·m / Σ|C̃m|²·m²       (Eq. A.7, energy-weighted LS estimate)

Note: m=0 is explicitly excluded from all sums (DC carries no rotation info).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ..spectral.gates import EnergyGate, ObservabilityGate


class RotationalMotionLoss(nn.Module):
    """Ring concentration + tilted-line energy ratio in the (m, ωt) plane.

    Paper reference: Section 3.5, Appendix A.4, Appendix D.2.

    Args:
        Nr:       Number of concentric rings (default 20, App A.8, conf 0.85).
        M:        Number of angular bins (default 24, App A.8, conf 0.85).
        delta:    Tilted-line bandwidth in temporal-freq bins (default 1, App A.8, conf 0.99).
        tau_E:    Energy gate threshold (App A.6, conf 0.90).
        gate_f:   Energy gate sharpness (App A.6, conf 0.90).
        obs_lam:  Observability gate lambda for m (App A.6, conf 0.90).
        epsilon:  Numerical stability constant.
    """

    def __init__(
        self,
        Nr: int = 20,
        M: int = 24,
        delta: float = 1.0,
        tau_E: float = 0.10,
        gate_f: float = 10.0,
        obs_lam: float = 1.0,
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        self.Nr = Nr
        self.M = M
        self.delta = delta
        self.epsilon = epsilon
        self.energy_gate = EnergyGate(tau_E=tau_E, f=gate_f)
        self.obs_gate = ObservabilityGate(lam=obs_lam)

    def forward(
        self,
        polar_sequence: Tensor,
        ring_energies: Tensor,
    ) -> Tensor:
        """Compute rotational motion loss.

        Args:
            polar_sequence: Polar spectrum [T_lp, Nr, M], complex64.
            ring_energies:  Ring energies [Nr, T_lp], float32.

        Returns:
            Scalar L_rot ∈ [0, 1]. Zero iff motion is perfectly rotational.
        """
        assert polar_sequence.dim() == 3, (
            f"Expected [T_lp, Nr, M], got {polar_sequence.shape}"
        )
        assert polar_sequence.is_complex(), "polar_sequence must be complex"
        assert ring_energies.dim() == 2, (
            f"Expected [Nr, T_lp], got {ring_energies.shape}"
        )

        T_lp, Nr, M = polar_sequence.shape

        # --- Step 1: Angular DFT → C_m(ρ, t) ---
        # App A.4: C_m(ρ,t) = (1/2π) ∫ V̂(ρ,θ,t) e^{−imθ} dθ
        # Discrete: DFT over M angular bins (axis=-1, which is the θ axis)
        C_m = torch.fft.fft(polar_sequence, dim=-1)  # [T_lp, Nr, M], complex

        # --- Step 2: Temporal DFT → C̃_m(ρ, ωt) ---
        # App A.4: C̃_m(ρ, ωt) = DFT_t{C_m(ρ,t)}
        C_m_tilde = torch.fft.fft(C_m, dim=0)        # [T_lp, Nr, M], complex

        # Rearrange to [Nr, M, T_lp] for easier indexing
        C_m_tilde = C_m_tilde.permute(1, 2, 0)       # [Nr, M, T_lp]

        # --- Step 3: Estimate angular velocity Ω* (Eq. A.7) ---
        omega_star = self._estimate_omega(C_m_tilde)  # scalar

        # --- Step 4: Tilted-line energy ratio C_rot ---
        C_rot = self._tilted_line_ratio(C_m_tilde, omega_star)  # scalar

        # --- Step 5: Ring concentration C_ring ---
        C_ring = self._ring_concentration(ring_energies)  # scalar

        # --- Step 6: Composite loss (Section 3.5) ---
        # L_rot = 1 − (C_ring + C_rot) / 2
        L_rot = 1.0 - 0.5 * (C_ring + C_rot)

        return L_rot.clamp(0.0, 1.0)

    def _estimate_omega(self, C_m_tilde: Tensor) -> Tensor:
        """Energy-weighted LS estimate of angular velocity Ω*.

        Paper reference: Equation A.7:
            Ω* = −Σ_{ρ,m≠0,ωt} |C̃_m|² · ωt · m
                / Σ_{ρ,m≠0,ωt} |C̃_m|² · m²

        m=0 is excluded (DC carries no rotational information, Sec 3.5).

        Args:
            C_m_tilde: [Nr, M, T_lp] complex.

        Returns:
            Scalar estimate of Ω*.
        """
        Nr, M, T_lp = C_m_tilde.shape
        device = C_m_tilde.device

        # Build m and ωt index tensors
        m_idx = torch.fft.fftfreq(M, device=device) * M  # [M], integer-valued
        omega_t = torch.fft.fftfreq(T_lp, device=device)  # [T_lp]

        # Expand to [Nr, M, T_lp] for broadcasting
        m_grid = m_idx.view(1, M, 1).expand(Nr, M, T_lp)
        ot_grid = omega_t.view(1, 1, T_lp).expand(Nr, M, T_lp)

        # Observability gate: g_obs(m) = m²/(m²+λ) — down-weights near-zero m
        g_obs = self.obs_gate(m_grid)  # [Nr, M, T_lp]

        E = C_m_tilde.abs() ** 2      # [Nr, M, T_lp]

        # Mask out m=0 (DC component — no rotation info, Sec 3.5)
        m_nonzero = (m_idx != 0).view(1, M, 1).expand(Nr, M, T_lp)
        E_masked = E * m_nonzero.float() * g_obs  # [Nr, M, T_lp]

        # Eq. A.7 numerator and denominator
        numerator   = (E_masked * ot_grid * m_grid).sum()
        denominator = (E_masked * m_grid ** 2).sum().clamp(min=self.epsilon)

        return -numerator / denominator  # scalar

    def _tilted_line_ratio(self, C_m_tilde: Tensor, omega_star: Tensor) -> Tensor:
        """Fraction of rotational energy on the line ωt + m·Ω* = 0.

        Paper reference: Eq. A.8, Appendix D.2:
            E_line = Σ_{ρ,m≠0,|ωt+m·Ω*|≤Δ} |C̃_m|²
            C_rot  = E_line / E_all

        Args:
            C_m_tilde:   [Nr, M, T_lp] complex.
            omega_star:  Scalar Ω* estimate.

        Returns:
            Scalar C_rot ∈ [0, 1].
        """
        Nr, M, T_lp = C_m_tilde.shape
        device = C_m_tilde.device

        m_idx  = torch.fft.fftfreq(M, device=device) * M    # [M]
        omega_t = torch.fft.fftfreq(T_lp, device=device)    # [T_lp]

        m_grid = m_idx.view(1, M, 1).expand(Nr, M, T_lp)
        ot_grid = omega_t.view(1, 1, T_lp).expand(Nr, M, T_lp)

        # Distance from each (m, ωt) point to the tilted line ωt + m·Ω* = 0
        dist = (ot_grid + m_grid * omega_star).abs()  # [Nr, M, T_lp]

        # App A.8: Δ = 1 temporal-frequency bin
        # ωt is normalised to [−0.5, 0.5], so 1 bin = 1/T_lp
        delta_norm = self.delta / T_lp
        on_line = (dist <= delta_norm).float()         # [Nr, M, T_lp]

        E = C_m_tilde.abs() ** 2
        # Exclude m=0
        m_nonzero = (m_idx != 0).view(1, M, 1).expand(Nr, M, T_lp).float()

        E_line = (E * on_line * m_nonzero).sum()
        E_all  = (E * m_nonzero).sum().clamp(min=self.epsilon)

        return E_line / E_all  # scalar C_rot

    def _ring_concentration(self, ring_energies: Tensor) -> Tensor:
        """Annular spectral concentration via temporal-mean entropy.

        Paper reference: Appendix D.2, Eq. D.1:
            H_ring(t) = −Σ_k Ek(t) log(Ek(t) + ε)
            H̄_ring   = mean_t H_ring(t)
            C_ring    = 1 − H̄_ring / log(Nr)

        Perfect annular concentration → H̄_ring = 0 → C_ring = 1.

        Args:
            ring_energies: [Nr, T_lp], float32. Unnormalised ring energies.

        Returns:
            Scalar C_ring ∈ [0, 1].
        """
        Nr, T_lp = ring_energies.shape

        # Normalise per time step so Ek(t) sums to 1 over rings
        E_sum = ring_energies.sum(dim=0, keepdim=True).clamp(min=self.epsilon)
        E_norm = ring_energies / E_sum  # [Nr, T_lp]

        # Per-frame entropy: H(t) = −Σ_k Ek(t) log(Ek(t) + ε)
        H_per_t = -(E_norm * torch.log(E_norm + self.epsilon)).sum(dim=0)  # [T_lp]

        H_bar = H_per_t.mean()                               # scalar
        log_Nr = torch.log(torch.tensor(Nr, dtype=torch.float32, device=ring_energies.device))

        C_ring = 1.0 - H_bar / log_Nr.clamp(min=self.epsilon)
        return C_ring.clamp(0.0, 1.0)

    def __repr__(self) -> str:
        return (
            f"RotationalMotionLoss(Nr={self.Nr}, M={self.M}, "
            f"delta={self.delta})"
        )
