"""
losses/translation_loss.py — Translational motion loss via energy-weighted ridge WLS.

Implements Section 3.4 and Appendix D.1 of arXiv 2506.02244v2.

Theory: constant-velocity translation V(x,y,t) = V0(x−vx·t, y−vy·t) concentrates
spectral energy on the plane:
    ωt + vx·ωx + vy·ωy + b0 = 0         (Eq. 3.3, translation slice of SIM(2))

The loss is the normalised WLS residual from fitting this plane:
    L_trans = Σ Wii(Ai·β̂ - bi)² / Σ Wii    (Section 3.4)

where rows Ai = (ωx,i, ωy,i, 1), bi = −ωt,i.

Fully differentiable: torch.linalg.solve propagates gradients through β̂
back into the spectrum and ultimately into model weights.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ..spectral.gates import EnergyGate


class TranslationalMotionLoss(nn.Module):
    """Energy-weighted ridge WLS plane fitting in (ωx, ωy, ωt) space.

    Paper reference: Section 3.4, Appendix D.1.
    SIM(2) slice: ωt + vx·ωx + vy·ωy + b0 = 0

    Args:
        ridge_lambda: Ridge regularisation λ = 1e-3 (App A.6, conf 0.97).
        epsilon:      Numeric jitter for stability (App A.6, conf 0.97).
        tau_E:        Energy gate threshold (App A.6, conf 0.90).
        gate_f:       Energy gate sharpness (App A.6, conf 0.90).
    """

    def __init__(
        self,
        ridge_lambda: float = 1e-3,
        epsilon: float = 1e-8,
        tau_E: float = 0.10,
        gate_f: float = 10.0,
    ) -> None:
        super().__init__()
        self.ridge_lambda = ridge_lambda
        self.epsilon = epsilon
        self.energy_gate = EnergyGate(tau_E=tau_E, f=gate_f)

    def forward(self, lowpass_cube: Tensor) -> Tensor:
        """Compute translational motion loss.

        Args:
            lowpass_cube: Low-pass complex spectrum, shape [T_lp, H_lp, W_lp].

        Returns:
            Scalar loss L_trans ≥ 0.  Zero iff motion is perfectly translational.
        """
        assert lowpass_cube.dim() == 3, (
            f"Expected [T_lp, H_lp, W_lp], got {lowpass_cube.shape}"
        )
        T_lp, H_lp, W_lp = lowpass_cube.shape

        # --- Build frequency coordinate grids ---
        # Sec 3.4: design matrix Ai = (ωx,i, ωy,i, 1), bi = −ωt,i
        omega_t = torch.fft.fftfreq(T_lp, device=lowpass_cube.device)  # [T_lp]
        omega_y = torch.fft.fftfreq(H_lp, device=lowpass_cube.device)  # [H_lp]
        omega_x = torch.fft.fftfreq(W_lp, device=lowpass_cube.device)  # [W_lp]

        # Expand to [T_lp, H_lp, W_lp] via broadcasting
        ot = omega_t.view(T_lp, 1, 1).expand(T_lp, H_lp, W_lp)
        oy = omega_y.view(1, H_lp, 1).expand(T_lp, H_lp, W_lp)
        ox = omega_x.view(1, 1, W_lp).expand(T_lp, H_lp, W_lp)

        # --- Energy and weights ---
        E = lowpass_cube.abs() ** 2  # [T_lp, H_lp, W_lp]
        E_max = E.max().clamp(min=self.epsilon)
        gate = self.energy_gate(E, E_max)  # [T_lp, H_lp, W_lp]
        W_flat = gate.reshape(-1)          # [N]

        # --- Build design matrix (App D.1) ---
        N = T_lp * H_lp * W_lp
        ox_flat = ox.reshape(N)
        oy_flat = oy.reshape(N)
        ot_flat = ot.reshape(N)

        # A ∈ R^{N×3}: rows are [ωx, ωy, 1]
        ones = torch.ones(N, device=lowpass_cube.device)
        A = torch.stack([ox_flat, oy_flat, ones], dim=1)  # [N, 3]
        b = -ot_flat                                       # [N]  (bi = −ωt,i)

        # --- Weighted ridge least squares (fully differentiable) ---
        beta_hat = self._ridge_wls(A, b, W_flat)  # [3]

        # --- Normalised residual (Sec 3.4) ---
        residuals = A @ beta_hat - b               # [N]
        w_sum = W_flat.sum().clamp(min=self.epsilon)
        L_trans = (W_flat * residuals ** 2).sum() / w_sum

        return L_trans

    def _ridge_wls(self, A: Tensor, b: Tensor, W: Tensor) -> Tensor:
        """Energy-weighted ridge least-squares: β̂ = argmin Σ Wi(Ai·β−bi)² + λ‖β‖².

        Paper reference: Section 3.4, App D.1.
        Fully differentiable via torch.linalg.solve.

        Args:
            A: Design matrix [N, 3].
            b: Target vector [N].
            W: Diagonal weights [N].

        Returns:
            Coefficient vector β̂ = [vx, vy, b0], shape [3].
        """
        # Weighted normal equations: (AᵀWA + λI) β = AᵀW b
        WA = W.unsqueeze(1) * A           # [N, 3]
        AtWA = A.t() @ WA                 # [3, 3]
        AtWb = WA.t() @ b                 # [3]

        # Ridge regularisation: M = AᵀWA + λI  (App A.6: λ=1e-3)
        lam_I = self.ridge_lambda * torch.eye(3, device=A.device, dtype=A.dtype)
        M = AtWA + lam_I                  # [3, 3]

        # Solve M·β = AᵀWb (differentiable)
        try:
            beta_hat = torch.linalg.solve(M, AtWb)  # [3]
        except torch.linalg.LinAlgError:
            # Fallback to pseudo-inverse if M is ill-conditioned (App A.6)
            beta_hat = torch.linalg.lstsq(M, AtWb.unsqueeze(1)).solution.squeeze(1)

        return beta_hat

    def __repr__(self) -> str:
        return (
            f"TranslationalMotionLoss(ridge_lambda={self.ridge_lambda}, "
            f"epsilon={self.epsilon})"
        )
