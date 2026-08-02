"""
losses/physics_motion_loss.py — Top-level physics motion loss orchestrator.

Implements the full pipeline from Figure 1 of arXiv 2506.02244v2:

    x̂₀ → 3D FFT (per channel, Hann window)
        → low-pass cube (ϱ=0.3)
        → [L_trans branch] WLS plane fitting
        → [L_rot branch]   polar resample → angular DFT → temporal DFT → ring/line stats
        → [L_scale branch] ring energies → gradient alignment + centroid trend
        → adaptive softmax weighting
        → L_motion (scalar, added to backbone denoising loss × weight=0.1)

Key implementation details (Sec 4.1, App A.1):
  - RGB processed channel-wise; energies summed across channels.
  - Spectral/solver blocks run in FP32 regardless of backbone precision.
  - Loss computed on the denoised prediction x̂₀ at every diffusion timestep.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from ..spectral.fft_utils import SpectralProcessor
from ..utils.precision import FP32Context
from .adaptive_composite import AdaptiveMotionLoss
from .rotation_loss import RotationalMotionLoss
from .scaling_loss import ScalingMotionLoss
from .translation_loss import TranslationalMotionLoss


class PhysicsMotionLoss(nn.Module):
    """Full frequency-domain physics motion regulariser.

    Paper reference: Figure 1, Section 3, Section 4.1.

    Plugs into any video diffusion training loop:

        total_loss = denoising_loss + physics_weight * physics_loss(x0_hat)

    Args:
        rho:                Low-pass fraction per dim (conf 0.97, Sec 4.1).
        Nr:                 Number of concentric rings (conf 0.85, App A.8).
        M:                  Angular bins (conf 0.85, App A.8).
        Nxi:                Log-radius bins (conf 0.85, App A.8).
        delta:              Tilted-line bandwidth in bins (conf 0.99, App A.8).
        tau:                Adaptive weighting temperature (conf 0.97, Table 5).
        ridge_lambda:       WLS ridge regularisation (conf 0.97, App A.6).
        epsilon:            Numeric jitter (conf 0.97, App A.6).
        epsilon_stab:       Centroid denominator stability (conf 0.97, Sec 3.6).
        tau_E:              Energy gate threshold (conf 0.90, App A.6).
        gate_f:             Energy gate sharpness (conf 0.90, App A.6).
        obs_lam:            Observability gate lambda (conf 0.90, App A.6).
        stop_grad_weights:  Stop-grad on adaptive weights (conf 0.70, assumed False).
        window:             Temporal window type (conf 0.99).
        T_min_scaling:      Min frames for scaling loss (conf 0.99, Sec 3.6).
    """

    def __init__(
        self,
        rho: float = 0.3,
        Nr: int = 20,
        M: int = 24,
        Nxi: int = 24,
        delta: float = 1.0,
        tau: float = 0.1,
        ridge_lambda: float = 1e-3,
        epsilon: float = 1e-8,
        epsilon_stab: float = 1e-8,
        tau_E: float = 0.10,
        gate_f: float = 10.0,
        obs_lam: float = 1.0,
        stop_grad_weights: bool = False,
        window: str = "hann",
        T_min_scaling: int = 3,
    ) -> None:
        super().__init__()

        self.spectral = SpectralProcessor(
            rho=rho, Nr=Nr, M=M, Nxi=Nxi, window=window
        )
        self.trans_loss = TranslationalMotionLoss(
            ridge_lambda=ridge_lambda, epsilon=epsilon,
            tau_E=tau_E, gate_f=gate_f,
        )
        self.rot_loss = RotationalMotionLoss(
            Nr=Nr, M=M, delta=delta,
            tau_E=tau_E, gate_f=gate_f, obs_lam=obs_lam, epsilon=epsilon,
        )
        self.scale_loss = ScalingMotionLoss(
            epsilon_stab=epsilon_stab, epsilon=epsilon, T_min=T_min_scaling,
        )
        self.adaptive = AdaptiveMotionLoss(
            tau=tau, stop_grad_weights=stop_grad_weights
        )

        self._fp32 = FP32Context()

    @classmethod
    def from_config(cls, cfg) -> "PhysicsMotionLoss":
        """Construct from an OmegaConf DictConfig (from configs/config.yaml)."""
        return cls(
            rho=cfg.spectral.low_pass_rho,
            Nr=cfg.spectral.Nr_rings,
            M=cfg.spectral.M_angular_bins,
            Nxi=cfg.spectral.Nxi_log_radius_bins,
            delta=cfg.spectral.delta_tolerance,
            tau=cfg.losses.softmax_temperature,
            ridge_lambda=cfg.losses.ridge_lambda,
            epsilon=cfg.losses.numeric_epsilon,
            epsilon_stab=cfg.losses.epsilon_stab,
            tau_E=cfg.gates.energy_threshold_tau_E,
            gate_f=cfg.gates.energy_smoothing_f,
            obs_lam=cfg.gates.observability_lambda_rot,
            stop_grad_weights=cfg.losses.stop_grad_weights,
            window=cfg.spectral.temporal_window,
            T_min_scaling=cfg.spectral.T_min_scaling,
        )

    def forward(
        self, x0_hat: Tensor
    ) -> Dict[str, Tensor]:
        """Compute the composite physics motion loss.

        Processes each RGB channel independently, summing energies across channels
        per Appendix A.1: "RGB is handled channel-wise with energies summed."

        Args:
            x0_hat: Denoised video prediction, shape [B, C, T, H, W].
                    C=1 or C=3. float32 or bfloat16 (cast to fp32 internally).

        Returns:
            Dict with keys:
              'loss':     Scalar L_motion (the value to add to training loss).
              'L_trans':  Scalar translation component.
              'L_rot':    Scalar rotation component.
              'L_scale':  Scalar scaling component.
              'w_trans':  Adaptive weight for translation.
              'w_rot':    Adaptive weight for rotation.
              'w_scale':  Adaptive weight for scaling.
        """
        assert x0_hat.dim() == 5, (
            f"Expected [B, C, T, H, W], got {x0_hat.shape}"
        )
        B, C, T, H, W = x0_hat.shape

        # Accumulate per-sample, per-channel losses
        L_trans_acc = torch.zeros(1, device=x0_hat.device)
        L_rot_acc   = torch.zeros(1, device=x0_hat.device)
        L_scale_acc = torch.zeros(1, device=x0_hat.device)
        count = 0

        # Disable autocast: spectral/solver blocks must run in FP32 (Sec 4.1)
        with self._fp32:
            x_fp32 = x0_hat.float()  # ensure FP32

            for b in range(B):
                # Accumulate channel-wise energies (App A.1)
                for c in range(C):
                    video = x_fp32[b, c]  # [T, H, W]

                    L_t, L_r, L_s = self._process_channel(video)
                    L_trans_acc = L_trans_acc + L_t
                    L_rot_acc   = L_rot_acc   + L_r
                    L_scale_acc = L_scale_acc + L_s
                    count += 1

        # Average over batch × channels
        n = max(count, 1)
        L_trans = L_trans_acc / n
        L_rot   = L_rot_acc   / n
        L_scale = L_scale_acc / n

        L_motion, weights = self.adaptive(
            L_trans.squeeze(), L_rot.squeeze(), L_scale.squeeze()
        )

        return {
            "loss":    L_motion,
            "L_trans": L_trans.squeeze(),
            "L_rot":   L_rot.squeeze(),
            "L_scale": L_scale.squeeze(),
            "w_trans": weights[0],
            "w_rot":   weights[1],
            "w_scale": weights[2],
        }

    def _process_channel(self, video: Tensor) -> tuple:
        """Run the full spectral pipeline on a single [T, H, W] channel.

        Returns:
            Tuple (L_trans, L_rot, L_scale) — three scalar losses.
        """
        # Step 1: 3D FFT with Hann window (App A.1)
        spectrum = self.spectral.compute_spectrum(video)   # [T, H, W] complex

        # Step 2: Low-pass cube truncation (Sec 4.1)
        lp_cube = self.spectral.apply_lowpass_cube(spectrum)  # [T_lp, H_lp, W_lp]

        # Step 3: Ring energies (used by both rotation and scaling branches)
        ring_E = self.spectral.get_ring_energies(lp_cube)    # [Nr, T_lp]

        # Step 4: Translation loss — WLS plane fit in (ωx, ωy, ωt)
        L_trans = self.trans_loss(lp_cube)                   # scalar

        # Step 5: Rotation loss — polar resample → angular+temporal DFT
        polar_seq = self.spectral.to_polar_sequence(lp_cube)  # [T_lp, Nr, M]
        L_rot = self.rot_loss(polar_seq, ring_E)               # scalar

        # Step 6: Scaling loss — radial gradient alignment + centroid trend
        L_scale = self.scale_loss(ring_E)                      # scalar

        return L_trans, L_rot, L_scale

    def __repr__(self) -> str:
        return (
            f"PhysicsMotionLoss(\n"
            f"  spectral={self.spectral},\n"
            f"  trans={self.trans_loss},\n"
            f"  rot={self.rot_loss},\n"
            f"  scale={self.scale_loss},\n"
            f"  adaptive={self.adaptive}\n"
            f")"
        )
