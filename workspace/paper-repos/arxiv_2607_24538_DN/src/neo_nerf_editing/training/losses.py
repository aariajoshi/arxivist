"""
training/losses.py

Implements every loss in the SIR's mathematical_spec (Sec. II-B, II-C):
  Eq. 3: L_rec  -- mask-conditioned reconstruction loss
  Eq. 4-5: L_reg -- density regularization (anti-reappearance inside removed-object bbox)
  Combined: L_masked = lambda_rec * L_rec + lambda_reg * L_reg
  Eq. 6: L_distill -- two-phase teacher-student distillation loss
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from neo_nerf_editing.models.language_field import OrientedBox


class MaskedReconstructionLoss(nn.Module):
    """Eq. 3:
        L_rec = E_{(I,M,Omega)~D} [ ell( R(F~_Theta, Omega), I; M ) ]
    Here `ell` is an L1 pixel loss over pixels selected by mask M (paper leaves the exact
    form of `ell` unspecified -- SIR mathematical_spec confidence 0.85, but the specific
    reduction 'L1 vs L2' is an undefined_symbol; L1 chosen as it is standard for masked image
    reconstruction / inpainting losses and more robust to the outlier pixels a heuristic
    inpainting backend can introduce)."""

    def forward(self, rendered: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rendered: [N,3] rendered RGB at masked pixels' rays.
            target: [N,3] ground-truth (or inpainted) RGB.
            mask: [N] float/bool selecting which pixels count toward the loss (1=include).
        """
        assert rendered.shape == target.shape, f"{rendered.shape} vs {target.shape}"
        mask = mask.to(rendered.dtype)
        assert mask.shape[0] == rendered.shape[0], "mask must align with per-pixel/ray dimension"
        per_pixel = torch.abs(rendered - target).mean(dim=-1)  # [N]
        denom = mask.sum().clamp(min=1.0)
        return (per_pixel * mask).sum() / denom


class DensityRegularizationLoss(nn.Module):
    """Eq. 4-5:
        L_reg = (1/n) * sum_i (sigma~_i - sigma~'_i)^2
        sigma~'(i) = 0 if p(i) in b else sigma~(i)
    Penalizes non-zero density predictions strictly within the removed-object bounding box b,
    discouraging density from reappearing there ('floaters')."""

    def forward(self, sigma: torch.Tensor, points: torch.Tensor, box: OrientedBox) -> torch.Tensor:
        """
        Args:
            sigma: [...,1] or [...] predicted densities sigma~ at sample points.
            points: [...,3] the corresponding 3D sample locations p(i).
            box: OrientedBox b.
        """
        if sigma.dim() == points.dim():
            sigma = sigma.squeeze(-1)
        assert sigma.shape == points.shape[:-1], f"sigma {sigma.shape} vs points leading dims {points.shape[:-1]}"
        inside = box.contains_torch(points)  # bool, same shape as sigma
        target = torch.where(inside, torch.zeros_like(sigma), sigma.detach())
        # Only samples inside b contribute a nonzero (sigma - 0)^2 term; samples outside b have
        # target == sigma.detach() so their gradient contribution is exactly zero, matching
        # "this loss applies only to samples within b" (paper, directly below Eq. 5).
        diff2 = (sigma - target) ** 2
        return diff2.mean()


class MaskedTrainingObjective(nn.Module):
    """L_masked = lambda_rec * L_rec + lambda_reg * L_reg (Sec. II-B, just above Sec. II-C).
    lambda_rec, lambda_reg numeric values are never given in the paper (SIR ambiguity #3);
    they are supplied by the caller from config.yaml (ASSUMED both = 1.0)."""

    def __init__(self):
        super().__init__()
        self.rec = MaskedReconstructionLoss()
        self.reg = DensityRegularizationLoss()

    def forward(
        self,
        rendered: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        sigma: torch.Tensor,
        points: torch.Tensor,
        box: OrientedBox,
        lambda_rec: float,
        lambda_reg: float,
    ) -> torch.Tensor:
        l_rec = self.rec(rendered, target, mask)
        l_reg = self.reg(sigma, points, box)
        return lambda_rec * l_rec + lambda_reg * l_reg


class TeacherStudentDistillationLoss(nn.Module):
    """Eq. 6 (two-phase):
        Phase 1 (field-level): (1/rm) * sum_{j,k} ( |sigma_jk - sigma_hat_jk| + ||c_jk - c_hat_jk||_1 )
        Phase 2 (image-level): (1/n) * sum_i ( ||C_i - C_hat_i||_2 + ||D_i - D_hat_i||_2 )
    """

    def forward_field(
        self, sigma_t: torch.Tensor, sigma_s: torch.Tensor, color_t: torch.Tensor, color_s: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            sigma_t, sigma_s: [r,m] teacher/student densities per (ray,sample).
            color_t, color_s: [r,m,3] teacher/student colors per (ray,sample).
        """
        assert sigma_t.shape == sigma_s.shape, f"{sigma_t.shape} vs {sigma_s.shape}"
        assert color_t.shape == color_s.shape, f"{color_t.shape} vs {color_s.shape}"
        density_term = torch.abs(sigma_t - sigma_s).mean()
        color_term = torch.abs(color_t - color_s).sum(dim=-1).mean()  # ||.||_1 over RGB, then mean over r*m
        return density_term + color_term

    def forward_image(
        self, rgb_t: torch.Tensor, rgb_s: torch.Tensor, depth_t: torch.Tensor, depth_s: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            rgb_t, rgb_s: [n,3] teacher/student rendered RGB per pixel.
            depth_t, depth_s: [n] (or [n,1]) teacher/student rendered depth per pixel.
        """
        assert rgb_t.shape == rgb_s.shape, f"{rgb_t.shape} vs {rgb_s.shape}"
        if depth_t.dim() == 1:
            depth_t = depth_t.unsqueeze(-1)
        if depth_s.dim() == 1:
            depth_s = depth_s.unsqueeze(-1)
        assert depth_t.shape == depth_s.shape, f"{depth_t.shape} vs {depth_s.shape}"
        rgb_term = torch.linalg.vector_norm(rgb_t - rgb_s, dim=-1).mean()  # ||.||_2 per pixel, mean over n
        depth_term = torch.linalg.vector_norm(depth_t - depth_s, dim=-1).mean()
        return rgb_term + depth_term

    def __repr__(self) -> str:
        return "TeacherStudentDistillationLoss()"
