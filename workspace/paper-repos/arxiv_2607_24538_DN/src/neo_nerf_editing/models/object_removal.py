"""
models/object_removal.py

Implements Sec. II-A, "Neural Field Resampling for Object Removal" -- the paper's first
core contribution, illustrated in Fig. 4:
  1. For each camera ray, compute its intersection with bounding box b and EXCLUDE that
     ray segment from sampling (unlike DFF [26], which samples the box and suppresses density).
  2. Concatenate remaining ray intervals (typically a near and far segment) into one
     continuous 1D sampling domain.
  3. Draw a fixed number of uniform samples from that domain (stage 1).
  4. Convert densities -> volumetric rendering weights -> normalized PDF -> draw a second,
     fixed set of hierarchical (PDF-based) samples from the SAME domain, never inside b.

This two-stage procedure is what mitigates the "floating points and distorted regions" the
paper shows DFF produces behind a removed object (Fig. 3b) by reallocating samples that would
have fallen inside b to the rest of the ray instead.
"""
from __future__ import annotations

from typing import List, Tuple

import torch

from neo_nerf_editing.models.language_field import OrientedBox


class RayBoxExcluder:
    """Ray/oriented-box intersection test + exclusion-interval construction (Sec. II-A)."""

    def intersect(
        self, rays_o: torch.Tensor, rays_d: torch.Tensor, box: OrientedBox, near: float, far: float
    ) -> torch.Tensor:
        """Slab-test intersection of each ray with the box's local (axis-aligned once
        de-rotated by -yaw) frame.

        Args:
            rays_o, rays_d: [N,3]
            box: OrientedBox to exclude.
        Returns:
            hits: [N,2] (t_enter, t_exit) clamped to [near,far]; t_enter > t_exit encodes "no hit".
        """
        assert rays_o.shape == rays_d.shape, f"{rays_o.shape} vs {rays_d.shape}"
        center = torch.as_tensor(box.center, dtype=rays_o.dtype, device=rays_o.device)
        extents = torch.as_tensor(box.extents, dtype=rays_o.dtype, device=rays_o.device)
        cos_t, sin_t = float(torch.cos(torch.tensor(-box.yaw))), float(torch.sin(torch.tensor(-box.yaw)))

        def to_local(v: torch.Tensor) -> torch.Tensor:
            x = v[..., 0] * cos_t - v[..., 2] * sin_t
            z = v[..., 0] * sin_t + v[..., 2] * cos_t
            y = v[..., 1]
            return torch.stack([x, y, z], dim=-1)

        o_local = to_local(rays_o - center)
        d_local = to_local(rays_d)

        # Slab method against axis-aligned box [-extents, +extents] in local frame.
        eps = 1e-8
        d_safe = torch.where(d_local.abs() < eps, torch.full_like(d_local, eps), d_local)
        t1 = (-extents - o_local) / d_safe
        t2 = (extents - o_local) / d_safe
        t_min = torch.minimum(t1, t2)
        t_max = torch.maximum(t1, t2)
        t_enter = t_min.max(dim=-1).values
        t_exit = t_max.min(dim=-1).values

        t_enter = torch.clamp(t_enter, min=near, max=far)
        t_exit = torch.clamp(t_exit, min=near, max=far)
        # If exit <= enter after clamping, there is no valid intersection within [near,far].
        no_hit = t_exit <= t_enter
        t_enter = torch.where(no_hit, torch.full_like(t_enter, float("nan")), t_enter)
        t_exit = torch.where(no_hit, torch.full_like(t_exit, float("nan")), t_exit)
        return torch.stack([t_enter, t_exit], dim=-1)

    def build_exclusion_intervals(
        self, near: float, far: float, hits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hits: [N,2] (t_enter,t_exit) from `intersect`, NaN rows mean "ray misses box".
        Returns:
            (seg_near, seg_far): each [N,2] giving the (start,end) of the near-side and
            far-side remaining ray segments. For rays that miss the box entirely, seg_near
            covers the whole [near,far] range and seg_far is a degenerate empty (far,far) segment.
        """
        n = hits.shape[0]
        device = hits.device
        t_enter, t_exit = hits[:, 0], hits[:, 1]
        missed = torch.isnan(t_enter)

        seg_near = torch.stack(
            [torch.full((n,), near, device=device), torch.where(missed, torch.full((n,), far, device=device), t_enter)],
            dim=-1,
        )
        seg_far = torch.stack(
            [torch.where(missed, torch.full((n,), far, device=device), t_exit), torch.full((n,), far, device=device)],
            dim=-1,
        )
        # Guard against degenerate/negative-length segments from floating point edge cases.
        seg_near[:, 1] = torch.maximum(seg_near[:, 1], seg_near[:, 0])
        seg_far[:, 1] = torch.maximum(seg_far[:, 1], seg_far[:, 0])
        return seg_near, seg_far


class TwoStageResampler:
    """Uniform + PDF-based hierarchical resampling over the ray intervals that remain after
    excluding the object's bounding box (Sec. II-A, Fig. 4c)."""

    def uniform_resample(
        self, seg_near: torch.Tensor, seg_far: torch.Tensor, n_uniform: int, perturb: bool = True
    ) -> torch.Tensor:
        """Draw `n_uniform` samples from the concatenated (seg_near ∪ seg_far) 1D domain per ray.

        Implementation: allocate samples proportionally to each segment's length (a ray whose
        near-segment is 90% of the remaining domain gets ~90% of its samples there), which is
        the natural reading of "draw a fixed number of uniform samples from [the concatenated
        domain]" -- concatenation implies length-proportional density, not a fixed 50/50 split.

        Returns:
            t_samples: [N, n_uniform], sorted ascending per ray, none inside the excluded box
            (up to floating-point boundary tolerance).
        """
        n_rays = seg_near.shape[0]
        device = seg_near.device
        len_near = (seg_near[:, 1] - seg_near[:, 0]).clamp(min=1e-6)
        len_far = (seg_far[:, 1] - seg_far[:, 0]).clamp(min=1e-6)
        total_len = len_near + len_far
        frac_near = (len_near / total_len).clamp(0.0, 1.0)  # [N]

        u = torch.linspace(0.0, 1.0, n_uniform, device=device).expand(n_rays, n_uniform).clone()
        if perturb:
            step = 1.0 / n_uniform
            u = u + (torch.rand_like(u) - 0.5) * step
            u = u.clamp(0.0, 1.0)

        in_near = u <= frac_near[:, None]
        # Map u in [0,frac_near) -> seg_near, u in [frac_near,1] -> seg_far.
        t_near = seg_near[:, 0:1] + (u / frac_near[:, None].clamp(min=1e-6)) * len_near[:, None]
        u_far = (u - frac_near[:, None]) / (1.0 - frac_near[:, None]).clamp(min=1e-6)
        t_far = seg_far[:, 0:1] + u_far * len_far[:, None]
        t = torch.where(in_near, t_near, t_far)
        t, _ = torch.sort(t, dim=-1)
        return t

    def hierarchical_resample(
        self,
        seg_near: torch.Tensor,
        seg_far: torch.Tensor,
        coarse_t: torch.Tensor,
        coarse_weights: torch.Tensor,
        n_hierarchical: int,
    ) -> torch.Tensor:
        """PDF-based (inverse-CDF) resampling using the coarse pass's rendering weights as an
        (unnormalized) density estimate, restricted to the same excluded-box-free domain.

        Args:
            coarse_t: [N, S] t-values from the uniform pass.
            coarse_weights: [N, S] volumetric rendering weights at those t-values.
        Returns:
            [N, n_hierarchical] new t-values, sorted, none inside the excluded box.
        """
        n_rays = coarse_t.shape[0]
        device = coarse_t.device
        weights = coarse_weights + 1e-5  # avoid zero-probability regions collapsing the PDF
        pdf = weights / weights.sum(dim=-1, keepdim=True)
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)  # [N, S+1]

        u = torch.rand(n_rays, n_hierarchical, device=device)
        inds = torch.searchsorted(cdf, u, right=True).clamp(1, cdf.shape[-1] - 1)
        below = inds - 1
        cdf_below = torch.gather(cdf, 1, below)
        cdf_above = torch.gather(cdf, 1, inds)
        t_below = torch.gather(coarse_t, 1, below.clamp(max=coarse_t.shape[-1] - 1))
        t_above = torch.gather(coarse_t, 1, inds.clamp(max=coarse_t.shape[-1] - 1))
        denom = (cdf_above - cdf_below).clamp(min=1e-8)
        frac = (u - cdf_below) / denom
        t_new = t_below + frac * (t_above - t_below)

        # Safety clamp: guarantee no sample lands inside the excluded interval even under
        # floating-point edge cases at segment boundaries.
        t_new = torch.clamp(t_new, min=seg_near[:, 0:1], max=seg_far[:, 1:2])
        t, _ = torch.sort(t_new, dim=-1)
        return t

    def __repr__(self) -> str:
        return "TwoStageResampler()"
