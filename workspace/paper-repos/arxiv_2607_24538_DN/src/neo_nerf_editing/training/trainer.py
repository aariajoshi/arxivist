"""
training/trainer.py

Volumetric rendering (standard NeRF formulation, referenced in Sec. II-A: "convert these
densities to volumetric rendering weights using the standard NeRF formulation [1]") and the
base NeRF training loop used to fit F_Theta (Eq. 1) and, later, the mask-conditioned
F~_Theta (Sec. II-B).
"""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn


class VolumetricRenderer:
    """Standard NeRF alpha-compositing renderer. Paper section: II-A, referencing [1]
    (Mildenhall et al.) for the density-to-weight conversion."""

    @staticmethod
    def weights_from_density(density: torch.Tensor, t_samples: torch.Tensor) -> torch.Tensor:
        """
        Args:
            density: [N_rays, N_samples, 1] (or [N_rays, N_samples])
            t_samples: [N_rays, N_samples] sample distances along each ray (sorted ascending).
        Returns:
            weights: [N_rays, N_samples] volumetric rendering weights (standard NeRF quadrature).
        """
        if density.dim() == 3:
            density = density.squeeze(-1)
        assert density.shape == t_samples.shape, (
            f"density {density.shape} and t_samples {t_samples.shape} must match"
        )
        deltas = t_samples[..., 1:] - t_samples[..., :-1]
        deltas = torch.cat([deltas, torch.full_like(deltas[..., :1], 1e10)], dim=-1)  # last interval -> infinity
        alpha = 1.0 - torch.exp(-density * deltas)
        transmittance = torch.cumprod(
            torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1), dim=-1
        )[..., :-1]
        weights = alpha * transmittance
        return weights

    def render_rays(
        self, field: nn.Module, rays_o: torch.Tensor, rays_d: torch.Tensor, t_samples: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            field: a JointNeRFField-compatible module.
            rays_o: [N_rays, 3]
            rays_d: [N_rays, 3]
            t_samples: [N_rays, N_samples] sample distances (sorted ascending, no NaNs).
        Returns:
            dict with 'rgb' [N_rays,3], 'depth' [N_rays], 'lang_feat' [N_rays,d_l], 'weights' [N_rays,N_samples]
        """
        assert rays_o.shape == rays_d.shape, f"rays_o {rays_o.shape} != rays_d {rays_d.shape}"
        assert not torch.isnan(t_samples).any(), "t_samples contains NaN — check exclusion-interval logic upstream"

        points = rays_o[:, None, :] + t_samples[..., None] * rays_d[:, None, :]  # [N,S,3]
        view_dirs = rays_d[:, None, :].expand_as(points)
        out = field(points, view_dirs)
        weights = self.weights_from_density(out["density"], t_samples)  # [N,S]

        rgb = torch.sum(weights[..., None] * out["rgb"], dim=1)  # [N,3]
        depth = torch.sum(weights * t_samples, dim=1)  # [N]
        lang_feat = torch.sum(weights[..., None] * out["lang_feat"], dim=1)  # [N,d_l]
        acc = torch.sum(weights, dim=1)  # [N] accumulated opacity, useful for masking background

        return {
            "rgb": rgb,
            "depth": depth,
            "lang_feat": lang_feat,
            "weights": weights,
            "acc": acc,
            "raw_density": out["density"],
        }

    def __repr__(self) -> str:
        return "VolumetricRenderer()"


def stratified_samples(
    n_rays: int, near: float, far: float, n_samples: int, device: str, perturb: bool = True
) -> torch.Tensor:
    """Standard stratified NeRF sampling over [near, far]. Returns [n_rays, n_samples]."""
    t = torch.linspace(0.0, 1.0, n_samples, device=device)
    t = near * (1.0 - t) + far * t
    t = t.expand(n_rays, n_samples).clone()
    if perturb:
        mids = 0.5 * (t[..., 1:] + t[..., :-1])
        upper = torch.cat([mids, t[..., -1:]], dim=-1)
        lower = torch.cat([t[..., :1], mids], dim=-1)
        rand = torch.rand_like(t)
        t = lower + (upper - lower) * rand
    return t


class NeRFTrainer:
    """Base training loop: samples ray batches from a dataset, renders, backprops an RGB
    (+ optional language-feature) reconstruction loss, and logs/checkpoints."""

    def __init__(self, lr: float = 5e-4, log_every: int = 50):
        self.renderer = VolumetricRenderer()
        self.lr = lr
        self.log_every = log_every

    def fit(
        self,
        field: nn.Module,
        dataset: Dict[str, torch.Tensor],
        n_steps: int,
        near: float,
        far: float,
        n_samples: int,
        batch_size: int = 1024,
        device: str = "cpu",
        loss_fn: Optional[Callable] = None,
        log_fn: Optional[Callable[[int, float], None]] = None,
    ) -> Dict[str, List[float]]:
        """
        Args:
            field: JointNeRFField (or compatible) to train, already on `device`.
            dataset: dict with 'rays_o' [M,3], 'rays_d' [M,3], 'rgb' [M,3] (flattened over all views/pixels).
            n_steps: number of optimizer steps.
            loss_fn: optional custom loss(rendered, target) -> scalar; defaults to MSE on RGB.
        Returns:
            {'loss': [n_steps floats]}
        """
        assert dataset["rays_o"].shape[0] == dataset["rgb"].shape[0], "rays and targets must align"
        optimizer = torch.optim.Adam(field.parameters(), lr=self.lr)
        history: Dict[str, List[float]] = {"loss": []}
        n_total = dataset["rays_o"].shape[0]
        loss_fn = loss_fn or (lambda pred, target: torch.mean((pred - target) ** 2))

        t0 = time.time()
        for step in range(n_steps):
            idx = torch.randint(0, n_total, (min(batch_size, n_total),), device=device)
            rays_o = dataset["rays_o"][idx]
            rays_d = dataset["rays_d"][idx]
            target_rgb = dataset["rgb"][idx]

            t_samples = stratified_samples(rays_o.shape[0], near, far, n_samples, device)
            out = self.renderer.render_rays(field, rays_o, rays_d, t_samples)
            loss = loss_fn(out["rgb"], target_rgb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            history["loss"].append(loss.item())
            if log_fn is not None and step % self.log_every == 0:
                log_fn(step, loss.item())

        history["elapsed_sec"] = [time.time() - t0]
        return history

    def __repr__(self) -> str:
        return f"NeRFTrainer(lr={self.lr})"
