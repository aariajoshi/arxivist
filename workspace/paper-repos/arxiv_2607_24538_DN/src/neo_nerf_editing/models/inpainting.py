"""
models/inpainting.py

Implements Sec. II-B, "Multiview-Consistent Progressive Inpainting" (Fig. 6):
  1. Augment training views with a virtual hemisphere around the removed-object bbox
     (4 azimuths, upward spiral of elevations).
  2. Tile 4 renderings into a 2x2 composite (inspired by NeRFiller [17]) with matching masks.
  3. Run a diffusion-based inpainting model on the composite, conditioned on the mask.
  4. Split the composite back into per-view completions; use them + originals to train a
     mask-conditioned NeRF (Eq. 3), progressively reducing inpainting strength xi each round.

SIR note: the paper names Stable Diffusion (Fig. 6 caption) as the inpainting backend. Stable
Diffusion checkpoints are unreachable from this sandbox's network allowlist (no huggingface.co
access). `InpaintingBackend` is an abstract interface so a real
`diffusers.StableDiffusionInpaintPipeline` backend is a drop-in replacement; the default
`HeuristicInpaintingBackend` below uses classical (non-generative) OpenCV inpainting purely so
the rest of the pipeline -- masking, tiling, the progressive xi schedule, mask-conditioned
training -- is faithfully exercised end-to-end. This is the single largest fidelity gap versus
the paper (see architecture_plan risk_assessment, severity High).
"""
from __future__ import annotations

import abc
from typing import Callable, List

import numpy as np
import torch

from neo_nerf_editing.data.rays import Camera, hemisphere_spiral_trajectory
from neo_nerf_editing.models.language_field import OrientedBox


class VirtualHemisphereSampler:
    """Samples virtual camera poses on a hemisphere around the removed-object bbox center
    (Sec. II-B: 'a virtual hemisphere centered at the bounding-box center ... at four opposing
    azimuths and varying elevations, forming a short upward spiral')."""

    def sample_views(self, box: OrientedBox, n_azimuths: int = 4, n_elevations: int = 3) -> List[Camera]:
        radius = float(np.linalg.norm(box.extents)) * 2.5  # far enough back to frame the whole box
        return hemisphere_spiral_trajectory(box.center, radius, n_azimuths=n_azimuths, n_elevations=n_elevations)

    def __repr__(self) -> str:
        return "VirtualHemisphereSampler()"


class InpaintingBackend(abc.ABC):
    """Pluggable inpainting backend interface. Swap in a real diffusion model by
    subclassing this and implementing `inpaint`."""

    @abc.abstractmethod
    def inpaint(self, composite_rgb: np.ndarray, composite_mask: np.ndarray, strength: float) -> np.ndarray:
        """
        Args:
            composite_rgb: [H,W,3] float32 in [0,1], the 2x2-tiled composite image.
            composite_mask: [H,W] bool, True where content must be (re)synthesized.
            strength: inpainting strength xi in [0,1]; 1.0 = fully resynthesize masked region
                from scratch, lower values blend more of the current rendering back in
                (Sec. II-B: 'we reduce the inpainting strength factor xi ... preserving more of
                the current rendering in later iterations').
        Returns:
            [H,W,3] float32 in [0,1], inpainted composite.
        """
        raise NotImplementedError


class HeuristicInpaintingBackend(InpaintingBackend):
    """SUBSTITUTE for Stable Diffusion (see module docstring). Uses OpenCV's classical
    Telea inpainting algorithm, blended with the original rendering by `strength` to emulate
    the paper's progressive strength-decay behavior. NOT a generative / semantically-aware
    inpainter -- it cannot hallucinate plausible new geometry/texture the way Stable Diffusion
    can; it only propagates nearby pixel statistics into the masked region.
    """

    def inpaint(self, composite_rgb: np.ndarray, composite_mask: np.ndarray, strength: float) -> np.ndarray:
        assert composite_rgb.ndim == 3 and composite_rgb.shape[-1] == 3, f"expected [H,W,3], got {composite_rgb.shape}"
        assert composite_mask.shape == composite_rgb.shape[:2], "mask/image spatial size mismatch"
        import cv2  # local import: keeps this an optional, swappable dependency

        img_u8 = np.clip(composite_rgb * 255.0, 0, 255).astype(np.uint8)
        mask_u8 = (composite_mask.astype(np.uint8)) * 255
        filled = cv2.inpaint(img_u8, mask_u8, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
        filled_f = filled.astype(np.float32) / 255.0

        # Blend by strength: strength=1 -> fully use the inpainted fill; strength->0 -> keep
        # more of the original rendering, matching the paper's decaying-xi behavior.
        blended = np.where(
            composite_mask[..., None],
            strength * filled_f + (1.0 - strength) * composite_rgb,
            composite_rgb,
        )
        return blended.astype(np.float32)

    def __repr__(self) -> str:
        return "HeuristicInpaintingBackend(backend='opencv_telea', SUBSTITUTE_for='stable_diffusion')"


def tile_2x2(images: List[np.ndarray], masks: List[np.ndarray]):
    """Tile exactly 4 [h,w,...] arrays into a single 2x2 composite (Sec. II-B, inspired by
    NeRFiller [17]). Returns (composite, ) with matching layout for images and masks."""
    assert len(images) == 4, f"tile_2x2 requires exactly 4 views, got {len(images)}"
    top = np.concatenate([images[0], images[1]], axis=1)
    bottom = np.concatenate([images[2], images[3]], axis=1)
    return np.concatenate([top, bottom], axis=0)


def split_2x2(composite: np.ndarray, h: int, w: int) -> List[np.ndarray]:
    """Inverse of `tile_2x2`."""
    top, bottom = composite[:h], composite[h : 2 * h]
    v0, v1 = top[:, :w], top[:, w : 2 * w]
    v2, v3 = bottom[:, :w], bottom[:, w : 2 * w]
    return [v0, v1, v2, v3]


class ProgressiveInpaintingLoop:
    """Drives the iterative re-render -> tile -> inpaint -> split -> (re)train loop of
    Sec. II-B, decaying the inpainting strength xi each round."""

    def __init__(self, backend: InpaintingBackend, xi_start: float = 1.0, xi_end: float = 0.2):
        self.backend = backend
        self.xi_start = xi_start
        self.xi_end = xi_end

    def xi_at(self, iteration: int, n_iterations: int) -> float:
        """Linear decay from xi_start to xi_end (ASSUMED schedule shape, SIR ambiguity #4)."""
        if n_iterations <= 1:
            return self.xi_start
        frac = iteration / (n_iterations - 1)
        return self.xi_start + frac * (self.xi_end - self.xi_start)

    def run_one_round(
        self,
        virtual_renders: List[np.ndarray],
        virtual_masks: List[np.ndarray],
        iteration: int,
        n_iterations: int,
    ) -> List[np.ndarray]:
        """Runs a single tile->inpaint->split round over a batch of exactly 4 virtual views."""
        assert len(virtual_renders) == 4 and len(virtual_masks) == 4, "expects batches of 4 virtual views"
        h, w = virtual_renders[0].shape[:2]
        composite_rgb = tile_2x2(virtual_renders, virtual_masks)
        composite_mask = tile_2x2(
            [m.astype(np.float32) for m in virtual_masks], [m.astype(np.float32) for m in virtual_masks]
        ).astype(bool)
        xi = self.xi_at(iteration, n_iterations)
        inpainted = self.backend.inpaint(composite_rgb, composite_mask, xi)
        return split_2x2(inpainted, h, w)

    def __repr__(self) -> str:
        return f"ProgressiveInpaintingLoop(backend={self.backend!r}, xi=[{self.xi_start},{self.xi_end}])"
