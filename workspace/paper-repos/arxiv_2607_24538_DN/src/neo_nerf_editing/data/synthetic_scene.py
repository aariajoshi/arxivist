"""
data/synthetic_scene.py

SUBSTITUTE for the NEO-Dataset (Sec. III): the real benchmark (12 registered tabletop scenes,
ArUco-marker-aligned pre/post-edit scans, YCB objects, two robot arms) is not publicly bundled
with the paper and was not reachable from this sandbox. This module procedurally generates an
analogous multi-view RGB-D tabletop scene with a small number of simple colored-box "objects"
sitting on a table plane, plus ground-truth pre/post-edit scans for a chosen object + motion --
enough structure to exercise the full removal -> inpainting -> relocation -> distillation ->
evaluation pipeline end-to-end and to compute the paper's own metrics against known ground truth.

This is explicitly NOT a substitute for real photorealistic scene complexity; see the
Reproducibility Notes in README.md and the Stage 6 comparison report.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from neo_nerf_editing.data.rays import Camera, RayGenerator, fixed_scan_trajectory


@dataclass
class ToyObject:
    name: str
    center: np.ndarray  # [3]
    half_extents: np.ndarray  # [3]
    color: np.ndarray  # [3] in [0,1]


@dataclass
class ToyScene:
    objects: List[ToyObject]
    table_half_extents: np.ndarray = field(default_factory=lambda: np.array([3.0, 0.05, 3.0], dtype=np.float32))
    table_color: np.ndarray = field(default_factory=lambda: np.array([0.6, 0.55, 0.5], dtype=np.float32))

    def sdf_scene_color_density(self, points: np.ndarray) -> Dict[str, np.ndarray]:
        """Analytic box-SDF-based 'ground truth' scene for a given set of points [...,3].
        Returns dict with 'density' [...] and 'rgb' [...,3], used both to render training
        views and to compute post-edit ground truth for evaluation."""
        density = np.zeros(points.shape[:-1], dtype=np.float32)
        rgb = np.zeros(points.shape[:-1] + (3,), dtype=np.float32)

        def box_occupancy(p, center, half_extents):
            d = np.abs(p - center) - half_extents
            inside = np.all(d <= 0, axis=-1)
            return inside

        table_inside = box_occupancy(points, np.array([0.0, -0.05, 0.0], dtype=np.float32), self.table_half_extents)
        density = np.where(table_inside, 50.0, density)
        rgb = np.where(table_inside[..., None], self.table_color, rgb)

        for obj in self.objects:
            inside = box_occupancy(points, obj.center, obj.half_extents)
            density = np.where(inside, 80.0, density)
            rgb = np.where(inside[..., None], obj.color, rgb)

        return {"density": density.astype(np.float32), "rgb": rgb.astype(np.float32)}


class ToySceneGenerator:
    """Procedural generator standing in for the NEO-Dataset's real robot-scanned scenes."""

    OBJECT_PALETTE = [
        ("soup_can", np.array([1.0, 0.15]), np.array([0.85, 0.15, 0.1])),
        ("lego_brick", np.array([0.6, 0.5]), np.array([0.9, 0.75, 0.1])),
        ("red_cup", np.array([0.5, 0.6]), np.array([0.8, 0.1, 0.1])),
    ]

    def generate_scene(self, seed: int = 0, objects: Optional[List[str]] = None) -> ToyScene:
        rng = np.random.default_rng(seed)
        names = objects or ["soup_can", "lego_brick"]
        toy_objects = []
        for i, name in enumerate(names):
            radius, half_h = 0.3, 0.25
            for palette_name, dims, color in self.OBJECT_PALETTE:
                if palette_name == name:
                    radius, half_h = dims[0] * 0.3, dims[1] * 0.5
                    break
            angle = rng.uniform(0, 2 * np.pi)
            dist = rng.uniform(0.4, 1.2)
            center = np.array([dist * np.cos(angle), half_h, dist * np.sin(angle)], dtype=np.float32)
            half_extents = np.array([radius, half_h, radius], dtype=np.float32)
            palette_match = next((c for p_name, _, c in self.OBJECT_PALETTE if p_name == name), None)
            color = palette_match.astype(np.float32) if palette_match is not None else rng.uniform(0.2, 0.9, size=3).astype(np.float32)
            toy_objects.append(ToyObject(name=name, center=center, half_extents=half_extents, color=color))
        return ToyScene(objects=toy_objects)

    def render_views(
        self, scene: ToyScene, camera_trajectory: List[Camera], H: int = 128, W: int = 128,
        near: float = 2.0, far: float = 8.0, n_samples: int = 128,
    ) -> Dict[str, torch.Tensor]:
        """Analytic volumetric rendering of the toy scene (ground-truth renderer, NOT the
        learned NeRF -- used to generate training targets and evaluation ground truth)."""
        gen = RayGenerator()
        all_rgb, all_depth, all_rays_o, all_rays_d = [], [], [], []
        for cam in camera_trajectory:
            rays_o, rays_d = gen.rays_from_camera(cam, H, W)
            t = np.linspace(near, far, n_samples, dtype=np.float32)
            points = rays_o[:, None, :] + t[None, :, None] * rays_d[:, None, :]  # [HW,S,3]
            out = scene.sdf_scene_color_density(points)
            density, rgb = out["density"], out["rgb"]

            deltas = np.diff(t, append=t[-1] + 1e10)
            alpha = 1.0 - np.exp(-density * deltas[None, :])
            trans = np.cumprod(np.concatenate([np.ones_like(alpha[:, :1]), 1.0 - alpha + 1e-10], axis=1), axis=1)[:, :-1]
            weights = alpha * trans
            rgb_out = np.sum(weights[..., None] * rgb, axis=1)
            depth_out = np.sum(weights * t[None, :], axis=1)

            all_rgb.append(rgb_out.reshape(H, W, 3))
            all_depth.append(depth_out.reshape(H, W))
            all_rays_o.append(rays_o)
            all_rays_d.append(rays_d)

        return {
            "images": torch.from_numpy(np.stack(all_rgb)).float(),
            "depths": torch.from_numpy(np.stack(all_depth)).float(),
            "rays_o": torch.from_numpy(np.concatenate(all_rays_o)).float(),
            "rays_d": torch.from_numpy(np.concatenate(all_rays_d)).float(),
            "rgb_flat": torch.from_numpy(np.concatenate([r.reshape(-1, 3) for r in all_rgb])).float(),
        }

    def apply_manipulation(
        self, scene: ToyScene, object_name: str, motion: np.ndarray, remove_only: bool = False
    ) -> ToyScene:
        """Returns a NEW ToyScene with `object_name` either removed or moved by `motion`
        [x,y,z,roll,pitch,yaw] (roll/pitch ignored, matching the planar OrientedBox assumption
        used throughout -- see training/pipeline.py:relocated_box_from_motion)."""
        new_objects = []
        for obj in scene.objects:
            if obj.name != object_name:
                new_objects.append(obj)
                continue
            if remove_only:
                continue
            translation = np.asarray(motion[:3], dtype=np.float32)
            new_objects.append(
                ToyObject(name=obj.name, center=obj.center + translation, half_extents=obj.half_extents, color=obj.color)
            )
        return ToyScene(objects=new_objects, table_half_extents=scene.table_half_extents, table_color=scene.table_color)

    def __repr__(self) -> str:
        return "ToySceneGenerator()"
