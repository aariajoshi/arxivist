"""
data/rays.py

Camera representation and pinhole ray generation. Not a component described explicitly in
the SIR's architecture graph (it is standard NeRF plumbing referenced implicitly by
Sec. II, "camera poses Omega"), but required to exercise every other module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class Camera:
    """A pinhole camera pose + intrinsics.

    Args:
        c2w: [4,4] camera-to-world extrinsic matrix.
        focal: focal length in pixels (assumes square pixels, shared fx=fy).
    """

    c2w: np.ndarray  # [4,4]
    focal: float

    def __repr__(self) -> str:
        t = self.c2w[:3, 3]
        return f"Camera(pos=({t[0]:.2f},{t[1]:.2f},{t[2]:.2f}), focal={self.focal:.1f})"


class RayGenerator:
    """Generates per-pixel rays for a pinhole `Camera`. Paper section: implicit in Eq. 1
    (camera images I with poses Omega used to train F_Theta)."""

    def rays_from_camera(self, camera: Camera, H: int, W: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (rays_o, rays_d), each [H*W, 3], in world space."""
        i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing="xy")
        # Camera-space ray directions (OpenGL-style: -z forward, y up).
        dirs = np.stack(
            [
                (i - W * 0.5) / camera.focal,
                -(j - H * 0.5) / camera.focal,
                -np.ones_like(i),
            ],
            axis=-1,
        )  # [H,W,3]
        R = camera.c2w[:3, :3]
        rays_d = dirs @ R.T  # [H,W,3]
        rays_d = rays_d / (np.linalg.norm(rays_d, axis=-1, keepdims=True) + 1e-8)
        rays_o = np.broadcast_to(camera.c2w[:3, 3], rays_d.shape)
        return rays_o.reshape(-1, 3).astype(np.float32), rays_d.reshape(-1, 3).astype(np.float32)


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0.0, 1.0, 0.0])) -> np.ndarray:
    """Build a [4,4] camera-to-world matrix looking from `eye` toward `target`."""
    forward = eye - target  # camera looks down -z, so "backward" axis points from target to eye
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    right = np.cross(up, forward)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-6:
        # Degenerate: `up` parallel to `forward`. Nudge `up` to avoid a singular basis.
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(up, forward)
        right_norm = np.linalg.norm(right)
    right = right / right_norm
    true_up = np.cross(forward, right)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = true_up
    c2w[:3, 2] = forward
    c2w[:3, 3] = eye
    return c2w


def fixed_scan_trajectory(n_views: int, radius: float = 4.0, height: float = 2.0) -> list:
    """A circular scan trajectory around the world origin, mirroring the paper's 'robot arm
    moves a camera along a fixed trajectory' capture procedure (Sec. III)."""
    cams = []
    for k in range(n_views):
        theta = 2 * np.pi * k / n_views
        eye = np.array([radius * np.cos(theta), height, radius * np.sin(theta)], dtype=np.float32)
        c2w = look_at(eye, target=np.array([0.0, 0.0, 0.0]))
        cams.append(Camera(c2w=c2w, focal=110.0))
    return cams


def hemisphere_spiral_trajectory(
    center: np.ndarray, radius: float, n_azimuths: int = 4, n_elevations: int = 3
) -> list:
    """Virtual viewpoints on a hemisphere spiral around `center`, per Sec. II-B / Fig. 6:
    'four opposing azimuths and varying elevations, forming a short upward spiral'."""
    cams = []
    for e in range(n_elevations):
        elevation = np.pi / 2 * (e + 1) / (n_elevations + 1)  # strictly between 0 and pi/2
        for a in range(n_azimuths):
            azimuth = 2 * np.pi * a / n_azimuths + e * (np.pi / n_azimuths)  # spiral offset per level
            eye = center + radius * np.array(
                [
                    np.cos(elevation) * np.cos(azimuth),
                    np.sin(elevation),
                    np.cos(elevation) * np.sin(azimuth),
                ],
                dtype=np.float32,
            )
            c2w = look_at(eye, target=center)
            cams.append(Camera(c2w=c2w, focal=110.0))
    return cams
