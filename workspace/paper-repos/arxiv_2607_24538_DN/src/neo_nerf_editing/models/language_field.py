"""
models/language_field.py

Implements the language-guided object localization from Sec. II:
    phi(p) in R^{d_l}: text embedding of prompt p
    L(x) in R^{d_l}: language feature at surface point x
Cosine similarity between phi(p) and L(x) over surface points yields high-relevance points;
the largest cluster of those is fit to an oriented bounding box b=(c,theta).

SIR note: the real system uses CLIP + DINO (Sec. II-D uses these explicitly for the language
field UPDATE step; the paper does not restate which embedding model produces phi(p) for the
initial grounding, but LERF [14] -- which this paper builds on -- uses CLIP). CLIP weights are
unreachable from this sandbox's network allowlist, so `embed_prompt` below uses a deterministic
hash-based embedding as an explicitly documented substitute (see architecture_plan risk R2).
Swap `DeterministicHashEmbedder` for a real CLIP text encoder to restore paper fidelity.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch


@dataclass
class OrientedBox:
    """b = (c, theta) from Sec. II: 3D center c, planar yaw theta in SO(2), plus extents
    needed to actually carve out a finite region (the paper does not name a separate extents
    symbol, but a bounding box is meaningless without one -- SIR ambiguity, extents assumed
    from the clustered point spread)."""

    center: np.ndarray  # [3]
    yaw: float  # radians, rotation about vertical (y) axis
    extents: np.ndarray  # [3] half-widths along local (x,y,z) box axes

    def contains(self, points: np.ndarray) -> np.ndarray:
        """points: [...,3] world coords -> bool mask [...] of which points fall inside b."""
        rel = points - self.center
        cos_t, sin_t = np.cos(-self.yaw), np.sin(-self.yaw)
        x_local = rel[..., 0] * cos_t - rel[..., 2] * sin_t
        z_local = rel[..., 0] * sin_t + rel[..., 2] * cos_t
        y_local = rel[..., 1]
        inside = (
            (np.abs(x_local) <= self.extents[0])
            & (np.abs(y_local) <= self.extents[1])
            & (np.abs(z_local) <= self.extents[2])
        )
        return inside

    def contains_torch(self, points: torch.Tensor) -> torch.Tensor:
        """Same as `contains` but for torch tensors (used inside the training loop)."""
        center = torch.as_tensor(self.center, dtype=points.dtype, device=points.device)
        extents = torch.as_tensor(self.extents, dtype=points.dtype, device=points.device)
        rel = points - center
        cos_t, sin_t = float(np.cos(-self.yaw)), float(np.sin(-self.yaw))
        x_local = rel[..., 0] * cos_t - rel[..., 2] * sin_t
        z_local = rel[..., 0] * sin_t + rel[..., 2] * cos_t
        y_local = rel[..., 1]
        inside = (
            (x_local.abs() <= extents[0]) & (y_local.abs() <= extents[1]) & (z_local.abs() <= extents[2])
        )
        return inside

    def __repr__(self) -> str:
        return f"OrientedBox(center={self.center.round(3).tolist()}, yaw={self.yaw:.3f}, extents={self.extents.round(3).tolist()})"


class DeterministicHashEmbedder:
    """STUB substitute for a real CLIP/DINO text+image encoder (unreachable in this sandbox).
    Produces a deterministic, unit-norm pseudo-embedding from a string by hashing it into a
    fixed-size vector. This preserves the *interface* (text -> R^{d_l} vector, cosine-comparable
    against per-point language features) so the rest of the pipeline is exercised end-to-end,
    but it carries NO real semantic content -- it cannot actually distinguish "soup can" from
    "lego brick" beyond what the synthetic scene's own labels encode for evaluation purposes.
    Replace with `open_clip` / `transformers` CLIP + DINO on a machine with model-hub access.
    """

    def __init__(self, d_l: int = 64):
        self.d_l = d_l

    def embed_text(self, text: str) -> np.ndarray:
        digest = hashlib.sha256(text.strip().lower().encode("utf-8")).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        vec = rng.normal(size=self.d_l).astype(np.float32)
        return vec / (np.linalg.norm(vec) + 1e-8)


class LanguageGrounder:
    """Localizes a language prompt in the scene via cosine similarity + clustering (Sec. II)."""

    def __init__(self, embedder: DeterministicHashEmbedder):
        self.embedder = embedder

    def embed_prompt(self, prompt: str) -> torch.Tensor:
        """phi(p) in R^{d_l}."""
        return torch.from_numpy(self.embedder.embed_text(prompt))

    def localize(
        self,
        surface_points: torch.Tensor,
        surface_lang_feats: torch.Tensor,
        prompt_embed: torch.Tensor,
        top_frac: float = 0.05,
    ) -> OrientedBox:
        """
        Args:
            surface_points: [N,3] candidate 3D surface points.
            surface_lang_feats: [N,d_l] language field values at those points.
            prompt_embed: [d_l] text embedding phi(p).
            top_frac: fraction of highest-cosine-similarity points to cluster (largest cluster
                fit to the box; single-cluster simplification since the synthetic scene has one
                object of interest at a time -- multi-cluster k-means is the natural extension
                for scenes with distractor objects sharing similar language features).
        Returns:
            OrientedBox fit to the highest-relevance point cluster.
        """
        assert surface_points.shape[0] == surface_lang_feats.shape[0], "points/features count mismatch"
        assert surface_points.shape[0] > 0, "no surface points provided for localization"

        feats = surface_lang_feats / (surface_lang_feats.norm(dim=-1, keepdim=True) + 1e-8)
        prompt = prompt_embed / (prompt_embed.norm() + 1e-8)
        sims = feats @ prompt.to(feats.dtype)  # [N] cosine similarity

        k = max(1, int(top_frac * surface_points.shape[0]))
        top_idx = torch.topk(sims, k).indices
        cluster_points = surface_points[top_idx].detach().cpu().numpy()  # [k,3]

        center = cluster_points.mean(axis=0)
        centered = cluster_points - center
        # Fit yaw via PCA on the (x,z) ground-plane components -> dominant horizontal axis.
        xz = centered[:, [0, 2]]
        if xz.shape[0] >= 2 and np.linalg.norm(xz) > 1e-6:
            cov = np.cov(xz.T)
            eigvals, eigvecs = np.linalg.eigh(cov)
            principal = eigvecs[:, np.argmax(eigvals)]
            yaw = float(np.arctan2(principal[1], principal[0]))
        else:
            yaw = 0.0

        cos_t, sin_t = np.cos(-yaw), np.sin(-yaw)
        x_local = centered[:, 0] * cos_t - centered[:, 2] * sin_t
        z_local = centered[:, 0] * sin_t + centered[:, 2] * cos_t
        y_local = centered[:, 1]
        extents = np.array(
            [
                max(np.abs(x_local).max(), 1e-2),
                max(np.abs(y_local).max(), 1e-2),
                max(np.abs(z_local).max(), 1e-2),
            ],
            dtype=np.float32,
        ) * 1.15  # small margin so the box fully encloses (not just touches) the cluster

        return OrientedBox(center=center.astype(np.float32), yaw=yaw, extents=extents)

    def __repr__(self) -> str:
        return f"LanguageGrounder(d_l={self.embedder.d_l})"
