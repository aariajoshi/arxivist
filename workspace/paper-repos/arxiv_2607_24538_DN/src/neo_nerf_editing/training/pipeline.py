"""
training/pipeline.py

Implements Sec. II-C ("Knowledge Distillation Method for Scene Reconfiguration"), II-D
("Updating Language Fields"), and II-E ("Editing Many Times").

Core idea (Sec. II-C, Fig. 7): rendering is decomposed into three disjoint regions --
  (i)   outside the object bounding box       -> supervised by F_Theta (original teacher)
  (ii)  inside the removal bounding box b     -> supervised by F~_Theta (edited teacher)
  (iii) the relocated-object region           -> supervised by F_Theta under the desired motion
A student NeRF F_hat_Theta is distilled against this region-wise composite target S', in two
phases (field-level, then + image-level RGB/depth supervision -- Eq. 6).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from neo_nerf_editing.models.language_field import DeterministicHashEmbedder, OrientedBox
from neo_nerf_editing.training.losses import TeacherStudentDistillationLoss
from neo_nerf_editing.training.trainer import VolumetricRenderer, stratified_samples


def relocated_box_from_motion(box: OrientedBox, motion: np.ndarray) -> OrientedBox:
    """Compute the bounding box at the object's new pose after applying a 6-DoF motion
    tau = [x,y,z,roll,pitch,yaw] (Sec. II, 'reference object motion tau'). Roll/pitch are not
    modeled in the planar OrientedBox representation the paper itself uses for b (theta in
    SO(2), i.e. yaw-only) -- ASSUMED simplification consistent with the paper's own box
    parameterization; only translation + yaw are applied."""
    translation = np.asarray(motion[:3], dtype=np.float32)
    dyaw = float(motion[5]) if len(motion) > 5 else 0.0
    return OrientedBox(center=box.center + translation, yaw=box.yaw + dyaw, extents=box.extents)


class RegionCompositor:
    """Builds the region-wise target field S' (Sec. II-C, Fig. 7)."""

    def __init__(self, renderer: VolumetricRenderer):
        self.renderer = renderer

    def teacher_query(
        self, field: nn.Module, points: torch.Tensor, view_dirs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        return field(points, view_dirs)

    def compose(
        self,
        points: torch.Tensor,
        view_dirs: torch.Tensor,
        teacher_full: nn.Module,
        teacher_edited: nn.Module,
        box: OrientedBox,
        motion: np.ndarray,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            points: [...,3] query points (any leading shape, e.g. [rays, samples]).
            view_dirs: broadcastable to points.
        Returns:
            dict with 'density' [...,1], 'rgb' [...,3], 'lang_feat' [...,d_l] -- the composed
            target field S' at these points, ready to supervise the student.
        """
        relocated_box = relocated_box_from_motion(box, motion)
        inside_removal = box.contains_torch(points)
        inside_relocated = relocated_box.contains_torch(points)

        # Region (iii): transform relocated-region points back into the ORIGINAL object's frame
        # so we can query the object's original appearance from teacher_full, then place it at
        # the new pose (this is what "supervised by F_Theta under the desired motion" means).
        translation = torch.as_tensor(
            relocated_box.center - box.center, dtype=points.dtype, device=points.device
        )
        dyaw = float(relocated_box.yaw - box.yaw)
        cos_t, sin_t = float(np.cos(-dyaw)), float(np.sin(-dyaw))
        rel = points - translation - torch.as_tensor(box.center, dtype=points.dtype, device=points.device)
        x_l = rel[..., 0] * cos_t - rel[..., 2] * sin_t
        z_l = rel[..., 0] * sin_t + rel[..., 2] * cos_t
        y_l = rel[..., 1]
        points_in_object_frame = torch.stack(
            [x_l, y_l, z_l], dim=-1
        ) + torch.as_tensor(box.center, dtype=points.dtype, device=points.device)

        out_full_direct = self.teacher_query(teacher_full, points, view_dirs)
        out_full_relocated = self.teacher_query(teacher_full, points_in_object_frame, view_dirs)
        out_edited = self.teacher_query(teacher_edited, points, view_dirs)

        composed = {}
        for key in ("density", "rgb", "lang_feat"):
            v_outside = out_full_direct[key]
            v_removal = out_edited[key]
            v_relocated = out_full_relocated[key]
            v = torch.where(inside_removal.unsqueeze(-1), v_removal, v_outside)
            v = torch.where(inside_relocated.unsqueeze(-1), v_relocated, v)
            composed[key] = v
        return composed

    def __repr__(self) -> str:
        return "RegionCompositor()"


class StudentDistiller:
    """Two-phase distillation of a student NeRF F_hat_Theta against the region-wise composite
    target S' (Sec. II-C, Eq. 6)."""

    def __init__(self, lr: float = 5e-4):
        self.renderer = VolumetricRenderer()
        self.loss_fn = TeacherStudentDistillationLoss()
        self.lr = lr

    def distill(
        self,
        student: nn.Module,
        compositor: RegionCompositor,
        teacher_full: nn.Module,
        teacher_edited: nn.Module,
        box: OrientedBox,
        motion: np.ndarray,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        near: float,
        far: float,
        n_samples: int,
        n_steps_phase1: int,
        n_steps_phase2: int,
        batch_size: int = 512,
        device: str = "cpu",
    ) -> Dict[str, List[float]]:
        """Trains `student` in place. Returns loss history for both phases."""
        optimizer = torch.optim.Adam(student.parameters(), lr=self.lr)
        history: Dict[str, List[float]] = {"phase1": [], "phase2": []}
        n_total = rays_o.shape[0]

        # --- Phase 1: field-level supervision (density/color along rays) ---
        for _ in range(n_steps_phase1):
            idx = torch.randint(0, n_total, (min(batch_size, n_total),), device=device)
            o, d = rays_o[idx], rays_d[idx]
            t = stratified_samples(o.shape[0], near, far, n_samples, device)
            points = o[:, None, :] + t[..., None] * d[:, None, :]
            view_dirs = d[:, None, :].expand_as(points)

            with torch.no_grad():
                target = compositor.compose(points, view_dirs, teacher_full, teacher_edited, box, motion)
            student_out = student(points, view_dirs)

            loss = self.loss_fn.forward_field(
                target["density"].squeeze(-1), student_out["density"].squeeze(-1),
                target["rgb"], student_out["rgb"],
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            history["phase1"].append(loss.item())

        # --- Phase 2: + image-level RGB/depth supervision from full renders ---
        for _ in range(n_steps_phase2):
            idx = torch.randint(0, n_total, (min(batch_size, n_total),), device=device)
            o, d = rays_o[idx], rays_d[idx]
            t = stratified_samples(o.shape[0], near, far, n_samples, device)
            points = o[:, None, :] + t[..., None] * d[:, None, :]
            view_dirs = d[:, None, :].expand_as(points)

            with torch.no_grad():
                target = compositor.compose(points, view_dirs, teacher_full, teacher_edited, box, motion)
                t_weights = self.renderer.weights_from_density(target["density"], t)
                teacher_rgb = torch.sum(t_weights[..., None] * target["rgb"], dim=1)
                teacher_depth = torch.sum(t_weights * t, dim=1)

            student_render = self.renderer.render_rays(student, o, d, t)

            loss = self.loss_fn.forward_image(
                teacher_rgb, student_render["rgb"], teacher_depth, student_render["depth"]
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            history["phase2"].append(loss.item())

        return history

    def __repr__(self) -> str:
        return f"StudentDistiller(lr={self.lr})"


class LanguageFieldUpdater:
    """Sec. II-D: after geometric editing, keep the edited NeRF F_hat_Theta fixed and update
    only the language field L' using CLIP+DINO-style supervision rendered at the original poses.

    SIR note: real CLIP+DINO models are unreachable in this sandbox (see language_field.py
    docstring). `DeterministicHashEmbedder` is reused here as the supervision source, applied to
    a per-point "pseudo-caption" derived from which teacher region a point came from -- this
    preserves the *procedure* (render from fixed geometry -> supervise language head only) while
    substituting the actual semantic model."""

    def __init__(self, embedder: DeterministicHashEmbedder, lr: float = 5e-4):
        self.embedder = embedder
        self.lr = lr

    def update(
        self,
        student: nn.Module,
        target_lang_feats: torch.Tensor,
        query_points: torch.Tensor,
        n_iterations: int,
        device: str = "cpu",
    ) -> List[float]:
        """
        Args:
            student: the fixed-geometry edited NeRF F_hat_Theta (only `lang_head` params are
                updated; density/rgb params are frozen, matching 'we keep the edited NeRF
                F_hat_Theta fixed').
            target_lang_feats: [N,d_l] target language features to distill into L'.
            query_points: [N,3] points at which those targets were computed.
        """
        for p in student.parameters():
            p.requires_grad_(False)
        for p in student.lang_head.parameters():
            p.requires_grad_(True)

        optimizer = torch.optim.Adam(student.lang_head.parameters(), lr=self.lr)
        history = []
        dummy_dir = torch.zeros_like(query_points)
        dummy_dir[..., 2] = 1.0

        for _ in range(n_iterations):
            out = student(query_points, dummy_dir)
            loss = torch.mean((out["lang_feat"] - target_lang_feats) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            history.append(loss.item())

        for p in student.parameters():
            p.requires_grad_(True)
        return history

    def __repr__(self) -> str:
        return "LanguageFieldUpdater()"


class NEOEditPipeline:
    """Sec. II-E, 'Editing Many Times': the top-level orchestrator matching Eq. 2,
        (S',L') = NEO((S,L), p, tau)
    where the returned (S',L') "retains the same structure as the original input
    representation" and can be fed back in as (S,L) for the next edit."""

    def __init__(self, compositor: RegionCompositor, distiller: StudentDistiller, lang_updater: LanguageFieldUpdater):
        self.compositor = compositor
        self.distiller = distiller
        self.lang_updater = lang_updater

    def edit_many(self, edits: List[Dict]) -> List[nn.Module]:
        """Applies a sequence of edits, each returning a new (fully independent) student NeRF
        that becomes the "teacher_full" for the *next* edit -- this is what makes the pipeline
        reusable across sequential manipulation / assembly tasks (Sec. V-D)."""
        results = []
        for edit_spec in edits:
            student = edit_spec["build_fresh_student"]()
            self.distiller.distill(
                student=student,
                compositor=self.compositor,
                teacher_full=edit_spec["teacher_full"],
                teacher_edited=edit_spec["teacher_edited"],
                box=edit_spec["box"],
                motion=edit_spec["motion"],
                rays_o=edit_spec["rays_o"],
                rays_d=edit_spec["rays_d"],
                near=edit_spec["near"],
                far=edit_spec["far"],
                n_samples=edit_spec["n_samples"],
                n_steps_phase1=edit_spec["n_steps_phase1"],
                n_steps_phase2=edit_spec["n_steps_phase2"],
            )
            results.append(student)
        return results

    def __repr__(self) -> str:
        return "NEOEditPipeline()"
