"""
utils/grasp_stub.py

STUB for AnyGrasp [32], the external grasp-planning system NEO uses for actual grasp execution
(Sec. II: "For grasp execution, AnyGrasp [32] is used to generate candidate grasps"). AnyGrasp
is a separate, license-gated research system and is NOT reproduced here. This module exists
purely so the pipeline has a real call site to integrate a genuine AnyGrasp installation
against; `AnyGraspStub` returns simple geometrically plausible candidate poses (top-down and
side approaches at the bounding box surface) with no claim to real grasp-quality fidelity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from neo_nerf_editing.models.language_field import OrientedBox


@dataclass
class Grasp:
    """A candidate grasp pose.

    Args:
        position: [3] world-space gripper-center target.
        approach_dir: [3] unit vector, direction the gripper approaches from.
        score: heuristic plausibility score in [0,1] (NOT a real grasp-quality estimate).
    """

    position: np.ndarray
    approach_dir: np.ndarray
    score: float

    def __repr__(self) -> str:
        return f"Grasp(pos={self.position.round(3).tolist()}, score={self.score:.2f})"


class AnyGraspStub:
    """
    STUB: This component (AnyGrasp [32]) was not described in sufficient implementation
    detail in the NEO paper (it is used as an off-the-shelf external system, not re-derived),
    and the real AnyGrasp codebase/weights are proprietary/license-gated and unavailable in
    this sandbox. Replace this stub with a real AnyGrasp installation before relying on its
    grasp quality for anything beyond exercising the pipeline's call sites.
    """

    def generate_candidates(self, box: OrientedBox, n_candidates: int = 8) -> List[Grasp]:
        """Generates simple top-down + side-approach candidate grasps at the box surface.
        NOT a reproduction of AnyGrasp's actual grasp-synthesis network."""
        grasps: List[Grasp] = []
        top_position = box.center + np.array([0.0, box.extents[1], 0.0], dtype=np.float32)
        grasps.append(Grasp(position=top_position, approach_dir=np.array([0.0, -1.0, 0.0]), score=0.9))

        n_sides = max(0, n_candidates - 1)
        for i in range(n_sides):
            angle = 2 * np.pi * i / max(n_sides, 1)
            offset = box.extents[0] * np.array([np.cos(angle), 0.0, np.sin(angle)], dtype=np.float32)
            position = box.center + offset
            approach_dir = -offset / (np.linalg.norm(offset) + 1e-8)
            grasps.append(Grasp(position=position, approach_dir=approach_dir, score=0.6))
        return grasps

    def __repr__(self) -> str:
        return "AnyGraspStub(STUB, not_real_AnyGrasp=True)"
