"""
physics_motion_loss — Physics-Guided Motion Loss for Video Generation.

Implementation of arXiv 2506.02244v2:
"Physics-Guided Motion Loss for Video Generation Model"
Xue et al., 2025.

Quick start::

    from physics_motion_loss import PhysicsMotionLoss

    loss_fn = PhysicsMotionLoss()
    out = loss_fn(x0_hat)          # x0_hat: [B, C, T, H, W]
    total_loss = denoising_loss + 0.1 * out['loss']
    total_loss.backward()
"""

from .losses.physics_motion_loss import PhysicsMotionLoss
from .losses.translation_loss import TranslationalMotionLoss
from .losses.rotation_loss import RotationalMotionLoss
from .losses.scaling_loss import ScalingMotionLoss
from .losses.adaptive_composite import AdaptiveMotionLoss

__all__ = [
    "PhysicsMotionLoss",
    "TranslationalMotionLoss",
    "RotationalMotionLoss",
    "ScalingMotionLoss",
    "AdaptiveMotionLoss",
]

__version__ = "0.1.0"
