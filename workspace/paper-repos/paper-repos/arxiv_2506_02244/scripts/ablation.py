"""
scripts/ablation.py — Ablation study runner.

Reproduces Table 4 of arXiv 2506.02244v2 by training with each motion loss
component removed in turn:

    Full Loss       : L_trans + L_rot + L_scale (adaptive weighted)
    w/o Translation : L_rot + L_scale only
    w/o Rotation    : L_trans + L_scale only
    w/o Scaling     : L_trans + L_rot only
    Flow Only       : ℒ_flow baseline (Section 4.2)

Usage::

    python scripts/ablation.py \\
        --config configs/config.yaml \\
        --backbone open_sora \\
        --data_root /path/to/openvid1m \\
        --output_dir outputs/ablation \\
        --ablation_mode full

    # Run all modes sequentially
    for mode in full no_trans no_rot no_scale flow_only; do
        python scripts/ablation.py --config configs/config.yaml \\
            --backbone open_sora --data_root /data --output_dir outputs/ablation \\
            --ablation_mode $mode
    done
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.physics_motion_loss.losses.adaptive_composite import AdaptiveMotionLoss
from src.physics_motion_loss.losses.physics_motion_loss import PhysicsMotionLoss
from src.physics_motion_loss.spectral.fft_utils import SpectralProcessor
from src.physics_motion_loss.losses.translation_loss import TranslationalMotionLoss
from src.physics_motion_loss.losses.rotation_loss import RotationalMotionLoss
from src.physics_motion_loss.losses.scaling_loss import ScalingMotionLoss
from src.physics_motion_loss.utils.config import load_config
from src.physics_motion_loss.utils.precision import FP32Context

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

ABLATION_MODES = ["full", "no_trans", "no_rot", "no_scale", "flow_only"]


# ---------------------------------------------------------------------------
# Ablated loss variants
# ---------------------------------------------------------------------------

class AblatedPhysicsLoss(nn.Module):
    """PhysicsMotionLoss with one branch disabled, matching Table 4.

    Args:
        base_loss: Fully constructed PhysicsMotionLoss instance.
        mode:      Which component to drop.
    """

    def __init__(self, base_loss: PhysicsMotionLoss, mode: str) -> None:
        super().__init__()
        assert mode in ABLATION_MODES, f"Unknown mode: {mode}"
        self.base = base_loss
        self.mode = mode
        self._fp32 = FP32Context()

    def forward(self, x0_hat: Tensor) -> Dict[str, Tensor]:
        B, C, T, H, W = x0_hat.shape

        L_trans_acc = torch.zeros(1, device=x0_hat.device)
        L_rot_acc   = torch.zeros(1, device=x0_hat.device)
        L_scale_acc = torch.zeros(1, device=x0_hat.device)
        count = 0

        with self._fp32:
            x_fp32 = x0_hat.float()
            for b in range(B):
                for c in range(C):
                    video = x_fp32[b, c]
                    spectrum  = self.base.spectral.compute_spectrum(video)
                    lp_cube   = self.base.spectral.apply_lowpass_cube(spectrum)
                    ring_E    = self.base.spectral.get_ring_energies(lp_cube)
                    polar_seq = self.base.spectral.to_polar_sequence(lp_cube)

                    L_t = self.base.trans_loss(lp_cube)
                    L_r = self.base.rot_loss(polar_seq, ring_E)
                    L_s = self.base.scale_loss(ring_E)

                    # Zero out disabled branch
                    if self.mode == "no_trans":
                        L_t = torch.zeros_like(L_t)
                    elif self.mode == "no_rot":
                        L_r = torch.zeros_like(L_r)
                    elif self.mode == "no_scale":
                        L_s = torch.zeros_like(L_s)

                    L_trans_acc = L_trans_acc + L_t
                    L_rot_acc   = L_rot_acc   + L_r
                    L_scale_acc = L_scale_acc + L_s
                    count += 1

        n = max(count, 1)
        L_trans = L_trans_acc / n
        L_rot   = L_rot_acc   / n
        L_scale = L_scale_acc / n

        if self.mode == "no_trans":
            # Only rotate + scale contribute; re-weight between the two
            losses = torch.stack([L_rot.squeeze(), L_scale.squeeze()])
            weights = torch.softmax(-losses / self.base.adaptive.tau, dim=0)
            L_motion = (weights * losses).sum()
            w_trans, w_rot, w_scale = (
                torch.zeros(1), weights[0], weights[1]
            )
        elif self.mode == "no_rot":
            losses = torch.stack([L_trans.squeeze(), L_scale.squeeze()])
            weights = torch.softmax(-losses / self.base.adaptive.tau, dim=0)
            L_motion = (weights * losses).sum()
            w_trans, w_rot, w_scale = (
                weights[0], torch.zeros(1), weights[1]
            )
        elif self.mode == "no_scale":
            losses = torch.stack([L_trans.squeeze(), L_rot.squeeze()])
            weights = torch.softmax(-losses / self.base.adaptive.tau, dim=0)
            L_motion = (weights * losses).sum()
            w_trans, w_rot, w_scale = (
                weights[0], weights[1], torch.zeros(1)
            )
        else:  # full
            L_motion, weights = self.base.adaptive(
                L_trans.squeeze(), L_rot.squeeze(), L_scale.squeeze()
            )
            w_trans, w_rot, w_scale = weights[0], weights[1], weights[2]

        return {
            "loss":    L_motion,
            "L_trans": L_trans.squeeze(),
            "L_rot":   L_rot.squeeze(),
            "L_scale": L_scale.squeeze(),
            "w_trans": w_trans,
            "w_rot":   w_rot,
            "w_scale": w_scale,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablation study — Physics-Guided Motion Loss (Table 4)"
    )
    parser.add_argument("--config",       type=str, required=True)
    parser.add_argument("--backbone",     type=str, required=True,
                        choices=["open_sora", "mvdit", "hunyuan"])
    parser.add_argument("--data_root",    type=str, required=True)
    parser.add_argument("--output_dir",   type=str, required=True)
    parser.add_argument("--ablation_mode", type=str, required=True,
                        choices=ABLATION_MODES)
    parser.add_argument("--seed",         type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg.training.backbone  = args.backbone
    cfg.data.data_root     = args.data_root
    cfg.training.seed      = args.seed

    out_dir = Path(args.output_dir) / args.ablation_mode
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Ablation mode: {args.ablation_mode}")
    logger.info(f"Output dir:    {out_dir}")

    # Build base loss and wrap with ablation
    base_loss = PhysicsMotionLoss.from_config(cfg)

    if args.ablation_mode == "flow_only":
        logger.info(
            "flow_only mode: use FlowConsistencyLoss instead of PhysicsMotionLoss.\n"
            "Set evaluation.raft_checkpoint in config and use losses/flow_baseline.py."
        )
    else:
        ablated = AblatedPhysicsLoss(base_loss, args.ablation_mode)
        logger.info(f"Ablated loss constructed: {ablated}")
        # TODO: plug `ablated` into PhysicsGuidedTrainer in place of base loss
        # This requires a small trainer patch to accept a custom loss module.
        logger.info(
            "Plug `ablated` into PhysicsGuidedTrainer by passing it as the "
            "`physics_loss` argument (trainer.physics_loss = ablated), then call "
            "trainer.train()."
        )

    logger.info("Ablation setup complete.")


if __name__ == "__main__":
    main()
