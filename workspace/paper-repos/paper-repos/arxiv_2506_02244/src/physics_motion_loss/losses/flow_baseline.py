"""
losses/flow_baseline.py — Flow-based temporal consistency loss (ablation baseline).

Implements the strong baseline from Section 4.2 (Eq. 4.3) of arXiv 2506.02244v2:

    L_flow = ‖M ⊙ (x_{t+1} − W(x_t, F_{t→t+1}))‖₁
           + ‖M'⊙ (x_t   − W(x_{t+1}, F_{t+1→t}))‖₁
           + λ · L_smooth(F)

Where:
  - F_{t→t+1}, F_{t+1→t}: forward/backward optical flow estimated by RAFT
  - W: bilinear warping operator
  - M, M': forward/backward visibility masks
  - L_smooth: flow smoothness regulariser (TV on flow field)

Note: RAFT is used at inference only — no gradient flows through the flow
estimator (Sec 4.2: "RAFT inference only"). Requires a pretrained RAFT checkpoint.

This module is OPTIONAL — only needed for reproducing the ablation study.
Enable via config: evaluation.flow_baseline_enabled=True
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FlowConsistencyLoss(nn.Module):
    """RAFT-based temporal consistency regulariser.

    Paper reference: Section 4.2, Equation 4.3.

    STUB NOTE: This implementation requires a RAFT optical flow model.
    Provide a path to RAFT weights via config: evaluation.raft_checkpoint
    If no checkpoint is available, this module raises NotImplementedError.

    Args:
        raft_checkpoint: Path to RAFT model weights (.pth). If None, raises
                         NotImplementedError on forward().
        smoothness_lambda: Weight for flow smoothness term (λ in Eq. 4.3).
                           Not specified in paper — ASSUMED 0.1 (conf < 0.6).
        epsilon_charb: Epsilon for Charbonnier loss robustification (Sec 4.2).
                       Not specified in paper — ASSUMED 1e-3 (conf < 0.6).
    """

    def __init__(
        self,
        raft_checkpoint: Optional[str] = None,
        smoothness_lambda: float = 0.1,   # ASSUMED — not stated in paper (conf < 0.6)
        epsilon_charb: float = 1e-3,       # ASSUMED — not stated in paper (conf < 0.6)
    ) -> None:
        super().__init__()
        self.smoothness_lambda = smoothness_lambda
        self.epsilon_charb = epsilon_charb
        self._raft = None

        if raft_checkpoint is not None:
            self._raft = self._load_raft(raft_checkpoint)

    def forward(self, frames: Tensor) -> Tensor:
        """Compute flow-based temporal consistency loss.

        Args:
            frames: Video tensor [B, T, C, H, W], float32.

        Returns:
            Scalar L_flow.

        Raises:
            NotImplementedError: If no RAFT checkpoint was provided.
        """
        if self._raft is None:
            raise NotImplementedError(
                "FlowConsistencyLoss requires a RAFT checkpoint. "
                "Set evaluation.raft_checkpoint in your config. "
                "See: https://github.com/princeton-vl/RAFT"
            )

        assert frames.dim() == 5, f"Expected [B, T, C, H, W], got {frames.shape}"
        B, T, C, H, W = frames.shape

        total_loss = torch.tensor(0.0, device=frames.device)
        n_pairs = 0

        for t in range(T - 1):
            x_t   = frames[:, t]      # [B, C, H, W]
            x_tp1 = frames[:, t + 1]  # [B, C, H, W]

            # RAFT inference only — no gradient through flow estimation
            with torch.no_grad():
                F_fwd = self._estimate_flow(x_t, x_tp1)    # [B, 2, H, W]
                F_bwd = self._estimate_flow(x_tp1, x_t)    # [B, 2, H, W]

            # Bilinear warping
            x_t_warped   = self._warp(x_t,   F_fwd)  # [B, C, H, W]
            x_tp1_warped = self._warp(x_tp1, F_bwd)  # [B, C, H, W]

            # Visibility masks (forward/backward consistency check)
            M_fwd = self._visibility_mask(x_tp1, x_t_warped)    # [B, 1, H, W]
            M_bwd = self._visibility_mask(x_t,   x_tp1_warped)  # [B, 1, H, W]

            # Eq. 4.3: masked Charbonnier reprojection error
            err_fwd = self._charbonnier(M_fwd * (x_tp1 - x_t_warped))
            err_bwd = self._charbonnier(M_bwd * (x_t   - x_tp1_warped))

            # Flow smoothness: total variation on both flow fields
            smooth = self._flow_smoothness(F_fwd) + self._flow_smoothness(F_bwd)

            # Eq. 4.3: L_flow per pair
            pair_loss = err_fwd + err_bwd + self.smoothness_lambda * smooth
            total_loss = total_loss + pair_loss.mean()
            n_pairs += 1

        return total_loss / max(n_pairs, 1)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _estimate_flow(self, frame_a: Tensor, frame_b: Tensor) -> Tensor:
        """Estimate optical flow from frame_a to frame_b using RAFT."""
        # RAFT expects images in [0, 255] range
        a = (frame_a * 255.0).clamp(0, 255)
        b = (frame_b * 255.0).clamp(0, 255)
        _, flow = self._raft(a, b, iters=20, test_mode=True)
        return flow  # [B, 2, H, W]

    @staticmethod
    def _warp(x: Tensor, flow: Tensor) -> Tensor:
        """Bilinear warp x by a 2-channel flow field."""
        B, C, H, W = x.shape
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=x.device, dtype=torch.float32),
            torch.arange(W, device=x.device, dtype=torch.float32),
            indexing="ij",
        )
        # Add flow offsets
        new_x = grid_x.unsqueeze(0) + flow[:, 0]  # [B, H, W]
        new_y = grid_y.unsqueeze(0) + flow[:, 1]  # [B, H, W]

        # Normalise to [-1, 1]
        new_x = 2.0 * new_x / (W - 1) - 1.0
        new_y = 2.0 * new_y / (H - 1) - 1.0

        grid = torch.stack([new_x, new_y], dim=-1)  # [B, H, W, 2]
        return F.grid_sample(x, grid, mode="bilinear", align_corners=True)

    @staticmethod
    def _visibility_mask(target: Tensor, warped: Tensor) -> Tensor:
        """Simple photometric visibility mask: 1 where warping error is small."""
        err = (target - warped).abs().mean(dim=1, keepdim=True)  # [B, 1, H, W]
        return (err < 0.1).float()

    def _charbonnier(self, x: Tensor) -> Tensor:
        """Charbonnier/pseudo-Huber loss: √(x² + ε²). Sec 4.2."""
        return torch.sqrt(x ** 2 + self.epsilon_charb ** 2).mean()

    @staticmethod
    def _flow_smoothness(flow: Tensor) -> Tensor:
        """Total-variation smoothness regulariser on the flow field."""
        dx = (flow[:, :, :, 1:] - flow[:, :, :, :-1]).abs().mean()
        dy = (flow[:, :, 1:, :] - flow[:, :, :-1, :]).abs().mean()
        return dx + dy

    @staticmethod
    def _load_raft(checkpoint_path: str):
        """Load pretrained RAFT model. Requires RAFT installed in environment."""
        try:
            from torchvision.models.optical_flow import raft_large
            model = raft_large(pretrained=False)
            state = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(state)
            model.eval()
            return model
        except ImportError:
            raise ImportError(
                "RAFT requires torchvision >= 0.16. "
                "Install with: pip install torchvision>=0.16.0"
            )

    def __repr__(self) -> str:
        return (
            f"FlowConsistencyLoss(smoothness_lambda={self.smoothness_lambda}, "
            f"raft_loaded={self._raft is not None})"
        )
