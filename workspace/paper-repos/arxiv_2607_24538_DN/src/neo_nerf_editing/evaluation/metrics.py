"""
evaluation/metrics.py

Standard NeRF quality + geometry metrics used in Sec. IV/V: PSNR, SSIM, depth RMSE, and
photometric reprojection error E_rep [34], each computable over the full image or a masked
(Out/In) region, matching Tables I-III's Full/Out/In columns.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from skimage.metrics import structural_similarity as sk_ssim


class NeRFEvalMetrics:
    """Metric implementations. All methods accept either full-image or masked-region inputs;
    pass `mask` to restrict computation to a region (e.g. the object-removal bounding box's
    projected footprint, per Sec. IV-A: 'metrics are computed only within the object removal
    region ... referred to as Out')."""

    def psnr(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
        """pred, target: [H,W,3] in [0,1]. mask: optional [H,W] bool."""
        assert pred.shape == target.shape, f"{pred.shape} vs {target.shape}"
        diff2 = (pred - target) ** 2
        if mask is not None:
            mask = mask.to(torch.bool)
            if mask.sum() == 0:
                return float("nan")
            mse = diff2[mask].mean().item()
        else:
            mse = diff2.mean().item()
        if mse <= 1e-12:
            return 99.0  # cap to avoid inf on a perfect (degenerate) match
        return float(10.0 * np.log10(1.0 / mse))

    def ssim(self, pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> float:
        """Computes SSIM over the full image (skimage requires a rectangular array); if `mask`
        is given, SSIM is computed on the tight bounding crop of the mask instead (a standard
        practical relaxation of 'masked SSIM', since SSIM's local-window formulation does not
        have a canonical sparse-pixel-set extension)."""
        assert pred.shape == target.shape, f"{pred.shape} vs {target.shape}"
        p = pred.detach().cpu().numpy()
        t = target.detach().cpu().numpy()
        if mask is not None:
            m = mask.detach().cpu().numpy().astype(bool)
            if m.sum() == 0:
                return float("nan")
            ys, xs = np.where(m)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            p, t = p[y0:y1, x0:x1], t[y0:y1, x0:x1]
        min_side = min(p.shape[0], p.shape[1])
        win_size = min(7, min_side if min_side % 2 == 1 else min_side - 1)
        if win_size < 3:
            return float("nan")
        return float(sk_ssim(p, t, channel_axis=-1, data_range=1.0, win_size=win_size))

    def depth_rmse(
        self, pred_depth: torch.Tensor, target_depth: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> float:
        """pred_depth, target_depth: [H,W] in meters."""
        assert pred_depth.shape == target_depth.shape, f"{pred_depth.shape} vs {target_depth.shape}"
        diff2 = (pred_depth - target_depth) ** 2
        if mask is not None:
            mask = mask.to(torch.bool)
            if mask.sum() == 0:
                return float("nan")
            mse = diff2[mask].mean().item()
        else:
            mse = diff2.mean().item()
        return float(np.sqrt(mse))

    def reprojection_error(
        self, pred_points: torch.Tensor, target_points: torch.Tensor, poses: Optional[torch.Tensor] = None
    ) -> float:
        """Photometric/geometric reprojection error E_rep [34]: mean 3D point-to-point distance
        between reconstructed surface points and ground-truth surface points, both expressed in
        the same (ArUco-registered, in the real paper) world frame. `poses` is accepted for
        interface completeness (a full LSD-SLAM-style reprojection would use them to reproject
        into each camera and compare pixel-space error) but is unused in this simplified
        point-to-point 3D form -- ASSUMED simplification, SIR mathematical_spec confidence n/a
        (E_rep's exact formula is not given in the NEO paper itself, only cited to [34])."""
        assert pred_points.shape == target_points.shape, f"{pred_points.shape} vs {target_points.shape}"
        dists = torch.linalg.vector_norm(pred_points - target_points, dim=-1)
        return float(dists.mean().item())

    def __repr__(self) -> str:
        return "NeRFEvalMetrics()"
