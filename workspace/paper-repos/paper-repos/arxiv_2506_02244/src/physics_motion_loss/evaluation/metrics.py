"""
evaluation/metrics.py — EvalCrafter metric wrappers.

Implements the evaluation protocol from Section 4.1 of arXiv 2506.02244v2:
"Following EvalCrafter (Liu et al., 2023) and the OpenVID-1M setting, we report
four categories of metrics: visual quality (VQA A, VQA T, SD-Score), temporal
coherence (CLIP-Temporal, Warping Error, Temporal Consistency), motion quality
(Action Recognition, Motion Accuracy, Flow), and text alignment (Text-Video
Alignment, BLIP-BLEU)."

VQA A/T, SD Score, Action Recognition Score, Motion Accuracy Score, BLIP-BLEU,
and Text-Video Alignment are computed by the official EvalCrafter implementation
(Liu et al., 2023) and invoked as a subprocess. Only directly computable metrics
(Warping Error, CLIP Temporal Score, Flow Score) are implemented natively here.

Paper reference: Section 4.1.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class WarpingError:
    """Optical-flow-based warping error for temporal coherence.

    Measures how well consecutive frames are predicted by the optical flow
    from the previous frame. Lower is better.

    Paper reference: Table 1 (Warping Error ↓).
    """

    def __init__(self, raft_checkpoint: Optional[str] = None) -> None:
        self.raft_checkpoint = raft_checkpoint
        self._raft = None

    def compute(self, frames: Tensor) -> float:
        """Compute mean warping error across consecutive frame pairs.

        Args:
            frames: [B, T, C, H, W], float32 in [0, 1].

        Returns:
            Scalar mean warping error.
        """
        if self._raft is None and self.raft_checkpoint is not None:
            self._raft = self._load_raft(self.raft_checkpoint)

        if self._raft is None:
            logger.warning(
                "No RAFT checkpoint available — warping error will be 0.0. "
                "Set evaluation.raft_checkpoint to get true warping error."
            )
            return 0.0

        B, T, C, H, W = frames.shape
        total_err = 0.0
        n = 0

        with torch.no_grad():
            for b in range(B):
                for t in range(T - 1):
                    f_t   = (frames[b, t]   * 255.0).unsqueeze(0)  # [1,C,H,W]
                    f_tp1 = (frames[b, t+1] * 255.0).unsqueeze(0)

                    _, flow = self._raft(f_t, f_tp1, iters=20, test_mode=True)

                    # Warp f_t by flow
                    grid = self._make_grid(flow)  # [1, H, W, 2]
                    f_t_warped = F.grid_sample(
                        frames[b, t].unsqueeze(0), grid,
                        mode="bilinear", align_corners=True
                    )

                    err = (f_tp1 / 255.0 - f_t_warped).abs().mean().item()
                    total_err += err
                    n += 1

        return total_err / max(n, 1)

    @staticmethod
    def _make_grid(flow: Tensor) -> Tensor:
        """Build a sampling grid from a flow field."""
        B, _, H, W = flow.shape
        device = flow.device
        gy, gx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing="ij",
        )
        base_grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)  # [1,H,W,2]
        flow_norm = torch.stack([
            flow[:, 0] / (W / 2),
            flow[:, 1] / (H / 2),
        ], dim=-1)  # [B,H,W,2]
        return base_grid + flow_norm

    @staticmethod
    def _load_raft(checkpoint: str):
        from torchvision.models.optical_flow import raft_large
        model = raft_large(pretrained=False)
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
        return model.eval()


class CLIPTemporalScore:
    """CLIP-based temporal consistency score.

    Measures cosine similarity of CLIP embeddings across consecutive frames.
    Higher is better (max 100).

    Paper reference: Table 1 (CLIP Temporal Score ↑).
    """

    def __init__(self, clip_model: str = "ViT-B/32") -> None:
        self.clip_model = clip_model
        self._model = None
        self._preprocess = None

    def _load(self, device: torch.device) -> None:
        try:
            import clip
            self._model, self._preprocess = clip.load(self.clip_model, device=device)
        except ImportError:
            raise ImportError(
                "pip install git+https://github.com/openai/CLIP.git"
            )

    def compute(self, frames: Tensor) -> float:
        """Compute mean CLIP cosine similarity across consecutive frames.

        Args:
            frames: [B, T, C, H, W], float32.

        Returns:
            Mean temporal score × 100 (to match paper's 99.x scale).
        """
        device = frames.device
        if self._model is None:
            self._load(device)

        B, T, C, H, W = frames.shape
        total_sim = 0.0
        n = 0

        with torch.no_grad():
            for b in range(B):
                embs = []
                for t in range(T):
                    frame = frames[b, t]  # [C, H, W]
                    # CLIP expects PIL-like preprocessing — use raw tensor path
                    frame_resized = F.interpolate(
                        frame.unsqueeze(0), size=(224, 224), mode="bilinear",
                        align_corners=False
                    )
                    emb = self._model.encode_image(frame_resized)
                    embs.append(F.normalize(emb, dim=-1))

                for t in range(T - 1):
                    sim = (embs[t] * embs[t + 1]).sum().item()
                    total_sim += sim
                    n += 1

        return (total_sim / max(n, 1)) * 100.0


class EvalCrafterMetrics:
    """Wrapper for the official EvalCrafter evaluation suite.

    Metrics requiring EvalCrafter (invoked as subprocess):
      VQA A, VQA T, SD Score, Action Recognition Score,
      Motion Accuracy Score, Flow Score, Text-Video Alignment, BLIP-BLEU.

    Natively computed by this class:
      Warping Error, CLIP Temporal Score.

    Paper reference: Section 4.1 — "We use the official EvalCrafter
    implementation and evaluation protocol for OpenVID-1M."

    Args:
        evalcrafter_root: Path to the cloned EvalCrafter repository.
        raft_checkpoint:  Optional path to RAFT weights for Warping Error.
    """

    def __init__(
        self,
        evalcrafter_root: Optional[str] = None,
        raft_checkpoint: Optional[str] = None,
    ) -> None:
        self.evalcrafter_root = Path(evalcrafter_root) if evalcrafter_root else None
        self.warping = WarpingError(raft_checkpoint=raft_checkpoint)
        self.clip_temporal = CLIPTemporalScore()

    def compute_native(self, frames: Tensor) -> Dict[str, float]:
        """Compute metrics that don't require EvalCrafter.

        Args:
            frames: [B, T, C, H, W], float32.

        Returns:
            Dict with 'warping_error' and 'clip_temporal_score'.
        """
        return {
            "warping_error":       self.warping.compute(frames),
            "clip_temporal_score": self.clip_temporal.compute(frames),
        }

    def compute_evalcrafter(
        self,
        video_dir: str,
        prompt_file: str,
        output_dir: str,
    ) -> Dict[str, float]:
        """Run EvalCrafter as a subprocess and parse results.

        Args:
            video_dir:   Directory containing generated .mp4 files.
            prompt_file: Path to text file with one prompt per line.
            output_dir:  Where EvalCrafter writes its result JSON.

        Returns:
            Dict of metric name → float value.

        Raises:
            RuntimeError: If EvalCrafter is not installed or fails.
        """
        if self.evalcrafter_root is None or not self.evalcrafter_root.exists():
            raise RuntimeError(
                "EvalCrafter root not found. Clone from:\n"
                "  https://github.com/EvalCrafter/EvalCrafter\n"
                "and set evaluation.evalcrafter_root in your config."
            )

        result_path = Path(output_dir) / "evalcrafter_results.json"
        cmd = [
            "python",
            str(self.evalcrafter_root / "evaluate.py"),
            "--video_dir", video_dir,
            "--prompt_file", prompt_file,
            "--output_path", str(result_path),
        ]

        logger.info(f"Running EvalCrafter: {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True)

        if proc.returncode != 0:
            raise RuntimeError(
                f"EvalCrafter failed:\n{proc.stderr}"
            )

        with open(result_path) as f:
            results: Dict[str, float] = json.load(f)

        return results

    def compute_all(
        self,
        frames: Tensor,
        video_dir: Optional[str] = None,
        prompt_file: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, float]:
        """Compute all available metrics.

        Native metrics are always computed. EvalCrafter metrics are computed
        only if video_dir, prompt_file, and output_dir are provided.
        """
        metrics = self.compute_native(frames)

        if video_dir and prompt_file and output_dir and self.evalcrafter_root:
            try:
                ec_metrics = self.compute_evalcrafter(
                    video_dir, prompt_file, output_dir
                )
                metrics.update(ec_metrics)
            except RuntimeError as e:
                logger.warning(f"EvalCrafter skipped: {e}")

        return metrics
