"""
scripts/visualize_spectrum.py — Debug visualisation of spectral pipeline.

Loads a video, runs it through the SpectralProcessor, and plots:
  1. 3D FFT low-pass cube (spatial slice at t=0)
  2. Ring energies Ek(t) over time (heatmap)
  3. Polar resampling (Nr × M) at t=0
  4. Radial gradient alignment fields (C_flow debug)
  5. Per-clip scalar losses {L_trans, L_rot, L_scale, L_motion}

Usage::

    python scripts/visualize_spectrum.py \\
        --video /path/to/video.mp4 \\
        --output_dir outputs/spectral_debug \\
        --config configs/config.yaml

    # Without a real video: generate a synthetic SIM(2) test clip
    python scripts/visualize_spectrum.py --synthetic --output_dir outputs/spectral_debug
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
logger = logging.getLogger(__name__)


def make_synthetic_video(
    motion: str = "translation",
    T: int = 16, H: int = 64, W: int = 64,
) -> torch.Tensor:
    """Generate a synthetic SIM(2) video clip for unit testing.

    Args:
        motion: One of 'translation', 'rotation', 'scaling'.
        T, H, W: Clip dimensions.

    Returns:
        Float32 tensor [T, H, W] in [0, 1].
    """
    frames = []
    # Base pattern: a bright Gaussian blob
    cy, cx = H // 2, W // 2
    sigma = min(H, W) / 8.0
    yy, xx = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )
    blob = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))

    for t in range(T):
        if motion == "translation":
            shift = int(t * W / (T * 2))
            frame = torch.roll(blob, shifts=shift, dims=1)
        elif motion == "rotation":
            angle = t * (360.0 / T)
            import torchvision.transforms.functional as TF
            pil_like = blob.unsqueeze(0)
            frame = TF.rotate(pil_like, angle=angle).squeeze(0)
        elif motion == "scaling":
            scale = 0.5 + t / T
            s = int(H * scale)
            scaled = torch.nn.functional.interpolate(
                blob.unsqueeze(0).unsqueeze(0),
                size=(max(s, 4), max(s, 4)),
                mode="bilinear", align_corners=False
            ).squeeze()
            # Crop or pad to H×W
            frame = torch.zeros(H, W)
            sh, sw = min(scaled.shape[0], H), min(scaled.shape[1], W)
            oh, ow = (H - sh) // 2, (W - sw) // 2
            frame[oh:oh+sh, ow:ow+sw] = scaled[:sh, :sw]
        else:
            raise ValueError(f"Unknown motion type: {motion}")
        frames.append(frame)

    return torch.stack(frames)  # [T, H, W]


def plot_results(
    video: torch.Tensor,
    loss_dict: dict,
    ring_energies: torch.Tensor,
    polar_frame: torch.Tensor,
    output_dir: Path,
    title: str = "",
) -> None:
    """Generate and save diagnostic plots."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        logger.warning("matplotlib not installed — skipping plots. pip install matplotlib")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    T, H, W = video.shape
    Nr, T_lp = ring_energies.shape

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f"Physics Motion Loss — Spectral Debug{' | ' + title if title else ''}",
                 fontsize=14, fontweight="bold")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # 1. Mid-frame of the video
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(video[T // 2].numpy(), cmap="gray", vmin=0, vmax=1)
    ax1.set_title(f"Video frame t={T//2}")
    ax1.axis("off")

    # 2. Ring energies heatmap [Nr × T_lp]
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(ring_energies.numpy(), aspect="auto",
                    origin="lower", cmap="viridis")
    ax2.set_title("Ring energies Ek(t)")
    ax2.set_xlabel("Time (T_lp)")
    ax2.set_ylabel("Ring k")
    plt.colorbar(im, ax=ax2, fraction=0.046)

    # 3. Polar frame at t=0 (magnitude)
    ax3 = fig.add_subplot(gs[0, 2])
    if polar_frame.is_complex():
        polar_mag = polar_frame.abs().numpy()
    else:
        polar_mag = polar_frame.numpy()
    ax3.imshow(polar_mag, aspect="auto", cmap="inferno", origin="lower")
    ax3.set_title("Polar resampling |V̂(ρ,θ)| at t=0")
    ax3.set_xlabel("Angular bin (θ)")
    ax3.set_ylabel("Ring (ρ)")

    # 4. Radial spectral centroid ρc(t)
    ax4 = fig.add_subplot(gs[1, 0])
    E_sum = ring_energies.sum(dim=0).clamp(min=1e-12)
    k = torch.arange(Nr, dtype=torch.float32)
    rho_c = (k.unsqueeze(1) * ring_energies).sum(dim=0) / E_sum
    ax4.plot(rho_c.numpy(), marker="o", markersize=3, linewidth=1.5, color="steelblue")
    ax4.set_title(f"Radial centroid ρc(t) | S_trend={loss_dict.get('S_trend', '?'):.3f}")
    ax4.set_xlabel("Time step")
    ax4.set_ylabel("ρc")
    ax4.grid(alpha=0.3)

    # 5. Loss bar chart
    ax5 = fig.add_subplot(gs[1, 1])
    loss_names  = ["L_trans", "L_rot", "L_scale", "L_motion"]
    loss_values = [
        loss_dict.get("L_trans",  0.0),
        loss_dict.get("L_rot",    0.0),
        loss_dict.get("L_scale",  0.0),
        loss_dict.get("loss",     0.0),
    ]
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]
    bars = ax5.bar(loss_names, loss_values, color=colors, alpha=0.85, edgecolor="black")
    for bar, val in zip(bars, loss_values):
        ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax5.set_title("Physics loss values")
    ax5.set_ylim(0, 1.1)
    ax5.set_ylabel("Loss value")
    ax5.grid(axis="y", alpha=0.3)

    # 6. Adaptive weights
    ax6 = fig.add_subplot(gs[1, 2])
    weight_names  = ["w_trans", "w_rot", "w_scale"]
    weight_values = [
        loss_dict.get("w_trans", 1/3),
        loss_dict.get("w_rot",   1/3),
        loss_dict.get("w_scale", 1/3),
    ]
    ax6.bar(weight_names, weight_values, color=colors[:3], alpha=0.85, edgecolor="black")
    ax6.set_title(f"Adaptive weights (τ={loss_dict.get('tau', 0.1):.2f})")
    ax6.set_ylim(0, 1.0)
    ax6.set_ylabel("Weight")
    ax6.grid(axis="y", alpha=0.3)

    save_path = output_dir / "spectral_debug.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved diagnostic plot: {save_path}")


def run_pipeline(video: torch.Tensor, cfg) -> dict:
    """Run the full spectral pipeline and return losses + intermediates."""
    from src.physics_motion_loss.losses.physics_motion_loss import PhysicsMotionLoss
    from src.physics_motion_loss.spectral.fft_utils import SpectralProcessor
    from src.physics_motion_loss.utils.precision import FP32Context

    loss_fn = PhysicsMotionLoss.from_config(cfg)
    proc    = SpectralProcessor(
        rho=cfg.spectral.low_pass_rho,
        Nr=cfg.spectral.Nr_rings,
        M=cfg.spectral.M_angular_bins,
    )

    x0_hat = video.unsqueeze(0).unsqueeze(0)  # [1, 1, T, H, W]
    with FP32Context():
        out = loss_fn(x0_hat.float())
        # Intermediates for plotting
        spec   = proc.compute_spectrum(video.float())
        lp     = proc.apply_lowpass_cube(spec)
        ring_E = proc.get_ring_energies(lp)
        polar  = proc.to_polar_sequence(lp)
        polar0 = polar[0]  # [Nr, M] at t=0

    result = {k: v.item() if isinstance(v, torch.Tensor) and v.dim() == 0 else v
              for k, v in out.items()}
    result["tau"] = cfg.losses.softmax_temperature
    return result, ring_E.detach(), polar0.detach()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise spectral pipeline for Physics-Guided Motion Loss"
    )
    parser.add_argument("--video",       type=str, default=None,
                        help="Path to input video .mp4")
    parser.add_argument("--synthetic",   action="store_true",
                        help="Use synthetic SIM(2) clip instead of real video")
    parser.add_argument("--motion",      type=str, default="translation",
                        choices=["translation", "rotation", "scaling"],
                        help="Synthetic motion type (with --synthetic)")
    parser.add_argument("--output_dir",  type=str, required=True)
    parser.add_argument("--config",      type=str, default="configs/config.yaml")
    parser.add_argument("--T",           type=int, default=16)
    parser.add_argument("--H",           type=int, default=64)
    parser.add_argument("--W",           type=int, default=64)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s")
    args = parse_args()
    output_dir = Path(args.output_dir)

    from src.physics_motion_loss.utils.config import load_config
    cfg = load_config(args.config)

    if args.synthetic:
        logger.info(f"Generating synthetic {args.motion} video ({args.T}×{args.H}×{args.W})")
        video = make_synthetic_video(args.motion, args.T, args.H, args.W)
        title = f"Synthetic — {args.motion}"
    elif args.video:
        logger.info(f"Loading video: {args.video}")
        try:
            import torchvision.io as tvio
            from torchvision.transforms.functional import resize
            frames, _, _ = tvio.read_video(args.video, pts_unit="sec",
                                           output_format="TCHW")
            indices = torch.linspace(0, len(frames)-1, args.T).long()
            frames = frames[indices]
            frames = resize(frames, [args.H, args.W])
            video  = frames[:, 0].float() / 255.0  # single channel, [T,H,W]
        except Exception as e:
            logger.error(f"Failed to load video: {e}")
            sys.exit(1)
        title = Path(args.video).stem
    else:
        logger.error("Provide --video or --synthetic")
        sys.exit(1)

    logger.info("Running spectral pipeline...")
    loss_dict, ring_energies, polar_frame = run_pipeline(video, cfg)

    logger.info("Loss values:")
    for k, v in loss_dict.items():
        if isinstance(v, (int, float)):
            logger.info(f"  {k}: {v:.4f}")

    plot_results(video, loss_dict, ring_energies, polar_frame, output_dir, title=title)
    logger.info("Done.")


if __name__ == "__main__":
    main()
