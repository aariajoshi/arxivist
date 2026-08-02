"""
train.py — Main fine-tuning entrypoint for Physics-Guided Motion Loss.

Paper reference: arXiv 2506.02244v2, Section 4.1.

Example usage::

    # Full training (Open-Sora backbone, 4× A100)
    python train.py \\
        --config configs/config.yaml \\
        --backbone open_sora \\
        --data_root /path/to/openvid1m \\
        --output_dir outputs/open_sora_physics

    # Hunyuan with LoRA
    python train.py \\
        --config configs/config.yaml \\
        --backbone hunyuan \\
        --use_lora \\
        --data_root /path/to/openvid1m \\
        --output_dir outputs/hunyuan_lora

    # Quick debug run (small data, few steps)
    python train.py --config configs/config.yaml --backbone open_sora \\
        --data_root /path/to/openvid1m --output_dir outputs/debug --debug

    # Dry run (validates setup without training)
    python train.py --config configs/config.yaml --backbone open_sora \\
        --data_root /path/to/openvid1m --output_dir outputs/dry --dry_run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# ── project imports ──────────────────────────────────────────────────────────
from src.physics_motion_loss.data.dataset import OpenVIDDataset
from src.physics_motion_loss.training.lora_wrapper import LoRAWrapper
from src.physics_motion_loss.training.trainer import PhysicsGuidedTrainer
from src.physics_motion_loss.utils.config import load_config, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backbone loading stubs
# ---------------------------------------------------------------------------

def load_backbone(backbone_name: str, device: torch.device):
    """Load a pretrained video diffusion backbone.

    Each backbone has its own loading API. This function dispatches to the
    appropriate loader. Extend this function when adding new backbones.

    Paper reference: Section 4.1 — Open-Sora, MVDIT, Hunyuan.

    STUB: The exact model loading APIs depend on the backbone repositories.
    Install them separately and adapt these loaders as needed.
    """
    if backbone_name == "open_sora":
        return _load_open_sora(device)
    elif backbone_name == "mvdit":
        return _load_mvdit(device)
    elif backbone_name == "hunyuan":
        return _load_hunyuan(device)
    else:
        raise ValueError(
            f"Unknown backbone '{backbone_name}'. "
            "Choose from: open_sora, mvdit, hunyuan"
        )


def _load_open_sora(device):
    """
    STUB: Load Open-Sora backbone.
    Install: https://github.com/hpcaitech/Open-Sora
    Replace this stub with the actual Open-Sora loading code.
    """
    try:
        # Placeholder — replace with actual Open-Sora import
        raise NotImplementedError(
            "Open-Sora loader stub. Install Open-Sora and implement this loader.\n"
            "See: https://github.com/hpcaitech/Open-Sora"
        )
    except ImportError:
        raise ImportError(
            "Open-Sora not installed. See https://github.com/hpcaitech/Open-Sora"
        )


def _load_mvdit(device):
    """STUB: Load MVDIT backbone. See OpenVID-1M repository."""
    raise NotImplementedError(
        "MVDIT loader stub. Install MVDIT and implement this loader.\n"
        "Reference: Nan et al. (2024), arXiv:2407.02371"
    )


def _load_hunyuan(device):
    """STUB: Load Hunyuan backbone."""
    raise NotImplementedError(
        "Hunyuan loader stub. Install HunyuanVideo and implement this loader.\n"
        "See: https://github.com/Tencent/HunyuanVideo"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Physics-Guided Motion Loss — Video Diffusion Fine-tuning"
    )
    parser.add_argument("--config",      type=str, required=True,
                        help="Path to config YAML (configs/config.yaml)")
    parser.add_argument("--backbone",    type=str, required=True,
                        choices=["open_sora", "mvdit", "hunyuan"],
                        help="Backbone model to fine-tune")
    parser.add_argument("--data_root",   type=str, required=True,
                        help="Path to OpenVID-1M root directory")
    parser.add_argument("--output_dir",  type=str, required=True,
                        help="Directory for checkpoints and logs")
    parser.add_argument("--resume",      type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--seed",        type=int, default=None,
                        help="Random seed override (default: from config)")
    parser.add_argument("--use_lora",    action="store_true",
                        help="Enable LoRA fine-tuning (for Hunyuan)")
    parser.add_argument("--lora_rank",   type=int, default=None,
                        help="LoRA rank override (ASSUMED: 16 if not set)")
    parser.add_argument("--physics_weight", type=float, default=None,
                        help="Physics loss mixing weight override")
    parser.add_argument("--epochs",      type=int, default=None,
                        help="Number of training epochs override")
    parser.add_argument("--lr",          type=float, default=None,
                        help="Learning rate override")
    parser.add_argument("--num_gpus",    type=int, default=None,
                        help="Number of GPUs override")
    parser.add_argument("--debug",       action="store_true",
                        help="Debug mode: small dataset, few steps")
    parser.add_argument("--dry_run",     action="store_true",
                        help="Validate setup without training")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # --- Load and patch config ---
    cfg = load_config(args.config)

    # CLI overrides
    cfg.training.backbone = args.backbone
    cfg.data.data_root    = args.data_root
    if args.seed is not None:
        cfg.training.seed = args.seed
    if args.lora_rank is not None:
        cfg.lora.rank = args.lora_rank
    if args.physics_weight is not None:
        cfg.losses.physics_loss_weight = args.physics_weight
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    if args.lr is not None:
        cfg.training.learning_rate = args.lr
    if args.num_gpus is not None:
        cfg.hardware.num_gpus = args.num_gpus
    if args.use_lora:
        cfg.lora.enabled = True
    if args.debug:
        cfg.data.max_samples = 64
        cfg.training.epochs  = 1
        cfg.training.log_every_n_steps  = 5
        cfg.training.save_every_n_steps = 20
        logger.info("DEBUG MODE: max_samples=64, epochs=1")

    set_seed(cfg.training.seed, cfg.training.deterministic)

    # --- Save resolved config ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "resolved_config.yaml")

    # --- Dataset & DataLoader ---
    logger.info("Building dataset...")
    train_ds = OpenVIDDataset(
        data_root=cfg.data.data_root,
        split="train",
        T=cfg.data.T,
        H=cfg.data.H,
        W=cfg.data.W,
        max_samples=cfg.data.get("max_samples", None),
    )
    val_ds = OpenVIDDataset(
        data_root=cfg.data.data_root,
        split="val",
        T=cfg.data.T,
        H=cfg.data.H,
        W=cfg.data.W,
        max_samples=cfg.data.get("max_samples", None),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=4,   # ASSUMED: not specified in paper; adjust per GPU memory
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=4,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
    )

    # --- Backbone ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading backbone: {cfg.training.backbone}")

    if args.dry_run:
        logger.info("DRY RUN: backbone loading skipped. Setup validated.")
        logger.info(f"  Train samples: {len(train_ds)}")
        logger.info(f"  Val samples:   {len(val_ds)}")
        logger.info(f"  Output dir:    {output_dir}")
        logger.info("DRY RUN COMPLETE — no training performed.")
        return

    model = load_backbone(cfg.training.backbone, device)

    # --- LoRA wrapping ---
    if cfg.lora.enabled:
        logger.info("Applying LoRA adapters...")
        lora = LoRAWrapper(
            rank=cfg.lora.rank,
            alpha=cfg.lora.alpha,
            target_modules=list(cfg.lora.target_modules),
            dropout=cfg.lora.dropout,
        )
        model = lora.wrap(model)

    # --- Multi-GPU ---
    if cfg.hardware.num_gpus > 1 and torch.cuda.device_count() > 1:
        logger.info(f"Using {cfg.hardware.num_gpus} GPUs via DataParallel")
        model = torch.nn.DataParallel(model)

    # --- Trainer ---
    trainer = PhysicsGuidedTrainer(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=str(output_dir),
    )

    if args.resume:
        logger.info(f"Resuming from: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # --- Train ---
    trainer.train()


if __name__ == "__main__":
    main()
