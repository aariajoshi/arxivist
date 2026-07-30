#!/usr/bin/env python
"""
train.py

Trains the base joint scene+language NeRF F_Theta (Eq. 1) on a single (synthetic) multi-view
scan, mirroring Sec. II: "The network F_Theta is trained on a set of images I ... captured by
the robot in a single scan, together with the corresponding set of camera poses Omega."

Example:
    python train.py --config configs/config.yaml --scene-seed 0 --out-dir runs/base_nerf
    python train.py --config configs/config.yaml --debug --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from neo_nerf_editing.data.rays import fixed_scan_trajectory
from neo_nerf_editing.data.synthetic_scene import ToySceneGenerator
from neo_nerf_editing.models.nerf_field import JointNeRFField
from neo_nerf_editing.training.trainer import NeRFTrainer
from neo_nerf_editing.utils.config import NEOConfig, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the base NEO joint scene+language NeRF (F_Theta).")
    p.add_argument("--config", type=str, default="configs/config.yaml", help="Path to YAML config")
    p.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from")
    p.add_argument("--seed", type=int, default=None, help="Override config seed")
    p.add_argument("--scene-seed", type=int, default=0, help="Seed for synthetic scene generation")
    p.add_argument("--out-dir", type=str, default="runs/base_nerf", help="Output directory")
    p.add_argument("--debug", action="store_true", help="Reduce dataset/steps for a quick local smoke test")
    p.add_argument("--dry-run", action="store_true", help="Build all components but do not train")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = NEOConfig.from_yaml(args.config)
    seed = args.seed if args.seed is not None else cfg.seed
    set_seed(seed, cfg.deterministic)
    device = cfg.device()
    os.makedirs(args.out_dir, exist_ok=True)

    n_views = cfg.data["n_train_views"] if not args.debug else 4
    n_steps = cfg.training["n_steps_base_nerf"] if not args.debug else 20
    # NOTE: CPU forward-pass cost for the full MLP (hidden_dim=128, 6 layers) measured at
    # ~6ms/ray for a full 192-sample render on this sandbox's single-core CPU; --debug shrinks
    # image resolution too (not just view count/steps) or a "quick" smoke test would still take
    # many minutes. Real (non-debug) runs use the full configs/config.yaml resolution.
    image_h = cfg.data["image_h"] if not args.debug else 32
    image_w = cfg.data["image_w"] if not args.debug else 32
    n_samples_uniform = cfg.training["n_samples_uniform"] if not args.debug else 32

    print("=" * 60)
    print("NEO base-NeRF training")
    print(f"  device            : {device}")
    print(f"  scene seed        : {args.scene_seed}")
    print(f"  n_train_views     : {n_views}")
    print(f"  image size        : {image_h}x{image_w}")
    print(f"  n_steps           : {n_steps}")
    print("=" * 60)

    scene_gen = ToySceneGenerator()
    scene = scene_gen.generate_scene(seed=args.scene_seed, objects=["soup_can", "lego_brick"])
    trajectory = fixed_scan_trajectory(n_views, radius=4.0, height=2.0)
    dataset = scene_gen.render_views(
        scene, trajectory, H=image_h, W=image_w, near=cfg.data["near"], far=cfg.data["far"]
    )
    dataset = {"rays_o": dataset["rays_o"].to(device), "rays_d": dataset["rays_d"].to(device), "rgb": dataset["rgb_flat"].to(device)}

    field = JointNeRFField(
        d_l=cfg.model["d_l"],
        hidden_dim=cfg.model["mlp_hidden_dim"],
        num_layers=cfg.model["mlp_num_layers"],
        posenc_num_freqs=cfg.model["posenc_num_freqs"],
        posenc_include_input=cfg.model["posenc_include_input"],
        skip_layers=cfg.model["mlp_skip_layers"],
    ).to(device)

    if args.resume:
        field.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed weights from {args.resume}")

    n_params = sum(p.numel() for p in field.parameters())
    print(f"Model: {field!r}")
    print(f"Dataset size (rays): {dataset['rays_o'].shape[0]}")
    print(f"Steps per 'epoch' (dataset_size / batch_size): {dataset['rays_o'].shape[0] // cfg.training['batch_size_rays']}")

    if args.dry_run:
        print("--dry-run set: components built successfully, skipping optimization.")
        return

    trainer = NeRFTrainer(lr=cfg.training["learning_rate"], log_every=cfg.logging["log_every_n_steps"])

    def log_fn(step: int, loss: float) -> None:
        print(f"  step {step:5d} | loss {loss:.6f}")

    history = trainer.fit(
        field,
        dataset,
        n_steps=n_steps,
        near=cfg.data["near"],
        far=cfg.data["far"],
        n_samples=n_samples_uniform,
        batch_size=cfg.training["batch_size_rays"],
        device=device,
        log_fn=log_fn,
    )

    ckpt_path = os.path.join(args.out_dir, "base_nerf.pt")
    torch.save(field.state_dict(), ckpt_path)
    with open(os.path.join(args.out_dir, "train_history.json"), "w") as f:
        json.dump({"loss": history["loss"], "elapsed_sec": history["elapsed_sec"]}, f, indent=2)

    print(f"Saved checkpoint to {ckpt_path}")
    print(f"Final loss: {history['loss'][-1]:.6f}  |  elapsed: {history['elapsed_sec'][0]:.1f}s")


if __name__ == "__main__":
    main()
