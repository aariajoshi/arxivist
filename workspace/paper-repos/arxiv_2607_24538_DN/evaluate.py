#!/usr/bin/env python
"""
evaluate.py

Computes PSNR / SSIM / depth-RMSE / E_rep between an edited NeRF's renders and the
ground-truth post-edit synthetic scan, over Full / Out / In regions (Sec. IV-A, IV-B):
  - "Out": the object's original bounding-box footprint (did we plausibly reconstruct what
    was revealed once the object left?)
  - "In": the object's new bounding-box footprint after relocation (did we plausibly render
    the relocated object there?)
  - "Full": the whole image.

Example:
    python evaluate.py --edited-checkpoint runs/edit_0/student_nerf.pt \\
        --edit-metadata runs/edit_0/edit_metadata.json --out-json results/metrics.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from neo_nerf_editing.data.rays import fixed_scan_trajectory
from neo_nerf_editing.data.synthetic_scene import ToySceneGenerator
from neo_nerf_editing.evaluation.metrics import NeRFEvalMetrics
from neo_nerf_editing.models.language_field import OrientedBox
from neo_nerf_editing.models.nerf_field import JointNeRFField
from neo_nerf_editing.training.trainer import VolumetricRenderer, stratified_samples
from neo_nerf_editing.utils.config import NEOConfig, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate an edited NEO checkpoint against synthetic ground truth.")
    p.add_argument("--edited-checkpoint", type=str, required=True)
    p.add_argument("--edit-metadata", type=str, required=True, help="edit_metadata.json produced by edit.py")
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--out-json", type=str, default="results/metrics.json")
    p.add_argument("--n-eval-views", type=int, default=6)
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def render_field_view(field, renderer, ray_gen, cam, H, W, near, far, n_samples, device, chunk=4096):
    rays_o, rays_d = ray_gen.rays_from_camera(cam, H, W)
    rays_o = torch.from_numpy(rays_o).to(device)
    rays_d = torch.from_numpy(rays_d).to(device)
    outs = []
    for start in range(0, rays_o.shape[0], chunk):
        end = min(start + chunk, rays_o.shape[0])
        t = stratified_samples(end - start, near, far, n_samples, device, perturb=False)
        with torch.no_grad():
            outs.append(renderer.render_rays(field, rays_o[start:end], rays_d[start:end], t))
    rgb = torch.cat([o["rgb"] for o in outs], dim=0).reshape(H, W, 3)
    depth = torch.cat([o["depth"] for o in outs], dim=0).reshape(H, W)
    return rgb, depth


def project_box_mask(box: OrientedBox, cam, H: int, W: int, near: float, far: float, ray_gen) -> torch.Tensor:
    """Approximate per-pixel mask of whether a ray through this pixel intersects `box`,
    used to define the 'Out'/'In' masked evaluation regions (Sec. IV-A)."""
    from neo_nerf_editing.models.object_removal import RayBoxExcluder

    rays_o, rays_d = ray_gen.rays_from_camera(cam, H, W)
    rays_o_t, rays_d_t = torch.from_numpy(rays_o), torch.from_numpy(rays_d)
    hits = RayBoxExcluder().intersect(rays_o_t, rays_d_t, box, near, far)
    mask = (~torch.isnan(hits[:, 0])).reshape(H, W)
    return mask


def main() -> None:
    args = parse_args()
    cfg = NEOConfig.from_yaml(args.config)
    set_seed(cfg.seed, cfg.deterministic)
    device = cfg.device()
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)

    with open(args.edit_metadata) as f:
        meta = json.load(f)

    H = W = 32 if args.debug else cfg.data["image_h"]
    n_samples = 32 if args.debug else cfg.training["n_samples_uniform"] + cfg.training["n_samples_hierarchical"]
    n_views = 2 if args.debug else args.n_eval_views

    field_kwargs = dict(
        d_l=cfg.model["d_l"], hidden_dim=cfg.model["mlp_hidden_dim"], num_layers=cfg.model["mlp_num_layers"],
        posenc_num_freqs=cfg.model["posenc_num_freqs"], posenc_include_input=cfg.model["posenc_include_input"],
        skip_layers=cfg.model["mlp_skip_layers"],
    )
    student = JointNeRFField(**field_kwargs).to(device)
    student.load_state_dict(torch.load(args.edited_checkpoint, map_location=device))
    student.eval()

    scene_gen = ToySceneGenerator()
    scene = scene_gen.generate_scene(seed=meta["scene_seed"], objects=["soup_can", "lego_brick"])
    post_edit_scene = scene_gen.apply_manipulation(
        scene, meta["object_name"], np.array(meta["motion"], dtype=np.float32), remove_only=meta["remove_only"]
    )

    box = OrientedBox(
        center=np.array(meta["box_center"], dtype=np.float32), yaw=meta["box_yaw"],
        extents=np.array(meta["box_extents"], dtype=np.float32),
    )
    from neo_nerf_editing.training.pipeline import relocated_box_from_motion

    in_box = relocated_box_from_motion(box, np.array(meta["motion"], dtype=np.float32))

    trajectory = fixed_scan_trajectory(n_views, radius=4.5, height=2.2)  # slightly different views than training, for held-out eval
    metrics = NeRFEvalMetrics()
    renderer = VolumetricRenderer()
    from neo_nerf_editing.data.rays import RayGenerator

    ray_gen = RayGenerator()

    per_view_results = []
    for cam in trajectory:
        pred_rgb, pred_depth = render_field_view(
            student, renderer, ray_gen, cam, H, W, cfg.data["near"], cfg.data["far"], n_samples, device
        )
        gt_out = scene_gen.render_views(post_edit_scene, [cam], H=H, W=W, near=cfg.data["near"], far=cfg.data["far"])
        gt_rgb = gt_out["images"][0].to(device)
        gt_depth = gt_out["depths"][0].to(device)

        out_mask = project_box_mask(box, cam, H, W, cfg.data["near"], cfg.data["far"], ray_gen)
        in_mask = project_box_mask(in_box, cam, H, W, cfg.data["near"], cfg.data["far"], ray_gen)

        result = {
            "psnr_full": metrics.psnr(pred_rgb, gt_rgb),
            "ssim_full": metrics.ssim(pred_rgb, gt_rgb),
            "depth_rmse_full": metrics.depth_rmse(pred_depth, gt_depth),
            "psnr_out": metrics.psnr(pred_rgb, gt_rgb, out_mask),
            "ssim_out": metrics.ssim(pred_rgb, gt_rgb, out_mask),
            "depth_rmse_out": metrics.depth_rmse(pred_depth, gt_depth, out_mask),
        }
        if not meta["remove_only"]:
            result.update(
                {
                    "psnr_in": metrics.psnr(pred_rgb, gt_rgb, in_mask),
                    "ssim_in": metrics.ssim(pred_rgb, gt_rgb, in_mask),
                    "depth_rmse_in": metrics.depth_rmse(pred_depth, gt_depth, in_mask),
                }
            )
        per_view_results.append(result)

    def mean_of(key):
        vals = [r[key] for r in per_view_results if key in r and not np.isnan(r[key])]
        return float(np.mean(vals)) if vals else None

    all_keys = set().union(*[set(r.keys()) for r in per_view_results])
    aggregated = {k: mean_of(k) for k in sorted(all_keys)}

    output = {"per_view": per_view_results, "aggregated": aggregated, "meta": meta}
    with open(args.out_json, "w") as f:
        json.dump(output, f, indent=2)

    print("=" * 60)
    print("NEO evaluation results (synthetic toy scene -- see README Reproducibility Notes)")
    for k, v in aggregated.items():
        print(f"  {k:20s}: {v:.4f}" if v is not None else f"  {k:20s}: N/A")
    print(f"Saved full results to {args.out_json}")


if __name__ == "__main__":
    main()
