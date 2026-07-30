#!/usr/bin/env python
"""
edit.py

Runs one full NEO scene edit given a trained base NeRF checkpoint, a language prompt, and a
planned 6-DoF object motion (Sec. II, Eq. 2: (S',L') = NEO((S,L), p, tau)). Pipeline:

  1. (Sec. II)    Localize target object -> oriented bounding box b.
  2. (Sec. II-A)  Neural field resampling: inference-time ray-resampling on the frozen base
                  NeRF that excludes b, giving an immediate "object removed" render.
  3. (Sec. II-B)  Train a NEW mask-conditioned NeRF F~_Theta (initialized from the base NeRF)
                  using L_masked = lambda_rec*L_rec + lambda_reg*L_reg, supervised by (a) the
                  Step-2 resampled renders at original camera poses and (b) progressively
                  inpainted virtual object-centered views.
  4. (Sec. II-C)  Region-wise teacher-student distillation: compose S' from {F_Theta outside b,
                  F~_Theta inside b, F_Theta-under-motion in the relocated region} and distill a
                  student F_hat_Theta.
  5. (Sec. II-D)  Update the language field on the fixed-geometry student.

SIR fidelity notes (see README "Reproducibility Notes" for full detail):
  - Localization uses the known synthetic-scene object registry (an "oracle" substitute) rather
    than the paper's real CLIP-grounded language field, because CLIP/DINO weights are
    unreachable in this sandbox. `--localization language_field` exercises the real
    cosine-similarity code path but will NOT reliably find the right object, since the
    language field's substitute text/scene embeddings carry no true semantic content.
  - Inpainting uses a classical OpenCV fallback, not Stable Diffusion (see models/inpainting.py).

Example:
    python edit.py --checkpoint runs/base_nerf/base_nerf.pt --prompt "move the soup can" \\
        --motion 0.4,0.0,0.3,0,0,0 --out-dir runs/edit_0
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from neo_nerf_editing.data.rays import fixed_scan_trajectory
from neo_nerf_editing.data.synthetic_scene import ToySceneGenerator
from neo_nerf_editing.models.inpainting import (
    HeuristicInpaintingBackend,
    ProgressiveInpaintingLoop,
    VirtualHemisphereSampler,
)
from neo_nerf_editing.models.language_field import DeterministicHashEmbedder, LanguageGrounder, OrientedBox
from neo_nerf_editing.models.nerf_field import JointNeRFField
from neo_nerf_editing.models.object_removal import RayBoxExcluder, TwoStageResampler
from neo_nerf_editing.training.losses import MaskedTrainingObjective
from neo_nerf_editing.training.pipeline import RegionCompositor, StudentDistiller, LanguageFieldUpdater
from neo_nerf_editing.training.trainer import VolumetricRenderer, stratified_samples
from neo_nerf_editing.utils.config import NEOConfig, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one NEO edit (Eq. 2).")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to trained base NeRF checkpoint")
    p.add_argument("--prompt", type=str, required=True, help="Language prompt identifying the target object")
    p.add_argument("--motion", type=str, required=True, help="6-DoF motion 'x,y,z,roll,pitch,yaw'")
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--scene-seed", type=int, default=0, help="Must match the seed used for train.py")
    p.add_argument("--out-dir", type=str, default="runs/edit_0")
    p.add_argument(
        "--localization", choices=["oracle", "language_field"], default="oracle",
        help="'oracle' uses synthetic-scene ground truth object registry (recommended, see docstring); "
             "'language_field' exercises the real (but semantically-inert substitute) cosine-similarity path",
    )
    p.add_argument("--remove-only", action="store_true", help="Only remove the object (Sec. IV-A protocol), skip relocation")
    p.add_argument("--debug", action="store_true", help="Reduce steps for a quick local smoke test")
    return p.parse_args()


def object_name_from_prompt(prompt: str, known_objects: list) -> str:
    """Best-effort keyword match from prompt text to a known synthetic-scene object name.
    Used only in --localization oracle mode."""
    lowered = prompt.lower()
    for name in known_objects:
        if name.replace("_", " ") in lowered or name in lowered:
            return name
    raise ValueError(f"Could not match prompt '{prompt}' to any known object in {known_objects}")


def main() -> None:
    args = parse_args()
    cfg = NEOConfig.from_yaml(args.config)
    set_seed(cfg.seed, cfg.deterministic)
    device = cfg.device()
    os.makedirs(args.out_dir, exist_ok=True)
    motion = np.array([float(x) for x in args.motion.split(",")], dtype=np.float32)
    assert motion.shape == (6,), f"--motion must have 6 comma-separated values, got {motion}"

    n_edit_steps = cfg.training["n_steps_edit_stage"] if not args.debug else 8
    n_phase1 = cfg.training["n_steps_distill_phase1"] if not args.debug else 8
    n_phase2 = cfg.training["n_steps_distill_phase2"] if not args.debug else 8
    n_lang_iters = cfg.training["n_steps_lang_field_update"] if not args.debug else 5
    n_inpaint_rounds = 3 if not args.debug else 2
    image_h = cfg.data["image_h"] if not args.debug else 24
    image_w = cfg.data["image_w"] if not args.debug else 24
    n_samples_uniform = cfg.training["n_samples_uniform"] if not args.debug else 24
    n_samples_hierarchical = cfg.training["n_samples_hierarchical"] if not args.debug else 32
    batch_size_rays = cfg.training["batch_size_rays"] if not args.debug else 128
    n_train_views = cfg.data["n_train_views"] if not args.debug else 4

    print("=" * 60)
    print("NEO edit pipeline")
    print(f"  prompt      : {args.prompt}")
    print(f"  motion      : {motion.tolist()}")
    print(f"  localization: {args.localization}")
    print(f"  device      : {device}")
    print("=" * 60)

    # --- Rebuild the same synthetic scene + base NeRF used in train.py ---
    scene_gen = ToySceneGenerator()
    scene = scene_gen.generate_scene(seed=args.scene_seed, objects=["soup_can", "lego_brick"])
    known_object_names = [o.name for o in scene.objects]

    field_kwargs = dict(
        d_l=cfg.model["d_l"], hidden_dim=cfg.model["mlp_hidden_dim"], num_layers=cfg.model["mlp_num_layers"],
        posenc_num_freqs=cfg.model["posenc_num_freqs"], posenc_include_input=cfg.model["posenc_include_input"],
        skip_layers=cfg.model["mlp_skip_layers"],
    )
    teacher_full = JointNeRFField(**field_kwargs).to(device)
    teacher_full.load_state_dict(torch.load(args.checkpoint, map_location=device))
    teacher_full.eval()

    # --- Step 1: Localize target object -> oriented bounding box b ---
    embedder = DeterministicHashEmbedder(d_l=cfg.model["d_l"])
    grounder = LanguageGrounder(embedder)
    if args.localization == "oracle":
        obj_name = object_name_from_prompt(args.prompt, known_object_names)
        target_obj = next(o for o in scene.objects if o.name == obj_name)
        box = OrientedBox(center=target_obj.center.copy(), yaw=0.0, extents=target_obj.half_extents * 1.15)
        print(f"[oracle localization] matched prompt to object '{obj_name}', box={box!r}")
    else:
        # Exercise the real code path: sample candidate surface points near known object extents
        # (a full implementation would extract these from the trained density field via
        # marching-cubes-style surface extraction; we approximate with a coarse grid near each
        # object for tractability) and query the (untrained-language-head) field's lang_feat.
        obj_name = known_object_names[0]
        pts = []
        for o in scene.objects:
            grid = np.stack(np.meshgrid(*[np.linspace(-1, 1, 5)] * 3, indexing="ij"), axis=-1).reshape(-1, 3)
            pts.append(o.center + grid * o.half_extents)
        surface_points = torch.from_numpy(np.concatenate(pts).astype(np.float32)).to(device)
        with torch.no_grad():
            dummy_dir = torch.zeros_like(surface_points)
            dummy_dir[:, 2] = 1.0
            lang_feats = teacher_full(surface_points, dummy_dir)["lang_feat"]
        prompt_embed = grounder.embed_prompt(args.prompt).to(device)
        box = grounder.localize(surface_points, lang_feats, prompt_embed, top_frac=0.1)
        print(f"[language_field localization] (NOTE: substitute embeddings are not semantically "
              f"meaningful yet -- box below is not expected to reliably match '{args.prompt}') box={box!r}")

    # --- Training data: same fixed scan trajectory as train.py ---
    trajectory = fixed_scan_trajectory(n_train_views, radius=4.0, height=2.0)
    render_out = scene_gen.render_views(
        scene, trajectory, H=image_h, W=image_w, near=cfg.data["near"], far=cfg.data["far"]
    )
    rays_o = render_out["rays_o"].to(device)
    rays_d = render_out["rays_d"].to(device)
    original_rgb_flat = render_out["rgb_flat"].to(device)

    # --- Step 2 (Sec. II-A): inference-time neural field resampling on the frozen teacher ---
    excluder = RayBoxExcluder()
    resampler = TwoStageResampler()
    renderer = VolumetricRenderer()

    RENDER_CHUNK = 4096  # ASSUMED: rays-per-chunk cap to keep MLP activation memory bounded on CPU

    def resampled_render_chunk(field: torch.nn.Module, o: torch.Tensor, d: torch.Tensor) -> dict:
        hits = excluder.intersect(o, d, box, cfg.data["near"], cfg.data["far"])
        seg_near, seg_far = excluder.build_exclusion_intervals(cfg.data["near"], cfg.data["far"], hits)
        t_u = resampler.uniform_resample(seg_near, seg_far, n_samples_uniform)
        with torch.no_grad():
            coarse = renderer.render_rays(field, o, d, t_u)
        t_h = resampler.hierarchical_resample(seg_near, seg_far, t_u, coarse["weights"], n_samples_hierarchical)
        t_all, _ = torch.sort(torch.cat([t_u, t_h], dim=-1), dim=-1)
        return renderer.render_rays(field, o, d, t_all)

    def resampled_render(field: torch.nn.Module, o: torch.Tensor, d: torch.Tensor) -> dict:
        """Chunked wrapper: full-dataset forward passes (e.g. all 131072 rays at once) blew up
        CPU memory (~13GB of MLP activations for [N,S,hidden]) -- real bug caught by actually
        running the pipeline. Chunking over RENDER_CHUNK rays keeps peak memory bounded."""
        outs = []
        n = o.shape[0]
        for start in range(0, n, RENDER_CHUNK):
            end = min(start + RENDER_CHUNK, n)
            outs.append(resampled_render_chunk(field, o[start:end], d[start:end]))
        return {k: torch.cat([out[k] for out in outs], dim=0) for k in outs[0]}

    with torch.no_grad():
        removed_render = resampled_render(teacher_full, rays_o, rays_d)
    print(f"[Step 2] object-removal resampling done over {rays_o.shape[0]} rays "
          f"(mean removed-render RGB: {removed_render['rgb'].mean().item():.4f})")

    # --- Step 3 (Sec. II-B): train mask-conditioned F~_Theta ---
    edited_field = JointNeRFField(**field_kwargs).to(device)
    edited_field.load_state_dict(copy.deepcopy(teacher_full.state_dict()))  # init from base NeRF

    hemi_sampler = VirtualHemisphereSampler()
    virtual_cams = hemi_sampler.sample_views(box, n_azimuths=4, n_elevations=3)
    inpaint_loop = ProgressiveInpaintingLoop(HeuristicInpaintingBackend(), xi_start=cfg.training["xi_start"], xi_end=cfg.training["xi_end"])
    objective = MaskedTrainingObjective()
    optimizer = torch.optim.Adam(edited_field.parameters(), lr=cfg.training["learning_rate"])

    from neo_nerf_editing.data.rays import RayGenerator

    ray_gen = RayGenerator()
    H, W = image_h, image_w
    hits_orig = excluder.intersect(rays_o, rays_d, box, cfg.data["near"], cfg.data["far"])
    mask_o = torch.isnan(hits_orig[:, 0])  # True where ray misses box -> safe to supervise with removed_render

    edit_history = {"l_masked": []}
    steps_per_round = max(1, n_edit_steps // n_inpaint_rounds)
    for round_idx in range(n_inpaint_rounds):
        # Render current edited_field from the 4 virtual views, tile+inpaint, split back.
        virtual_renders, virtual_masks = [], []
        for cam in virtual_cams[:4]:
            o_v, d_v = ray_gen.rays_from_camera(cam, H, W)
            o_v_t = torch.from_numpy(o_v).to(device)
            d_v_t = torch.from_numpy(d_v).to(device)
            with torch.no_grad():
                out_v = resampled_render(edited_field, o_v_t, d_v_t)
            img_v = out_v["rgb"].reshape(H, W, 3).detach().cpu().numpy()
            virtual_renders.append(img_v)
            m = torch.zeros(H * W, dtype=torch.bool)
            hits_v = excluder.intersect(o_v_t, d_v_t, box, cfg.data["near"], cfg.data["far"])
            m = (~torch.isnan(hits_v[:, 0])).reshape(H, W).cpu().numpy()
            virtual_masks.append(m)
        inpainted_views = inpaint_loop.run_one_round(virtual_renders, virtual_masks, round_idx, n_inpaint_rounds)

        for step in range(steps_per_round):
            # Half the steps supervise original-view background (mask_o), half supervise
            # inpainted virtual-view content (mask=1 within the box footprint).
            idx = torch.randint(0, rays_o.shape[0], (batch_size_rays,), device=device)
            o_b, d_b = rays_o[idx], rays_d[idx]
            t_b = stratified_samples(o_b.shape[0], cfg.data["near"], cfg.data["far"], n_samples_uniform, device)
            out_b = renderer.render_rays(edited_field, o_b, d_b, t_b)

            target_o = removed_render["rgb"][idx]
            mask_b = mask_o[idx]
            loss = objective(
                out_b["rgb"], target_o, mask_b, out_b["raw_density"],
                o_b[:, None, :] + t_b[..., None] * d_b[:, None, :], box,
                cfg.training["lambda_rec"], cfg.training["lambda_reg"],
            )

            v_idx = np.random.randint(0, 4)
            target_v_np = inpainted_views[v_idx].reshape(-1, 3)
            n_pix = target_v_np.shape[0]
            pix_idx = np.random.choice(n_pix, size=min(256, n_pix), replace=False)
            cam_v = virtual_cams[v_idx]
            o_v, d_v = ray_gen.rays_from_camera(cam_v, H, W)
            o_v_b = torch.from_numpy(o_v[pix_idx]).to(device)
            d_v_b = torch.from_numpy(d_v[pix_idx]).to(device)
            t_v_b = stratified_samples(o_v_b.shape[0], cfg.data["near"], cfg.data["far"], n_samples_uniform, device)
            out_v_b = renderer.render_rays(edited_field, o_v_b, d_v_b, t_v_b)
            target_v = torch.from_numpy(target_v_np[pix_idx]).to(device)
            mask_v = torch.ones(o_v_b.shape[0], dtype=torch.bool, device=device)
            loss_v = objective(
                out_v_b["rgb"], target_v, mask_v, out_v_b["raw_density"],
                o_v_b[:, None, :] + t_v_b[..., None] * d_v_b[:, None, :], box,
                cfg.training["lambda_rec"], cfg.training["lambda_reg"],
            )

            total_loss = loss + loss_v
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            edit_history["l_masked"].append(total_loss.item())

        print(f"[Step 3] inpainting round {round_idx+1}/{n_inpaint_rounds} "
              f"(xi={inpaint_loop.xi_at(round_idx, n_inpaint_rounds):.2f}) "
              f"final L_masked={edit_history['l_masked'][-1]:.6f}")

    # --- Step 4 (Sec. II-C): region-wise teacher-student distillation ---
    student = JointNeRFField(**field_kwargs).to(device)
    student.load_state_dict(copy.deepcopy(teacher_full.state_dict()))
    compositor = RegionCompositor(renderer)
    distiller = StudentDistiller(lr=cfg.training["learning_rate"])
    dist_history = distiller.distill(
        student=student, compositor=compositor, teacher_full=teacher_full, teacher_edited=edited_field,
        box=box, motion=(np.zeros(6, dtype=np.float32) if args.remove_only else motion),
        rays_o=rays_o, rays_d=rays_d, near=cfg.data["near"], far=cfg.data["far"],
        n_samples=n_samples_uniform, n_steps_phase1=n_phase1, n_steps_phase2=n_phase2,
        batch_size=batch_size_rays, device=device,
    )
    print(f"[Step 4] distillation done. phase1 final={dist_history['phase1'][-1]:.6f}, "
          f"phase2 final={dist_history['phase2'][-1]:.6f}")

    # --- Step 5 (Sec. II-D): update the language field on the fixed-geometry student ---
    lang_updater = LanguageFieldUpdater(embedder, lr=cfg.training["learning_rate"])
    sample_points = rays_o[:512] + 3.0 * rays_d[:512]  # arbitrary mid-ray points as supervision anchors
    with torch.no_grad():
        target_embed = grounder.embed_prompt(f"scene after editing: {obj_name}").to(device)
        target_lang = target_embed.unsqueeze(0).expand(sample_points.shape[0], -1).float()
    lang_history = lang_updater.update(student, target_lang, sample_points, n_lang_iters, device=device)
    print(f"[Step 5] language field update done. final loss={lang_history[-1]:.6f}")

    # --- Save outputs ---
    ckpt_path = os.path.join(args.out_dir, "student_nerf.pt")
    torch.save(student.state_dict(), ckpt_path)
    meta = {
        "prompt": args.prompt, "motion": motion.tolist(), "object_name": obj_name,
        "box_center": box.center.tolist(), "box_yaw": box.yaw, "box_extents": box.extents.tolist(),
        "remove_only": args.remove_only, "scene_seed": args.scene_seed,
        "edit_history_final": edit_history["l_masked"][-1],
        "distill_phase1_final": dist_history["phase1"][-1], "distill_phase2_final": dist_history["phase2"][-1],
        "lang_update_final": lang_history[-1],
    }
    with open(os.path.join(args.out_dir, "edit_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved edited student checkpoint to {ckpt_path}")
    print(f"Saved edit metadata to {os.path.join(args.out_dir, 'edit_metadata.json')}")


if __name__ == "__main__":
    main()
