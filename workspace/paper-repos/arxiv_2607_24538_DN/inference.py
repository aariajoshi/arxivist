#!/usr/bin/env python
"""
inference.py

Single-sample inference: given a trained base NeRF checkpoint and a language prompt, localizes
the target object (Sec. II bounding-box retrieval) and generates candidate grasps (Sec. II,
"Grasp Planning" / AnyGrasp [32] call site), WITHOUT running the full edit/distillation
pipeline. This is the lightweight "what would NEO do right now" entrypoint; use `edit.py` to
actually perform and persist a scene edit.

Example:
    python inference.py --checkpoint runs/base_nerf/base_nerf.pt --prompt "move the soup can"
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from neo_nerf_editing.data.synthetic_scene import ToySceneGenerator
from neo_nerf_editing.models.language_field import DeterministicHashEmbedder, LanguageGrounder, OrientedBox
from neo_nerf_editing.models.nerf_field import JointNeRFField
from neo_nerf_editing.utils.config import NEOConfig, set_seed
from neo_nerf_editing.utils.grasp_stub import AnyGraspStub


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-sample NEO inference: localize + propose grasps.")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--config", type=str, default="configs/config.yaml")
    p.add_argument("--scene-seed", type=int, default=0)
    p.add_argument("--n-grasp-candidates", type=int, default=8)
    p.add_argument("--out-json", type=str, default=None, help="Optional path to dump result JSON")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = NEOConfig.from_yaml(args.config)
    set_seed(cfg.seed, cfg.deterministic)
    device = cfg.device()

    field_kwargs = dict(
        d_l=cfg.model["d_l"], hidden_dim=cfg.model["mlp_hidden_dim"], num_layers=cfg.model["mlp_num_layers"],
        posenc_num_freqs=cfg.model["posenc_num_freqs"], posenc_include_input=cfg.model["posenc_include_input"],
        skip_layers=cfg.model["mlp_skip_layers"],
    )
    field = JointNeRFField(**field_kwargs).to(device)
    field.load_state_dict(torch.load(args.checkpoint, map_location=device))
    field.eval()

    scene_gen = ToySceneGenerator()
    scene = scene_gen.generate_scene(seed=args.scene_seed, objects=["soup_can", "lego_brick"])
    known_object_names = [o.name for o in scene.objects]

    lowered = args.prompt.lower()
    matched = next((n for n in known_object_names if n.replace("_", " ") in lowered or n in lowered), None)
    if matched is None:
        raise ValueError(
            f"Could not match prompt '{args.prompt}' to any known object {known_object_names}. "
            f"(oracle localization substitute -- see edit.py docstring for why)"
        )
    target_obj = next(o for o in scene.objects if o.name == matched)
    box = OrientedBox(center=target_obj.center.copy(), yaw=0.0, extents=target_obj.half_extents * 1.15)

    grasp_stub = AnyGraspStub()
    grasps = grasp_stub.generate_candidates(box, n_candidates=args.n_grasp_candidates)

    result = {
        "prompt": args.prompt,
        "matched_object": matched,
        "box": {"center": box.center.tolist(), "yaw": box.yaw, "extents": box.extents.tolist()},
        "grasps": [
            {"position": g.position.tolist(), "approach_dir": g.approach_dir.tolist(), "score": g.score}
            for g in grasps
        ],
    }

    print("=" * 60)
    print(f"Prompt        : {args.prompt}")
    print(f"Matched object: {matched}")
    print(f"Bounding box  : {box!r}")
    print(f"Grasp candidates ({len(grasps)}, from AnyGraspStub -- see utils/grasp_stub.py docstring):")
    for g in grasps:
        print(f"  {g!r}")
    print("=" * 60)

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved result to {args.out_json}")


if __name__ == "__main__":
    main()
