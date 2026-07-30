#!/usr/bin/env python
"""
data/download.py

The real NEO-Dataset is not publicly bundled/mirrored anywhere this sandbox could reach (see
data/README_data.md). This script does not fabricate a fake download -- it checks whether the
user has already placed real data at `data/raw/neo_dataset/`, and if not, prints clear
instructions and offers to generate the synthetic stand-in scene used by the rest of this repo.

Usage:
    python data/download.py                    # checks for real data, offers synthetic fallback
    python data/download.py --synthetic-only    # skip the check, just materialize synthetic scene
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default=os.path.join(os.path.dirname(__file__), "raw", "neo_dataset"))
    p.add_argument("--synthetic-only", action="store_true")
    args = p.parse_args()

    if not args.synthetic_only and os.path.isdir(args.data_root) and os.listdir(args.data_root):
        print(f"Found existing data at {args.data_root} -- assuming this is real NEO-Dataset content.")
        print("Nothing to download; proceed with train.py / edit.py using a real-data adapter (see data/README_data.md).")
        return

    if not args.synthetic_only:
        print("=" * 70)
        print("Real NEO-Dataset not found, and it is not reachable for automated download")
        print("from this environment (see data/README_data.md for exactly why + what to do")
        print("if you have access to it).")
        print("=" * 70)
        print()
        print("Falling back to generating the synthetic stand-in scene used throughout this repo...")

    from neo_nerf_editing.data.synthetic_scene import ToySceneGenerator

    scene_gen = ToySceneGenerator()
    scene = scene_gen.generate_scene(seed=0, objects=["soup_can", "lego_brick"])
    manifest = {
        "note": "Synthetic stand-in scene (NOT real NEO-Dataset). See data/README_data.md.",
        "objects": [
            {"name": o.name, "center": o.center.tolist(), "half_extents": o.half_extents.tolist(), "color": o.color.tolist()}
            for o in scene.objects
        ],
    }
    out_dir = os.path.join(os.path.dirname(__file__), "processed")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "synthetic_scene_manifest.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote synthetic scene manifest to {out_path}")
    print("train.py / edit.py / evaluate.py regenerate this scene on the fly from --scene-seed;")
    print("this manifest is just a human-readable record of what seed=0 currently contains.")


if __name__ == "__main__":
    main()
