# Data — NEO-Dataset

The paper's real benchmark, **NEO-Dataset** (Sec. III), consists of 12 registered tabletop
scenes (11 single-object + 1 multi-object) captured with a Franka Emika Panda and a UR5 arm,
scanned before and after object removal/manipulation, registered via four ArUco markers. Object
classes are drawn from the [YCB Object and Model Set](https://www.ycbbenchmarks.com/).

## Status in this repository

The NEO-Dataset is referenced via the paper's project page (`https://csiro-robotics.github.io/NEO/`)
but was **not bundled with the paper and was not reachable from this reproduction's sandboxed
network environment** (the egress allowlist does not include the project page's host or any
robotics-lab data mirror). No download script is provided for it here because there is nothing
automatable to script against without that access.

**If you have access to the real NEO-Dataset:**
1. Place each scene under `data/raw/neo_dataset/{scene_name}/{pre,post}/` with `images/`,
   `poses.json` (or your preferred camera-pose format), and `aruco_transform.json`.
2. Write a small adapter implementing the same interface as
   `src/neo_nerf_editing/data/synthetic_scene.py::ToySceneGenerator.render_views` (i.e. return a
   dict with `images`, `depths`, `rays_o`, `rays_d`, `rgb_flat`) so `train.py` / `edit.py` /
   `evaluate.py` can consume it without further changes.
3. Also replace `models/language_field.py::DeterministicHashEmbedder` with a real CLIP text
   encoder, and `models/inpainting.py::HeuristicInpaintingBackend` with a real
   `diffusers.StableDiffusionInpaintPipeline`, to restore full paper fidelity (both interfaces
   are designed as one-line swaps — see their docstrings).

## What ships instead

`src/neo_nerf_editing/data/synthetic_scene.py::ToySceneGenerator` procedurally generates an
analogous multi-view RGB-D tabletop scene (colored boxes on a table, standing in for YCB
objects) with known ground truth, specifically so the full pipeline and every metric in
`evaluation/metrics.py` can be exercised and tested end-to-end without the real data. This is
explicitly **not** a substitute for the real benchmark's scene complexity or photorealism — see
`comparison/comparison_report.md` for how this affects the reproducibility score.
