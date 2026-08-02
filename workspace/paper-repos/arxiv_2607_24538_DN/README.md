# NEO: NeRF It Once, Edit It Many Times for Continuous Object Manipulation

**Unofficial ArXivist reproduction scaffold** for:

> Mikołaj Zieliński, David Hall, Dominik Belter, Peyman Moghadam.
> *NEO: NeRF It Once, Edit It Many Times for Continuous Object Manipulation.*
> Submitted to IEEE Robotics and Automation Letters, 2026. [arXiv:2607.24538](https://arxiv.org/abs/2607.24538)

## What this paper does

NEO edits a Neural Radiance Field (NeRF) representation of a tabletop scene directly at the
weight level, so a robot can remove, inpaint, and relocate objects — and predict what the scene
will look like after each action — **without re-scanning**. Given a language prompt ("move the
soup can") and a planned motion, NEO (i) removes the target object from the NeRF via a
resampling trick that avoids the floating-point artifacts of prior methods, (ii) fills in the
newly-revealed, previously-occluded surfaces using multiview-consistent diffusion inpainting,
and (iii) distills the original and edited NeRFs into a single updated network via a
teacher-student scheme, so edits persist and compose across a sequence of manipulations. The
paper also introduces NEO-Dataset, the first benchmark with registered pre/post-edit scans for
quantitatively evaluating NeRF scene editing in robotic manipulation.

## ⚠️ Read this before running anything: Reproducibility Notes

This reproduction was built and run in a **fully offline, CPU-only sandbox with no access to**:
the real **NEO-Dataset** (unreleased/unreachable), **Stable Diffusion** or **CLIP/DINO**
checkpoints (no model-hub network access), or **AnyGrasp** (proprietary/license-gated). Rather
than silently faking results, every one of these gaps is implemented as an explicit, documented,
swappable substitute:

| Paper component | What ships here | Why |
|---|---|---|
| NEO-Dataset (12 real scanned scenes) | `ToySceneGenerator`: procedural synthetic tabletop scene | Real dataset not bundled/reachable — see `data/README_data.md` |
| Stable Diffusion inpainting | `HeuristicInpaintingBackend` (OpenCV classical inpainting) | SD checkpoints unreachable from this sandbox; interface (`InpaintingBackend`) is a one-line swap for `diffusers.StableDiffusionInpaintPipeline` |
| CLIP-grounded language field + CLIP/DINO field update | `DeterministicHashEmbedder` (deterministic hash → unit vector) | CLIP/DINO weights unreachable; carries no real semantic content — see caveat below |
| AnyGrasp [32] | `AnyGraspStub` (simple top-down/side candidate poses) | Proprietary, not re-derived by the paper itself either |

**The algorithmic contributions that make NEO *NEO*** — the two-stage box-exclusion ray
resampling (Sec. II-A), mask-conditioned reconstruction + density regularization (Eq. 3–5), and
the region-wise teacher-student distillation (Eq. 6, Sec. II-C) — **are implemented faithfully
to the SIR** and are exercised end-to-end against the synthetic scene. What's substituted is
specifically the large *pretrained-model* dependencies, not the paper's own novel machinery.

**Language grounding caveat:** because `DeterministicHashEmbedder` has no real semantic content,
`edit.py` defaults to `--localization oracle`, which looks up the target object directly from
the synthetic scene's own object registry (a labeled stand-in for what a real trained CLIP
language field would localize). `--localization language_field` exercises the real
cosine-similarity code path (`models/language_field.py`) but will not reliably find the correct
object — this is expected and documented, not a bug.

**Hyperparameters the paper never states numerically** (optimizer, learning rate, batch size,
step counts, λ_rec/λ_reg loss weights, virtual-view count, re-inpaint interval `n`, ξ decay
schedule) are filled with standard NeRF-literature defaults, each marked `# ASSUMED: <basis>` in
`configs/config.yaml`. See `sir-registry/arxiv_2607_24538/sir.json` → `ambiguities` for the full
list with alternatives considered.

Full detail, including the actual measured (synthetic, debug-scale) run numbers from building
this repo, is in [`comparison/comparison_report.md`](comparison/comparison_report.md).

## Quick start

```bash
pip install -r requirements.txt
pip install -e .

# 1. Train the base joint scene+language NeRF on the synthetic scene
python train.py --config configs/config.yaml --scene-seed 0 --out-dir runs/base_nerf

# 2. Run one NEO edit: remove + relocate an object given a prompt + motion
python edit.py --checkpoint runs/base_nerf/base_nerf.pt --prompt "move the soup can" \
    --motion 0.4,0.0,0.3,0,0,0 --out-dir runs/edit_0

# 3. Evaluate against synthetic ground truth (PSNR/SSIM/depth-RMSE/E_rep, Full/Out/In)
python evaluate.py --edited-checkpoint runs/edit_0/student_nerf.pt \
    --edit-metadata runs/edit_0/edit_metadata.json --out-json results/metrics.json

# 4. (Optional) Single-sample inference: localize + propose grasps, no training
python inference.py --checkpoint runs/base_nerf/base_nerf.pt --prompt "move the lego brick"

# 5. (Optional) Table III ablation sweep
python run_ablation.py --config configs/config.yaml --out-dir runs/ablation
```

Every command above also accepts `--debug` for a fast (~1-2 min on CPU) smoke test with a
shrunk image resolution / step count — start there before a full run.

## Installation

**pip:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

**conda:**
```bash
conda env create -f environment.yaml
conda activate neo-nerf-editing
```

**Docker:**
```bash
docker compose -f docker/docker-compose.yml up train
docker compose -f docker/docker-compose.yml up notebook   # Jupyter at localhost:8888
```

## Repository layout

```
src/neo_nerf_editing/
├── models/         # JointNeRFField (F_Theta), LanguageGrounder, object-removal resampling,
│                   # multiview inpainting, all per SIR architecture graph
├── data/           # camera/ray utilities + synthetic scene generator (NEO-Dataset stand-in)
├── training/        # losses (Eq. 3-6), base NeRF trainer, region-wise distillation pipeline
├── evaluation/       # PSNR / SSIM / depth-RMSE / E_rep
└── utils/           # config + seeding, AnyGrasp interface stub

train.py / edit.py / evaluate.py / inference.py / run_ablation.py   # entrypoints
configs/config.yaml                                                  # all hyperparameters
```

## Expected results (paper's own reported numbers, Table I/II)

These are the **paper's** results on the real NEO-Dataset — not reproduced numbers (see
Reproducibility Notes above and the comparison report for what this repo actually measures).

**Object removal, 5 scenes (Table I):**

| Method | PSNR Full↑ | PSNR Out↑ | SSIM Full↑ | SSIM Out↑ | RMSE Full↓ | RMSE Out↓ | E_rep Full↓ | E_rep Out↓ |
|---|---|---|---|---|---|---|---|---|
| NeRFiller | 20.40 | 21.71 | 0.696 | 0.671 | 0.252 | 0.254 | 0.104 | 0.080 |
| Seal-3D | 27.08 | 20.05 | 0.866 | 0.709 | 0.162 | 0.140 | 0.079 | 0.089 |
| DFF | 26.91 | 21.64 | 0.878 | 0.747 | 0.048 | 0.047 | 0.068 | 0.074 |
| **NEO (paper)** | **27.20** | **25.43** | **0.881** | **0.845** | 0.056 | **0.033** | **0.067** | 0.071 |

**Pick-and-place, 6 scenes (Table II):**

| Method | PSNR Out↑ | PSNR In↑ | SSIM Out↑ | SSIM In↑ |
|---|---|---|---|---|
| Seal-3D | 18.94 | 18.56 | 0.695 | 0.589 |
| **NEO (paper)** | **21.99** | **19.26** | **0.783** | **0.619** |

## Citation

```bibtex
@article{zielinski2026neo,
  title   = {NEO: NeRF It Once, Edit It Many Times for Continuous Object Manipulation},
  author  = {Zieli{\'n}ski, Miko{\l}aj and Hall, David and Belter, Dominik and Moghadam, Peyman},
  journal = {IEEE Robotics and Automation Letters (submitted)},
  year    = {2026},
  eprint  = {2607.24538},
  archivePrefix = {arXiv}
}
```

## License

This reproduction scaffold is provided as-is for research reproducibility purposes. Refer to the
original paper for licensing of NEO itself; `AnyGraspStub` and `HeuristicInpaintingBackend` are
original substitute code written for this repo, not derived from any proprietary system.
