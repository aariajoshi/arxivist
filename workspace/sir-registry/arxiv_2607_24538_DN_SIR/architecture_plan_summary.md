# Architecture Plan Summary — NEO (arxiv_2607_24538)

## Framework
PyTorch, Python 3.10+, CPU-first (auto-CUDA if available), YAML config. No HuggingFace hard dependency — the real Stable Diffusion / CLIP / DINO checkpoints this paper relies on are unreachable from this sandbox's network allowlist, so pluggable interfaces are used instead (see Risks).

## Module Hierarchy (12 files)
- `models/nerf_field.py` — `JointNeRFField` (F_Θ): positional-encoding MLP → (density, RGB, language feature)
- `models/language_field.py` — `LanguageGrounder`: cosine-similarity localization → oriented bbox
- `models/object_removal.py` — `RayBoxExcluder` + `TwoStageResampler`: the paper's core Sec. II-A contribution
- `models/inpainting.py` — `VirtualHemisphereSampler`, pluggable `InpaintingBackend` (OpenCV heuristic default), `ProgressiveInpaintingLoop`
- `training/losses.py` — `L_rec`, `L_reg`, `L_masked`, two-phase `L_distill` (Eq. 3–6, implemented exactly as specified)
- `training/pipeline.py` — `RegionCompositor`, `StudentDistiller`, `LanguageFieldUpdater`, `NEOEditPipeline` (the "editing many times" loop)
- `training/trainer.py` — `VolumetricRenderer`, `NeRFTrainer`
- `data/synthetic_scene.py` — `ToySceneGenerator`: stands in for the unreleased NEO-Dataset
- `data/rays.py` — camera/ray utilities
- `evaluation/metrics.py` — PSNR / SSIM / depth-RMSE / E_rep over Full/Out/In regions
- `utils/config.py`, `utils/grasp_stub.py` — config loading; interface-only `AnyGraspStub`

## Entrypoints
`train.py` (base NeRF) → `edit.py` (one NEO edit) → `evaluate.py` (metrics vs ground truth) → `run_ablation.py` (Table III variants)

## Key Assumptions (flagged `# ASSUMED` in config.yaml)
- Optimizer/LR/batch/steps: paper states none of these (training_pipeline confidence 0.45) → standard NeRF-literature defaults used
- λ_rec = λ_reg = 1.0 (values never given)
- Backbone: plain positional-encoding MLP (paper never names hash-grid vs MLP)
- Virtual view count: 12 (4 azimuths × 3 elevations)
- Re-inpaint interval n = 100 steps, ξ linearly decayed 1.0→0.2

## Biggest Fidelity Gaps (see risk_assessment for full detail)
1. **NEO-Dataset unavailable** → all quantitative results here run on a procedurally generated synthetic toy scene, not the real benchmark.
2. **Stable Diffusion / CLIP / DINO unreachable** → OpenCV heuristic inpainting and a deterministic hash-based language embedding are used as documented, swappable substitutes.
3. **AnyGrasp is proprietary** → `AnyGraspStub` provides an interface-compatible but explicitly non-faithful placeholder.

These three gaps are the primary drivers of the reproducibility score in Stage 6 — the *algorithmic* contributions (resampling, mask-conditioned training, region-wise distillation) are implemented faithfully to the SIR; the *pretrained-model* dependencies are substituted and labeled.
