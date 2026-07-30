# Hallucination Report
**Paper**: NEO: NeRF It Once, Edit It Many Times for Continuous Object Manipulation
**Paper ID**: arxiv_2607_24538
**Date**: 2026-07-29

Reviews the generated architecture plan + code against the SIR to identify components that
were invented, assumed, or omitted relative to the paper.

## Structural Hallucinations (components NOT in the SIR)

**None identified.** Every class in `architecture_plan.json → module_hierarchy` maps to a named
SIR architecture module (see `sir_modules_mapped` on each entry), except for standard supporting
infrastructure that is required to exercise the SIR's modules but is not itself a scientific
contribution of the paper:

| Supporting class | Why it exists | Risk |
|---|---|---|
| `PositionalEncoding` | Standard NeRF (Mildenhall et al. [1], which the paper itself cites) input encoding; paper doesn't restate its own encoding | Minor — this is the universal default, not an invention |
| `Camera`, `RayGenerator`, `fixed_scan_trajectory`, `hemisphere_spiral_trajectory` | Camera/ray plumbing implied by "camera images I with poses Ω" (Eq. 1) but never itself specified as a class in the paper | None — necessary glue code |
| `ToyScene`, `ToyObject`, `ToySceneGenerator` | Synthetic-data substitute for the real NEO-Dataset (explicitly and repeatedly labeled as such) | None if used only for smoke-testing; **would be a Critical risk if anyone mistook these numbers for paper-comparable results** — see `benchmark_comparison.md` |
| `Grasp`, `AnyGraspStub` | Interface-compatible stand-in call site for the external AnyGrasp [32] dependency | None — explicitly documented as a non-faithful stub |

## Parametric Hallucinations (assumed hyperparameters)

All 7 below are marked `# ASSUMED` in `configs/config.yaml` and listed in
`sir.json → implementation_assumptions` / `ambiguities`. None have been validated against the
paper (impossible without the numbers it never published) or tuned.

| # | Parameter | Assumed value | SIR confidence | Severity if wrong |
|---|---|---|---|---|
| 1 | Optimizer | Adam | 0.6 | Minor — Adam is the near-universal NeRF default |
| 2 | Learning rate | 5e-4 | 0.2 | Moderate — could easily be off by 2-5x for this specific architecture |
| 3 | Backbone width/depth (`mlp_hidden_dim=128`, `mlp_num_layers=6`) | plain MLP | 0.5 | Significant — if the real backbone is an Instant-NGP hash grid, absolute quality/speed differ substantially |
| 4 | λ_rec, λ_reg (combined objective weights) | both 1.0 | 0.4 | Moderate — an imbalanced true ratio would change how strongly floaters are suppressed vs. reconstruction fidelity is prioritized |
| 5 | Virtual view count (4 azimuths × 3 elevations = 12) | 12 | 0.5 | Minor — affects inpainting supervision density, not correctness of the mechanism |
| 6 | Re-inpaint interval `n` and ξ decay schedule | n=100, linear 1.0→0.2 | 0.35 | Minor — affects convergence speed/final polish, not mechanism correctness |
| 7 | Two-stage ray-sample counts (uniform=64, hierarchical=128) | 64 / 128 | 0.4 | Minor — standard NeRF defaults; real values could differ |

None of these coincide with a *measured* Moderate/Significant/Critical deviation in this report,
because (per `benchmark_comparison.md`) no metric comparison against the paper's real numbers
was possible at all — the dataset gap dominates and would mask any effect from these regardless.

## Omission Hallucinations (SIR components present but stubbed/substituted)

| Component (SIR) | What ships instead | Severity | Evidence | Suggested fix |
|---|---|---|---|---|
| Stable Diffusion inpainting (Fig. 6 caption) | `HeuristicInpaintingBackend` (OpenCV Telea) | **Significant** | Classical inpainting cannot hallucinate plausible new geometry/texture; this is precisely what Sec. II-B's inpainting step needs for previously-unobserved regions (paper's own Fig. 5 example) | Swap in `diffusers.StableDiffusionInpaintPipeline` behind the existing `InpaintingBackend` interface (one-line change in `edit.py`) |
| CLIP-grounded language field + CLIP/DINO field update (Sec. II, II-D) | `DeterministicHashEmbedder` (hash → unit vector, no real semantics) | **Significant** | Cannot actually distinguish "soup can" from "lego brick" by meaning; `edit.py` defaults to `--localization oracle` (ground-truth object registry) specifically to route around this for functional testing | Replace with a real CLIP text/image encoder + DINO feature extractor; both call sites (`LanguageGrounder.embed_prompt`, `LanguageFieldUpdater.update`) are isolated for a straightforward swap |
| AnyGrasp [32] (external, Sec. II) | `AnyGraspStub` (simple top-down/side candidates) | **Minor** | AnyGrasp is proprietary/license-gated and not re-derived by the paper itself either — the paper treats it as an off-the-shelf black box, same as this repo does | Install real AnyGrasp and implement `AnyGraspStub`'s interface against it |

No SIR architecture module was silently dropped — every one of the 10 modules in
`sir.json → architecture.modules` has a corresponding file/class in the generated repo (cross-
checked against `architecture_plan.json → module_hierarchy`, 12 entries, one-to-many mapping
where a single SIR module split across a couple of implementation files, e.g. the two-stage
resampler's excluder + resampler classes).

## Summary

Zero structural or omission hallucinations of the paper's *own* novel contributions were found —
the resampling, mask-conditioned training, and region-wise distillation machinery are all present
and implemented per the SIR's mathematical spec. The 3 Significant/Minor omissions are all
*pretrained external models* the paper itself depends on but doesn't re-derive, substituted here
because they are unreachable in this sandbox, and each is clearly labeled with a one-line-swap
path back to the real dependency.
