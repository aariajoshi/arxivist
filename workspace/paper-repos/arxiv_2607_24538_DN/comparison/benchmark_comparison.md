# Benchmark Comparison Report
**Paper**: NEO: NeRF It Once, Edit It Many Times for Continuous Object Manipulation
**Paper ID**: arxiv_2607_24538
**Comparison Date**: 2026-07-29
**Reproducibility Score**: 0.00 (High confidence that this score is correctly near-zero — see below)

## Why the score is near-zero, and what it does and does not mean

The paper's 16 reported quantitative results (Tables I–III) are **all** measured on the real
**NEO-Dataset** (12 robot-scanned tabletop scenes) using the real **Stable Diffusion** inpainting
backend and a real **CLIP/DINO**-grounded language field. None of that is reachable in this
sandbox (see README "Reproducibility Notes"). The only run available for this comparison used:

- a procedurally generated **synthetic toy scene** (not NEO-Dataset)
- `--debug` scale: 24×24 renders, 4 training views, 15–20 base-NeRF steps, 8 edit/distillation
  steps per phase (a smoke test, not a real training run)
- the `HeuristicInpaintingBackend` (OpenCV) and `DeterministicHashEmbedder` substitutes

Under the Stage 6 matching rule (metric name **+ dataset + split**), every one of the paper's
16 reported results is **UNMATCHED** — there is no "NEO-Dataset, object removal, 5 scenes" run
to compare against, only a synthetic debug run. Per the scoring formula, zero matched pairs
yields `base_score = 0`, and the full `unmatched_penalty` (16/16 × 0.2 = 0.2) applies on top,
giving `reproducibility_score = max(0, 0 − 0.0465 − 0.2) = 0.00`.

**This score reflects data/model availability, not a demonstrated implementation defect.** The
paper's own algorithmic contributions (Sec. II-A resampling, Eq. 3–5 masked training, Eq. 6
distillation) were implemented and unit-verified against their mathematical specification (see
"Algorithmic Fidelity Checks" below) — that verification is real, even though it can't be
expressed as a percentage deviation against the paper's Table I/II numbers.

## Metric Comparison

| Metric | Dataset (paper) | Paper Value | Dataset (this run) | This Run's Value | Deviation | Severity |
|---|---|---|---|---|---|---|
| PSNR (Full) | NEO-Dataset, object removal, 5 scenes | 27.20 | synthetic toy, debug | 14.96 | UNMATCHED (different dataset) | N/A |
| PSNR (Out) | NEO-Dataset, object removal, 5 scenes | 25.43 | synthetic toy, debug | 13.92 | UNMATCHED | N/A |
| SSIM (Full) | NEO-Dataset, object removal, 5 scenes | 0.881 | synthetic toy, debug | 0.456 | UNMATCHED | N/A |
| SSIM (Out) | NEO-Dataset, object removal, 5 scenes | 0.845 | synthetic toy, debug | 0.056 | UNMATCHED | N/A |
| Depth RMSE (Full) | NEO-Dataset, object removal, 5 scenes | 0.056 | synthetic toy, debug | 1.043 | UNMATCHED | N/A |
| Depth RMSE (Out) | NEO-Dataset, object removal, 5 scenes | 0.033 | synthetic toy, debug | 0.635 | UNMATCHED | N/A |
| E_rep (Full) | NEO-Dataset, object removal, 5 scenes | 0.067 | synthetic toy, debug | — (not computed; requires registered 3D GT points) | UNMATCHED | N/A |
| PSNR (Out), pick-and-place | NEO-Dataset, 6 scenes | 21.99 | synthetic toy, debug | 10.47 (In) / 13.92 (Out, removal run) | UNMATCHED | N/A |
| Table III ablation (all 4 variants × 3 regions × 3 metrics) | NEO-Dataset | see paper Table III | synthetic toy, debug | see `raw_debug_run_results.json` | UNMATCHED | N/A |

*("Deviation"/"Severity" columns are intentionally marked UNMATCHED/N/A throughout, per Step 2–3
of the comparison methodology: these are not comparable measurements, and computing a percentage
deviation between them would misrepresent what changed. Depth RMSE units also differ in
interpretation here — this run's synthetic scene uses arbitrary world units, not the paper's
calibrated meters — compounding why a numeric deviation would be meaningless.)*

## Algorithmic Fidelity Checks (supplementary — not part of the reproducibility_score)

These are qualitative, code-level checks that the paper's own novel mechanisms behave as
specified, independent of final image quality:

| Check | Result |
|---|---|
| Resampled ray-box exclusion: do any of the 2-stage resampled points land inside the excluded bounding box? | **Pass** — 0/40 in isolated test, 0/N across all debug runs (see `notebooks/explore_arxiv_2607_24538.ipynb`, Section 2) |
| L_masked (Eq. 3–5) decreases during mask-conditioned training? | **Pass** — 0.0146 → 0.0036 over the debug run's inpainting rounds |
| Distillation phase 1 (field-level) and phase 2 (image-level) losses computed and both run? | **Pass** — both phases execute; phase 2 loss (0.138) is higher than phase 1 (0.016), consistent with it supervising full-image RGB+depth rather than per-sample field values (different loss scale, not a regression) |
| All 4 Table III ablation variants (w/o virtual-view-dist, w/o grid-inp, w/o mask-cond-train, field-distill-only) execute without crashing and produce distinguishable code paths? | **Pass** — all 5 variants (incl. full) ran; see `raw_debug_run_results.json` |
| Language field update (Sec. II-D) loss computed with geometry frozen? | **Pass** — only `lang_head` parameters received gradients (verified via `requires_grad` toggling in `LanguageFieldUpdater.update`) |

## Summary

This reproduction faithfully implements NEO's core algorithmic machinery (verified via the checks
above) but **cannot be numerically compared to the paper's reported results** because the real
dataset, inpainting model, and language-grounding model are all unavailable in this environment.
The reproducibility_score of 0.00 is a direct, honest consequence of zero matched (metric,
dataset, split) triples — not a claim that the implementation is broken.

## Root Cause Analysis

**Critical — Data mismatch (probability: High).** The paper's Tables I–III are exclusively
NEO-Dataset results; this run used a synthetic stand-in. No amount of debugging this repo's code
closes this gap — it requires either the real dataset (see `data/README_data.md`) or accepting
synthetic-scene numbers as a different, non-comparable benchmark.

**Significant — Pretrained-model substitution (probability: Medium-High).** Even with the real
dataset, swapping Stable Diffusion → OpenCV heuristic inpainting and CLIP/DINO → hash embedder
would still depress PSNR/SSIM specifically in the "Out"/newly-revealed regions (exactly where
this run's SSIM (Out) = 0.056 is worst), since classical inpainting cannot hallucinate plausible
new texture the way a diffusion model can.

**Moderate — Training convergence (probability: High, for the absolute low numbers observed).**
The debug run used 8-20 optimization steps per stage (a smoke test), not the thousands of steps
a real NeRF training run needs. The absolute PSNR (~11-15 dB) and SSIM (~0.04-0.46) values
measured are consistent with a severely undertrained NeRF, independent of any of the above gaps.

**Minor — Hyperparameter assumptions (probability: Low-Medium contribution).** λ_rec, λ_reg,
backbone width/depth, and the re-inpaint interval `n` are all `# ASSUMED` (SIR ambiguities
#1-#4, #6). None of these were tuned or swept; a real reproduction attempt should sweep them
once real data is available.

## Recommended Actions (most impactful first)

1. **Obtain the real NEO-Dataset** (or an equivalent robot-scanned tabletop scene set) — this is
   the single blocking issue for any meaningful metric comparison. See `data/README_data.md`.
2. **Swap in real Stable Diffusion + CLIP/DINO** on a machine with model-hub network access —
   both interfaces (`InpaintingBackend`, language embedding) are designed as one-line swaps.
3. **Run full (non-debug) training**: `python train.py --config configs/config.yaml` (no
   `--debug`) for a properly converged base NeRF before editing.
4. **Sweep the ASSUMED hyperparameters** (`lambda_rec`, `lambda_reg`, `mlp_hidden_dim`,
   `reinpaint_interval_n`) once (1) and (3) are addressed, using `run_ablation.py` as a template
   for a broader hyperparameter sweep harness.
