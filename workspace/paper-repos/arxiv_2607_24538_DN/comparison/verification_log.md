# Verification Log
**Paper ID**: arxiv_2607_24538
**Comparison run timestamp**: 2026-07-29T00:00:00Z (session date; see conversation for exact run times)

## Artifact versions used
- SIR version: 1 (`sir-registry/arxiv_2607_24538/sir.json`, overall confidence 0.69)
- Architecture plan version: 1 (`sir-registry/arxiv_2607_24538/architecture_plan.json`)

## Input to this comparison
No external user-supplied results file was provided. Instead, this comparison audits
**self-generated verification runs** produced while building and smoke-testing the repo in
Stage 4/5 (debug-scale, synthetic scene — see `README.md` Reproducibility Notes). These are
recorded verbatim in `comparison/raw_debug_run_results.json`.

- SHA256 (`raw_debug_run_results.json`): `cee36f0b516d5788277947dcef1b56a6b3d9649dccd84b9a745951a790e45b9e`

## Paper metrics available for matching
16 total reported results extracted into `sir.json → evaluation_protocol.reported_results`:
- 8 from Table I (object removal, Full/Out × PSNR/SSIM/RMSE/E_rep)
- 8 from Table II (pick-and-place, Out/In × PSNR/SSIM/RMSE/E_rep, plus non-primary rows)

## Matching outcome
- Metrics compared: 16
- Metrics matched (same metric + dataset + split): **0**
- Reason: run used `synthetic_toy_scene (debug)`, paper's metrics are all
  `NEO-Dataset (5 or 6 scenes)` — no dataset-level match possible.

## Runs actually executed during this session (audit trail)
1. `train.py --config configs/config.yaml --debug --dry-run` — component construction check, no training
2. `train.py --config configs/config.yaml --debug --out-dir runs/base_nerf_debug` — 20 steps, loss 0.191 → 0.075
3. `edit.py --checkpoint runs/base_nerf_debug/base_nerf.pt --prompt "move the soup can" --motion 0.4,0.0,0.3,0,0,0 --out-dir runs/edit_debug --debug` — full 5-step pipeline (localize → resample → mask-conditioned train → distill → language update), all steps completed and logged
4. `evaluate.py --edited-checkpoint runs/edit_debug/student_nerf.pt --edit-metadata runs/edit_debug/edit_metadata.json --out-json results/metrics_debug.json --debug` — Full/Out/In PSNR/SSIM/depth-RMSE computed
5. `inference.py --checkpoint runs/base_nerf_debug/base_nerf.pt --prompt "move the lego brick"` — localization + grasp-stub candidates
6. `run_ablation.py --config configs/config.yaml --out-dir runs/ablation_debug --debug --variants full` then `--variants wo_virtual_view_dist wo_grid_inp wo_mask_cond_train field_distill_only` — all 5 Table III variants executed
7. `notebooks/reproduce_arxiv_2607_24538.ipynb` — executed via `nbclient`, 0 errors, loss-decrease assertion passed
8. `notebooks/explore_arxiv_2607_24538.ipynb` — executed via `nbclient`, 0 errors, resampling-exclusion assertion passed (0/40 points inside excluded box)

## Bugs found and fixed during this verification (real, not hypothetical)
1. **`data/synthetic_scene.py`**: tuple-unpacking bug (`for _, _, c in ...` reused `_` across two
   different positions in the tuple), causing a numpy "ambiguous truth value" crash on the very
   first `generate_scene()` call. Fixed by using distinct variable names and `next()`.
2. **`edit.py`**: full-dataset (131,072-ray) forward passes without chunking attempted to
   allocate ~13GB of MLP activations on this sandbox's single-core CPU (only 3.9GB RAM available),
   causing the process to be OOM-killed. Fixed by adding a chunked-rendering wrapper
   (`RENDER_CHUNK=4096`) used everywhere a full ray set is rendered at once.
3. **Tractability, not a correctness bug**: initial `--debug` settings (8 training views at full
   128×128 resolution) were measured at ~25s per 4096-ray chunk on this CPU — far too slow for a
   "quick smoke test." Reduced debug-mode image resolution to 24×24-32×32 and view counts to 4,
   documented inline with the measured per-chunk timing that motivated the change.

## User-reported config modifications
None — no external user run was supplied for this comparison; all runs used
`configs/config.yaml` defaults with `--debug` overrides as documented above.

## Confidence in this verification log
**High** — every run listed above was actually executed in this session (not simulated or
inferred), with console output captured and cross-checked against the JSON artifacts saved
alongside this report.
