# Architecture Plan Summary — arxiv_2607_20415
## Flux-Corrected Diagonal Frog: second order and positivity at all time steps (Itkin, 2026)

## 1. Framework
**Python + NumPy/SciPy** (not PyTorch/JAX/TF — this is a numerical PDE scheme paper, not
a neural network). All linear solves are 1D banded systems, matching the paper's repeated
claim of "O(n) banded solve" cost, so `scipy.linalg.solve_banded` is the direct implementation
target. No GPU, no autograd, no datasets to download — everything is generated synthetically
(OU process, smooth advection-diffusion, unresolved front) exactly as in the paper's Section 6.

## 2. Module Hierarchy (15 files)
```
src/fcdf_diagonal_frog/
├── operators/
│   ├── grid.py            Grid1D: uniform mesh, cell Peclet number
│   ├── df_operator.py     DFOperator: assembles A1 (M-matrix core), A2 (full 2nd-order),
│   │                      C = A2-A1 (antidiffusive correction), banded core solves
│   └── chang_cooper.py    ChangCooperOperator: exponentially-fitted baseline
├── limiter/
│   └── zalesak.py         ZalesakLimiter: Eq.15 budget rule / Eq. A.2 clamp form / Eq.21 combined clamp
├── schemes/
│   ├── fcdf_a.py           Global-stopping-rule Picard (Eq. 12)
│   ├── fcdf_b.py           Per-interface Zalesak-limited Picard (Eqs.13-15) — PRIMARY scheme
│   ├── fcdf_dc.py           Defect-corrected 2-stage 2nd-order-in-time (Eqs.17-21)
│   ├── active_set.py        Semismooth-Newton active-set solver (Eqs.24-26)
│   ├── monotone_core.py      Backward Euler on A1 alone (1st-order floor baseline)
│   └── unlimited.py          Backward Euler on A2 alone (positivity-failure baseline)
├── linear_windows/
│   └── thresholds.py        gamma_0, gamma_r via bisection on resolvent/Pade sign (Section 6.1 method)
├── benchmarks/
│   ├── ou_process.py         OU exact solution, sign-changing drift
│   └── advection_diffusion.py Smooth Gaussian benchmark + unresolved-front benchmark
├── evaluation/
│   └── metrics.py            L1 error, observed order, mass defect, positivity check
└── utils/
    └── config.py             YAML config loading
```

## 3. Tensor Flows
Three core forward-pass specs are fully written out in `architecture_plan.json`:
- FCDF-B single implicit step (the Picard sweep of Proposition 1)
- FCDF-DC one full time step (predictor + defect-corrected corrector, Proposition 3)
- Active-set solve (unlimited-solve-first, then pattern-update loop, Proposition 5)

## 4. Config Schema
`configs/config.yaml` exposes: scheme selection, Zalesak kappa split (=2, SIR confidence 0.93),
Picard tolerance (ASSUMED 1e-12 — not stated numerically in the paper, SIR confidence 0.5),
active-set pattern-update cap (ASSUMED 25, safeguard for the paper's own open nonsingularity
question above gamma_pic), dt/T_horizon, and all three benchmark parameter sets (OU, front,
smooth-advection Peclet sweep) taken directly from Section 6.

## 5. Dependencies
numpy, scipy (banded solves + bisection), matplotlib (Figures 1-2), pyyaml (config), pandas
(Table 2-9 export). Dev: pytest, pytest-cov, black, ruff. No CUDA, no torch/jax/tf.

## 6. Entrypoints
- `train.py` — repo-template name kept for consistency; there is no training. It runs one
  scheme through one benchmark to horizon T and saves diagnostics (the PDE-solver analogue of
  a training run).
- `evaluate.py` — reproduces Tables 2-9 and Figures 1-2.
- `inference.py` — single implicit step / single full integration for a user-supplied IC.

## 7. Docker
`python:3.11-slim` base, `build-essential` + `libopenblas-dev` for scipy's banded solver
backend, default CMD runs the full evaluation suite.

## 8. Risk Assessment (5 risks, all Medium/Low — see architecture_plan.json for full detail)
| Risk | Severity | Mitigation |
|---|---|---|
| gamma_0/gamma_r can't be cross-checked against unavailable companion paper | Medium | Wide bisection bracket, report booleans not just exact digits |
| Active-set nonsingularity above gamma_pic is an open question in the paper itself | Medium | Guard every banded solve, fallback/report non-convergence rather than crash |
| Sign-changing-drift stencil not spelled out for OU benchmark | Medium | Per-node upwind by sign(mu), unit-test conservation (1^T A2=0) and core M-matrix property |
| No stated Picard/Newton tolerance | Low | Default 1e-12, expose in config, sanity-check against paper's reported ~1e-13 to 1e-15 residuals |
| Sub-cell boundary-mass regime falls outside nodal L1 | Low | Document as known limitation; none of the reproduced benchmarks enter this regime |

## Handling of low-confidence SIR sections
- `training_pipeline` (SIR confidence 0.55): repurposed as the time-marching/solve config, not
  an ML training loop — documented explicitly in `config.yaml` comments.
- Active-set solver (SIR confidence 0.82, open nonsingularity question): implemented behind a
  swappable interface with a hard iteration cap and explicit `converged` flag, never silently
  assumed to succeed.
- Linear-window thresholds (SIR confidence 0.7, imported from unavailable companion paper):
  implemented as a numerically-measured quantity (bisection), flagged in code comments as
  "measured, not derived here."
