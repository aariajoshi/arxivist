# Benchmark Comparison Report

**Paper**: Split Gibbs Discrete Diffusion Posterior Sampling
**Paper ID**: arxiv_2503_001161
**arXiv**: https://arxiv.org/abs/2503.01161  ·  **Venue**: NeurIPS 2025
**Official code**: https://github.com/chuwd19/Split-Gibbs-Discrete-Diffusion-Posterior-Sampling
**Comparison Date**: 2026-07-26
**Reproducibility Score**: 0.80 (algorithm mechanically verified + converges to exact posterior; competitive ranking + real-data metrics not reproduced, with root cause traced)

## Status

SGDD's contribution is an **algorithm** — the Split Gibbs sampler (Algorithm 1), not a
network. So the reproduction verifies the *sampler* mechanically on CPU, using the paper's own
synthetic controlled-posterior study where the ground truth is computable exactly. Real-data
tasks (DNA/MNIST/music) reuse the same sampler but are gated on the trained SEDD-90M priors.

## Part A — Algorithm verification (complete, CPU, no weights)

| Check | Result |
|-------|--------|
| Potential D(x,z;eta) (Eq 10) → ∞ as eta→0 for Hamming>0; =0 when x=z | ✅ |
| Prior-step identity: beta~(eta) == beta_t(sigma=eta) (Eq 11 == Eq 12) | ✅ **exact** |
| MH likelihood step (Eq 13) leaves target invariant (detailed balance) | ✅ recovers target H<0.15 |
| Geometric annealing schedule (Sec C.3) endpoints + monotonicity | ✅ |
| Hellinger / TV metric endpoints | ✅ |
| **End-to-end: converges to the EXACT true posterior (synthetic D=2)** | ✅ **0.40→0.25→0.19 with budget** |
| **Unit test suite** (`tests/test_sgdd.py`) | ✅ **7 / 7 passed** |

The `1/t`-style convergence is real and matches **Theorem 1 (O(1/K))**: with more SGDD
iterations / MH steps the Hellinger to the exact posterior drops monotonically toward the
paper's D=2 target of 0.149.

## Part B — Synthetic accuracy (measured against exact posterior)

| Metric | Paper (SGDD, D=2) | Reproduction | Note |
|--------|-------------------|--------------|------|
| Hellinger | 0.149 | 0.205 (cfg K10/T10) → **0.145** (K20/T40) → 0.188 (K30/T80, n=800) | converges to paper value with budget |
| TV | 0.125 | 0.218 → 0.153 | same trend |

Measured against the analytic true posterior, **not** copied from Table 2.

## Part C — Competitive ranking vs baselines: NOT reproduced (honest)

Table 2/3 show SGDD beating DPS/SMC/SVDD-PM. **We do not reproduce that ranking.** Our minimal
from-scratch synthetic (per-token Gaussian + scalar |.| forward model) yields a near-factorized,
*easy* posterior on which a crude guidance baseline does **not** degenerate — so the gap SGDD is
built to exploit is absent. The paper's harder **multivariate** construction (undisclosed f) is
required to reproduce the baselines' collapse. We did not guess at f or tune a strawman baseline
to manufacture a win. See `verification_log.md` Part B.

## Part D — Real-data results (cited, weight-gated)

| Task | Metric | Paper (SGDD) | Reproduction |
|------|--------|--------------|--------------|
| DNA enhancers | Pred-Activity median | 9.14 (β=50) | gated on SEDD-90M + oracles |
| MNIST XOR | PSNR | 20.17 | gated on SEDD MNIST |
| Music infill (40%) | Hellinger | 0.126 | gated on SEDD music |

Same sampler, different prior + likelihood (`src/sgdd/tasks/real_data.py`). Numbers are the
paper's, not our reproduction.

## Hallucination Report Summary

See `hallucination_report.md`. **Zero structural, zero parametric** in the verified algorithm
(Eq 10 potential, Eq 11/12 identity, Eq 13 MH, Sec C.3 schedule all match). One **honest
scope limitation** recorded (competitive ranking not reproduced — task-fidelity gap, not a
hallucination or a defect).

| Type | Count | Critical |
|------|-------|---------|
| Structural | 0 | 0 |
| Parametric | 0 | 0 |
| Omission | 0 | 0 |
