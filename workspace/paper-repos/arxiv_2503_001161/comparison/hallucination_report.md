# Hallucination Report — SGDD (arxiv_2503_001161)

**Comparison Date**: 2026-07-26
**SIR version**: 1
**Scope**: verified algorithm (Part A) + measured synthetic accuracy (Part B). Real-data
metrics cited, weight-gated.

## Summary

| Type | Count | Critical | Notes |
|------|-------|----------|-------|
| Structural | 0 | 0 | Algorithm 1 + Eq 10/11/12/13 match the paper mechanically |
| Parametric | 0 | 0 | Table 7 hyperparameters (K, MH_T, H, N, schedule) taken verbatim |
| Omission | 0 | 0 | competitive ranking not reproduced but explicitly disclosed, not omitted |

## Structural (0)

Re-derived from the paper and checked mechanically:

- **Split Gibbs alternation** (Algorithm 1, Sec 3.4): likelihood step (MH) then prior step
  (partial reverse diffusion), annealed over {eta_k}. ✅
- **Potential** (Eq 10): `D = Hamming(x,z) * log[(1+(N-1)e^{-eta})/((N-1)(1-e^{-eta}))]`,
  diverging as eta→0. ✅ verified numerically.
- **Prior-step identity** (Eq 11 == Eq 12): `beta~(eta) = (N-1)/N (1-e^{-eta})` equals the
  forward corruption `beta_t` at `sigma=eta`, so the prior step is a partial denoise. ✅ exact.
- **Likelihood step** (Eq 13): MH on `exp(-f(z;y) - D(x,z;eta))`, gradient-free. ✅ invariance
  verified by detailed balance.
- **Geometric schedule** (Sec C.3): eta_min=1e-4, eta_max=20. ✅

## Parametric (0)

All hyperparameters verbatim from Table 7: synthetic K=10 / MH_T=10 / H=20 / N=50; DNA K=50 /
MH_T=200 / β∈{30,50}; MNIST K∈{50,100} / MH_T∈{2000,5000}; music K=100 / MH_T=5000; Euler H=20
throughout. No value fabricated.

## Omission (0) — with one disclosed scope limitation

The reproduction covers the full algorithm and the synthetic accuracy study. Two parts are
deliberately **not** reproduced, and both are stated openly (so neither is a hidden omission):

1. **Competitive ranking vs DPS/SMC/SVDD-PM (Table 2).** Our minimal synthetic task is too easy
   to reproduce the baselines' degenerate-to-prior failure; reproducing the ranking needs the
   paper's undisclosed multivariate forward model. We did **not** fabricate a favorable baseline.
2. **Real-data metrics (Tables 3-5).** Gated on the SEDD-90M priors + task oracles. Cited as the
   paper's, labeled gated.

## Notes

- The synthetic Hellinger/TV are measured against the analytic true posterior; the convergence
  trend toward the paper's 0.149 is real and seed-fixed.
- The most important honesty point: SGDD's algorithm is **verified**, but the headline
  *comparative* claim is **not** reproduced here, and we say so plainly rather than manufacture
  a win on an easy task. The gap is traced to task fidelity (undisclosed multivariate f), not to
  any defect in the sampler.
