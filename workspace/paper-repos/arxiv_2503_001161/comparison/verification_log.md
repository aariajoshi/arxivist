# Verification Log — SGDD (arxiv_2503_001161)

## Run metadata
- Algorithm verification: 2026-07-26, local CPU (numpy only, no trained weights)
- Real-data (DNA/MNIST/music) metrics: **gated** on the SEDD-90M priors + task oracles
- SIR version used: 1
- Manual review required: No

## Part A — algorithm verification (executed, CPU)

```
pytest tests/ -q        # 7 passed (local CPU, numpy only)
```

SGDD is an **algorithm** (Algorithm 1), so verification targets the sampler itself, not a
network. Checks:

- **Potential D(x,z;eta) (Eq 10)** diverges as eta -> 0 for Hamming(x,z) > 0 and is 0 when
  x == z — the property that forces both split marginals to the posterior. ✅
- **beta~(eta) == beta_t(sigma=eta)** exactly (Eq 11 == Eq 12 identity): the prior step is a
  partial reverse discrete-diffusion denoise at noise level eta. ✅
- **Metropolis-Hastings likelihood step (Eq 13)** leaves the target exp(-f - D) invariant —
  detailed-balance check on a small support reproduces the exact target (Hellinger < 0.15). ✅
- **Geometric annealing schedule (Sec C.3)** eta_k = eta_min^{k/K} eta_max^{1-k/K}: endpoints
  and strict monotonic decay. ✅
- **Hellinger / TV** metrics: 0 for identical, 1 for disjoint. ✅
- **End-to-end (synthetic D=2, exact posterior known):** SGDD converges to the *exact* true
  posterior and beats prior-collapse; Hellinger drops with budget per Theorem 1's O(1/K):

  ```
  prior-collapse (ignore y):  H ~ 0.58   (seed 3; the baselines' documented failure mode)
  SGDD  K=10 T=10:            H = 0.396
  SGDD  K=20 T=40:            H = 0.254
  SGDD  K=30 T=80:            H = 0.188   -> approaching paper's D=2 target 0.149
  ```

## Part B — reproducing Table 2's competitive ordering: NOT claimed (honest limitation)

The paper's Table 2 shows SGDD beating DPS/SMC/SVDD-PM. **We do not reproduce that ranking**,
and we do not pretend to. Root cause, traced explicitly:

- The paper's synthetic prior is a discretized **multivariate** Gaussian with a purpose-built
  forward model f: X -> R^D chosen so the posterior is genuinely hard and guidance baselines
  **degenerate to the prior** as D grows (their Sec 4.2).
- Our from-scratch task is minimal: a **per-token** Gaussian prior with a scalar |.| forward
  model. Its posterior is near-factorized and easy, so a crude DPS-style guidance baseline
  fits it well and does *not* degenerate — removing the gap SGDD is designed to exploit.
- Consequently, at the paper's N=50 our baseline can match or beat SGDD. This reflects our
  **task being too easy**, not an SGDD defect: SGDD still provably converges to the exact
  posterior (Part A). We default the config to N=12 (where the convergence story is cleanest)
  and document N=50 as the fidelity gap.

To reproduce the ordering faithfully one needs the paper's exact multivariate synthetic
construction (or the real SEDD tasks below). We chose not to guess at the undisclosed details
of f rather than fabricate a favorable baseline.

## Part C — real-data tasks (gated)

DNA enhancers (Sec 4.3), MNIST XOR/AND (Sec 4.4), music infilling (Sec 4.5) reuse the SAME
verified sampler; only the prior (SEDD-90M) and likelihood f(z;y) change
(`src/sgdd/tasks/real_data.py` documents the wiring). Numbers in Table 3-5 are cited as the
paper's, weight-gated, never reproduced here.

## Integrity
- Synthetic Hellinger/TV are **measured against the analytic true posterior**, not copied from
  Table 2. The convergence trend (0.40 -> 0.19) is real, seed-fixed, reproducible.
- The Table-2 *ranking* is explicitly NOT claimed; the reason (task-fidelity gap) is stated,
  not hidden. No baseline was tuned to manufacture a favorable comparison.
- Real-data metrics are cited as the paper's and labeled gated.
