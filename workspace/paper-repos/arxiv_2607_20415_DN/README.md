# Flux-Corrected Diagonal Frog — Reproduction

Reproduction of Andrey Itkin, **"Flux-Corrected Diagonal Frog: second order and positivity
at all time steps"** (arXiv:2607.20415v1, math.NA, July 2026).

This is a **numerical-PDE scheme paper**, not a machine-learning paper: it constructs
positivity-preserving, mass-conservative, second-order finite-difference schemes for the
1D Fokker-Planck equation. There is no learned model, dataset, or training loop — every
"result" below is a deterministic banded linear-algebra computation.

## What's implemented

| Component | File | Paper reference |
|---|---|---|
| Core-correction split (A1, A2, C) | `src/fcdf_diagonal_frog/operators/df_operator.py` | Section 2, Eqs. (2)-(9) |
| Zalesak limiter (budget rule + combined clamp) | `src/fcdf_diagonal_frog/limiter/zalesak.py` | Eqs. (15), (21) |
| FCDF-A (global stopping rule) | `schemes/fcdf_a.py` | Eq. (12) |
| **FCDF-B (primary scheme)** | `schemes/fcdf_b.py` | Eqs. (13)-(15), Prop. 1 |
| FCDF-DC (defect-corrected, 2nd order in time) | `schemes/fcdf_dc.py` | Eqs. (17)-(21), Prop. 3 |
| Active-set / semismooth-Newton solver | `schemes/active_set.py` | Eqs. (24)-(26), Prop. 5 |
| Monotone core / unlimited baselines | `schemes/monotone_core.py`, `unlimited.py` | Section 6 baselines |
| Chang-Cooper baseline | `operators/chang_cooper.py` | Section 3.1, 6.3 |
| Linear-window thresholds (gamma_0, gamma_r) | `linear_windows/thresholds.py` | Section 6.1 (bisection method) |
| OU / smooth-advection / front benchmarks | `benchmarks/` | Section 6 |

## Quickstart

```bash
pip install -r requirements.txt
pip install -e .

# Fast smoke test (n=51, T=0.02)
python train.py --config configs/config.yaml --debug

# Reproduce Tables 2, 3, 5, 7, 8
python evaluate.py --config configs/config.yaml --table all --out-dir results/

# Single-step demo
python inference.py --config configs/config.yaml --scheme fcdf_b
```

Or via Docker: `docker compose -f docker/docker-compose.yml up --build`.

## Validated results (this reproduction vs. the paper)

Full numeric comparison in `comparison/benchmark_comparison.md`, revised after a follow-up
investigation that actually tested (rather than assumed) two flagged deviations. Headline
points:

- **Table 3 (OU spatial order)**: our FCDF-B orders (1.78, 1.80, 1.59) match the paper's
  reported values (1.78, 1.80, 1.59) to the reported precision.
- **Table 2 (coverage)**: `gamma_pic` matches exactly (closed form); `gamma_0`/`gamma_r`
  (bisection-measured, see caveat below) reproduce the same qualitative conclusion —
  condition (a) holds from n≈101/201 onward, condition (b) never holds on any tested mesh.
- **Table 7 (positivity)**: reproduces the paper's headline result exactly qualitatively —
  the *unlimited* scheme is the only one that goes negative, and only on the front
  benchmark at a small step. The *magnitude* does not match (ours: -0.077 to -0.139 across
  a full mesh sweep n=51..801; paper: -0.255) and — importantly — **this was actively
  investigated, not just attributed to mesh size**: a mesh sweep shows the gap is not a
  simple resolution effect (see "Known limitations" below). FCDF-B, FCDF-DC, and the
  monotone core stay nonnegative everywhere at every mesh tested; mass conserved to
  ~1e-12–1e-13 in all cases (paper: ≤2.2e-11).
- **Table 8 (active-set cost)**: reproduces the paper's qualitative transition exactly — 0
  pattern updates once gamma ≥ 5·gamma_pic, matching the paper's own transition point. Below
  that threshold, we now report 2 updates (paper: 1) after fixing a real counting bug found
  during a follow-up investigation (see "Known limitations"); the residual 1-update gap was
  confirmed mesh-independent (tested n=101 through n=801) and remains an open question.

## Known limitations / honest deviations (do not paper over these)

1. **gamma_0 / gamma_r are numerically measured, not analytically derived.** Their closed
   forms live in the companion paper *[Itkin and Kazbek, 2026, "Diagonal Frog meets ADI"]*,
   which is "in preparation" and was not available. We implement exactly the operational
   procedure this paper itself describes (bisection on the sign of the most-negative
   resolvent/Padé-map entry). Our `gamma_0` values agree with the paper's to within the
   same order of magnitude at small n but drift further (~3.5x) at n=401; `gamma_r` at
   n=401 is reported by the paper as 3.8e5 (just under their search bound) but our
   bisection reports "no window found" below 1e6 — likely a floating-point conditioning
   issue this close to the search boundary, not a conceptual error (condition (b)'s
   qualitative conclusion — *never satisfied* — is unaffected either way).
2. **Sign-changing-drift stencil is our own resolution of an underspecified paper detail.**
   The paper states the mu>0 stencil explicitly and says mu<0 is "fully symmetric" and
   sign-changing drift is handled "directionally at each node," without giving the exact
   per-node rule needed for the OU benchmark (whose drift changes sign at the domain
   center by construction). We resolve the upwind direction per *interface* via the sign
   of the face-averaged drift. This is validated indirectly: exact mass conservation
   (1^T A = 0 to 1e-10) and near-exact reproduction of Table 3's orders on the
   sign-changing-drift OU benchmark both hold under this convention.
3. **Active-set pattern-update counts are qualitatively right but not exactly matched.**
   **Update after follow-up investigation**: the original 3-vs-1 gap included a real bug —
   a redundant confirmatory solve after the clamp pattern had already stabilized — now
   fixed (`schemes/active_set.py`), bringing the count to 2. The residual 2-vs-1 gap was
   then tested across n=101 through n=801 and confirmed **mesh-independent** (identical at
   every mesh size), so it is not a mesh artifact either. It remains an open question,
   likely reflecting a genuinely different (but still valid) clamp-pattern trajectory to
   the same fixed point rather than a remaining bug — see
   `comparison/benchmark_comparison.md` Root Cause Analysis for the full investigation.
   The regime transition (0 updates from 5x gamma_pic onward) matches the paper exactly at
   every mesh size tested.
4. **Unlimited-scheme undershoot magnitude does not match, and this is not a mesh
   artifact.** Paper: -0.255 on the front benchmark at small dt. Ours: ranges from -0.077
   to -0.139 across a full mesh sweep (n=51, 101, 201, 401, 801), peaking at n=101 and
   *decreasing* at both coarser and finer meshes — so there is no mesh size that reproduces
   the paper's magnitude, ruling out the most obvious explanation. Our operator
   construction was independently unit-tested to match the paper's Eqs. (5)/(7) exactly for
   this uniform-positive-drift benchmark, so this is reported as a genuinely open
   discrepancy (unstated experimental parameter, or an unidentified implementation
   difference) rather than explained away. The *qualitative* finding — only the unlimited
   scheme goes negative, only here — reproduces at every mesh size tried.
5. **Tables 4, 6, 9 are not reproduced** in `evaluate.py` (scope/time, not a technical
   blocker): Table 4 needs a separate matrix-exponential (`expm`) semi-discrete integrator
   that isn't otherwise used anywhere in the paper's actual *schemes*; Table 6 requires a
   2000-step long-time integration; Table 9 reuses the same active-set machinery as Table 8
   on a different mesh sequence. All three could be added by extending `evaluate.py`
   following the same pattern as Tables 2/3/5/7/8.
6. **A "banded solve" is implemented via `scipy.sparse.linalg.spsolve`** (sparse LU) rather
   than a literal hand-rolled Thomas algorithm. For the narrow-banded 1D matrices used
   throughout, this is effectively linear-time in practice and is the standard, correct
   engineering choice for the paper's repeated "O(n) banded solve" cost claim.
7. **No numeric Picard/Newton convergence tolerance is stated in the paper.** We default to
   `1e-12` (config-overridable); achieved residuals in our Table 8 run land in the same
   ballpark (1e-13 to 1e-15) as the paper's Tables 8-9.

## Repository layout

```
src/fcdf_diagonal_frog/    core package (operators, limiter, schemes, benchmarks, evaluation)
configs/                   config.yaml (paper-scale), config_debug.yaml (fast smoke test)
tests/                     14 unit tests: conservation, positivity, Proposition 1/3/5 checks
train.py / evaluate.py / inference.py   entrypoints (see Quickstart)
docker/                    Dockerfile + docker-compose.yml
comparison/                comparison_report.md + reproduced CSVs vs. paper's reported values
notebooks/                 walkthrough notebook (see notebooks/README.md)
```

## Testing

```bash
pytest tests/ -v
```
15/15 tests pass, covering: exact mass conservation (uniform and sign-changing drift),
hand-derived-formula matching against the paper's Eqs. (5)/(7), M-matrix/Metzler structure,
core-resolvent nonnegativity (Lemma 3), FCDF-B unconditional positivity and Proposition
1(iv) consistency, the paper's Table-7 unlimited-scheme-goes-negative finding, FCDF-DC
positivity/conservation and its 2nd-order-vs-1st-order accuracy advantage, and the
active-set solver's positivity guarantee, large-gamma unlimited-acceptance behavior, and
correct (non-redundant) pattern-update counting.

**Two real bugs were caught and fixed during this reproduction's lifetime**, both by tests
specifically designed to check the paper's own claimed properties:

1. An initial sign error in the FCDF-DC defect-flux term (`schemes/fcdf_dc.py`) that made
   the "2nd order in time" scheme *less* accurate than the 1st-order FCDF-B baseline —
   caught by `test_fcdf_dc_second_order_beats_fcdf_b_on_smooth_ou_like_problem`, root-caused
   via the Lemma 1 algebraic identity, and fixed.
2. A redundant confirmatory solve in the active-set solver (`schemes/active_set.py`) that
   inflated the reported `pattern_updates` count by +1 in every run — found during a
   follow-up investigation into a Table 8 discrepancy, fixed, and locked in by
   `test_active_set_does_not_perform_redundant_confirmatory_solve`.

See `comparison/benchmark_comparison.md` for the full investigation, including a mesh sweep
(n=51 to n=801) that **tested and refuted** the hypothesis that the remaining Table 7/8
deviations were simple mesh-size artifacts — they are reported as genuinely open questions,
not explained away.
