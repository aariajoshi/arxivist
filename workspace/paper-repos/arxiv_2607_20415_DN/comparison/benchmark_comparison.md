# Benchmark Comparison Report

**Paper**: Flux-Corrected Diagonal Frog: second order and positivity at all time steps
**Paper ID**: arxiv_2607_20415
**arXiv**: https://arxiv.org/abs/2607.20415 (v1, math.NA)
**Comparison Date**: 2026-07-26 (revised same-day after follow-up investigation)
**SIR Version Used**: 1 (overall confidence 0.84)
**Architecture Plan Version Used**: 1

**Note on provenance**: this comparison is self-generated -- the "user results" are this
session's own execution of `evaluate.py` against the generated repository. This report was
**revised after a follow-up investigation** requested by the user, to actually verify
(not just hypothesize about) the two deviations flagged in the first pass. That
investigation:

- **Found and fixed a second real bug** in the active-set solver: a redundant
  confirmatory solve that inflated the reported pattern-update count by +1 in every case
  (see Root Cause Analysis below).
- **Tested and refuted** the original "mesh-size artifact" hypothesis for the unlimited
  scheme's undershoot magnitude, via an explicit mesh sweep from n=51 to n=801. The paper's
  reported magnitude is not reproduced at *any* mesh size tested -- this remains a genuine,
  unresolved discrepancy, not an explained one.

Raw outputs are in `comparison/table*.csv` (regenerated after both fixes; see
`verification_log.md` for checksums).

---

## Reproducibility Score

| Score | Confidence | Metrics Compared | Matched |
|-------|------------|-----------------|---------|
| **0.71** / 1.0 | High | 12 | 9 (+3 unmatched, not reproduced this session) |

**Interpretation** (per ArXivist scale): 0.60-0.74 = **Partial reproduction -- review
moderate deviations**. The numeric score is unchanged from the first pass -- both flagged
items remain in the same deviation-severity bucket even after improvement, because the
scoring formula saturates once a deviation exceeds 50% -- but the *evidentiary basis* is
now substantially stronger: one deviation shrank via a confirmed bug fix (not a
hypothesis), and the other was actively tested and shown to be a genuine open question
rather than an assumed mesh artifact. The paper's core qualitative claims -- positivity,
conservation, 2nd-order accuracy, Chang-Cooper's high-Peclet degradation, and the
active-set solver's cost-transition point -- are all reproduced correctly and are not in
question.

---

## Metric Comparison Table

| Metric | Dataset | Paper Value | Our Value | Deviation | Severity |
|---|---|---|---|---|---|
| Coverage condition (b) holds (ever) | OU, n=51..401 | False | False | 0% | Excellent |
| FCDF-B spatial order, n=101 | OU benchmark | 1.78 | 1.777 | -0.17% | Excellent |
| FCDF-B spatial order, n=201 | OU benchmark | 1.80 | 1.803 | +0.17% | Excellent |
| FCDF-B spatial order, n=401 | OU benchmark | 1.59 | 1.585 | -0.31% | Excellent |
| FCDF-DC spatial order, n=201 (spot check) | OU benchmark | 1.90 | 1.897 | -0.16% | Excellent |
| FCDF-DC temporal order, first dt-refinement | OU benchmark, n=401 | 2.15 | 2.08 | -3.26% | Good |
| Active-set: pattern updates at gamma/gamma_pic=5 (transition) | Front, n=401 | 0 | 0 | 0% | Excellent |
| Active-set: pattern updates, gamma/gamma_pic in [0.01,2] | Front, n=401 | 1 | **2** (was 3 pre-fix) | +100% (was +200%) | Critical, but confirmed-not-mesh-related, root cause partly identified |
| Unlimited scheme min value, front/small-dt | Front, n=401 | -0.255 | **-0.077** (mesh sweep: -0.130 to -0.139 at n=51..201, does not reach -0.255 anywhere) | -70% at n=401 (worse than the n=201 report); **confirmed NOT a simple mesh-size effect** | Critical -- genuinely unresolved, flagged for follow-up |
| Max mass defect, all schemes/scenarios | OU + front, n=401 | <=2.2e-11 | <=1.05e-12 | N/A -- both at noise floor | Excellent (by judgment; see note) |
| Coverage condition (a) holds from n= | OU benchmark | 101 | 201 | off by one mesh level | Moderate |
| CC/FCDF-B error ratio, highest Peclet tested | Smooth advection | 21.9 (at Peh=3.79) | 18.3 (at Peh=8.44) | -16.4% (different Peh sampled) | Significant (confounded) |

**Unmatched (not reproduced this session -- see README "Known limitations")**:
- Table 6: long-time front-smearing order over 2000 steps (paper: FCDF-B order 1.25 at T=0.4)
- Table 9: active-set pattern-update count in the *coverage gap* (paper: 1 update at gamma=sqrt(gamma_pic*gamma0))
- Section 6.3 front-benchmark FCDF-B order at its finest mesh (paper: 1.47)

---

## Deviation Summary

| Severity | Count |
|----------|-------|
| Excellent (<=2%) | 6 |
| Good (2-5%) | 1 |
| Moderate (5-15%) | 1 |
| Significant (15-30%) | 1 |
| Critical (>30%) | 2 |
| Unmatched | 3 |

---

## Summary

The reproduction remains **strong on everything checkable in closed form**: spatial
convergence orders (Table 3) match the paper to within 0.3% at every tested mesh point, the
qualitative Table 7 finding (only the *unlimited* scheme goes negative, only on the front
benchmark, only at small dt) reproduces at every mesh size tried, and the active-set
solver's regime transition (0 pattern updates once gamma >= 5*gamma_pic) matches the
paper's own transition point exactly.

Two items remain genuinely open after this follow-up investigation -- this is an honest
downgrade from the first pass's "likely a mesh artifact" framing, not an upgrade:

1. **Active-set pattern-update count** (2 vs. paper's 1): **partially resolved**. A real
   counting bug was found and fixed (see below), closing about a third of the gap (3 -> 2).
   The residual 1-update difference was tested across n=101 to n=801 and is **not**
   mesh-dependent -- it persists identically at every mesh size. This points to a genuine,
   still-unidentified difference in exactly which clamp pattern our solver converges
   through on its first solve, versus the paper's. Not yet root-caused to a specific line
   of code.
2. **Unlimited-scheme undershoot magnitude** (-0.077 to -0.139 across meshes, vs. paper's
   -0.255): **hypothesis refuted, not resolved**. A mesh-size explanation was actively
   tested (sweep n=51/101/201/401/801) and found false -- the closest we get to the paper's
   magnitude is -0.139 at n=101, and magnitude *decreases* at both coarser (n=51) and finer
   (n=401, n=801) meshes, so there is no monotonic "match the paper's mesh and it'll agree"
   story. Since our operator construction was independently unit-tested to match the
   paper's Eqs. (5)/(7) exactly for this benchmark's uniform positive drift (no
   sign-changing-drift ambiguity applies here), the remaining gap likely reflects either an
   unstated experimental parameter (domain length, a different dt, or a different specific
   mesh) or a genuine discretization difference not yet identified. Flagged honestly as
   **unresolved**.

Both are now backed by real, executed investigation (mesh sweeps, a fixed bug, a locked-in
regression test) rather than a plausible-sounding guess.

## Root Cause Analysis

### Active-set pattern updates, gamma/gamma_pic in [0.01, 2] -- CONFIRMED bug fixed, residual gap open

**Bug found and fixed**: the original `ActiveSetSolver.solve()` checked for pattern
convergence *after* performing a solve with the newly-computed pattern, comparing it to the
*previous* iteration's pattern. This meant that even when a pattern was already
self-consistent (i.e., solving with it reproduced the same pattern), the code performed one
more redundant solve before recognizing convergence -- inflating the reported count by
exactly +1 in every run. Fixed in `schemes/active_set.py` by checking pattern stability
*before* solving, comparing the freshly-computed pattern against the pattern *actually used
in the most recent solve*; if they match, no new solve is needed. This is now locked in by
`tests/test_fcdf_dc_and_active_set.py::test_active_set_does_not_perform_redundant_confirmatory_solve`.

**Effect**: pattern-update count in this regime dropped from 3 to 2 at every mesh size
tested (101, 201, 401, 801 -- confirmed via a dedicated mesh sweep, see
`verification_log.md`). The mesh-independence of this result also rules out mesh size as an
explanation for the *residual* gap (2 vs. paper's 1).

**Residual gap (2 vs. 1), not yet explained**: possible remaining causes, in order of
plausibility:
1. **A genuinely different first clamp pattern.** Our solver initializes from the unlimited
   solution's implied pattern; if the paper's realization arrives at a self-consistent
   pattern on the very first attempt (1 solve total) while ours requires one real pattern
   change before stabilizing (2 solves total), the two algorithms are doing the same thing
   correctly but landing on the active set via a different path. This would not be a bug,
   just a different (still valid, per Proposition 5) trajectory to the same fixed point.
   Not yet confirmed -- would require inspecting the actual clamp pattern at each step,
   which this session did not have time to add as a diagnostic.
2. **A subtly different cap/budget definition.** Eq. (15)'s caps are re-derived here from
   `b` and `gamma`; a small difference in how ties (flux exactly at a cap boundary) are
   broken could shift which interfaces are flagged as clamped on the first pass.

### Unlimited scheme min value, front/small-dt -- hypothesis refuted, root cause still open

The original report attributed the -0.123 vs. -0.255 gap (at n=201) to mesh resolution and
suggested re-running at the paper's n=401. That rerun gave -0.077 -- a *smaller* magnitude,
the opposite of what the "finer mesh should look more like the paper's presumably-similar
setup" hypothesis predicted. A full mesh sweep (`n` in {51, 101, 201, 401, 801}) shows the
undershoot peaks around n=101 (-0.139) and shrinks in both directions from there -- meaning
no single "right" mesh choice within a plausible range reproduces -0.255. Since:
- our `A1`/`A2` operators are unit-tested to match the paper's Eqs. (5)/(7) exactly for
  this benchmark (uniform positive drift, no sign-changing-drift generalization in play),
- the benchmark parameters (mu=1, D=1e-4, plateau on [0.1,0.4], domain [0,1], T=0.2,
  dt=2e-4) are all taken directly from the paper's stated values,

the most likely remaining explanations are an **unstated experimental detail** (the paper
does not give the mesh size used for Table 7 specifically, unlike Table 8 which states
n=401) or a **genuine implementation difference not yet localized**. This is reported as an
open item, not swept under a plausible-sounding rug.

### Coverage condition (a) -- unchanged from first pass (not re-investigated this round)

At n=101 we measure `gamma_0=0.01024` against `gamma_pic=0.01000` (paper: `gamma_0=0.00984`
against the same `gamma_pic`), a ~2-3% difference right at the threshold crossing. Since
gamma_0 is bisection-measured rather than analytically derived (its closed form lives in an
unavailable companion paper -- SIR ambiguity #1), this remains the most likely explanation,
but tightening the bisection tolerance to confirm it was not re-tested this round (time
budget prioritized the two items the user specifically asked about).

---

## Hallucination Report Summary

See `hallucination_report.md` for the full report (updated to note the second bug fix).

| Type | Count | Critical |
|------|-------|---------|
| Structural | 0 | 0 |
| Parametric | 3 | 0 |
| Omission | 3 | 0 |

No structural hallucinations. Two real bugs have now been found and fixed via the test
suite across this reproduction's lifetime (FCDF-DC defect-flux sign error, and this
session's active-set redundant-solve counting bug) -- both caught by tests specifically
designed to check the paper's own claimed properties, not by accident.

---

## Recommended Actions

Prioritized by expected impact:

1. **Add a clamp-pattern diagnostic** to `ActiveSetSolver.solve()` (e.g. return the sequence
   of patterns tried, not just the count) to determine whether the residual active-set gap
   (2 vs. 1) is a genuinely different-but-valid trajectory or an actual remaining bug.
2. **Investigate the unlimited-scheme undershoot gap directly** rather than via mesh size:
   check whether a different dt, domain length, or T changes the magnitude toward -0.255;
   the mesh sweep already rules out mesh size as the lever to pull.
3. **Tighten the bisection tolerance** in `linear_windows/thresholds.py` near suspected
   threshold crossings (e.g. n=101 on the OU benchmark) to check the coverage-condition-(a)
   borderline mismatch.
4. **Implement Tables 4, 6, 9** in `evaluate.py` to close the 3 unmatched-metric gap.

---

## Implementation Notes

*From the SIR -- sections with confidence < 0.7 that may affect these results:*

- **Linear-window thresholds gamma_0/gamma_r (confidence 0.7)**: numerically bisected, not
  analytically derived -- directly implicated in the coverage-condition-(a) borderline
  mismatch (unchanged from first pass).
- **Active-set solver's general mixed-pattern nonsingularity above gamma_pic (confidence
  0.82, an open question in the paper itself)**: not implicated in either flagged deviation
  above (all runs converged with `converged=True`); the counting-convention bug was a
  separate, unrelated implementation defect, now fixed.

---

## Verification Log Summary

- Comparison run at: 2026-07-26T12:40:00Z (initial), revised 2026-07-26T13:20:00Z
  (follow-up investigation)
- Results source: self-generated (`evaluate.py` executed in this session)
- Two bugs found and fixed during this reproduction's lifetime; both locked in by
  regression tests
- Manual review still recommended for the two items flagged Critical above -- they are now
  well-characterized open questions, not resolved findings

Full audit trail in `verification_log.md`.
