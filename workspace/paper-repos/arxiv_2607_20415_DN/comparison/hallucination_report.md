# Hallucination Report — arxiv_2607_20415

**Comparison date**: 2026-07-26
**Scope**: audit of `paper-repos/arxiv_2607_20415/src/` against `sir.json` and
`architecture_plan.json`.

---

## Structural Hallucinations

**None found.** Every module in `architecture_plan.json → module_hierarchy` corresponds to
a real component described in the SIR (`architecture.components` or `mathematical_spec`),
and every file in `src/fcdf_diagonal_frog/` corresponds to a planned module. No component
was invented that isn't traceable to a specific paper section.

One item worth noting, not a hallucination but a documented *generalization*: the operator
assembly in `operators/df_operator.py` is built via a general interface-flux-differencing
construction (per-interface upwind direction chosen from the sign of the face-averaged
drift) rather than literally transcribing the paper's mu>0-only Eqs. (4)-(9) row by row.
This was a necessary, explicitly-flagged design decision (SIR ambiguity #4) to handle the
OU benchmark's sign-changing drift, which the paper itself does not give an explicit
per-node formula for. It is validated indirectly (exact conservation to 1e-10, and
near-exact reproduction of Table 3's convergence orders on that same sign-changing-drift
benchmark), so it is a *documented interpretation*, not a hallucination.

---

## Parametric Hallucinations

Three assumed hyperparameters, all flagged `# ASSUMED` in `configs/config.yaml` from the
start (i.e., before any comparison was run — not retrofitted to explain a deviation):

1. **`picard_tol: 1.0e-12`** (SIR implementation_assumption #6, confidence 0.5). The paper
   reports *achieved* residuals of ~1e-13 to 1e-15 (Tables 8-9) but never states a target
   tolerance. Our achieved residuals (Table 8 CSV: 1.2e-14 to 1.3e-12) land in a broadly
   similar range. **Not implicated** in any flagged deviation — the active-set pattern-count
   discrepancy is a counting-convention issue (see `benchmark_comparison.md` root cause
   analysis), not a tolerance issue, since all our runs report `converged=True` with tight
   residuals well below this tolerance.
2. **`active_set_max_pattern_updates: 25`** (safeguard for Proposition 5's open
   nonsingularity question above gamma_pic). Never triggered in any run — all active-set
   solves converged in ≤3 pattern updates, far below the cap.
3. **`zalesak_kappa: 2`** (the even budget split, SIR confidence 0.93 — this is actually the
   *only* split value the paper analyzes in Proposition 1 and all appendix proofs, so this
   is closer to a directly-stated paper value than a free assumption).

None of these three coincide with the Critical or Significant deviations flagged in
`benchmark_comparison.md`; those are attributed instead to mesh-size differences and a
convergence-counting convention (see Root Cause Analysis there).

---

## Omission Hallucinations

Three paper metrics/tables present in the SIR's `evaluation_protocol.reported_results` but
**not computed** in this session's `evaluate.py` run:

1. **Table 6** (long-time front smearing, 2000-step integration) — `evaluate.py` does not
   implement this table at all yet. Severity: **Minor**. Not a silent stub — the function
   simply does not exist; `README.md` documents this openly under "Known limitations."
   Suggested fix: add a `table6_long_time_smearing()` function following the same pattern
   as `table7_positivity_conservation()`, integrating to T∈{0.05,0.1,0.2,0.4} at fixed
   dt=2e-4 and recording the observed order at each horizon.
2. **Table 9** (active-set cost in the coverage gap, on the front problem across a mesh
   sequence) — same situation, not implemented. Severity: **Minor**. Suggested fix: reuse
   `ActiveSetSolver` at `gamma = sqrt(gamma_pic * gamma_0)` across the mesh sequence
   `[101, 201, 401, 801]`, following `table8_active_set_cost()`'s pattern.
3. **FCDF-B order at the front benchmark's finest mesh** (Section 6.3, paper value 1.47) —
   `evaluate.py`'s Table 5 only covers the *smooth* advection-diffusion Peclet sweep; the
   separate front-benchmark order study (a different sub-experiment in the same paper
   section) was not implemented as its own table function. Severity: **Minor**. Suggested
   fix: add a `table5b_front_order()` function running FCDF-B and Chang-Cooper on the front
   benchmark across a mesh-refinement sequence and reporting observed orders.

All three omissions are scope/time-boxing decisions made and disclosed in `README.md`
*before* this comparison was run, not discovered gaps papered over after the fact.

---

## Overall Assessment

No evidence of fabricated results, invented architecture components, or silently-stubbed
functionality that claims to work but doesn't. The three omissions are honestly documented
absences, not hallucinations in the sense of false claims. The parametric assumptions are
standard, low-risk defaults for unspecified numerical tolerances, disclosed upfront, and not
implicated in any of the deviations found during comparison.

## Update (follow-up investigation)

A second real bug was found and fixed during a follow-up investigation requested by the
user: the active-set solver (`schemes/active_set.py`) performed one redundant confirmatory
banded solve after its clamp pattern had already stabilized, inflating the reported
`pattern_updates` count by +1 in every run. This is now fixed and locked in by a regression
test (`test_active_set_does_not_perform_redundant_confirmatory_solve`). This is the second
real, test-caught bug found across this reproduction's lifetime (the first being the
FCDF-DC defect-flux sign error from the initial code-generation pass) — both are evidence
that the test suite is doing genuine verification work, not rubber-stamping.

The follow-up investigation also tested (and refuted) the hypothesis that the unlimited
scheme's undershoot-magnitude deviation was a mesh-size artifact. This is reported as an
honest **non-finding**: the hypothesis was wrong, the discrepancy remains open, and this is
stated plainly in `benchmark_comparison.md` rather than replaced with a new unverified guess.
