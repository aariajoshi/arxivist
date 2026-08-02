# Benchmark Comparison Report

**Paper**: Quantum Cryptanalysis on IBM Quantum Hardware (arxiv_2607_18340)
**arXiv**: https://arxiv.org/abs/2607.18340
**Comparison Date**: 2026-07-22
**SIR Version Used**: 1

---

## Reproducibility Score

| Score | Confidence | Metrics Compared | Matched |
|-------|------------|-------------------|---------|
| **0.704** / 1.0 | medium | 6 | 5 |

**Interpretation**: 0.65-0.85 = Good reproduction with minor deviations.

**Read this before the number surprises you** (in a good way, this time): this
paper is structurally different from a typical ML benchmark. Its core,
*disclosed* content (five textbook quantum algorithms) reproduces essentially
perfectly. Its one *undisclosed* component (a withheld noise-mitigation
technique) cannot be tested at all and is marked `unmatched`, not scored as a
failure. And testing surfaced a genuine, verifiable error in the paper's own
stated CBC-MAC forgery formula.

---

## Metric Comparison Table

| Metric | Paper Value | Repo Value | Deviation | Severity |
|--------|-------------|------------|-----------|----------|
| Bernstein-Vazirani single-query success | 1.0 (claimed) | 1.0 (achieved) | 0% | ✅ Excellent |
| Grover iteration count matches (π/4)√(2ⁿ/M) | 1.0 | 1.0 | 0% | ✅ Excellent |
| Even-Mansour clean/rank-1 at n≤5 | 1.0 | 1.0 (achieved up to n=8) | 0% | ✅ Excellent |
| 3-round Feistel clean recovery, block 6/8 | 1.0 | 1.0 | 0% | ✅ Excellent |
| CBC-MAC period formula `s=Ek(a)⊕Ek(b)` correctness | 1.0 (claimed) | 0.0 (verified false) | -100% | 🔴 Critical |
| Even-Mansour hybrid recovery, n=6-10, real hardware | rank 63/1023 | rank 1 (noiseless sim) | — | ⬜ Unmatched* |

\* Marked unmatched, not scored numerically: the paper's n=6-10 result depends
on a withheld technique run on real noisy hardware; this repo runs on a
noiseless simulator by default (no hardware access available), where clean
recovery happens trivially for a completely different reason (absence of
noise, not the withheld technique). Comparing these numbers directly would be
comparing two different experiments, not measuring reproduction fidelity.

---

## Deviation Summary

| Severity | Count |
|----------|-------|
| ✅ Excellent | 4 |
| 🔴 Critical | 1 |
| ⬜ Unmatched | 1 |

---

## Root Cause Analysis

### The 4 "Excellent" matches

All four disclosed, testable algorithmic claims reproduce cleanly:
Bernstein-Vazirani, Grover (with the correct multi-marked-item iteration
formula), Even-Mansour's clean regime, and 3-round Feistel. These aren't
just "ran without crashing" — each was verified against the paper's specific
stated formulas (oracle constructions, iteration counts, period equations)
via direct algebraic derivation and empirical testing (16/16 across the
repo's benchmark, robust across 3+ random seeds).

### The 1 "Critical" — CBC-MAC period formula

**This is a genuine discrepancy discovered during implementation, not an
implementation mistake.** The paper states:

> f(x) = Ek(x⊕c·a) ⊕ Ek(x⊕c·b) with period s = Ek(a) ⊕ Ek(b)

Direct testing (see `oracles/simon_cbc_mac.py` and
`tests/test_simon_algorithms.py::test_cbc_mac_period_discrepancy_documented`)
shows this oracle does **not** satisfy Simon's 2-to-1 promise with the
paper's stated period for a generic permutation Ek. It **does** satisfy the
promise with period **s = a⊕b** (the input-space difference) — the same
algebraic identity that makes the Even-Mansour reduction work. This was
verified empirically across multiple test permutations, not just asserted.

*Likely explanation*: either a typo/error in the paper (swapping the
input-space period with an output-space quantity that resembles the
Even-Mansour case's k2-recovery step), or the real CBC-MAC forgery attack
(Kaplan et al.) relies on additional structure from actual CBC-chaining that
this generic-oracle formulation doesn't capture as literally written.

### The 1 "Unmatched" — Even-Mansour N=6-10 real-hardware hybrid

Not scored, because it cannot be meaningfully compared:
- The paper's result depends on a technique explicitly withheld pending an IP decision.
- This repo has no access to real IBM hardware (`ibm_kingston`).
- Running the *disclosed* algorithm on a noiseless simulator trivially gets rank-1 for the wrong reason (no noise at all, rather than a noise-compensation technique).

---

## Hallucination Report Summary

See `hallucination_report.md` for the full report.

| Type | Count | Critical |
|------|-------|---------|
| Structural | 1 | 0 |
| Parametric | 3 | 0 |
| Omission | 1 | 1 |

The one Critical omission (the withheld technique itself) is the paper's own
explicit withholding — not something ArXivist's generation process dropped.

---

## Recommended Actions

1. **Flag the CBC-MAC period-formula discrepancy to the paper's authors** — this looks like a genuine error worth a correction/clarification, independent of anything else in this reproduction.
2. **No action needed on the four "Excellent" matches** — they're solid.
3. **Treat the N=6-10 real-hardware result as unverifiable pending disclosure.** If the withheld technique is published later, re-run this comparison against it directly.
4. **If real IBM Quantum access becomes available**, swap `experiment.backend_mode: ibm_real` in `configs/config.yaml` to test the disclosed algorithms against actual hardware noise (still won't reproduce the withheld technique, but will show how the *disclosed* algorithms alone perform on real noise).

---

## Verification Log Summary

- Comparison run at: 2026-07-22T10:30:00Z
- Full audit trail, including the exact config used and result hash, in `verification_log.md`.
