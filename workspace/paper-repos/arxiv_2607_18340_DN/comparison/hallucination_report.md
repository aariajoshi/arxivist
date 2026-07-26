# Hallucination Report

**Paper**: Quantum Cryptanalysis on IBM Quantum Hardware (arxiv_2607_18340)
**Comparison Date**: 2026-07-22

---

## Structural Hallucinations

### 1. Feistel oracle qubit layout (1+2m instead of paper's stated 1+4m)

- **Location**: `src/quantum_cryptanalysis/oracles/simon_feistel.py::FeistelOracle`
- **Severity**: Minor
- **Evidence**: The paper states "the oracle requires 1+4m qubits for block
  size 2m" without providing a circuit diagram. This implementation uses
  1+2m qubits (1 selector + m input + m output), computing F1's
  intermediate value classically into a lookup table rather than
  representing it as separate quantum registers. The period-recovery
  semantics (s=(1,gamma)) are preserved exactly; only the literal qubit
  count differs from the paper's stated total.
- **Suggested fix**: If the authors publish a circuit diagram, check whether
  the extra 2m qubits serve a purpose (e.g. explicit F1 evaluation
  registers, uncomputation ancillas) and add them if bit-exact qubit-count
  matching is required for some downstream use.

*No other structural hallucinations found* — every other oracle/algorithm
module maps directly to an explicit paper equation or the SIR's
architecture modules.

---

## Parametric Hallucinations

### 1. `experiment.shots: 4096`

- **Severity**: Minor
- **Evidence**: SIR ambiguities[1], confidence 0.4 — the paper's actual
  per-job shot counts are unrecoverable `[measurement]` placeholders in the
  provided PDF's Supplementary Table 4.
- **Suggested fix**: Sweep shot counts if exact hardware-statistics matching is ever needed.

### 2. `ToySPNCipher` (Grover's target construction)

- **Severity**: Significant
- **Evidence**: SIR ambiguities[3], confidence 0.4 — the paper never
  specifies the reduced SPN's S-box, round count, or key schedule for the
  Grover attack, unlike every other attack (which get explicit oracle
  formulas).
- **Suggested fix**: Substitute the authors' actual construction if
  disclosed. Current results demonstrate Grover's correctness against *a*
  valid SPN-like target, not necessarily the paper's specific one.

### 3. `TopKHybridRanker` scoring function

- **Severity**: Significant (but intentionally so — see note)
- **Evidence**: SIR ambiguities[0]/[2], confidence 0.05 (real method) / 0.45
  (this substitute). This is explicitly labeled throughout the codebase and
  README as a best-effort stand-in for the paper's withheld technique, not
  an attempt to reproduce it. Its middling-to-poor empirical performance
  under synthetic noise (e.g. true-key rank 62/63 in one notebook run) is
  expected and not a target for further tuning until the real technique is
  disclosed.
- **Suggested fix**: Not applicable until disclosure.

---

## Omission Hallucinations

### 1. Hardware-aware circuit-conditioning and readout post-selection technique

- **SIR location**: Section 5, Supplementary S4; SIR
  `architecture.modules["withheld_hardware_scaling_technique"]`, confidence 0.05.
- **Severity**: Critical
- **Note**: This is the paper's **own explicit withholding** ("pending an
  intellectual-property decision"), not a gap introduced during ArXivist's
  parsing or code generation. It is flagged here, in `README.md`, and in
  `architecture_plan.json`'s risk assessment for maximum visibility, since
  it's the single most consequential missing piece relative to the paper's
  headline claim (Even-Mansour recovery at N=10).
- **Suggested fix**: Re-run this comparison once/if the technique is published.

*No other omissions found* — every other disclosed algorithm, oracle,
and evaluation metric from the SIR is implemented and tested in this repo.

---

## Summary

| Type | Count | Critical | Significant | Minor |
|------|-------|----------|--------------|-------|
| Structural | 1 | 0 | 0 | 1 |
| Parametric | 3 | 0 | 2 | 1 |
| Omission | 1 | 1 | 0 | 0 |

The single Critical finding is the paper's own disclosed withholding, not an
ArXivist-introduced gap. Everything else traces to documented SIR
ambiguities with their own confidence scores. One additional finding — the
CBC-MAC period-formula discrepancy — is scored in
`benchmark_comparison.md`'s metric table (as a Critical *metric* deviation)
rather than here, since it's a verified paper-correctness issue rather than
a code-generation hallucination.
