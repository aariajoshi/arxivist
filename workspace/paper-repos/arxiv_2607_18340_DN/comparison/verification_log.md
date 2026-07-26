# Verification Log

**Paper ID**: arxiv_2607_18340
**Comparison run at**: 2026-07-22T10:30:00Z

---

## Provenance

| Field | Value |
|---|---|
| SIR version used | 1 (`sir-registry/arxiv_2607_18340/sir.json`) |
| Architecture plan version used | 1 (`sir-registry/arxiv_2607_18340/architecture_plan.json`) |
| Paper metrics compared | 6 |
| Matched (scored) pairs | 5 |
| Unmatched | 1 (Even-Mansour N=6-10 real-hardware hybrid — different experimental conditions) |
| Results hash | `sha256:e789313e4ff0da1fb024509b364f5c4a329d0c271bb57ebc8e1e9aec61728657` |

## How the results were obtained

Ran `python run_benchmark.py --config configs/config.yaml --output
results/stage6_benchmark.json` — a full (non-quick) pass covering:
- Bernstein-Vazirani at n=8, 16
- Grover SPN at n=6, 8
- Simon-Even-Mansour at n=3, 4, 5, 6, 7, 8
- Simon-CBC-MAC forgery at n=4, 6
- Simon-3-round-Feistel at half-block m=3, 4, 5, 6 (block sizes 6, 8, 10, 12)

All 16 runs on the default `aer_simulator_noiseless` backend, 4096 shots,
seed=42. Result: **16/16 clean (rank-1) recoveries.**

## Config used (unmodified from repo defaults)

- `experiment.backend_mode`: aer_simulator_noiseless
- `experiment.shots`: 4096
- `experiment.seed`: 42
- All attack size lists as shipped in `configs/config.yaml`

No config modifications were needed for this comparison — unlike the
LATTICE paper's Stage 6 run, this paper's disclosed algorithms are cheap
enough to run at full configured scale in the sandbox.

## Requires manual review: **Yes**

Reasons:
1. **The paper's central claim (Even-Mansour N=6-10, real hardware) is fundamentally untestable here** — it depends on a technique the paper itself withholds, and this environment has no real IBM Quantum hardware access regardless.
2. **A likely paper error was found and should be flagged to the authors**: the stated CBC-MAC forgery period formula (`s=Ek(a)⊕Ek(b)`) does not satisfy Simon's algorithm's 2-to-1 promise under direct testing; the actual period is `s=a⊕b`. This was verified via `oracles/promise_utils.py::verify_two_to_one_promise` across multiple test instances, not just asserted from a single example.
3. **Grover's target SPN cipher is a labeled stand-in** (`ToySPNCipher`), since the paper never specifies its actual construction.

## What would upgrade this to a higher-confidence comparison

- **Disclosure of the withheld technique** (Section 5) — would let this comparison actually test the paper's headline claim instead of marking it unmatched.
- **Real IBM Quantum hardware access** — would let the disclosed algorithms be tested against genuine hardware noise (still not the withheld technique, but a meaningfully closer comparison than a noiseless simulator).
- **Author clarification on the CBC-MAC period formula** — would resolve whether this is a paper typo, a real construction ArXivist's implementation doesn't fully capture, or something else.
- **The paper's actual SPN construction**, if disclosed, to replace `ToySPNCipher`.
