# Architecture Plan — Quantum Cryptanalysis on IBM Hardware (arxiv_2607_18340)

## Framework
**Qiskit + Qiskit Aer** (numpy for classical GF(2)/cipher reference code), Python 3.10+, no GPU
required. *(Schema note: the architecture-plan schema's `framework.primary` field only allows
pytorch/jax/tensorflow, since it was designed for neural-network papers. `pytorch` is set as a
placeholder to satisfy the schema — the real framework is Qiskit, documented in `reasoning` and
used throughout the actual repo.)*

## ⚠️ The most important thing to know
The paper's real-hardware headline result (Even-Mansour recovery at N=6–10) depends on a
**"hardware-aware circuit-conditioning and readout post-selection technique" that the paper
explicitly withholds** pending an IP decision (Section 5). This cannot be reproduced from the
paper as written. This repo:
- Faithfully implements the **disclosed** algorithms (BV, Grover, Simon for EM/CBC-MAC/Feistel) exactly as specified.
- Defaults to a **noiseless Qiskit Aer simulator**, where these textbook algorithms recover periods/secrets cleanly (rank-1) by construction — this validates algorithmic correctness.
- Includes a clearly-labeled **best-effort substitute** (`TopKHybridRanker`) for the withheld technique, for experimenting with noisy backends — but this is explicitly *not* a reproduction of the paper's real method.

## Module Hierarchy
```
src/quantum_cryptanalysis/
├── oracles/
│   ├── bernstein_vazirani.py   BVOracle
│   ├── grover_spn.py            ToySPNCipher, GroverSPNOracle (SPN construction ASSUMED — paper never specifies one)
│   ├── simon_even_mansour.py    EvenMansourOracle
│   ├── simon_cbc_mac.py          CBCMACForgeryOracle
│   └── simon_feistel.py           FeistelOracle
├── algorithms/
│   ├── bv_circuit.py               BernsteinVazirani
│   ├── grover_circuit.py            GroverSearch
│   └── simon_circuit.py              SimonAlgorithm (+ GF(2) period solver)
├── postprocessing/
│   ├── gf2_solver.py                  GF(2) Gaussian elimination utilities
│   └── hybrid_ranking.py               TopKHybridRanker — labeled substitute for the withheld technique
├── backend/
│   ├── execution.py                     BackendFactory (Aer by default; real-IBM hook available)
│   └── mitigation.py                      ReadoutMitigator (standard Qiskit mitigation only — NOT the withheld technique)
├── evaluation/
│   └── metrics.py                          rank computation, birthday-bound comparisons
└── utils/
    └── config.py                             YAML loading + seeding
```

## Key Equation → Code Mapping
| Paper element | Code location |
|---|---|
| BV oracle `f(x)=a·x` | `oracles/bernstein_vazirani.py` |
| Grover iteration count `(π/4)√(2ⁿ)` | `algorithms/grover_circuit.py` |
| EM Simon reduction | `oracles/simon_even_mansour.py` |
| CBC-MAC forgery `f(x)=Ek(x⊕c·a)⊕Ek(x⊕c·b)` | `oracles/simon_cbc_mac.py` |
| 3-round Feistel oracle + period `s=(1,γ)` | `oracles/simon_feistel.py` |
| GF(2) linear solve for hidden period | `postprocessing/gf2_solver.py` |
| Top-K hybrid + classical verification (n=6–10) | `postprocessing/hybrid_ranking.py` — **labeled substitute, not the paper's real method** |
| Readout error mitigation (Nation et al.) | `backend/mitigation.py` |

## Config Highlights
- `experiment.backend_mode: aer_simulator_noiseless` (default) — real IBM hardware access isn't available in this environment; a real-backend path exists via `qiskit-ibm-runtime` for users with their own IBM Quantum credentials.
- `experiment.shots: 4096` — **ASSUMED**, since the paper's actual per-job shot counts are unrecoverable `[measurement]` placeholders in the provided PDF.
- `grover_spn` uses a small toy SPN construction, since the paper never specifies its S-box/rounds/key schedule.

## Entrypoints
- `run_attack.py --attack {bv,grover,simon_em,simon_cbcmac,simon_feistel} [--n N] [--backend-mode MODE]`
- `run_benchmark.py --config configs/config.yaml --output results/benchmark_table.json` — reproduces a Table 1/2-style summary across all five attacks.

## Risk Assessment
| Severity | Risk | Mitigation |
|---|---|---|
| High | Withheld hardware-scaling technique (core N=6-10 claim) | Reproduce disclosed algorithms only, on noiseless sim by default; labeled substitute ranker included for experimentation |
| Medium | No real IBM hardware access in this environment | Aer simulator default; real-backend hook documented |
| Medium | Grover's target SPN construction undefined in paper | Small toy SPN defined and clearly labeled as an assumption |
| Low | Shot counts unrecoverable from paper | Defaulted to 4096, configurable |
| Low | n=9,10 (27-30 qubits) slow/memory-heavy on CPU Aer | Included in config but off by default; documented in README |
