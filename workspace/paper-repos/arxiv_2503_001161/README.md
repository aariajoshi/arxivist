# SGDD — Split Gibbs Discrete Diffusion Posterior Sampling (reproduction)

Reproduction of **Chu, Wu, Chen, Song, Yue — "Split Gibbs Discrete Diffusion Posterior
Sampling"** (NeurIPS 2025), [arXiv:2503.01161](https://arxiv.org/abs/2503.01161).
Official code: <https://github.com/chuwd19/Split-Gibbs-Discrete-Diffusion-Posterior-Sampling>.

## What this is

SGDD is an **algorithm**, not a network: a plug-and-play posterior sampler for **discrete**
diffusion models built on the split-Gibbs principle. It alternates

1. a **likelihood step** — `z ~ pi(x=x_k, z; eta)` via Metropolis-Hastings (Eq 13), needing
   *no gradient* of `p(y|x)` (the key enabler for categorical data); and
2. a **prior step** — `x ~ pi(x, z=z_k; eta)`, which is exactly a *partial* reverse
   discrete-diffusion denoise from noise level `eta` (Eq 11 == Eq 12 identity),

under a geometric annealing schedule `eta_k` (Sec C.3). Both split variables provably
converge to `p(x|y)` (Theorem 1, `O(1/K)`).

## What this repo verifies (CPU, no weights)

The paper's **own controlled study** (Sec 4.2, Table 2): a discretized-Gaussian posterior
whose ground truth is computable *exactly*. We build the sampler from scratch and check it
mechanically against that ground truth — this is a direct, honest test of the core claim.

```bash
pip install -e .
pytest tests/ -q                       # 7 tests: potential limits, Eq11==Eq12 identity,
                                       # MH detailed balance, schedule, metrics, end-to-end
python run_synthetic.py --config configs/config.yaml
```

`run_synthetic.py` reports **Hellinger / TV to the exact true posterior** and compares
against a DPS-style guidance baseline (the Table-2 ordering SGDD should win).

## What is gated (needs trained weights)

The DNA-enhancer (Sec 4.3), MNIST XOR/AND (Sec 4.4), and music-infilling (Sec 4.5)
experiments reuse the **same sampler** but require the trained **SEDD-small (~90M)** priors
and task oracles/classifiers from the official repo. `src/sgdd/tasks/real_data.py` documents
the exact wiring (prior adapter + likelihood `f(z;y)`); no algorithm change is needed to run
them once weights are supplied.

| Task | Prior | N | D | K | MH_T | H |
|------|-------|---|---|---|------|---|
| Synthetic (this repo) | closed-form Gaussian | 50 | 2/5/10 | 10 | 10 | 20 |
| DNA enhancers | SEDD-90M | 4 | 200 | 50 | 200 | 20 |
| MNIST XOR/AND | SEDD-90M | 2 | 1024 | 50/100 | 2000/5000 | 20 |
| Music infilling | SEDD-90M | 129 | 256 | 100 | 5000 | 20 |

## Layout

```
src/sgdd/
  samplers/potential.py            # D(x,z;eta) (Eq 10), beta~, prior reweight (Eq 11)
  samplers/metropolis_hastings.py  # likelihood step (Eq 13)
  samplers/split_gibbs.py          # Algorithm 1 + geometric schedule (Sec C.3)
  samplers/baseline_dps.py         # DPS-style guidance baseline (reference, for ordering)
  models/discrete_diffusion.py     # uniform-kernel diffusion + closed-form denoiser (Eq 12)
  tasks/synthetic.py               # discretized-Gaussian prior + exact true posterior
  tasks/real_data.py               # SEDD-gated DNA / MNIST / music wiring (documented)
  metrics.py                       # Hellinger, total variation
tests/test_sgdd.py                 # mechanical verification (7 tests)
run_synthetic.py                   # Table-2 reproduction entrypoint
```

## Honesty

The synthetic Hellinger/TV we print are **measured against the analytic true posterior**,
not copied from Table 2. All real-data numbers referenced in `comparison/` are cited as the
*paper's*, explicitly labeled weight-gated, and never presented as our reproduction.
