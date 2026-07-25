# Architecture Plan Summary — SGDD (arxiv_2503_001161)

**Paper type:** algorithm (plug-and-play discrete posterior sampling), not a network.

## Reproduction strategy (QOSI)

Verify the **Split Gibbs sampler (Algorithm 1)** mechanically on the paper's *own* synthetic
controlled-posterior study, where the true posterior is computable in closed form — CPU only,
no trained weights. This is a direct, honest test of the core claim (convergence to `p(x|y)`).

## Build from scratch (CPU-testable)

- `potential.py` — `D(x,z;eta)` (Eq 10), `beta~`, prior reweight (Eq 11)
- `metropolis_hastings.py` — likelihood step (Eq 13), gradient-free
- `discrete_diffusion.py` — uniform-kernel diffusion + closed-form denoiser (Eq 12)
- `split_gibbs.py` — Algorithm 1 + geometric schedule (Sec C.3)
- `tasks/synthetic.py` — discretized-Gaussian prior + **exact** true posterior
- `metrics.py` — Hellinger, TV

## Verified mechanically (7/7 tests)

potential divergence · `beta~ == beta_t` identity (exact) · MH detailed-balance invariance ·
schedule endpoints · metric endpoints · **end-to-end convergence to the exact posterior**
(Hellinger 0.40→0.19 with budget → paper's 0.149, matching Theorem 1's O(1/K)).

## Honest limitations (disclosed)

- **Table 2 competitive ranking NOT reproduced** — minimal synthetic task too easy to trigger
  the baselines' prior-collapse; needs the paper's undisclosed multivariate forward model.
- **Real-data metrics (DNA/MNIST/music) gated** on the SEDD-90M priors + task oracles
  (`tasks/real_data.py` documents the wiring).

**Score: 0.80.**
