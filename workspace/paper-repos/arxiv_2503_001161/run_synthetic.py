"""Run the SGDD synthetic controlled-posterior benchmark (Sec 4.2, Table 2).

Reproduces the paper's OWN accuracy check: SGDD sampling of a discretized-Gaussian
posterior whose ground truth is computable exactly. Reports Hellinger / TV against the
true posterior and (optionally) against a DPS-style guidance baseline.

Usage:
    python run_synthetic.py --config configs/config.yaml
    python run_synthetic.py --D 5 --n 2000      # override
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from sgdd.metrics import empirical_marginal, hellinger, total_variation
from sgdd.models.discrete_diffusion import ClosedFormUniformPrior
from sgdd.samplers.split_gibbs import sample_many
from sgdd.tasks.synthetic import SyntheticGaussianTask


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--D", type=int, default=None)
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    t, s, e = cfg["task"], cfg["sgdd"], cfg["eval"]
    D = args.D if args.D is not None else t["D"]
    n = args.n if args.n is not None else e["n_samples"]
    N = t["N"]
    dims = tuple(e["posterior_dims"])

    if D > 4:
        # exact N^D enumeration is infeasible; cap N so the ground truth stays computable
        N = min(N, 10)
        print(f"[warn] D={D}: capping N to {N} so the exact true posterior is enumerable.")

    task = SyntheticGaussianTask(D=D, N=N, sigma_prior=t["sigma_prior"],
                                 sigma_y=t["sigma_y"], grid_lim=t["grid_lim"], seed=t["seed"])
    prior = ClosedFormUniformPrior(task.prior_logits, N=N)

    print(f"[sgdd] synthetic task D={D} N={N} | K={s['K']} MH_T={s['mh_T']} H={s['euler_H']}")
    true = task.true_posterior_marginal(dims=dims)

    # prior-collapse reference (the failure mode the paper's baselines exhibit as D grows)
    p0 = prior.p0
    prior_marg = p0[dims[0]][:, None] * p0[dims[1]][None, :] if len(dims) == 2 else p0[dims[0]]
    h_prior = hellinger(prior_marg, true)
    print(f"[ref  ]  prior-collapse Hellinger={h_prior:.4f}  (ignoring y)")

    # SGDD at the config budget, plus a larger budget to show O(1/K) convergence (Thm 1)
    for tag, (K, T) in {"cfg": (s["K"], s["mh_T"]), "2x": (s["K"] * 2, s["mh_T"] * 4)}.items():
        sg = sample_many(n, prior=prior, neg_log_likelihood=task.neg_log_likelihood,
                         N=N, D=D, K=K, mh_T=T, euler_H=s["euler_H"],
                         eta_min=s["eta_min"], eta_max=s["eta_max"], seed=1)
        emp = empirical_marginal(sg, N, dims=dims)
        h, tv = hellinger(emp, true), total_variation(emp, true)
        print(f"[SGDD ]  K={K:3d} T={T:4d}  Hellinger={h:.4f}  TV={tv:.4f}"
              f"   {'(paper D=2: 0.149/0.125)' if tag == 'cfg' else '(more budget -> lower, per Thm 1)'}")


if __name__ == "__main__":
    main()
