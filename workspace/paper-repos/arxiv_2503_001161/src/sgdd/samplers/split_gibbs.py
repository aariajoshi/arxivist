"""SGDD — Algorithm 1: Split Gibbs Discrete Diffusion Posterior Sampling (Sec 3.4).

Alternates a likelihood step (Metropolis-Hastings, Eq 13) and a prior step (partial
reverse discrete diffusion, Eq 11) under a geometric annealing schedule eta_k that
decays eta_max -> eta_min. Both split variables x and z provably converge to p(x|y)
(Theorem 1, O(1/K)).
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from ..models.discrete_diffusion import ClosedFormUniformPrior, euler_prior_step
from .metropolis_hastings import mh_likelihood_step


def geometric_schedule(K: int, eta_min: float = 1e-4, eta_max: float = 20.0) -> np.ndarray:
    """Annealing schedule eta_k = eta_min^{k/K} * eta_max^{1-k/K}  (Sec C.3).

    eta_0 = eta_max, and eta_k decays geometrically toward eta_min as k -> K.
    """
    ks = np.arange(K)
    return (eta_min ** (ks / K)) * (eta_max ** (1.0 - ks / K))


def sgdd(
    prior: ClosedFormUniformPrior,
    neg_log_likelihood: Callable[[np.ndarray], float],
    N: int,
    D: int,
    K: int = 10,
    mh_T: int = 10,
    euler_H: int = 20,
    eta_min: float = 1e-4,
    eta_max: float = 20.0,
    x0: np.ndarray | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Run SGDD and return a single posterior sample x_K (Algorithm 1)."""
    rng = np.random.default_rng(seed)
    x = rng.integers(N, size=D) if x0 is None else x0.copy()
    z = x.copy()
    schedule = geometric_schedule(K, eta_min, eta_max)

    for k in range(K):
        eta = float(schedule[k])
        # (1) likelihood step (Eq 13): z ~ pi(x=x_k, z; eta) via MH
        z = mh_likelihood_step(x, z, neg_log_likelihood, eta, N, mh_T, rng)
        # (2) prior step (Eq 11): x ~ pi(x, z=z_k; eta) via partial reverse diffusion
        x = euler_prior_step(z, eta, prior, N, euler_H, rng)
    return x


def sample_many(n: int, **kwargs) -> np.ndarray:
    """Draw n independent SGDD samples (varying the seed). Returns [n, D]."""
    base_seed = kwargs.pop("seed", 0)
    out = []
    for i in range(n):
        out.append(sgdd(seed=base_seed + i, **kwargs))
    return np.stack(out, axis=0)
