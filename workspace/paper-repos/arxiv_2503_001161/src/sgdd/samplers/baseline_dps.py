"""A discrete DPS-style guidance baseline (Sec 4.1) for the Table-2 ordering check.

The paper's discrete DPS analogy adds a likelihood-guidance term to the reverse
diffusion instead of the split-Gibbs decoupling. We implement a simple, faithful stand-in:
reverse-diffuse from the prior while reweighting the per-token x0 posterior by a
one-step likelihood approximation exp(-f(x0_hat;y)). It captures the failure mode the
paper highlights — guidance-based methods degenerate toward the prior as D grows — so
SGDD should beat it on Hellinger. This is a REFERENCE baseline, not a faithful DPS
re-implementation, and is labeled as such wherever its numbers appear.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from ..models.discrete_diffusion import ClosedFormUniformPrior, forward_marginal


def dps_like(
    prior: ClosedFormUniformPrior,
    neg_log_likelihood: Callable[[np.ndarray], float],
    N: int,
    D: int,
    steps: int = 50,
    eta_max: float = 20.0,
    guidance: float = 1.0,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    sigmas = np.linspace(eta_max, 0.0, steps + 1)
    x = rng.integers(N, size=D)
    for h in range(steps):
        s, s_next = sigmas[h], sigmas[h + 1]
        post = prior.posterior_x0(x, s)                    # p(x0|x_s) [D,N]
        x0_map = post.argmax(axis=1)
        # crude likelihood guidance: bias the categorical toward lower f(x0;y)
        f0 = neg_log_likelihood(x0_map)
        log_bias = np.zeros((D, N))
        for i in range(D):
            for a in range(N):
                cand = x0_map.copy(); cand[i] = a
                log_bias[i, a] = -guidance * (neg_log_likelihood(cand) - f0)
        logp = np.log(post + 1e-12) + log_bias
        logp -= logp.max(axis=1, keepdims=True)
        p = np.exp(logp); p /= p.sum(axis=1, keepdims=True)
        x0_hat = np.array([rng.choice(N, p=p[i]) for i in range(D)])
        x = x0_hat if s_next <= 0 else forward_marginal(x0_hat, s_next, N, rng)
    return x


def sample_many_dps(n: int, **kwargs) -> np.ndarray:
    base = kwargs.pop("seed", 0)
    return np.stack([dps_like(seed=base + i, **kwargs) for i in range(n)], axis=0)
