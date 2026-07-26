"""Likelihood sampling step via Metropolis-Hastings (Sec 3.3, Eq 13).

The likelihood step draws
    z ~ pi(x=x_k, z; eta) ~ exp( -f(z;y) - d(x_k,z) * w(eta) ),   f(z;y) = -log p(y|z).
The unnormalized log-density is fully accessible, so we sample it with MH using a
single-site (one-token) uniform proposal. Crucially this needs NO gradient of the
likelihood, which is why it works on discrete/categorical data.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .potential import potential_weight


def mh_likelihood_step(
    x_fixed: np.ndarray,
    z_init: np.ndarray,
    neg_log_likelihood: Callable[[np.ndarray], float],
    eta: float,
    N: int,
    steps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Run `steps` single-site MH updates targeting exp(-f(z;y) - d(x_fixed,z) w(eta)).

    Parameters
    ----------
    x_fixed : the current split variable x_k (held fixed this step).
    z_init  : starting z (typically the previous z or a copy of x_k).
    neg_log_likelihood : f(z;y) = -log p(y|z), any callable on a token array.
    eta, N  : noise level and vocab size for the coupling weight w(eta).
    steps   : number of MH sweeps proposals (Table 7's `Metropolis-Hastings T`).
    """
    w = potential_weight(eta, N)
    z = z_init.copy()
    D = z.shape[0]

    def energy(zz: np.ndarray) -> float:
        # -log pi(x_fixed, zz) up to const = f(zz;y) + Hamming(x_fixed,zz) * w
        return neg_log_likelihood(zz) + float(np.sum(x_fixed != zz)) * w

    e_cur = energy(z)
    for _ in range(steps):
        i = int(rng.integers(D))
        old = z[i]
        new = int(rng.integers(N))
        if new == old:
            continue
        z[i] = new
        e_new = energy(z)
        # accept w.p. min(1, exp(e_cur - e_new))
        if e_new <= e_cur or rng.random() < np.exp(e_cur - e_new):
            e_cur = e_new
        else:
            z[i] = old  # reject
    return z
