"""Regularization potential D(x,z;eta) and prior-step reweighting (Sec 3.2, Eq 10-12).

The potential couples the split variables x and z by their Hamming distance and
is the mechanism that forces both marginals to the posterior as eta -> 0.
"""
from __future__ import annotations

import math

import numpy as np


def beta_tilde(eta: float, N: int) -> float:
    """beta~ = (N-1)/N * (1 - e^{-eta})   (Sec 3.2, below Eq 11).

    This is the per-token flip probability of the uniform kernel at noise level eta.
    beta~ -> 0 as eta -> 0 (no noise); beta~ -> (N-1)/N as eta -> inf (uniform).
    """
    return (N - 1) / N * (1.0 - math.exp(-eta))


def potential_weight(eta: float, N: int) -> float:
    """The scalar log-weight multiplying the Hamming distance in D(x,z;eta) (Eq 10).

    D(x,z;eta) = d(x,z) * w(eta),  w(eta) = log[ (1 + (N-1)e^{-eta}) /
                                                 ((N-1)(1-e^{-eta})) ].
    w(eta) -> +inf as eta -> 0+, so D blows up unless d(x,z)=0. w(eta) -> 0 as eta grows.
    """
    e = math.exp(-eta)
    num = 1.0 + (N - 1) * e
    den = (N - 1) * (1.0 - e)
    return math.log(num / den)


def potential(x: np.ndarray, z: np.ndarray, eta: float, N: int) -> float:
    """D(x,z;eta) = Hamming(x,z) * potential_weight(eta,N)  (Eq 10)."""
    d = int(np.sum(x != z))
    return d * potential_weight(eta, N)


def prior_reweight_logprob(z: np.ndarray, x: np.ndarray, eta: float, N: int) -> float:
    """log of the prior-step reweighting factor (beta~/(1-beta~))^{d(z,x)}  (Eq 11).

    pi(x, z=z_k; eta) ~ p0(x) * (beta~/(1-beta~))^{d(z_k,x)}.
    Returns the log of the (beta~/(1-beta~))^{d} factor only (the p0(x) term is
    supplied by the diffusion prior).
    """
    bt = beta_tilde(eta, N)
    d = int(np.sum(z != x))
    return d * (math.log(bt) - math.log(1.0 - bt))
