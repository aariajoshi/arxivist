"""Uniform-kernel discrete diffusion + prior sampling step (Sec 2.1, 3.2).

The prior step of SGDD (Eq 11) is equivalent to a *partial* reverse discrete-diffusion
sampler started at noise level sigma_t = eta from state x_t = z_k (via the Eq 11 == Eq 12
identity). For the synthetic benchmark we do not need a trained network: the prior p0 is
a factorized categorical (discretized Gaussian) whose posterior p(x0 | xt) is available in
closed form, giving an exact concrete-score prior. Real-data tasks would swap in a trained
SEDD model here.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np


def beta_t(sigma: float, N: int) -> float:
    """Per-token corruption prob of the uniform kernel at noise sigma (Sec 2.1)."""
    return (N - 1) / N * (1.0 - math.exp(-sigma))


def forward_marginal(x0: np.ndarray, sigma: float, N: int, rng: np.random.Generator) -> np.ndarray:
    """Sample x_t ~ p(x_t | x0) under the uniform kernel: each token flips to a
    uniform random symbol w.p. beta_t, else stays (Sec 2.1, Eq 5)."""
    bt = beta_t(sigma, N)
    xt = x0.copy()
    flip = rng.random(x0.shape) < bt
    xt[flip] = rng.integers(N, size=int(flip.sum()))
    return xt


class ClosedFormUniformPrior:
    """Factorized categorical prior with an EXACT p(x0 | xt) denoiser.

    p0(x) = prod_i p0_i(x_i) given as per-token logits `prior_logits` [D, N].
    Under the uniform kernel, p(x0=a | xt) ~ p0(a) * [ (1-beta_t) if a==xt else beta_t/(N-1) ],
    which we normalize per token. This is the closed-form concrete score used for the
    synthetic study (no training).
    """

    def __init__(self, prior_logits: np.ndarray, N: int) -> None:
        self.D, self.Nlog = prior_logits.shape
        assert self.Nlog == N
        self.N = N
        # normalized prior probs per token
        p = np.exp(prior_logits - prior_logits.max(axis=1, keepdims=True))
        self.p0 = p / p.sum(axis=1, keepdims=True)  # [D, N]

    def posterior_x0(self, xt: np.ndarray, sigma: float) -> np.ndarray:
        """Return p(x0 | xt) as a [D, N] categorical (Eq 12)."""
        bt = beta_t(sigma, N=self.N)
        stay = 1.0 - bt
        move = bt / (self.N - 1)
        like = np.full((self.D, self.N), move)
        like[np.arange(self.D), xt] = stay
        post = self.p0 * like
        post /= post.sum(axis=1, keepdims=True)
        return post


def euler_prior_step(
    z_k: np.ndarray,
    eta: float,
    prior: ClosedFormUniformPrior,
    N: int,
    H: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Prior step (Eq 11): partial reverse diffusion from sigma=eta, x_t=z_k, H Euler steps.

    We anneal sigma from eta -> 0 over H steps; at each step we compute the exact
    p(x0 | x_sigma), take an expected-flow Euler update by re-noising the predicted x0
    to the next (lower) sigma. At sigma=0 we sample the final x0 from p(x0 | x_current).
    """
    sigmas = np.linspace(eta, 0.0, H + 1)
    x = z_k.copy()
    for h in range(H):
        s, s_next = sigmas[h], sigmas[h + 1]
        post = prior.posterior_x0(x, s)                       # p(x0 | x_s)   [D,N]
        x0_hat = np.array([rng.choice(N, p=post[i]) for i in range(prior.D)])
        if s_next <= 0.0:
            x = x0_hat
        else:
            x = forward_marginal(x0_hat, s_next, N, rng)      # re-noise to next level
    return x
