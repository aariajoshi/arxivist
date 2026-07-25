"""Synthetic controlled-posterior benchmark (Sec 4.2, Table 2).

Prior p(x): a discretized 1-D Gaussian N(0, sigma^2) applied per token over a grid of
N symbols (each symbol = a grid point value). Forward model G(x) = ||f(x)||_1 with a
simple linear f. Because both the prior and the (soft) likelihood are factorized/low-dim,
the TRUE posterior p(x|y) is computable exactly on the first dimensions, so Hellinger/TV
against ground truth are exact — this is the paper's own accuracy check for SGDD.
"""
from __future__ import annotations

import numpy as np


class SyntheticGaussianTask:
    def __init__(self, D: int = 2, N: int = 50, sigma_prior: float = 1.0,
                 sigma_y: float = 0.3, grid_lim: float = 3.0, seed: int = 0) -> None:
        self.D, self.N = D, N
        self.sigma_prior, self.sigma_y = sigma_prior, sigma_y
        # symbol -> real value grid
        self.grid = np.linspace(-grid_lim, grid_lim, N)  # [N]
        # per-token prior logits = log N(grid; 0, sigma_prior^2)
        logp = -0.5 * (self.grid / sigma_prior) ** 2
        self.prior_logits = np.tile(logp, (D, 1))         # [D, N]
        rng = np.random.default_rng(seed)
        # linear f(x) = w . values ; G(x) = |f(x)| (l1 of a scalar -> abs)
        self.w = rng.normal(size=D)
        # draw a ground-truth x_true and its measurement y
        p0 = np.exp(logp - logp.max()); p0 /= p0.sum()
        self.x_true = np.array([rng.choice(N, p=p0) for _ in range(D)])
        self.y = self._forward(self.x_true) + rng.normal(scale=sigma_y)

    def _values(self, x: np.ndarray) -> np.ndarray:
        return self.grid[x]

    def _forward(self, x: np.ndarray) -> float:
        return float(np.abs(np.dot(self.w, self._values(x))))

    def neg_log_likelihood(self, z: np.ndarray) -> float:
        """f(z;y) = -log p(y|z) = ||G(z) - y|| / sigma_y  (Sec 4.1 inverse-problem form)."""
        return abs(self._forward(z) - self.y) / self.sigma_y

    def prior_probs(self) -> np.ndarray:
        p = np.exp(self.prior_logits - self.prior_logits.max(axis=1, keepdims=True))
        return p / p.sum(axis=1, keepdims=True)

    # ---- exact posterior over the FULL joint (feasible for small D*N) ----
    def true_posterior_marginal(self, dims=(0, 1)) -> np.ndarray:
        """Exact p(x|y) marginalized onto `dims` by enumerating all N^D states.

        Only call for small D (the paper visualizes the first 2 dims); the full
        enumeration is N^D so keep D <= ~4 for this exact check.
        """
        from itertools import product
        p0 = self.prior_probs()
        shape = tuple(self.N for _ in range(len(dims)))
        marg = np.zeros(shape)
        for state in product(range(self.N), repeat=self.D):
            x = np.array(state)
            logw = np.sum(np.log(p0[np.arange(self.D), x])) - self.neg_log_likelihood(x)
            key = tuple(x[d] for d in dims)
            marg[key] += np.exp(logw)
        marg /= marg.sum()
        return marg
