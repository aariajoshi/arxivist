"""Ornstein-Uhlenbeck benchmark.

Paper: Itkin (2026), Section 6.1-6.2.

    dX_t = -alpha*X_t dt + sigma dW_t

Exact solution given a Gaussian initial condition N(x0, v0) remains Gaussian with
mean x0*exp(-alpha*t) and variance (sigma^2/(2*alpha))*(1-exp(-2*alpha*t)) + v0*exp(-2*alpha*t).
The domain spans 6 stationary standard deviations either side of the origin (Section 6.1),
so the drift mu(x) = -alpha*x changes sign at the domain center -- this is what exercises the
sign-changing-drift stencil (SIR ambiguity #4).
"""
from __future__ import annotations

import numpy as np


class OUBenchmark:
    def __init__(self, alpha: float = 1.0, sigma: float = 1.0, x0: float = 0.5, v0: float = 1.0e-2):
        self.alpha = alpha
        self.sigma = sigma
        self.x0 = x0
        self.v0 = v0

    @property
    def stationary_std(self) -> float:
        return self.sigma / np.sqrt(2.0 * self.alpha)

    def domain(self, n_std: float = 6.0) -> tuple[float, float]:
        L = n_std * self.stationary_std
        return -L, L

    def drift(self, x: np.ndarray) -> np.ndarray:
        return -self.alpha * np.asarray(x)

    def diffusion(self, x: np.ndarray) -> np.ndarray:
        return np.full_like(np.asarray(x, dtype=float), 0.5 * self.sigma ** 2)

    def initial_condition(self, x: np.ndarray) -> np.ndarray:
        return _gaussian(x, self.x0, self.v0)

    def exact_density(self, x: np.ndarray, t: float) -> np.ndarray:
        mean = self.x0 * np.exp(-self.alpha * t)
        var = (self.sigma ** 2 / (2 * self.alpha)) * (1 - np.exp(-2 * self.alpha * t)) + self.v0 * np.exp(
            -2 * self.alpha * t
        )
        return _gaussian(x, mean, var)


def _gaussian(x: np.ndarray, mean: float, var: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * (x - mean) ** 2 / var) / np.sqrt(2 * np.pi * var)
