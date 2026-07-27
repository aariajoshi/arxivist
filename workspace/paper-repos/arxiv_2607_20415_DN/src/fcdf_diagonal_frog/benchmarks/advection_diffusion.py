"""Advection-diffusion benchmarks.

Paper: Itkin (2026), Section 6.3-6.4.

SmoothAdvectionDiffusionBenchmark: constant-coefficient advection-diffusion; a Gaussian
initial density stays Gaussian with mean x0+mu*t and variance v0+2*D*t. Used for the Peclet
sweep (Table 5) where FCDF's stencil order is contrasted against Chang-Cooper's.

FrontBenchmark: mu=1, D=1e-4, unit-height plateau on [0.1, 0.4] -- an order-one unresolved
front (Section 3.1 taxonomy) with no closed-form reference; used for Tables 6, 8, 9 and the
monotone-floor / active-set-cost studies.
"""
from __future__ import annotations

import numpy as np


class SmoothAdvectionDiffusionBenchmark:
    def __init__(self, mu: float = 1.0, D: float = 1.0e-2, x0: float = 0.3, v0: float = 4.0e-3):
        self.mu = mu
        self.D = D
        self.x0 = x0
        self.v0 = v0

    def drift(self, x: np.ndarray) -> np.ndarray:
        return np.full_like(np.asarray(x, dtype=float), self.mu)

    def diffusion(self, x: np.ndarray) -> np.ndarray:
        return np.full_like(np.asarray(x, dtype=float), self.D)

    def initial_condition(self, x: np.ndarray) -> np.ndarray:
        return _gaussian(x, self.x0, self.v0)

    def exact_density(self, x: np.ndarray, t: float) -> np.ndarray:
        mean = self.x0 + self.mu * t
        var = self.v0 + 2 * self.D * t
        return _gaussian(x, mean, var)

    def domain(self, n_std: float = 8.0, t_final: float = 0.1) -> tuple[float, float]:
        """Pad by n_std standard deviations of the *final* profile on both sides
        (Section 6.3: 'the domain is padded by eight standard deviations of the final
        profile on both sides, so the zero-flux boundaries are never reached')."""
        final_var = self.v0 + 2 * self.D * t_final
        final_std = np.sqrt(final_var)
        final_mean = self.x0 + self.mu * t_final
        lo = min(self.x0, final_mean) - n_std * final_std
        hi = max(self.x0, final_mean) + n_std * final_std
        return lo, hi


class FrontBenchmark:
    def __init__(self, mu: float = 1.0, D: float = 1.0e-4, plateau_lo: float = 0.1, plateau_hi: float = 0.4):
        self.mu = mu
        self.D = D
        self.plateau_lo = plateau_lo
        self.plateau_hi = plateau_hi

    def drift(self, x: np.ndarray) -> np.ndarray:
        return np.full_like(np.asarray(x, dtype=float), self.mu)

    def diffusion(self, x: np.ndarray) -> np.ndarray:
        return np.full_like(np.asarray(x, dtype=float), self.D)

    def initial_condition(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.where((x >= self.plateau_lo) & (x <= self.plateau_hi), 1.0, 0.0)


def _gaussian(x: np.ndarray, mean: float, var: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * (x - mean) ** 2 / var) / np.sqrt(2 * np.pi * var)
