"""Uniform 1D grid utility.

Paper: Itkin (2026), "Flux-Corrected Diagonal Frog", Section 2.
Implements the uniform grid x_i = x_min + (i-1)*h, i=1..n used throughout the paper.
"""
from __future__ import annotations

import numpy as np


class Grid1D:
    """Uniform 1D grid with n nodes on [x_min, x_max].

    Args:
        x_min: left endpoint of the domain.
        x_max: right endpoint of the domain.
        n: number of grid nodes (n >= 5 required by the 2nd-order stencils' 2-point
           backward/forward reach near boundaries).
    """

    def __init__(self, x_min: float, x_max: float, n: int) -> None:
        if n < 5:
            raise ValueError(f"Grid1D requires n >= 5 for the 2nd-order stencils, got n={n}")
        self.x_min = x_min
        self.x_max = x_max
        self.n = n
        self.h = (x_max - x_min) / (n - 1)
        self.x = x_min + np.arange(n) * self.h

    def cell_peclet(self, mu: np.ndarray, D: np.ndarray) -> np.ndarray:
        """Elementwise cell Peclet number Pe_h = |mu|*h/D (Section 3.1).

        D=0 entries are mapped to +inf (pure advection, unresolved regardless of h).
        """
        D = np.asarray(D, dtype=float)
        mu = np.asarray(mu, dtype=float)
        with np.errstate(divide="ignore"):
            pe = np.where(D > 0, np.abs(mu) * self.h / np.where(D > 0, D, 1.0), np.inf)
        return pe

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"Grid1D(x_min={self.x_min}, x_max={self.x_max}, n={self.n}, h={self.h:.6g})"
