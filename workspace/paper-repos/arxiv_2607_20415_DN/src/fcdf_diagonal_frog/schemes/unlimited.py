"""Baseline: backward Euler on the full 2nd-order operator A2, with no limiting at all.

Paper: Itkin (2026), Section 6 -- "the unlimited linear scheme ... included to show where
positivity fails." This is the scheme Godunov's theorem forbids from being simultaneously
positive and 2nd-order for all gamma; Table 7 shows it produces a negative density at a
small step on the front benchmark.
"""
from __future__ import annotations

import numpy as np

from fcdf_diagonal_frog.operators.df_operator import DFOperator


class UnlimitedSolver:
    def step(self, op: DFOperator, b: np.ndarray, gamma: float) -> np.ndarray:
        return op.unlimited_solve(b, gamma)
