"""Baseline: backward Euler on the monotone core A1 alone.

Paper: Itkin (2026), Section 6 -- "the monotone core, the first-order positive floor."
Unconditionally positive (A1 is an M-matrix for every gamma>0) but only ever first-order
accurate, since it discards the antidiffusive correction entirely.
"""
from __future__ import annotations

import numpy as np

from fcdf_diagonal_frog.operators.df_operator import DFOperator


class MonotoneCoreSolver:
    def step(self, op: DFOperator, b: np.ndarray, gamma: float) -> np.ndarray:
        return op.core_solve(b, gamma)
