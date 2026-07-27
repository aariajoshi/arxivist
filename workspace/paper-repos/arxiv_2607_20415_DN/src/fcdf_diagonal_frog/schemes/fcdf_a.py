"""Scheme FCDF-A: global-stopping-rule Picard iteration.

Paper: Itkin (2026), Section 3, Eq. (12).

    (I - gamma*A1) p^[k+1] = b + gamma*C p^[k],   p^[0] = b

Terminates at the last iterate whose right-hand side is entrywise nonnegative -- a direct
analogue of the runtime sign-monitoring rule in the companion Diagonal Frog paper. Cheap,
but a single node's sign failure at some sweep discards the *entire* correction for that
sweep (this is the defect Scheme FCDF-B is designed to fix -- see fcdf_b.py).
"""
from __future__ import annotations

import numpy as np

from fcdf_diagonal_frog.operators.df_operator import DFOperator


class FCDF_A_Solver:
    """Global-stopping-rule Picard solver (Eq. 12)."""

    def step(self, op: DFOperator, b: np.ndarray, gamma: float, max_sweeps: int = 50) -> dict:
        if np.any(b < -1e-14):
            raise ValueError("FCDF-A requires a nonnegative base b")
        b = np.clip(b, 0.0, None)
        p = b.copy()
        last_good = b.copy()
        for k in range(max_sweeps):
            rhs = b + gamma * (op.C @ p)
            if np.all(rhs >= -1e-13):
                p_new = op.core_solve(rhs, gamma)
                last_good = p_new
                if np.linalg.norm(p_new - p, 1) < 1e-12 * max(1.0, np.linalg.norm(p, 1)):
                    p = p_new
                    break
                p = p_new
            else:
                # sign failure: discard this sweep's correction entirely, stop at last_good
                break
        return {"p": last_good, "sweeps": k + 1}
