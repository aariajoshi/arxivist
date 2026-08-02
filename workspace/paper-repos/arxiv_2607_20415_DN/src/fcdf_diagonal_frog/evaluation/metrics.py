"""Evaluation metrics used throughout Section 6.

Discrete L1 norm ||v||_{1,h} = h * sum_i |v_i|, successive-halving observed order (not a
single least-squares slope, per the paper's stated methodology), mass-defect and
positivity checks.
"""
from __future__ import annotations

import numpy as np


class Metrics:
    @staticmethod
    def l1_error(p: np.ndarray, p_exact: np.ndarray, h: float) -> float:
        return float(h * np.sum(np.abs(p - p_exact)))

    @staticmethod
    def l1_norm(v: np.ndarray, h: float) -> float:
        return float(h * np.sum(np.abs(v)))

    @staticmethod
    def observed_order(errors: list) -> list:
        """Successive-halving convergence order: order_k = log2(err_{k-1}/err_k), assuming
        each successive mesh in `errors` halves h relative to the previous one."""
        errors = np.asarray(errors, dtype=float)
        orders = [None]
        for k in range(1, len(errors)):
            if errors[k] <= 0 or errors[k - 1] <= 0:
                orders.append(None)
            else:
                orders.append(float(np.log2(errors[k - 1] / errors[k])))
        return orders

    @staticmethod
    def mass_defect(p: np.ndarray, p0: np.ndarray) -> float:
        return float(abs(np.sum(p) - np.sum(p0)))

    @staticmethod
    def min_nodal_value(p: np.ndarray) -> float:
        return float(np.min(p))
