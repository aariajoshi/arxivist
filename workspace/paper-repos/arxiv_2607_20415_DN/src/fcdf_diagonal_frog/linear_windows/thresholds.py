"""Linear positivity-window thresholds gamma_0 (backward-Euler resolvent) and gamma_r
(Pade(0,2) map), computed numerically by bisection on the sign of the most negative entry
of the respective map -- exactly the operational procedure described in Section 6.1.

IMPORTANT (SIR ambiguity #1 / implementation_assumption #1): the closed-form
characterizations of gamma_0 and gamma_r live in the companion paper
[Itkin and Kazbek, 2026], which is "in preparation" and not available to cross-check. This
module measures the thresholds numerically rather than re-deriving them analytically --
documented as a deliberate, flagged implementation choice, not a paper-derived formula.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from fcdf_diagonal_frog.operators.df_operator import DFOperator


def _is_nonnegative_map(M_inv_apply, n: int, tol: float = -1e-10) -> bool:
    """Check entrywise nonnegativity of M^{-1} by applying it to each unit basis vector
    (M^{-1} is dense in general, so for the modest mesh sizes used here -- up to a few
    thousand -- this direct column-by-column check is affordable and exact)."""
    for k in range(n):
        e_k = np.zeros(n)
        e_k[k] = 1.0
        col = M_inv_apply(e_k)
        if np.any(col < tol):
            return False
    return True


class LinearWindowThresholds:
    """Bisection search for gamma_0 and gamma_r (Section 6.1)."""

    def __init__(self, lo: float = 1e-8, hi: float = 1e6, xtol: float = 1e-3, max_bisections: int = 60):
        self.lo = lo
        self.hi = hi
        self.xtol = xtol
        self.max_bisections = max_bisections

    def gamma_0(self, op: DFOperator) -> float | None:
        """Backward-Euler resolvent threshold: smallest gamma such that (I-gamma*A2)^-1 >= 0
        entrywise, and remains so for all larger gamma (eventual positivity, monotone search)."""
        n = op.grid.n

        def resolvent_ok(gamma: float) -> bool:
            M = (sp.identity(n, format="csr") - gamma * op.A2).tocsc()
            lu = spla.splu(M)
            return _is_nonnegative_map(lambda e: lu.solve(e), n)

        return self._bisect_monotone_threshold(resolvent_ok)

    def gamma_r(self, op: DFOperator) -> float | None:
        """Pade(0,2) threshold: smallest gamma such that
        r02(gamma*A2) = (I - gamma*A2 + gamma^2/2*A2^2)^-1 >= 0 entrywise."""
        n = op.grid.n
        A2 = op.A2

        def pade_ok(gamma: float) -> bool:
            M = (sp.identity(n, format="csr") - gamma * A2 + (gamma ** 2 / 2.0) * (A2 @ A2)).tocsc()
            lu = spla.splu(M)
            return _is_nonnegative_map(lambda e: lu.solve(e), n)

        return self._bisect_monotone_threshold(pade_ok)

    def _bisect_monotone_threshold(self, predicate) -> float | None:
        """Assumes `predicate(gamma)` is False below the threshold and True above it
        (eventual positivity). Returns None if `predicate(hi)` is still False (paper's
        'no window found below the search bound' case, Table 2/Figure 1 open symbols)."""
        lo, hi = self.lo, self.hi
        if not predicate(hi):
            return None
        if predicate(lo):
            return lo
        for _ in range(self.max_bisections):
            mid = 0.5 * (lo + hi)
            if predicate(mid):
                hi = mid
            else:
                lo = mid
            if (hi - lo) < self.xtol * max(1.0, hi):
                break
        return hi
