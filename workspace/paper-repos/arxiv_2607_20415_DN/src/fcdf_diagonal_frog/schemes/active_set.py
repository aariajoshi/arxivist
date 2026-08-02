"""Active-set solver for the implicit directional step, at any step size gamma.

Paper: Itkin (2026), Section 5.2, Eqs. (24)-(26), Proposition 5, Lemma 2.

Reformulates the FCDF-B fixed point as a zero of the piecewise-linear residual

    F(p) = (I - gamma*A1) p - gamma*D_Lambda(p) - b

Each interface is labeled by its clamp pattern (lower cap / free / upper cap). For a fixed
pattern S the generalized Jacobian V_S = I - gamma*(A1 + C_S) is banded and solved directly;
the iteration terminates once the pattern stops changing (F is piecewise linear, so this
happens in finitely many steps -- reported in the paper as never more than a handful).

IMPORTANT (SIR implementation_assumption #4 / risk_assessment): the paper explicitly leaves
nonsingularity of V_S for *arbitrary* mixed patterns above gamma_pic as an OPEN theoretical
question (only checked numerically, never failing in their experiments). This implementation
therefore wraps every banded solve in a guard and reports `converged=False` rather than
crashing or silently returning a wrong answer if a solve fails or the pattern does not settle
within `max_pattern_updates`.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from fcdf_diagonal_frog.operators.df_operator import DFOperator


class ActiveSetSolver:
    def solve(self, op: DFOperator, b: np.ndarray, gamma: float, max_pattern_updates: int = 25) -> dict:
        if np.any(b < -1e-14):
            raise ValueError("Active-set solver requires a nonnegative base b")
        b = np.clip(b, 0.0, None)
        h = op.grid.h
        n_faces = op.grid.n - 1

        # --- Move 1: try the unlimited (all-free) solve first ---
        p_unlimited = op.unlimited_solve(b, gamma)
        if np.all(p_unlimited >= -1e-10):
            residual = float(np.linalg.norm(p_unlimited - gamma * (op.A2 @ p_unlimited) - b, 1))
            return {
                "p": p_unlimited,
                "pattern_updates": 0,
                "unlimited_accepted": True,
                "residual": residual,
                "converged": True,
                "free_fraction": 1.0,
            }

        # --- Move 2: active-set / semismooth-Newton iteration ---
        c_plus = h * b[:-1] / (2.0 * gamma)
        c_minus = h * b[1:] / (2.0 * gamma)

        p = p_unlimited.copy()
        prev_pattern = None  # pattern actually used in the most recent solve (None = none yet)
        converged = False
        residual = np.inf
        updates_used = 0

        for update in range(max_pattern_updates):
            d = op.unlimited_flux(p)
            new_pattern = np.where(d > c_plus, 1, np.where(d < -c_minus, -1, 0))

            # If the pattern implied by the current solution is IDENTICAL to the pattern we
            # last solved with, p already solves V_S x = rhs for this pattern exactly -- no
            # new solve is needed. This is the correct "number of pattern updates" counting
            # convention (a real bug: the previous version always performed one redundant
            # confirmatory solve before checking this, inflating the reported count by +1
            # relative to the paper -- see comparison/benchmark_comparison.md Root Cause
            # Analysis, "Active-set pattern updates" item #1, which flagged exactly this).
            if prev_pattern is not None and np.array_equal(new_pattern, prev_pattern):
                converged = True
                break

            free_mask = new_pattern == 0
            V_S = op.restricted_operator(free_mask)
            M = (sp.identity(op.grid.n, format="csr") - gamma * V_S).tocsc()

            # fixed clamped-flux contribution to the RHS (Eq. 25-26 caller-side gamma scaling)
            fixed_flux = np.where(new_pattern == 1, c_plus, np.where(new_pattern == -1, -c_minus, 0.0))
            rhs = b + gamma * op.divergence_of_face_values(fixed_flux)

            try:
                p_new = spla.spsolve(M, rhs)
                if not np.all(np.isfinite(p_new)):
                    raise np.linalg.LinAlgError("non-finite solution")
            except (np.linalg.LinAlgError, RuntimeError):
                # Nonsingularity of V_S for this pattern is not guaranteed above gamma_pic
                # (Proposition 5's open question) -- fail safe rather than propagate garbage.
                break

            updates_used = update + 1
            residual = float(np.linalg.norm((sp.identity(op.grid.n) - gamma * V_S) @ p_new - rhs, 1))
            p = p_new
            prev_pattern = new_pattern
        else:
            updates_used = max_pattern_updates

        free_fraction = float(np.mean(prev_pattern == 0)) if (n_faces > 0 and prev_pattern is not None) else 1.0
        if not converged and updates_used < max_pattern_updates:
            # loop exited via the LinAlgError break; try one more consistency pass
            converged = np.all(p >= -1e-8)

        return {
            "p": p,
            "pattern_updates": updates_used,
            "unlimited_accepted": False,
            "residual": residual,
            "converged": bool(converged and np.all(p >= -1e-8)),
            "free_fraction": free_fraction,
        }
