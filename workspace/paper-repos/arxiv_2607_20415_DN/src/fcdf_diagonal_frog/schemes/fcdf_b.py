"""Scheme FCDF-B: per-interface Zalesak-limited Picard iteration.

Paper: Itkin (2026), Section 3, Eqs. (13)-(15), Proposition 1.

    Lambda_{i+1/2} = clamp(d_{i+1/2}(p), caps from b)          (Eq. 15 / Lemma 4)
    (I - gamma*A1) p^[k+1] = b + gamma * Div(Lambda(p^[k]))    (Eq. 14)
    p^[0] = b

This is the primary scheme of the paper: unconditionally positive and exactly
mass-conservative for *every* gamma > 0 and every limiter value (Proposition 1(i)-(ii)),
with a Picard sweep that contracts geometrically in discrete l1 norm for
gamma < gamma_pic = h/(2*mu_bar) (Proposition 1(iii)).
"""
from __future__ import annotations

import numpy as np

from fcdf_diagonal_frog.operators.df_operator import DFOperator
from fcdf_diagonal_frog.limiter.zalesak import ZalesakLimiter


class FCDF_B_Solver:
    """Primary scheme of the paper (Proposition 1)."""

    def step(
        self,
        op: DFOperator,
        limiter: ZalesakLimiter,
        b: np.ndarray,
        gamma: float,
        tol: float = 1e-12,
        max_sweeps: int = 200,
    ) -> dict:
        if np.any(b < -1e-14):
            raise ValueError("FCDF-B requires a nonnegative base b (Proposition 1 hypothesis)")
        b = np.clip(b, 0.0, None)
        h = op.grid.h
        p = b.copy()
        residual = np.inf
        sweeps_used = 0
        for k in range(max_sweeps):
            d = op.unlimited_flux(p)  # Eq. 9, unlimited antidiffusive interface flux
            Lambda = limiter.clamp(d, b, gamma, h)  # Eq. 15 / Lemma 4 clamp form
            rhs = b + gamma * op.divergence_of_face_values(Lambda)
            p_new = op.core_solve(rhs, gamma)
            residual = float(np.linalg.norm(p_new - p, 1))
            p = p_new
            sweeps_used = k + 1
            if residual < tol * max(1.0, np.linalg.norm(b, 1)):
                break
        return {"p": p, "sweeps": sweeps_used, "residual": residual}

    @staticmethod
    def picard_contraction_bound(mu_bar: float, h: float) -> float:
        """gamma_pic = h / (2*mu_bar), Proposition 1(iii)."""
        if mu_bar <= 0:
            raise ValueError("mu_bar must be > 0")
        return h / (2.0 * mu_bar)
