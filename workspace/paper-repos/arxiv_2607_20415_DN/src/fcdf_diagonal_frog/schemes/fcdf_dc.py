"""Scheme FCDF-DC: defect-corrected two-stage time stepper, 2nd order in time.

Paper: Itkin (2026), Section 4, Eqs. (17)-(21), Lemma 1, Proposition 3.

Predictor (Eq. 17):   (I - dt*A2) Y0    = p^n
Corrector (Eq. 18):   (I - dt*A2) p^{n+1} = p^n - (dt^2/2) A2^2 Y0   (Lemma 1 defect identity)

Both stages are realized purely by limited backward-Euler-type solves -- never an explicit
half-step, which would be sign-indefinite under a parabolic-scale step. The combined clamp
(Eq. 21, Section 4.2) folds the fixed defect flux G into the same budget-based clamp used by
FCDF-B, with G=0 recovering exactly the FCDF-B predictor.
"""
from __future__ import annotations

import numpy as np

from fcdf_diagonal_frog.operators.df_operator import DFOperator
from fcdf_diagonal_frog.limiter.zalesak import ZalesakLimiter


class FCDF_DC_Solver:
    """Two-stage defect-corrected scheme (Proposition 3)."""

    def _limited_stage(
        self,
        op: DFOperator,
        limiter: ZalesakLimiter,
        p_n: np.ndarray,
        dt: float,
        G: np.ndarray,
        tol: float = 1e-12,
        max_sweeps: int = 200,
    ) -> dict:
        h = op.grid.h
        p = p_n.copy()
        residual = np.inf
        sweeps_used = 0
        for k in range(max_sweeps):
            d = op.unlimited_flux(p)
            Lambda = limiter.clamp_with_offset(d, G, p_n, dt, h)  # Eq. 21
            rhs = p_n + op.divergence_of_face_values(Lambda)
            p_new = op.core_solve(rhs, dt)
            residual = float(np.linalg.norm(p_new - p, 1))
            p = p_new
            sweeps_used = k + 1
            if residual < tol * max(1.0, np.linalg.norm(p_n, 1)):
                break
        return {"p": p, "sweeps": sweeps_used, "residual": residual}

    def step(self, op: DFOperator, limiter: ZalesakLimiter, p_n: np.ndarray, dt: float) -> dict:
        if np.any(p_n < -1e-14):
            raise ValueError("FCDF-DC requires a nonnegative base p_n (Proposition 3 hypothesis)")
        p_n = np.clip(p_n, 0.0, None)
        n_faces = op.grid.n - 1

        # Predictor (Eq. 17): G=0 recovers exactly the FCDF-B sweep.
        zero_G = np.zeros(n_faces)
        pred = self._limited_stage(op, limiter, p_n, dt, zero_G)
        Y0 = pred["p"]

        # Defect flux (Section 4.2): G_{i+1/2} = (dt^2/2) * Jhat_{i+1/2}(w), w = A2 @ Y0,
        # combined with s_i = -(G_{i+1/2}-G_{i-1/2})/h = divergence_of_face_values(G).
        # Algebraic check against Lemma 1 (s must equal -(dt^2/2)*A2*w exactly, since
        # p^{n+1} = p^n - (dt^2/2) A2^2 Y0): since divergence_of_face_values(total_flux_A2(w))
        # == A2 @ w by construction (both implement (A2 p)_i = -(Jhat_{i+1/2}-Jhat_{i-1/2})/h),
        # matching s requires the NEGATIVE sign below (caught by test_fcdf_dc_and_active_set.py
        # test_fcdf_dc_second_order_beats_fcdf_b_on_smooth_ou_like_problem, which failed loudly
        # with the positive sign: FCDF-DC was *less* accurate than 1st-order FCDF-B, impossible
        # for a correctly-signed 2nd-order defect correction).
        w = op.apply_A2(Y0)
        G = -(dt ** 2 / 2.0) * op.total_flux_A2(w)

        # Corrector (Eq. 18, via Lemma 1's defect identity).
        corr = self._limited_stage(op, limiter, p_n, dt, G)
        p_next = corr["p"]

        return {
            "p_next": p_next,
            "Y0": Y0,
            "sweeps_predictor": pred["sweeps"],
            "sweeps_corrector": corr["sweeps"],
            "residual_predictor": pred["residual"],
            "residual_corrector": corr["residual"],
        }
