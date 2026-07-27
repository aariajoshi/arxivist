"""Dispatch helper mapping a config scheme name to a runnable step function.

Not mentioned as a separate module in the architecture plan, but factored out here to avoid
duplicating the scheme-selection logic across train.py / evaluate.py / inference.py.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from fcdf_diagonal_frog.limiter.zalesak import ZalesakLimiter
from fcdf_diagonal_frog.operators.chang_cooper import ChangCooperOperator
from fcdf_diagonal_frog.operators.df_operator import DFOperator
from fcdf_diagonal_frog.schemes.active_set import ActiveSetSolver
from fcdf_diagonal_frog.schemes.fcdf_a import FCDF_A_Solver
from fcdf_diagonal_frog.schemes.fcdf_b import FCDF_B_Solver
from fcdf_diagonal_frog.schemes.fcdf_dc import FCDF_DC_Solver
from fcdf_diagonal_frog.schemes.monotone_core import MonotoneCoreSolver
from fcdf_diagonal_frog.schemes.unlimited import UnlimitedSolver

TWO_STAGE_SCHEMES = {"fcdf_dc"}


def build_backward_euler_stepper(scheme: str, cfg_model: dict) -> Callable[[DFOperator, np.ndarray, float], np.ndarray]:
    """Returns step(op, b, gamma) -> p_next for the one-stage (theta=1) schemes."""
    limiter = ZalesakLimiter(kappa=cfg_model.get("zalesak_kappa", 2))
    tol = cfg_model.get("picard_tol", 1e-12)
    max_sweeps = cfg_model.get("picard_max_sweeps", 200)
    max_updates = cfg_model.get("active_set_max_pattern_updates", 25)

    if scheme == "fcdf_a":
        solver = FCDF_A_Solver()
        return lambda op, b, gamma: solver.step(op, b, gamma, max_sweeps=max_sweeps)["p"]
    if scheme == "fcdf_b":
        solver = FCDF_B_Solver()
        return lambda op, b, gamma: solver.step(op, limiter, b, gamma, tol=tol, max_sweeps=max_sweeps)["p"]
    if scheme == "monotone_core":
        solver = MonotoneCoreSolver()
        return lambda op, b, gamma: solver.step(op, b, gamma)
    if scheme == "unlimited":
        solver = UnlimitedSolver()
        return lambda op, b, gamma: solver.step(op, b, gamma)
    if scheme == "active_set":
        solver = ActiveSetSolver()
        return lambda op, b, gamma: solver.solve(op, b, gamma, max_pattern_updates=max_updates)["p"]
    if scheme == "chang_cooper":
        cache: dict = {}

        def _cc_step(op: DFOperator, b: np.ndarray, gamma: float) -> np.ndarray:
            key = id(op)
            if key not in cache:
                cache[key] = ChangCooperOperator.assemble(op.grid, op.mu, op.D)
            return cache[key].step(b, gamma)

        return _cc_step
    raise ValueError(f"'{scheme}' is not a one-stage (backward-Euler) scheme; use build_two_stage_stepper")


def build_two_stage_stepper(cfg_model: dict):
    """Returns step(op, p_n, dt) -> dict (FCDF-DC, the only two-stage scheme)."""
    limiter = ZalesakLimiter(kappa=cfg_model.get("zalesak_kappa", 2))
    solver = FCDF_DC_Solver()
    return lambda op, p_n, dt: solver.step(op, limiter, p_n, dt)
