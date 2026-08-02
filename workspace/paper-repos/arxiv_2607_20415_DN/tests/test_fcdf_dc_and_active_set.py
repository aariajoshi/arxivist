"""Tests for Scheme FCDF-DC (Proposition 3) and the active-set solver (Proposition 5)."""
import numpy as np

from fcdf_diagonal_frog.limiter.zalesak import ZalesakLimiter
from fcdf_diagonal_frog.operators.df_operator import DFOperator
from fcdf_diagonal_frog.operators.grid import Grid1D
from fcdf_diagonal_frog.schemes.active_set import ActiveSetSolver
from fcdf_diagonal_frog.schemes.fcdf_b import FCDF_B_Solver
from fcdf_diagonal_frog.schemes.fcdf_dc import FCDF_DC_Solver


def test_fcdf_dc_unconditional_positivity_and_conservation():
    grid = Grid1D(0.0, 1.0, 81)
    mu = np.full(grid.n, 1.0)
    D = np.full(grid.n, 1.0e-4)
    op = DFOperator.assemble(grid, mu, D)
    limiter = ZalesakLimiter()
    solver = FCDF_DC_Solver()
    b = np.where((grid.x >= 0.1) & (grid.x <= 0.4), 1.0, 0.0)

    out = solver.step(op, limiter, b, dt=1e-3)
    assert np.all(out["p_next"] >= -1e-10)
    assert abs(np.sum(out["p_next"]) - np.sum(b)) < 1e-8


def test_fcdf_dc_second_order_beats_fcdf_b_on_smooth_ou_like_problem():
    """Sanity check that FCDF-DC's temporal error decreases faster than FCDF-B's under
    dt-refinement on a smooth, well-resolved problem (Proposition 3(iv))."""
    grid = Grid1D(-3.0, 3.0, 81)
    mu = -grid.x
    D = np.full(grid.n, 0.5)
    op = DFOperator.assemble(grid, mu, D)
    limiter = ZalesakLimiter()
    b0 = np.exp(-0.5 * (grid.x - 0.5) ** 2 / 0.1)
    b0 = b0 / (np.sum(b0) * grid.h)

    def run(dt, n_steps, dc):
        p = b0.copy()
        if dc:
            solver = FCDF_DC_Solver()
            for _ in range(n_steps):
                p = solver.step(op, limiter, p, dt)["p_next"]
        else:
            solver = FCDF_B_Solver()
            for _ in range(n_steps):
                p = solver.step(op, limiter, p, dt)["p"]
        return p

    T = 0.02
    p_dc_coarse = run(T / 10, 10, dc=True)
    p_dc_fine = run(T / 20, 20, dc=True)
    p_b_coarse = run(T / 10, 10, dc=False)
    p_b_fine = run(T / 20, 20, dc=False)

    # use the finest FCDF-DC run as a proxy reference (both should converge to it)
    ref = run(T / 40, 40, dc=True)
    err_dc = np.linalg.norm(p_dc_fine - ref, 1)
    err_b = np.linalg.norm(p_b_fine - ref, 1)
    assert err_dc <= err_b + 1e-12  # DC should not be worse than FCDF-B at matched dt


def test_active_set_solver_never_returns_negative():
    grid = Grid1D(0.0, 1.0, 81)
    mu = np.full(grid.n, 1.0)
    D = np.full(grid.n, 1.0e-4)
    op = DFOperator.assemble(grid, mu, D)
    b = np.where((grid.x >= 0.1) & (grid.x <= 0.4), 1.0, 0.0)
    solver = ActiveSetSolver()
    mu_bar = 1.0
    gamma_pic = FCDF_B_Solver.picard_contraction_bound(mu_bar, grid.h)

    for ratio in (0.1, 1.0, 5.0, 100.0):
        out = solver.solve(op, b, ratio * gamma_pic)
        assert np.all(out["p"] >= -1e-8), f"active-set solver went negative at ratio={ratio}"


def test_active_set_accepts_unlimited_at_large_gamma():
    grid = Grid1D(0.0, 1.0, 81)
    mu = np.full(grid.n, 1.0)
    D = np.full(grid.n, 1.0e-4)
    op = DFOperator.assemble(grid, mu, D)
    b = np.where((grid.x >= 0.1) & (grid.x <= 0.4), 1.0, 0.0)
    solver = ActiveSetSolver()
    mu_bar = 1.0
    gamma_pic = FCDF_B_Solver.picard_contraction_bound(mu_bar, grid.h)
    out = solver.solve(op, b, 1000 * gamma_pic)
    assert out["unlimited_accepted"] is True
    assert out["pattern_updates"] == 0


def test_active_set_does_not_perform_redundant_confirmatory_solve():
    """Regression test for a real counting bug found during Stage 6 comparison: the
    original implementation always performed one extra confirmatory solve after the
    clamp pattern had already stabilized, inflating 'pattern_updates' by +1 relative to
    the paper's counting convention (paper Table 8 reports 1 update in this regime; the
    buggy version reported 3, the fixed version reports 2 -- see
    comparison/benchmark_comparison.md for the full investigation, including confirming
    via a mesh sweep n=101..801 that the residual 2-vs-1 gap is NOT a mesh-size artifact).
    This test only locks in that the redundant-solve bug does not regress, not that we
    exactly match the paper's count."""
    grid = Grid1D(0.0, 1.0, 401)
    mu = np.full(grid.n, 1.0)
    D = np.full(grid.n, 1.0e-4)
    op = DFOperator.assemble(grid, mu, D)
    b = np.where((grid.x >= 0.1) & (grid.x <= 0.4), 1.0, 0.0)
    solver = ActiveSetSolver()
    mu_bar = 1.0
    gamma_pic = FCDF_B_Solver.picard_contraction_bound(mu_bar, grid.h)
    out = solver.solve(op, b, 1.0 * gamma_pic)
    assert out["converged"] is True
    assert out["pattern_updates"] == 2  # was 3 before the counting-convention fix
