"""Tests for Scheme FCDF-B (Proposition 1)."""
import numpy as np

from fcdf_diagonal_frog.limiter.zalesak import ZalesakLimiter
from fcdf_diagonal_frog.operators.df_operator import DFOperator
from fcdf_diagonal_frog.operators.grid import Grid1D
from fcdf_diagonal_frog.schemes.fcdf_b import FCDF_B_Solver
from fcdf_diagonal_frog.schemes.unlimited import UnlimitedSolver


def _front_ic(x):
    return np.where((x >= 0.1) & (x <= 0.4), 1.0, 0.0)


def test_fcdf_b_unconditional_positivity_on_front_at_small_and_large_gamma():
    """Proposition 1(i): every FCDF-B iterate/fixed point is nonnegative for every gamma>0,
    even where the unlimited scheme (Table 7) is known to go negative."""
    grid = Grid1D(0.0, 1.0, 101)
    mu = np.full(grid.n, 1.0)
    D = np.full(grid.n, 1.0e-4)
    op = DFOperator.assemble(grid, mu, D)
    limiter = ZalesakLimiter()
    solver = FCDF_B_Solver()
    b = _front_ic(grid.x)

    for gamma in (1e-4, 5e-2, 10.0):
        out = solver.step(op, limiter, b, gamma)
        assert np.all(out["p"] >= -1e-10), f"FCDF-B went negative at gamma={gamma}"


def test_fcdf_b_exact_mass_conservation():
    """Proposition 1(ii): exact conservation for every limiter value."""
    grid = Grid1D(0.0, 1.0, 101)
    mu = np.full(grid.n, 1.0)
    D = np.full(grid.n, 1.0e-4)
    op = DFOperator.assemble(grid, mu, D)
    limiter = ZalesakLimiter()
    solver = FCDF_B_Solver()
    b = _front_ic(grid.x)
    out = solver.step(op, limiter, b, 5e-2)
    assert abs(np.sum(out["p"]) - np.sum(b)) < 1e-10


def test_unlimited_scheme_goes_negative_on_front_small_gamma():
    """Reproduces the paper's headline Table 7 finding: the unlimited scheme is the only
    one that can go negative, and it does so on the front benchmark at a small step."""
    grid = Grid1D(0.0, 1.0, 101)
    mu = np.full(grid.n, 1.0)
    D = np.full(grid.n, 1.0e-4)
    op = DFOperator.assemble(grid, mu, D)
    b = _front_ic(grid.x)
    unl = UnlimitedSolver()
    p = unl.step(op, b, 2.0e-4)
    assert np.min(p) < -1e-6, "expected the unlimited scheme to undershoot on this benchmark"


def test_fcdf_b_matches_unlimited_on_smooth_slack_case():
    """Proposition 1(iv): when caps are slack everywhere (smooth, well-resolved density),
    the FCDF-B fixed point coincides with the unlimited solution."""
    grid = Grid1D(-3.0, 3.0, 101)
    mu = -0.0 * grid.x  # zero drift, pure diffusion -> very well resolved, caps slack
    D = np.full(grid.n, 1.0)
    op = DFOperator.assemble(grid, mu, D)
    limiter = ZalesakLimiter()
    b = np.exp(-0.5 * grid.x**2)
    b = b / (np.sum(b) * grid.h)
    solver = FCDF_B_Solver()
    unl = UnlimitedSolver()
    p_b = solver.step(op, limiter, b, 1e-3)["p"]
    p_u = unl.step(op, b, 1e-3)
    assert np.allclose(p_b, p_u, atol=1e-8)


def test_picard_contraction_bound_formula():
    assert abs(FCDF_B_Solver.picard_contraction_bound(2.0, 0.1) - 0.025) < 1e-12
