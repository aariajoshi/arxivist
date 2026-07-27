"""Tests for DFOperator assembly (Section 2, Eqs. 4-9)."""
import numpy as np
import pytest

from fcdf_diagonal_frog.operators.df_operator import DFOperator
from fcdf_diagonal_frog.operators.grid import Grid1D


def test_conservation_uniform_positive_drift():
    grid = Grid1D(0.0, 1.0, 30)
    mu = np.full(grid.n, 1.0)
    D = np.full(grid.n, 0.05)
    op = DFOperator.assemble(grid, mu, D)
    for M in (op.A1, op.A2, op.C):
        col_sums = np.asarray(M.sum(axis=0)).ravel()
        assert np.max(np.abs(col_sums)) < 1e-10


def test_conservation_sign_changing_drift():
    """OU-like sign-changing drift (mu(x) = -x) must still conserve mass exactly."""
    grid = Grid1D(-3.0, 3.0, 41)
    mu = -grid.x
    D = np.full(grid.n, 0.5)
    op = DFOperator.assemble(grid, mu, D)
    for M in (op.A1, op.A2, op.C):
        col_sums = np.asarray(M.sum(axis=0)).ravel()
        assert np.max(np.abs(col_sums)) < 1e-10


def test_A1_A2_match_paper_formulas_interior_row():
    """Hand-check an interior row (mu>0 uniformly) against paper Eqs. (5) and (7)."""
    grid = Grid1D(0.0, 1.0, 20)
    mu = np.full(grid.n, 1.0)
    D = np.full(grid.n, 0.05)
    op = DFOperator.assemble(grid, mu, D)
    A1, A2 = op.A1.toarray(), op.A2.toarray()
    h = grid.h
    i = 6  # safely interior, i-2 >= 0

    beta_p = D[i - 1] / h**2 + mu[i - 1] / h
    gamma_p = -2 * D[i] / h**2 - mu[i] / h
    delta_p = D[i + 1] / h**2
    assert np.allclose([A1[i, i - 1], A1[i, i], A1[i, i + 1]], [beta_p, gamma_p, delta_p])

    alpha = -mu[i - 2] / (2 * h)
    beta = D[i - 1] / h**2 + 2 * mu[i - 1] / h
    gam = -2 * D[i] / h**2 - 3 * mu[i] / (2 * h)
    delta = D[i + 1] / h**2
    assert np.allclose([A2[i, i - 2], A2[i, i - 1], A2[i, i], A2[i, i + 1]], [alpha, beta, gam, delta])


def test_A1_is_metzler_with_negative_diagonal():
    """A1 Metzler (off-diagonals >= 0); (I - gamma*A1) should then be a nonsingular M-matrix."""
    grid = Grid1D(0.0, 1.0, 20)
    mu = np.full(grid.n, 1.0)
    D = np.full(grid.n, 0.05)
    op = DFOperator.assemble(grid, mu, D)
    A1 = op.A1.toarray()
    n = grid.n
    off_diag_mask = ~np.eye(n, dtype=bool)
    assert np.all(A1[off_diag_mask] >= -1e-12)


def test_core_resolvent_is_entrywise_nonnegative_and_unit_column_sum():
    """Lemma 3: M^{-1} > 0 entrywise, 1^T M^{-1} = 1^T, so induced l1 norm is exactly 1."""
    grid = Grid1D(0.0, 1.0, 25)
    mu = np.full(grid.n, 1.0)
    D = np.full(grid.n, 0.05)
    op = DFOperator.assemble(grid, mu, D)
    gamma = 0.01
    n = grid.n
    cols = []
    for k in range(n):
        e = np.zeros(n)
        e[k] = 1.0
        cols.append(op.core_solve(e, gamma))
    Minv = np.column_stack(cols)
    assert np.all(Minv > -1e-12)
    col_sums = Minv.sum(axis=0)
    assert np.allclose(col_sums, 1.0, atol=1e-8)
