"""Chang-Cooper (1970) exponentially-fitted baseline scheme.

Paper: Itkin (2026), Section 3.1 and 6.3 -- used as the "natural alternative" whose global,
solution-independent degradation to first order at large cell Peclet number contrasts with
FCDF-B's local, solution-dependent degradation confined to unresolved layers.

Standard Scharfetter-Gummel/Chang-Cooper flux at interface j (between nodes j, j+1):

    Pe_face = mu_face * h / D_face                      (D_face = 0.5*(D[j]+D[j+1]))
    delta(Pe) = 1/Pe - 1/(exp(Pe)-1)          (exponential fitting weight, Pe->0 gives 1/2)
    J_hat_{j+1/2} = mu_face*[(1-delta)*p[j] + delta*p[j+1]] - D_face*(p[j+1]-p[j])/h

This reduces, as Pe_face -> +-infinity, to pure upwind differencing (delta -> 0 or 1), which
is the mechanism behind its *global* first-order degradation at high Peclet number (Section
3.1) -- independent of the solution, unlike FCDF-B's solution-driven limiter.

Assembled interface-by-interface so it remains valid under variable and sign-changing drift
(needed for the OU benchmark, Section 6.1-6.2 Table 3).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from fcdf_diagonal_frog.operators.grid import Grid1D


def _cc_delta(pe: np.ndarray) -> np.ndarray:
    """Exponential-fitting weight delta(Pe) = 1/Pe - 1/(exp(Pe)-1), with the removable
    singularity at Pe=0 handled by a Taylor-series fallback (delta(0)=1/2)."""
    out = np.empty_like(pe)
    small = np.abs(pe) < 1e-8
    out[small] = 0.5 - pe[small] / 12.0  # Taylor expansion near Pe=0
    pe_reg = pe[~small]
    out[~small] = 1.0 / pe_reg - 1.0 / (np.exp(pe_reg) - 1.0)
    return out


@dataclass
class ChangCooperOperator:
    grid: Grid1D
    A: sp.csr_matrix

    @classmethod
    def assemble(cls, grid: Grid1D, mu: np.ndarray, D: np.ndarray) -> "ChangCooperOperator":
        n, h = grid.n, grid.h
        n_faces = n - 1
        mu_face = 0.5 * (mu[:-1] + mu[1:])
        D_face = 0.5 * (D[:-1] + D[1:])
        D_face_safe = np.where(D_face > 0, D_face, 1e-300)
        pe = mu_face * h / D_face_safe
        delta = _cc_delta(pe)

        F = sp.lil_matrix((n_faces, n))
        for j in range(n_faces):
            # advective part: mu_face*[(1-delta)*p_j + delta*p_{j+1}]
            F[j, j] += mu_face[j] * (1.0 - delta[j])
            F[j, j + 1] += mu_face[j] * delta[j]
            # diffusive part (same sign convention as DFOperator: +D[j]/h, -D[j+1]/h times p)
            F[j, j] += D_face[j] / h
            F[j, j + 1] += -D_face[j] / h
        F = F.tocsr()

        Div = sp.lil_matrix((n, n_faces))
        for i in range(n):
            if i - 1 >= 0:
                Div[i, i - 1] += 1.0 / h
            if i <= n_faces - 1:
                Div[i, i] += -1.0 / h
        Div = Div.tocsr()

        A = (Div @ F).tocsr()
        return cls(grid=grid, A=A)

    def step(self, p: np.ndarray, gamma: float) -> np.ndarray:
        """Backward-Euler step: (I - gamma*A) p_new = p."""
        n = self.grid.n
        M = (sp.identity(n, format="csr") - gamma * self.A).tocsc()
        return spla.spsolve(M, p)
