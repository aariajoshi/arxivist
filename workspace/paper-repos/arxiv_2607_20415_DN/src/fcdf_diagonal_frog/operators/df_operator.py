"""Diagonal Frog (DF) spatial operator assembly.

Paper: Itkin (2026), "Flux-Corrected Diagonal Frog", Section 2 (the 1D setting and the
core-correction split).

We assemble everything via conservative interface-flux differencing rather than literally
writing out Eqs. (4)-(5)/(7) row by row, because that is the only construction that (a)
generalizes cleanly to sign-changing drift (SIR ambiguity #4 -- the paper only states the
mu>0 case explicitly and says sign-changing drift is handled "directionally at each node"
without giving the exact per-node rule) and (b) *guarantees* exact telescoping mass
conservation (1^T A1 = 1^T A2 = 1^T C = 0) by construction, which Propositions 1-3 all
depend on.

Interface convention: interface j (0 <= j <= n-2) sits between node j and node j+1.
Domain-edge fluxes (before node 0, after node n-1) are fixed at zero -- the "zero-flux
closure" of Section 2.

For each interface we build two competing advective numerical fluxes:
  - g_hat_2nd (2nd-order upwind-biased, Eq. 2 in flux form): used by A2.
  - g_hat_1st (1st-order upwind, Eq. "g_hat^(1)"):            used by A1 (the monotone core).
Both share the same centered diffusive flux contribution (Eq. 3 in flux form).
C = A2 - A1 keeps only the *antidiffusive* difference g_hat_2nd - g_hat_1st (Eq. 8-9);
it carries no diffusion, exactly as the paper states.

Upwind direction per interface is chosen from the sign of the face-averaged drift
mu_face = 0.5*(mu_j + mu_{j+1}) (our explicit, documented resolution of SIR ambiguity #4).
Near a domain edge, where the 2nd-order stencil would need a node outside the grid, we
fall back to the 1st-order flux at that interface (matching the paper's stated near-boundary
treatment).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Union

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from fcdf_diagonal_frog.operators.grid import Grid1D

ArrayOrFunc = Union[np.ndarray, Callable[[np.ndarray], np.ndarray]]


def _as_array(grid: Grid1D, val: ArrayOrFunc) -> np.ndarray:
    if callable(val):
        return np.asarray(val(grid.x), dtype=float)
    arr = np.asarray(val, dtype=float)
    if arr.shape != (grid.n,):
        raise ValueError(f"Expected shape ({grid.n},), got {arr.shape}")
    return arr


@dataclass
class DFOperator:
    """Assembles and holds A1 (M-matrix core), A2 (full 2nd-order operator), and
    C = A2 - A1 (antidiffusive correction) for a fixed (mu, D) pair on a given grid.

    All three are stored as scipy.sparse.csr_matrix. Solves against (I - gamma*A1) use
    scipy.sparse.linalg.splu, which on these narrow-banded 1D matrices is the practical
    equivalent of the paper's "one banded M-matrix solve at O(n) cost" (documented
    implementation choice, SIR risk_assessment item on banded solves).
    """

    grid: Grid1D
    mu: np.ndarray
    D: np.ndarray
    A1: sp.csr_matrix
    A2: sp.csr_matrix
    C: sp.csr_matrix
    upwind_left: np.ndarray  # bool[n-1]: True if interface j is upwinded from the left (mu_face>=0)
    Fc: sp.csr_matrix = None  # (n-1, n) antidiffusive flux matrix, Fc @ p = d_{i+1/2}(p) (Eq. 9)
    Div: sp.csr_matrix = None  # (n, n-1) divergence operator with zero-flux boundary closure

    @classmethod
    def assemble(cls, grid: Grid1D, mu: ArrayOrFunc, D: ArrayOrFunc) -> "DFOperator":
        n, h = grid.n, grid.h
        mu_arr = _as_array(grid, mu)
        D_arr = _as_array(grid, D)
        if np.any(D_arr < 0):
            raise ValueError("Diffusion coefficient D must be >= 0 (paper Section 2)")

        n_faces = n - 1
        mu_face = 0.5 * (mu_arr[:-1] + mu_arr[1:])
        upwind_left = mu_face >= 0.0

        # Build sparse flux matrices F2, F1: shape (n_faces, n) such that
        # (F2 @ p)[j] = total 2nd-order flux at interface j, (F1 @ p)[j] = total 1st-order flux.
        F2 = sp.lil_matrix((n_faces, n))
        F1 = sp.lil_matrix((n_faces, n))

        for j in range(n_faces):
            # --- diffusive part, identical for F1 and F2 (Eq. 3 in flux form) ---
            # J = mu*p - d/dx(D*p); at the interface, -d/dx(D p) ~= (D[j]*p[j] - D[j+1]*p[j+1])/h.
            # So the diffusive contribution to the *total flux* has coefficient +D[j]/h on
            # p[j] and -D[j+1]/h on p[j+1] (verified by hand against paper Eqs. 5 and 7).
            diff_coeff_j = D_arr[j] / h
            diff_coeff_jp1 = -D_arr[j + 1] / h

            if upwind_left[j]:
                # mu>=0 at this face: upwind from the left.
                # 1st order: g_hat_1st = mu[j] * p[j]
                F1[j, j] += mu_arr[j]
                # 2nd order: g_hat_2nd = 0.5*(3*mu[j]*p[j] - mu[j-1]*p[j-1]) if j-1 exists,
                # else fall back to 1st order at this near-boundary interface.
                if j - 1 >= 0:
                    F2[j, j] += 1.5 * mu_arr[j]
                    F2[j, j - 1] += -0.5 * mu_arr[j - 1]
                else:
                    F2[j, j] += mu_arr[j]
            else:
                # mu<0 at this face: upwind from the right.
                # 1st order: g_hat_1st = mu[j+1] * p[j+1]
                F1[j, j + 1] += mu_arr[j + 1]
                # 2nd order: g_hat_2nd = 0.5*(3*mu[j+1]*p[j+1] - mu[j+2]*p[j+2]) if j+2 exists,
                # else fall back to 1st order at this near-boundary interface.
                if j + 2 <= n - 1:
                    F2[j, j + 1] += 1.5 * mu_arr[j + 1]
                    F2[j, j + 2] += -0.5 * mu_arr[j + 2]
                else:
                    F2[j, j + 1] += mu_arr[j + 1]

            # add the (negative-of-diffusive-difference) part to BOTH F1 and F2 (shared core)
            F1[j, j] += diff_coeff_j
            F1[j, j + 1] += diff_coeff_jp1
            F2[j, j] += diff_coeff_j
            F2[j, j + 1] += diff_coeff_jp1

        F1 = F1.tocsr()
        F2 = F2.tocsr()

        # Divergence operator Div: shape (n, n_faces), dp_i/dt = -(J_{i+1/2}-J_{i-1/2})/h
        # with zero-flux boundary closure (J_{-1/2} = J_{n-1/2} = 0).
        Div = sp.lil_matrix((n, n_faces))
        for i in range(n):
            if i - 1 >= 0:  # J_{i-1/2} = Fluxes[i-1] exists
                Div[i, i - 1] += 1.0 / h
            # else: J_{i-1/2} = 0 (left domain edge), contributes nothing
            if i <= n_faces - 1:  # J_{i+1/2} = Fluxes[i] exists
                Div[i, i] += -1.0 / h
            # else: J_{i+1/2} = 0 (right domain edge), contributes nothing
        Div = Div.tocsr()

        A1 = (Div @ F1).tocsr()
        A2 = (Div @ F2).tocsr()
        C = (A2 - A1).tocsr()
        Fc = (F2 - F1).tocsr()

        return cls(
            grid=grid, mu=mu_arr, D=D_arr, A1=A1, A2=A2, C=C,
            upwind_left=upwind_left, Fc=Fc, Div=Div,
        )

    def restricted_operator(self, free_mask: np.ndarray) -> sp.csr_matrix:
        """A1 + C restricted to the free interfaces of `free_mask` (bool[n-1]) -- the
        generalized-Jacobian matrix V_S = I - gamma*(A1+C_S) construction of Eq. (25)-(26)
        (gamma multiplication happens in the caller, e.g. active_set.py)."""
        Fc_masked = self.Fc.multiply(free_mask.astype(float)[:, None]).tocsr()
        C_restricted = (self.Div @ Fc_masked).tocsr()
        return (self.A1 + C_restricted).tocsr()

    # ------------------------------------------------------------------
    def unlimited_flux(self, p: np.ndarray) -> np.ndarray:
        """Antidiffusive interface flux d_{i+1/2}(p) = C-flux, length n-1 (Eq. 9).

        Recovered directly from C by reversing the divergence: since C = Div @ Fc with
        Fc the antidiffusive flux matrix, and Div has a simple +-1/h structure, we can
        recompute Fc @ p directly for efficiency instead of re-deriving Fc explicitly.
        """
        return self._Fc_matvec(p)

    def _Fc_matvec(self, p: np.ndarray) -> np.ndarray:
        # Recompute (F2 - F1) @ p on the fly using stored mu/upwind_left (avoids storing
        # a second (n-1, n) matrix; correctness verified against C in tests).
        n = self.grid.n
        mu = self.mu
        upwind_left = self.upwind_left
        d = np.zeros(n - 1)
        for j in range(n - 1):
            if upwind_left[j]:
                if j - 1 >= 0:
                    d[j] = 0.5 * (mu[j] * p[j] - mu[j - 1] * p[j - 1])
                else:
                    d[j] = 0.0
            else:
                if j + 2 <= n - 1:
                    d[j] = 0.5 * (mu[j + 1] * p[j + 1] - mu[j + 2] * p[j + 2])
                else:
                    d[j] = 0.0
        return d

    def total_flux_A2(self, w: np.ndarray) -> np.ndarray:
        """Total (2nd-order) numerical flux Jhat_{i+1/2}(w), length n-1 -- 'the total
        numerical flux of Section 2' referenced in Eq. (21)'s defect flux
        G_{i+1/2} = (dt^2/2) * Jhat_{i+1/2}(w), w = A2 @ Y0 (Section 4.2)."""
        n = self.grid.n
        h = self.grid.h
        mu, D, upwind_left = self.mu, self.D, self.upwind_left
        J = np.zeros(n - 1)
        for j in range(n - 1):
            diff = D[j] / h * w[j] - D[j + 1] / h * w[j + 1]
            if upwind_left[j]:
                if j - 1 >= 0:
                    g2 = 1.5 * mu[j] * w[j] - 0.5 * mu[j - 1] * w[j - 1]
                else:
                    g2 = mu[j] * w[j]
            else:
                if j + 2 <= n - 1:
                    g2 = 1.5 * mu[j + 1] * w[j + 1] - 0.5 * mu[j + 2] * w[j + 2]
                else:
                    g2 = mu[j + 1] * w[j + 1]
            J[j] = g2 + diff
        return J

    def divergence_of_face_values(self, flux: np.ndarray) -> np.ndarray:
        """Apply the same Div operator used in assembly to an arbitrary length-(n-1)
        face-value array (e.g. a *limited* flux), with zero-flux boundary closure."""
        n = self.grid.n
        h = self.grid.h
        out = np.zeros(n)
        out[1:] += flux / h
        out[:-1] += -flux / h
        return out

    # ------------------------------------------------------------------
    def core_solve(self, rhs: np.ndarray, gamma: float) -> np.ndarray:
        """Solve (I - gamma*A1) x = rhs. A1 is a nonsingular M-matrix (Lemma 3) for
        every gamma > 0, so this always has a unique solution."""
        n = self.grid.n
        M = sp.identity(n, format="csr") - gamma * self.A1
        return spla.spsolve(M.tocsc(), rhs)

    def apply_A1(self, p: np.ndarray) -> np.ndarray:
        return self.A1 @ p

    def apply_A2(self, p: np.ndarray) -> np.ndarray:
        return self.A2 @ p

    def unlimited_solve(self, rhs: np.ndarray, gamma: float) -> np.ndarray:
        """Solve (I - gamma*A2) x = rhs directly (the 'unlimited' scheme / baseline,
        and the first move of the active-set solver). May return a negative solution."""
        n = self.grid.n
        M = sp.identity(n, format="csr") - gamma * self.A2
        return spla.spsolve(M.tocsc(), rhs)
