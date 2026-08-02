"""Two-point Zalesak-type flux limiter.

Paper: Itkin (2026), Section 3 (Scheme FCDF-B) and Section 4.2 (combined clamp for FCDF-DC).

Implements the even-budget-split (kappa_j == 2, SIR eq_zalesak_limiter, confidence 0.93)
clamp used throughout the paper's propositions:

    Lambda_{i+1/2}(p) = median( -c^-_{i+1/2}, d_{i+1/2}(p), c^+_{i+1/2} )
    c^+_{i+1/2} = h*b_i   / (2*gamma)
    c^-_{i+1/2} = h*b_{i+1} / (2*gamma)

(Appendix A, Lemma 4 / Eq. A.1-A.2 -- the clamp form is exactly equivalent to the
budget-ratio form of Eq. 15 and is what we implement directly, since it is numerically
simpler and avoids a division-by-zero when d_{i+1/2}=0.)
"""
from __future__ import annotations

import numpy as np


class ZalesakLimiter:
    """Per-interface Zalesak limiter with the even budget split kappa_j = 2."""

    def __init__(self, kappa: float = 2.0) -> None:
        if kappa <= 0:
            raise ValueError("kappa (budget split) must be positive")
        self.kappa = kappa

    def caps(self, b: np.ndarray, gamma: float, h: float) -> tuple[np.ndarray, np.ndarray]:
        """Return (c_plus, c_minus), each length n-1, for the interfaces of a length-n
        nonnegative right-hand side b (Eq. A.1, with kappa_j == self.kappa == 2 by default)."""
        if gamma <= 0:
            raise ValueError("gamma must be > 0")
        c_plus = h * b[:-1] / (self.kappa * gamma)
        c_minus = h * b[1:] / (self.kappa * gamma)
        return c_plus, c_minus

    def clamp(self, flux: np.ndarray, b: np.ndarray, gamma: float, h: float) -> np.ndarray:
        """Clamp the unlimited antidiffusive flux d_{i+1/2}(p) onto [-c_minus, c_plus]
        (Lemma 4 clamp form, Eq. A.2). b must be entrywise >= 0."""
        if np.any(b < -1e-14):
            raise ValueError("Zalesak limiter requires a nonnegative base b (Proposition 1 hypothesis)")
        c_plus, c_minus = self.caps(np.clip(b, 0.0, None), gamma, h)
        return np.clip(flux, -c_minus, c_plus)

    def clamp_with_offset(
        self, iterated_flux: np.ndarray, defect_flux: np.ndarray, p_n: np.ndarray, dt: float, h: float
    ) -> np.ndarray:
        """FCDF-DC combined clamp (Eq. 21): clamps the *sum* of the iterated antidiffusive
        flux (scaled by dt) and a fixed defect flux G onto the same budget interval drawn
        from the common nonnegative base p_n (kappa==2 even split, Section 4.2, Lemma 5)."""
        if np.any(p_n < -1e-14):
            raise ValueError("FCDF-DC combined clamp requires a nonnegative base p_n")
        p_n_clipped = np.clip(p_n, 0.0, None)
        c_plus = h * p_n_clipped[:-1] / self.kappa
        c_minus = h * p_n_clipped[1:] / self.kappa
        total = dt * iterated_flux + defect_flux
        return np.clip(total, -c_minus, c_plus)
