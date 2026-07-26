"""
GF(2) linear algebra utilities.

Simon's algorithm (used for the Even-Mansour, CBC-MAC forgery, and 3-round
Feistel attacks) works by collecting measurement outcomes y_1, ..., y_{n-1}
each satisfying y_i . s = 0 (mod 2), where s is the hidden period. Once
n-1 linearly independent such vectors are collected, s is the unique
non-zero vector in their null space over GF(2).
"""

from __future__ import annotations

import numpy as np


def bitstring_to_vector(bitstring: str) -> np.ndarray:
    """Convert a bitstring (e.g. '0110') to a GF(2) column vector (numpy int array of 0/1)."""
    return np.array([int(b) for b in bitstring], dtype=np.int64)


def gf2_rank(vectors: list[np.ndarray]) -> int:
    """Compute the GF(2) rank of a list of bit vectors via Gaussian elimination mod 2."""
    if not vectors:
        return 0
    mat = np.array(vectors, dtype=np.int64) % 2
    rows, cols = mat.shape
    rank = 0
    for col in range(cols):
        pivot = None
        for r in range(rank, rows):
            if mat[r, col] == 1:
                pivot = r
                break
        if pivot is None:
            continue
        mat[[rank, pivot]] = mat[[pivot, rank]]
        for r in range(rows):
            if r != rank and mat[r, col] == 1:
                mat[r] = (mat[r] + mat[rank]) % 2
        rank += 1
        if rank == rows:
            break
    return rank


def solve_null_space_gf2(vectors: list[np.ndarray], n: int) -> np.ndarray | None:
    """Find a non-zero vector s in GF(2)^n orthogonal to every vector in `vectors`.

    Args:
        vectors: list of n-bit GF(2) vectors (each y_i satisfies y_i . s = 0).
        n: bit-length of the hidden period.

    Returns:
        The (unique, up to the trivial all-zero solution) non-zero solution
        vector s as a length-n int array, or None if fewer than n-1
        linearly independent constraints were provided (underdetermined).
    """
    if not vectors:
        return None

    mat = np.array(vectors, dtype=np.int64) % 2
    rank = gf2_rank(list(mat))
    if rank < n - 1:
        return None  # not enough independent equations yet

    # Reduce to row-echelon form (mod 2) to find the null space vector.
    rows, cols = mat.shape
    m = mat.copy()
    pivot_cols = []
    r = 0
    for col in range(cols):
        pivot = None
        for i in range(r, rows):
            if m[i, col] == 1:
                pivot = i
                break
        if pivot is None:
            continue
        m[[r, pivot]] = m[[pivot, r]]
        for i in range(rows):
            if i != r and m[i, col] == 1:
                m[i] = (m[i] + m[r]) % 2
        pivot_cols.append(col)
        r += 1
        if r == rows:
            break

    free_cols = [c for c in range(cols) if c not in pivot_cols]
    if not free_cols:
        return None  # only the trivial all-zero solution exists

    # Set the first free variable to 1, solve for pivot variables.
    s = np.zeros(n, dtype=np.int64)
    free_col = free_cols[0]
    s[free_col] = 1
    for i, pcol in enumerate(pivot_cols):
        if m[i, free_col] == 1:
            s[pcol] = 1

    if not s.any():
        return None
    return s


def vector_to_bitstring(vec: np.ndarray) -> str:
    """Convert a GF(2) vector back to a bitstring."""
    return "".join(str(int(b)) for b in vec)
