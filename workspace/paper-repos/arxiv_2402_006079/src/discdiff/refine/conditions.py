"""Absorb / Escape conditions (Table 3).

Absorb-Escape operates on per-position *max-softmax confidences*:
    L(D, s_i)  — the diffusion decoder's max softmax at position i,
    L(M, s_j)  — the autoregressive model's max softmax at position j.

A low L(D, .) marks a 'valley' where the diffusion model is unsure — a likely
single-nucleotide rounding error (Sec 4.2, Fig 4).
"""
from __future__ import annotations

import random


def absorb_condition(l_d_i: float, t_absorb: float) -> bool:
    """A(s_i) = true iff L(D, s_i) < T_absorb  (Table 3, Absorb / Threshold).

    Trigger refinement where the diffusion model is UNconfident.
    """
    return l_d_i < t_absorb


def escape_natural(l_d_j: float, l_m_j: float) -> bool:
    """E(s_j) = true iff L(D, s_j) > L(M, s_j)  (Table 3, Escape / Natural — DEFAULT).

    Stop autoregressive refinement once the diffusion model is again more
    confident than the AR model, i.e. we have left the low-confidence valley.
    """
    return l_d_j > l_m_j


def escape_threshold(l_m_j: float, t_escape: float) -> bool:
    """E(s_j, T_escape) = true iff L(M, s_j) < T_escape  (Table 3, Escape / Threshold)."""
    return l_m_j < t_escape


def escape_random(t_random: float, rng: random.Random | None = None) -> bool:
    """E(s_j, T_random) = true iff uniform(0,1) > T_random  (Table 3, Escape / Random)."""
    r = (rng or random).random()
    return r > t_random


def escape_max_length(j: int, start: int, max_sub_length: int) -> bool:
    """E(s_j, max_sub_length) = true iff (j - start) > max_sub_length (Table 3, Max Length)."""
    return (j - start) > max_sub_length
