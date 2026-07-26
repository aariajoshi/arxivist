"""
Evaluation metrics for the quantum cryptanalysis benchmark.

Paper reference: Section 4.1 (rank of true key), Section 4.3/Figure 2
(query-complexity separation), Section 4.4/Figure 5 (classical-simulation
memory wall).
"""

from __future__ import annotations

import math


def birthday_bound(n: int) -> float:
    """Classical period-finding query complexity: O(2^(n/2))."""
    return 2 ** (n / 2)


def simon_query_complexity(n: int) -> int:
    """Quantum (Simon's algorithm) query complexity: Theta(n). Uses n queries as the constant."""
    return n


def random_expectation_rank(n: int) -> float:
    """Expected rank of the true key under uniform-random guessing: 2^(n-1) (paper's Figure 3)."""
    return 2 ** (n - 1)


def grover_iteration_count(n: int) -> int:
    """Grover iteration count ~ (pi/4)*sqrt(2^n) (paper's Section 4.3/Figure 4)."""
    return max(1, round((math.pi / 4) * math.sqrt(2 ** n)))


def statevector_memory_bytes(n_qubits: int, bytes_per_amplitude: int = 16) -> int:
    """Statevector memory requirement: 2^n_qubits * bytes_per_amplitude (complex128 default).

    Paper reference: Section 4.4, Figure 5 (e.g. EM n=8 -> 24 qubits, 0.27 GB).
    """
    return (2 ** n_qubits) * bytes_per_amplitude


def summarize_rank_result(true_key_rank: int, total_candidates: int, n: int) -> dict:
    """Summarize a rank-recovery result against the paper's reference quantities.

    Args:
        true_key_rank: 1-indexed rank of the true key in the candidate ranking.
        total_candidates: total number of candidates ranked (2^n - 1 typically).
        n: security parameter / block size.

    Returns:
        dict with the rank, whether it's "clean" (rank==1), the birthday
        bound and random-expectation reference values for this n, and how
        the observed rank compares (as a ratio) to the birthday bound.
    """
    bb = birthday_bound(n)
    rand_exp = random_expectation_rank(n)
    return {
        "n": n,
        "true_key_rank": true_key_rank,
        "total_candidates": total_candidates,
        "is_clean_rank_1": true_key_rank == 1,
        "birthday_bound_2^(n/2)": bb,
        "random_expectation_2^(n-1)": rand_exp,
        "rank_to_birthday_bound_ratio": true_key_rank / bb if bb > 0 else None,
    }
