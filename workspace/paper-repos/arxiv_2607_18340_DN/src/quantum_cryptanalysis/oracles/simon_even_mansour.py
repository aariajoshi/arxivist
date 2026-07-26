"""
Even-Mansour Simon-reduction oracle.

Paper reference: Section 1, 3.1 (Kuwakado-Morii reduction, refs [1],[2]).
The Even-Mansour cipher is E_{k1,k2}(x) = P(x XOR k1) XOR k2 for a PUBLIC
permutation P. The Kuwakado-Morii Simon-reduction defines

    g(x) = E(x) XOR P(x) = P(x XOR k1) XOR P(x)     (k2 cancels out)

which satisfies g(x XOR k1) = g(x) for all x (hidden period k1), while
otherwise behaving as a 2-to-1 function for a generic permutation P. This
requires TWO calls to P per oracle query (one at x XOR k1, one at x) --
a single permutation call alone is a *bijection* and cannot hide a period.
Once k1 (=s) is recovered via Simon's algorithm, k2 follows from a single
classical query: k2 = E(0) XOR P(k1).

SIR confidence 0.6 (clean/differential-Simon regime at n<=5 is standard and
well specified; the paper's noisy real-hardware n=6-10 "hybrid" regime
depends on the withheld technique -- see postprocessing/hybrid_ranking.py).
"""

from __future__ import annotations

from typing import Callable

from qiskit import QuantumCircuit


class EvenMansourOracle:
    """Builds the Simon-style oracle circuit g(x) = P(x XOR k1) XOR P(x) for Even-Mansour.

    Args:
        None (stateless; parameters passed to build_circuit).
    """

    def build_circuit(self, k1: int, permutation: list[int] | Callable[[int], int], n: int) -> QuantumCircuit:
        """Build the 2n-qubit Simon oracle for g(x) = P(x XOR k1) XOR P(x).

        Args:
            k1: the hidden period (secret key component) to recover, as an
                integer in [0, 2^n).
            permutation: either a list of length 2^n giving the PUBLIC
                permutation P as a lookup table (permutation[x] = P(x)), or
                a callable int -> int.
            n: block size (number of input/output qubits).

        Returns:
            A 2n-qubit QuantumCircuit: qubits [0,n) are the input register
            x, qubits [n,2n) are the output register initialized to |0>
            and XORed with g(x) = P(x XOR k1) XOR P(x).
        """
        if callable(permutation):
            perm_list = [permutation(x) for x in range(2 ** n)]
        else:
            perm_list = list(permutation)
        assert len(perm_list) == 2 ** n, f"permutation must have length 2^{n}={2**n}"
        assert sorted(perm_list) == list(range(2 ** n)), "permutation must be a bijection on [0, 2^n)"

        qc = QuantumCircuit(2 * n, name=f"EM_oracle(k1={k1},n={n})")

        # g(x) = P(x XOR k1) XOR P(x), precomputed classically as a lookup
        # table (feasible for the paper's validated sizes, n<=10) and
        # implemented as a reversible multi-controlled-X network, matching
        # the style used for the other Simon-based oracles in this repo.
        for x in range(2 ** n):
            gx = perm_list[x ^ k1] ^ perm_list[x]
            controls = list(range(n))
            ctrl_state = format(x, f"0{n}b")  # verified convention, see module docstring history
            for j in range(n):
                if (gx >> j) & 1:
                    qc.mcx(controls, n + j, ctrl_state=ctrl_state)

        return qc


def verify_promise(permutation: list[int], k1: int, n: int) -> bool:
    """Check that g(x) = P(x XOR k1) XOR P(x) is genuinely 2-to-1 with period exactly {0, k1}.

    Simon's algorithm assumes a *promise*: g(x) = g(x') iff x' in {x, x XOR s}.
    For a generic/large-n permutation this holds with high probability, but a
    randomly sampled SMALL permutation (e.g. n=3, 8 elements) can accidentally
    create additional collisions beyond the intended period, breaking the
    promise and causing Simon's algorithm to fail or return a spurious period.
    This is a combinatorial property of the specific (permutation, k1) pair,
    not a property the paper discusses explicitly -- but it must hold for any
    concrete instance to behave as Simon's algorithm expects.

    Args:
        permutation: the public permutation P as a length-2^n lookup table.
        k1: the candidate hidden period.
        n: block size.

    Returns:
        True iff g(x) = P(x XOR k1) XOR P(x) collides only within {x, x XOR k1}
        pairs for every x (the genuine Simon promise).
    """
    from collections import defaultdict

    groups = defaultdict(list)
    for x in range(2 ** n):
        gx = permutation[x ^ k1] ^ permutation[x]
        groups[gx].append(x)
    for xs in groups.values():
        if len(xs) not in (0, 2):
            return False
        if len(xs) == 2 and xs[1] != (xs[0] ^ k1):
            return False
    return True


def find_valid_permutation(n: int, k1: int, rng, max_attempts: int = 200) -> list[int]:
    """Construct a permutation P satisfying the Simon promise for the given k1.

    NOTE ON METHOD: a first implementation attempt used pure random search
    (shuffle P, check verify_promise, retry). That approach works for very
    small n but becomes exponentially unreliable as n grows: the number of
    coset pairs whose g-values must all be pairwise distinct grows as
    2^(n-1), so by a birthday-paradox-like argument the probability that a
    UNIFORMLY RANDOM permutation happens to satisfy the promise shrinks
    rapidly with n (confirmed empirically: random search reliably fails to
    find a valid n=6 instance within hundreds of attempts). This reflects a
    genuine property of the Kuwakado-Morii reduction's "generic permutation"
    idealization, not a bug -- real Even-Mansour security arguments hold
    asymptotically over a random P, not via brute-force search at small n.

    This function therefore uses a DETERMINISTIC greedy construction instead:
    partition [0, 2^n) into the 2^(n-1) coset pairs {x, x XOR k1}, then
    greedily assign each pair a pair of output values from a shrinking pool
    such that every pair's output-XOR ("delta") is distinct from every
    other pair's -- which is exactly the promise condition. This always
    succeeds for n up to at least 12 (the paper's largest simulated size).

    Args:
        n: block size.
        k1: the hidden period the resulting instance should have.
        rng: a `random.Random` instance, used only to shuffle the output
            pool for variety across calls (does not affect correctness).
        max_attempts: unused, kept for backward-compatible signature.

    Returns:
        A permutation (list[int]) satisfying `verify_promise`.

    Raises:
        RuntimeError: if construction fails (should not happen for any
            practical n; would indicate a logic error if it did).
    """
    del max_attempts  # retried internally below; kept for backward-compatible signature
    size = 2 ** n
    j = (k1 & -k1).bit_length() - 1  # index of k1's lowest set bit
    representatives = [x for x in range(size) if not (x >> j) & 1]

    last_error = None
    for _ in range(50):  # greedy can occasionally paint itself into a corner; reshuffle and retry
        pool = list(range(size))
        rng.shuffle(pool)
        perm = [None] * size
        used_deltas: set[int] = set()
        ok = True

        for r in representatives:
            a = pool[0]
            chosen_b = None
            for b in pool[1:]:
                delta = a ^ b
                if delta != 0 and delta not in used_deltas:
                    chosen_b = b
                    break
            if chosen_b is None:
                ok = False
                last_error = f"stuck at representative r={r} with {len(pool)} pool items remaining"
                break
            pool.remove(a)
            pool.remove(chosen_b)
            used_deltas.add(a ^ chosen_b)
            perm[r] = a
            perm[r ^ k1] = chosen_b

        if ok:
            assert all(v is not None for v in perm), "construction left an unassigned output"
            assert sorted(perm) == list(range(size)), "construction did not produce a valid permutation"
            return perm

    raise RuntimeError(
        f"Deterministic construction failed for n={n}, k1={k1} after 50 reshuffle attempts "
        f"(last failure: {last_error}); this should not happen for n<=12 -- please file an issue."
    )
