"""
Generic Simon's-algorithm circuit assembly, shared by the Even-Mansour,
CBC-MAC forgery, and 3-round Feistel attacks.

Paper reference: Section 2.1 ("Simon's algorithm... finds a hidden period s
... using Theta(n) quantum queries"). Standard textbook Simon's algorithm:
H^n on the input register, apply the oracle, H^n again, measure -> outcome y
satisfies y . s = 0 (mod 2). Repeat until n-1 linearly independent y's are
collected, then solve for s via GF(2) linear algebra.
"""

from __future__ import annotations

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile

from quantum_cryptanalysis.postprocessing.gf2_solver import (
    bitstring_to_vector,
    gf2_rank,
    solve_null_space_gf2,
    vector_to_bitstring,
)


class SimonAlgorithm:
    """Runs Simon's algorithm against a given oracle circuit to recover the hidden period."""

    def collect_measurements(
        self, oracle_circuit: QuantumCircuit, n: int, backend, shots: int = 4096
    ) -> list[str]:
        """Run the Simon circuit and collect measured bitstrings.

        Args:
            oracle_circuit: a 2n-qubit oracle circuit (input register [0,n),
                output register [n,2n)), as built by simon_even_mansour /
                simon_cbc_mac / simon_feistel.
            n: input register width (bit-length of the hidden period s).
            backend: a Qiskit-compatible backend.
            shots: number of shots to run.

        Returns:
            List of distinct measured n-bit strings from the input
            register, each satisfying y . s = 0 (mod 2) in the noiseless
            case. (On a noisy backend, some outcomes may violate this due
            to hardware error -- see postprocessing/hybrid_ranking.py for
            how the paper's noisy n=6-10 regime is approximately handled.)
        """
        qr = QuantumRegister(2 * n)
        cr = ClassicalRegister(n)
        qc = QuantumCircuit(qr, cr)

        qc.h(range(n))
        qc.compose(oracle_circuit, inplace=True)
        qc.h(range(n))
        qc.measure(range(n), range(n))

        tqc = transpile(qc, backend)
        result = backend.run(tqc, shots=shots).result()
        counts = result.get_counts()

        # Return distinct outcomes, sorted by descending frequency (most
        # useful ordering for both the clean linear-solve path and the
        # noisy top-K hybrid path).
        return [bitstring for bitstring, _ in sorted(counts.items(), key=lambda kv: -kv[1])]

    def solve_period(self, measurements: list[str], n: int) -> str | None:
        """Solve for the hidden period s given a list of measured bitstrings.

        Args:
            measurements: list of n-bit strings, each expected to satisfy
                y . s = 0 (mod 2) (the all-zeros string, if present, is
                uninformative and is skipped).
            n: bit-length of s.

        Returns:
            The recovered period s as an n-bit string, or None if fewer
            than n-1 linearly independent (non-trivial) measurements were
            provided.
        """
        vectors = [
            bitstring_to_vector(m) for m in measurements if any(c == "1" for c in m)
        ]
        if gf2_rank(vectors) < n - 1:
            return None
        s = solve_null_space_gf2(vectors, n)
        return vector_to_bitstring(s) if s is not None else None
