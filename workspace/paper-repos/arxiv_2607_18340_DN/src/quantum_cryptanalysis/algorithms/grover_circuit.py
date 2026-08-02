"""
Grover search circuit assembly.

Paper reference: Section 4.3, Figure 4, Supplementary S2. Iteration count
~ (pi/4)*sqrt(2^n) (e.g. ~13 iterations at n=8, matching Figure 4).
"""

from __future__ import annotations

import math

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile


def _diffuser(n: int) -> QuantumCircuit:
    """Standard Grover diffuser (inversion about the mean) on n qubits."""
    qc = QuantumCircuit(n, name="diffuser")
    qc.h(range(n))
    qc.x(range(n))
    qc.h(n - 1)
    qc.mcx(list(range(n - 1)), n - 1)
    qc.h(n - 1)
    qc.x(range(n))
    qc.h(range(n))
    return qc


class GroverSearch:
    """Runs Grover's algorithm against a phase-flip oracle circuit."""

    def run(self, oracle_circuit: QuantumCircuit, n: int, backend, shots: int = 4096) -> int:
        """Run Grover's algorithm.

        Args:
            oracle_circuit: an n-qubit phase-flip oracle (see GroverSPNOracle.build_circuit).
                If `oracle_circuit.metadata` contains `num_marked`, the
                iteration count is corrected for M>1 marked items via
                (pi/4)*sqrt(2^n / M); otherwise M=1 is assumed (paper's
                stated formula, Section 4.3/Figure 4).
            n: number of qubits (key size).
            backend: a Qiskit-compatible backend.
            shots: number of shots for the final measurement.

        Returns:
            The recovered key as an integer (the most frequently measured outcome).
        """
        num_marked = 1
        if oracle_circuit.metadata and "num_marked" in oracle_circuit.metadata:
            num_marked = max(1, oracle_circuit.metadata["num_marked"])
        num_iterations = max(1, round((math.pi / 4) * math.sqrt((2 ** n) / num_marked)))

        qr = QuantumRegister(n)
        cr = ClassicalRegister(n)
        qc = QuantumCircuit(qr, cr)

        qc.h(range(n))
        diffuser = _diffuser(n)
        for _ in range(num_iterations):
            qc.compose(oracle_circuit, inplace=True)
            qc.compose(diffuser, inplace=True)
        qc.measure(range(n), range(n))

        tqc = transpile(qc, backend)
        result = backend.run(tqc, shots=shots).result()
        counts = result.get_counts()
        most_common = max(counts, key=counts.get)
        return int(most_common, 2)
