"""
Bernstein-Vazirani algorithm assembly: Hadamards + oracle + Hadamards, single query.
"""

from __future__ import annotations

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile


class BernsteinVazirani:
    """Assembles and runs the full BV circuit, recovering the secret in one query."""

    def run(self, oracle_circuit: QuantumCircuit, n: int, backend, shots: int = 1) -> str:
        """Run the BV algorithm.

        Args:
            oracle_circuit: an (n+1)-qubit oracle circuit (see BVOracle.build_circuit).
            n: number of input qubits (secret length).
            backend: a Qiskit-compatible backend (see backend/execution.py).
            shots: number of shots (1 suffices in the noiseless case, per
                the paper's "single query" claim; use more on noisy backends).

        Returns:
            The recovered secret bitstring (MSB-first, length n).
        """
        qr = QuantumRegister(n + 1)
        cr = ClassicalRegister(n)
        qc = QuantumCircuit(qr, cr)

        qc.x(n)
        qc.h(range(n + 1))
        qc.compose(oracle_circuit, inplace=True)
        qc.h(range(n))
        qc.measure(range(n), range(n))

        tqc = transpile(qc, backend)
        result = backend.run(tqc, shots=shots).result()
        counts = result.get_counts()
        most_common = max(counts, key=counts.get)
        return most_common
