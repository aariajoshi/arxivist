"""
Bernstein-Vazirani oracle.

Paper reference: Section 2.1, Supplementary S2. f(x) = a.x (mod 2); a single
quantum query followed by Hadamard transforms yields the secret a directly.
SIR confidence 0.9 (fully explicit, standard textbook construction).
"""

from __future__ import annotations

from qiskit import QuantumCircuit


class BVOracle:
    """Builds the Bernstein-Vazirani oracle circuit for a given secret string.

    Args:
        None (stateless; secret is passed to build_circuit).
    """

    def build_circuit(self, secret: str) -> QuantumCircuit:
        """Build the oracle U_f: |x>|y> -> |x>|y XOR (a.x)>.

        Args:
            secret: the hidden linear secret a, as a bitstring of length n
                (e.g. "1011").

        Returns:
            An (n+1)-qubit QuantumCircuit implementing the oracle only
            (no Hadamards -- those are added by BernsteinVazirani.run).
            Qubit n is the ancilla output qubit.
        """
        n = len(secret)
        qc = QuantumCircuit(n + 1, name=f"BV_oracle(a={secret})")
        for i, bit in enumerate(reversed(secret)):
            if bit == "1":
                qc.cx(i, n)
        return qc
