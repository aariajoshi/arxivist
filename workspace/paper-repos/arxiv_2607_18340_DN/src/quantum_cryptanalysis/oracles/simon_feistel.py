"""
3-round Feistel (DES-family) Simon oracle.

Paper reference: Section 3.1, Supplementary S2 (Kuwakado-Morii reduction).

    f(b, x) = LeftHalf(E(x, alpha_b)) XOR alpha_b = F2(x XOR F1(alpha_b))
    hidden period: s = (1, gamma), gamma = F1(alpha_0) XOR F1(alpha_1)

Clean (rank-1) if F2 is a permutation. The oracle requires 1+4m qubits for
block size 2m (1 bit for b, plus 2m for the x/output halves each -- packed
here as 1 selector qubit + m qubits for x + m qubits for the F1(alpha_b)
intermediate + m qubits for the F2 output, matching the paper's stated
1+4m qubit count when x and F1(alpha_b) are each m bits... see note below).

SIR confidence 0.85 (explicit construction and qubit-count formula). The
paper stresses: placing the variable on the left (x) and the constant on
the right (alpha_b) is essential -- reversing this admits no hidden period.
This constraint is preserved exactly in the implementation below.
"""

from __future__ import annotations

from typing import Callable

from qiskit import QuantumCircuit


class FeistelOracle:
    """Builds the Simon oracle circuit for the 3-round Feistel construction.

    NOTE on qubit count: the paper states "1+4m qubits for block size 2m"
    without a full circuit diagram. We interpret this as: 1 selector qubit
    (b) + m qubits for x (input register, matching Simon's algorithm
    convention of an m-bit input/output pair per half) + up to 2m ancilla/
    output qubits for intermediate F1/F2 evaluation and the final output
    register, giving a circuit of total width 1+4m for the *full* Simon
    construction (selector + x register + F1-intermediate + F2-output).
    This qubit-budget interpretation is a design choice made to match the
    paper's stated total, not a literal diagram from the paper (which is
    not provided) -- flagged here for transparency, distinct from the SIR's
    own ambiguities which concern F1/F2's mathematical definitions, not
    qubit layout.

    Args:
        None (stateless; parameters passed to build_circuit).
    """

    def build_circuit(
        self,
        f1: Callable[[int], int],
        f2: Callable[[int], int],
        alpha0: int,
        alpha1: int,
        m: int,
    ) -> QuantumCircuit:
        """Build the Simon oracle for f(b,x) = F2(x XOR F1(alpha_b)).

        Args:
            f1: round function F1: [0, 2^m) -> [0, 2^m).
            f2: round function F2: [0, 2^m) -> [0, 2^m). Must be a
                permutation for a clean (rank-1) period.
            alpha0, alpha1: the two fixed constant right-half inputs
                (alpha0 != alpha1), analogous to R0 in the paper's notation.
            m: half-block size (block size is 2m).

        Returns:
            A (1 + 2m)-qubit QuantumCircuit: qubit 0 is the selector b,
            qubits [1, 1+m) are the input register x, qubits [1+m, 1+2m)
            are the output register initialized to |0> and XORed with
            f(b, x). (A simplified 1+2m-qubit layout is used here rather
            than the paper's literal 1+4m, since we do not expose F1's
            intermediate register as separate qubits -- it is computed
            classically into the lookup table below. This preserves the
            paper's period-recovery semantics exactly while using fewer
            qubits than a literal in-place quantum F1/F2 evaluation would.)
        """
        assert alpha0 != alpha1, "alpha0 and alpha1 must differ (paper's assumption)"
        n_total = 1 + 2 * m
        qc = QuantumCircuit(n_total, name=f"Feistel_oracle(m={m})")

        selector = 0
        x_reg = list(range(1, 1 + m))
        out_reg = list(range(1 + m, 1 + 2 * m))

        for b in (0, 1):
            alpha_b = alpha0 if b == 0 else alpha1
            for x in range(2 ** m):
                fbx = f2(x ^ f1(alpha_b))
                # controls = [selector] + x_reg = qubits [0, 1, ..., m], increasing order.
                # Verified convention (see simon_even_mansour.py): for an increasing
                # controls list [0..w-1], ctrl_state = format(value, f'0{w}b') directly,
                # where value's bit0 = qubit0's desired state. Here bit0 = b (selector,
                # qubit0), bits[1:] = bits of x (qubit1..qubitm).
                value = b | (x << 1)
                ctrl_state = format(value, f"0{m + 1}b")
                controls = [selector] + x_reg
                for j in range(m):
                    if (fbx >> j) & 1:
                        qc.mcx(controls, out_reg[j], ctrl_state=ctrl_state)
        return qc
