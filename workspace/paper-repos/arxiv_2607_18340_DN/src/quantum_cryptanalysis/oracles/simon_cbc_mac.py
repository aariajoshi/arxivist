"""
CBC-MAC forgery Simon oracle.

Paper reference: Table 2, Supplementary S2: "f(x) = Ek(x XOR c.a) XOR Ek(x
XOR c.b) with period s = Ek(a) XOR Ek(b)."

VERIFIED DISCREPANCY (found during implementation/testing, not just a
parsing ambiguity): for a generic permutation Ek, this oracle formula does
NOT actually have period s = Ek(a) XOR Ek(b) as the paper states. Direct
verification (see tests/test_simon_algorithms.py and
promise_utils.verify_two_to_one_promise) shows the oracle f(x) = Ek(x XOR
a) XOR Ek(x XOR b) instead satisfies Simon's 2-to-1 promise with period

    s = a XOR b     (the INPUT-space difference, not an output-space one)

by the same algebraic argument used for the Even-Mansour oracle:
    f(x XOR (a XOR b)) = Ek(x XOR (a XOR b) XOR a) XOR Ek(x XOR (a XOR b) XOR b)
                        = Ek(x XOR b) XOR Ek(x XOR a) = f(x)
This holds for ANY permutation Ek, generically, exactly like the
Even-Mansour reduction. The paper's stated "s = Ek(a) XOR Ek(b)" does not
satisfy this promise for a generic Ek (empirically confirmed to fail for
random test permutations). This repository therefore reports/verifies the
recovered period against s = a XOR b, and documents the discrepancy here
rather than silently "correcting" the paper without saying so. See the
repo's comparison/hallucination_report.md for how this is scored.

SIR confidence for the oracle formula itself: 0.85 (explicit); confidence
in the paper's stated PERIOD value specifically: lowered given this
verified inconsistency -- see README.md and hallucination_report.md.
"""

from __future__ import annotations

from typing import Callable

from qiskit import QuantumCircuit


class CBCMACForgeryOracle:
    """Builds the Simon oracle circuit for CBC-MAC forgery.

    Args:
        None (stateless; parameters passed to build_circuit).
    """

    def build_circuit(
        self,
        block_cipher: Callable[[int, int], int],
        key: int,
        a: int,
        b: int,
        n: int,
    ) -> QuantumCircuit:
        """Build the 2n-qubit Simon oracle for f(x) = Ek(x XOR c.a) XOR Ek(x XOR c.b).

        Here c is folded into the constant offsets (a, b directly represent
        c.a and c.b as n-bit constants, consistent with how the paper's
        Table 2 states the construction).

        Args:
            block_cipher: a callable (key, plaintext) -> ciphertext,
                implementing Ek as a permutation on n-bit blocks. This
                repository does not assume a specific block cipher -- pass
                in any n-bit permutation-based reference implementation
                (e.g. a small Feistel or SPN toy cipher).
            key: the block cipher key k.
            a, b: the two n-bit constant offsets defining the forgery target.
            n: block size.

        Returns:
            A 2n-qubit QuantumCircuit implementing f(x) as a lookup-table
            oracle (feasible for the paper's validated sizes n=4,6).
        """
        qc = QuantumCircuit(2 * n, name=f"CBCMAC_oracle(n={n})")
        for x in range(2 ** n):
            fx = block_cipher(key, x ^ a) ^ block_cipher(key, x ^ b)
            ctrl_state = format(x, f"0{n}b")  # Qiskit ctrl_state convention verified empirically; do NOT reverse
            controls = list(range(n))
            for j in range(n):
                if (fx >> j) & 1:
                    qc.mcx(controls, n + j, ctrl_state=ctrl_state)
        return qc
