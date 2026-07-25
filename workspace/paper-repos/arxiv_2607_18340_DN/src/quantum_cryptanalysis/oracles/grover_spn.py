"""
Grover SPN oracle.

Paper reference: Table 2, Supplementary S2 ("Grover / SPN key search"). The
paper never specifies the reduced SPN's S-box, round count, or key schedule
(SIR ambiguities[3], confidence 0.4) -- only that the oracle marks the key
matching a known plaintext/ciphertext pair. This module defines a small,
clearly-labeled TOY SPN cipher as a stand-in, since no concrete construction
is given in the paper to reproduce exactly.
"""

from __future__ import annotations

from qiskit import QuantumCircuit


class ToySPNCipher:
    """A small, generic substitution-permutation-network cipher (ASSUMED stand-in).

    NOT the paper's actual construction (which is unspecified) -- this is a
    minimal, easily-reasoned-about SPN used only so the Grover attack has a
    concrete target to search over. Two rounds of (key-XOR -> 4-bit S-box
    substitution -> bit-permutation).

    Args:
        n_bits: block size in bits (must be a multiple of 4, since the
            S-box operates on 4-bit nibbles).
    """

    # A small fixed 4-bit S-box (values 0-15 -> 0-15 bijection), used purely
    # as a toy substitution layer.
    SBOX = [0xC, 0x5, 0x6, 0xB, 0x9, 0x0, 0xA, 0xD, 0x3, 0xE, 0xF, 0x8, 0x4, 0x7, 0x1, 0x2]

    def __init__(self, n_bits: int) -> None:
        self.n_bits = n_bits
        self.n_full_nibbles = n_bits // 4
        self.remainder_bits = n_bits % 4

    def _substitute(self, value: int) -> int:
        out = 0
        for i in range(self.n_full_nibbles):
            nibble = (value >> (4 * i)) & 0xF
            out |= self.SBOX[nibble] << (4 * i)
        if self.remainder_bits:
            # Leftover bits (n_bits not a multiple of 4, e.g. paper's n=6)
            # pass through the substitution layer unchanged.
            rem = (value >> (4 * self.n_full_nibbles)) & ((1 << self.remainder_bits) - 1)
            out |= rem << (4 * self.n_full_nibbles)
        return out

    def _permute_bits(self, value: int) -> int:
        # Simple fixed bit-rotation as the permutation layer.
        return ((value << 1) | (value >> (self.n_bits - 1))) & ((1 << self.n_bits) - 1)

    def encrypt(self, key: int, plaintext: int) -> int:
        """Encrypt a plaintext under `key` using 2 SPN rounds.

        Args:
            key: n_bits-wide integer key.
            plaintext: n_bits-wide integer plaintext.

        Returns:
            n_bits-wide integer ciphertext.
        """
        state = plaintext ^ key
        for _ in range(2):
            state = self._substitute(state)
            state = self._permute_bits(state)
            state ^= key
        return state


class GroverSPNOracle:
    """Builds a Grover oracle marking the key that maps a known plaintext to a known ciphertext.

    Since the SPN's encryption function is classically pre-computable for
    every candidate key (small n_bits <= 8 in the paper's validated sizes),
    the oracle is implemented as a phase-flip lookup table: for each
    candidate key k, if ToySPNCipher(n).encrypt(k, plaintext) == ciphertext,
    the oracle applies a phase of -1 to |k>.
    """

    def build_circuit(self, plaintext: int, ciphertext: int, n_bits: int) -> QuantumCircuit:
        """Build the n_bits-qubit Grover oracle (phase-flip convention).

        Args:
            plaintext: known plaintext (int in [0, 2^n_bits)).
            ciphertext: known ciphertext produced by the true (unknown) key.
            n_bits: key size in bits.

        Returns:
            An n_bits-qubit QuantumCircuit that flips the phase of the
            unique key(s) k satisfying ToySPNCipher(n_bits).encrypt(k, plaintext) == ciphertext.
            The number of matching keys M is stored as `circuit.metadata["num_marked"]`
            since Grover's optimal iteration count depends on M (a fixed
            (plaintext, ciphertext) pair can, in general, be produced by more
            than one key -- key->ciphertext at fixed plaintext is not
            guaranteed injective -- so M is not always 1).
        """
        cipher = ToySPNCipher(n_bits)
        qc = QuantumCircuit(n_bits, name=f"Grover_SPN_oracle(n={n_bits})")
        matching_keys = [
            k for k in range(2 ** n_bits) if cipher.encrypt(k, plaintext) == ciphertext
        ]
        qc.metadata = {"num_marked": len(matching_keys)}
        for k in matching_keys:
            ctrl_state = format(k, f"0{n_bits}b")
            # Multi-controlled-Z (phase flip) on |k>: implement via X-sandwiched MCZ.
            for i, bit in enumerate(ctrl_state[::-1]):
                if bit == "0":
                    qc.x(i)
            qc.h(n_bits - 1)
            qc.mcx(list(range(n_bits - 1)), n_bits - 1)
            qc.h(n_bits - 1)
            for i, bit in enumerate(ctrl_state[::-1]):
                if bit == "0":
                    qc.x(i)
        return qc
