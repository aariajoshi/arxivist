"""
Test suite: verifies all five disclosed quantum attacks recover the correct
secret/period on a noiseless simulator, across multiple random trials.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from quantum_cryptanalysis.algorithms.bv_circuit import BernsteinVazirani
from quantum_cryptanalysis.algorithms.grover_circuit import GroverSearch
from quantum_cryptanalysis.algorithms.simon_circuit import SimonAlgorithm
from quantum_cryptanalysis.backend.execution import BackendFactory
from quantum_cryptanalysis.oracles.bernstein_vazirani import BVOracle
from quantum_cryptanalysis.oracles.grover_spn import GroverSPNOracle, ToySPNCipher
from quantum_cryptanalysis.oracles.promise_utils import verify_two_to_one_promise
from quantum_cryptanalysis.oracles.simon_cbc_mac import CBCMACForgeryOracle
from quantum_cryptanalysis.oracles.simon_even_mansour import EvenMansourOracle, find_valid_permutation
from quantum_cryptanalysis.oracles.simon_feistel import FeistelOracle

BACKEND = BackendFactory("aer_simulator_noiseless").get_backend()


def test_bernstein_vazirani():
    for n in (4, 8):
        rng = random.Random(n)
        secret = "".join(rng.choice("01") for _ in range(n))
        oracle = BVOracle().build_circuit(secret)
        recovered = BernsteinVazirani().run(oracle, n=n, backend=BACKEND, shots=1)
        assert recovered == secret, f"BV failed for n={n}: {recovered} != {secret}"


def test_simon_even_mansour():
    simon = SimonAlgorithm()
    for n in (3, 4, 5, 6):
        rng = random.Random(n)
        k1 = rng.randint(1, 2 ** n - 1)
        perm = find_valid_permutation(n, k1, rng)
        oracle = EvenMansourOracle().build_circuit(k1=k1, permutation=perm, n=n)
        measurements = simon.collect_measurements(oracle, n=n, backend=BACKEND, shots=500)
        s = simon.solve_period(measurements, n=n)
        assert s == format(k1, f"0{n}b"), f"EM failed for n={n}: {s} != {k1}"


def test_simon_cbc_mac_forgery():
    simon = SimonAlgorithm()
    for n in (4, 6):
        rng = random.Random(n)
        a = rng.randint(1, 2 ** n - 1)
        b = 0
        perm = find_valid_permutation(n, a, rng)

        def block_cipher(key, pt, perm=perm):
            return perm[pt]

        true_s = a ^ b

        oracle = CBCMACForgeryOracle().build_circuit(block_cipher, 0, a, b, n)
        measurements = simon.collect_measurements(oracle, n=n, backend=BACKEND, shots=500)
        s = simon.solve_period(measurements, n=n)
        assert s == format(true_s, f"0{n}b"), f"CBC-MAC failed for n={n}: {s} != {true_s}"


def test_simon_feistel():
    simon = SimonAlgorithm()
    for m in (3, 4):
        rng = random.Random(m)
        perm1 = list(range(2 ** m)); rng.shuffle(perm1)
        perm2 = list(range(2 ** m)); rng.shuffle(perm2)
        f1, f2 = (lambda a: perm1[a]), (lambda a: perm2[a])
        alpha0 = rng.randint(0, 2 ** m - 1)
        alpha1 = rng.randint(0, 2 ** m - 1)
        while alpha1 == alpha0:
            alpha1 = rng.randint(0, 2 ** m - 1)
        gamma = f1(alpha0) ^ f1(alpha1)
        true_s = format(1 | (gamma << 1), f"0{1 + m}b")

        oracle = FeistelOracle().build_circuit(f1, f2, alpha0, alpha1, m)
        measurements = simon.collect_measurements(oracle, n=1 + m, backend=BACKEND, shots=500)
        s = simon.solve_period(measurements, n=1 + m)
        assert s == true_s, f"Feistel failed for m={m}: {s} != {true_s}"


def test_grover_spn():
    for n in (6, 8):
        rng = random.Random(n)
        cipher = ToySPNCipher(n)
        true_key = rng.randint(1, 2 ** n - 1)
        plaintext = None
        for candidate_pt in range(2 ** n):
            ct = cipher.encrypt(true_key, candidate_pt)
            if sum(1 for k in range(2 ** n) if cipher.encrypt(k, candidate_pt) == ct) == 1:
                plaintext = candidate_pt
                break
        assert plaintext is not None
        ciphertext = cipher.encrypt(true_key, plaintext)
        oracle = GroverSPNOracle().build_circuit(plaintext, ciphertext, n)
        recovered = GroverSearch().run(oracle, n=n, backend=BACKEND, shots=4096)
        assert recovered == true_key, f"Grover failed for n={n}: {recovered} != {true_key}"


def test_cbc_mac_period_discrepancy_documented():
    """Confirms the verified discrepancy: paper's stated s=Ek(a)^Ek(b) does NOT
    satisfy the promise for a generic permutation, while s=a^b does (using
    the same deterministic construction as Even-Mansour, with b=0)."""
    rng = random.Random(0)
    n = 4
    a = rng.randint(1, 2 ** n - 1)
    b = 0
    perm = find_valid_permutation(n, a, rng)

    def block_cipher(key, pt):
        return perm[pt]

    f = lambda x: block_cipher(0, x ^ a) ^ block_cipher(0, x ^ b)
    s_derived = a ^ b
    s_paper = block_cipher(0, a) ^ block_cipher(0, b)

    assert verify_two_to_one_promise(f, s_derived, n), "expected the derived a^b period to satisfy the promise"
    if s_paper != s_derived:
        assert not verify_two_to_one_promise(f, s_paper, n), (
            "paper's Ek(a)^Ek(b) formula should generically NOT satisfy the promise"
        )


if __name__ == "__main__":
    test_bernstein_vazirani()
    test_simon_even_mansour()
    test_simon_cbc_mac_forgery()
    test_simon_feistel()
    test_grover_spn()
    test_cbc_mac_period_discrepancy_documented()
    print("All tests passed.")
