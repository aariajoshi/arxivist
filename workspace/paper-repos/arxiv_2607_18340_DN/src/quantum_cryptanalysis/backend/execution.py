"""
Backend abstraction: Qiskit Aer simulator by default, with an optional real
IBM Quantum hardware hook.

Paper reference: Section 3.2 (real ibm_kingston hardware execution via
Qiskit) and Section 3.3 (genuine statevector simulation, paper's own
"QuantumOS on NVIDIA B200"). Real IBM hardware access requires the user's
own IBM Quantum account/credentials, which cannot be assumed available in a
reproduction environment (SIR implementation_assumptions[3], confidence
0.9) -- this defaults to Qiskit Aer instead.
"""

from __future__ import annotations

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error


class BackendFactory:
    """Creates a Qiskit-compatible backend for circuit execution.

    Args:
        mode: one of "aer_simulator_noiseless" (default), "aer_simulator_noisy",
            or "ibm_real" (requires `ibm_backend_name` and a configured
            IBM Quantum account via qiskit-ibm-runtime; not used by default).
    """

    def __init__(self, mode: str = "aer_simulator_noiseless") -> None:
        valid_modes = ("aer_simulator_noiseless", "aer_simulator_noisy", "ibm_real")
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got {mode!r}")
        self.mode = mode

    def get_backend(
        self,
        noise_model: "NoiseModel | None" = None,
        ibm_backend_name: str | None = None,
    ):
        """Return a backend object.

        Args:
            noise_model: optional custom Qiskit Aer NoiseModel; if None and
                mode="aer_simulator_noisy", a simple synthetic depolarizing
                noise model is used as a stand-in for real hardware noise
                (NOT a reproduction of ibm_kingston's actual noise profile).
            ibm_backend_name: required if mode="ibm_real" (e.g. "ibm_kingston").

        Returns:
            A Qiskit-Aer (or, if configured, real IBM Runtime) backend.
        """
        if self.mode == "aer_simulator_noiseless":
            return AerSimulator()

        if self.mode == "aer_simulator_noisy":
            if noise_model is None:
                noise_model = self._default_synthetic_noise_model()
            return AerSimulator(noise_model=noise_model)

        if self.mode == "ibm_real":
            if not ibm_backend_name:
                raise ValueError("ibm_backend_name is required when mode='ibm_real'")
            try:
                from qiskit_ibm_runtime import QiskitRuntimeService
            except ImportError as e:
                raise ImportError(
                    "qiskit-ibm-runtime is required for mode='ibm_real'. "
                    "Install it and configure your IBM Quantum credentials "
                    "(QiskitRuntimeService.save_account(...)) before using this mode."
                ) from e
            service = QiskitRuntimeService()
            return service.backend(ibm_backend_name)

        raise AssertionError("unreachable")  # mode already validated in __init__

    @staticmethod
    def _default_synthetic_noise_model(depolarizing_prob: float = 0.01) -> "NoiseModel":
        """A simple synthetic depolarizing noise model.

        NOTE: this is a generic stand-in for experimentation, NOT a
        reproduction of ibm_kingston's actual (Heron-generation) noise
        profile, which is not published in the paper beyond a single
        two-qubit gate fidelity data point (219/233 for the block-8 Feistel
        instance, Section 3.2).
        """
        noise_model = NoiseModel()
        error_1q = depolarizing_error(depolarizing_prob, 1)
        error_2q = depolarizing_error(depolarizing_prob * 2, 2)
        noise_model.add_all_qubit_quantum_error(error_1q, ["h", "x", "rz", "sx"])
        noise_model.add_all_qubit_quantum_error(error_2q, ["cx", "cz"])
        return noise_model
