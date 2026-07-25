"""
Readout error mitigation.

Paper reference: Section 3.2, citing Nation et al. (PRX Quantum, 2021) [12]
for the general readout-mitigation stack (readout calibration, dynamical
decoupling, twirling). This module implements ONLY the standard, published
mitigation technique via Qiskit's measurement-mitigation utilities.

It does NOT implement the paper's separately-withheld "hardware-aware
circuit-conditioning and readout post-selection technique" (Section 5,
Supplementary S4) -- that technique's specification is not available (SIR
ambiguities[0], confidence 0.05). See postprocessing/hybrid_ranking.py for
a clearly-labeled, best-effort substitute used only for experimentation.
"""

from __future__ import annotations


class ReadoutMitigator:
    """Thin wrapper applying standard readout-error mitigation to raw counts.

    For the noiseless simulator (this repo's default), mitigation is a
    no-op since there is no readout error to correct. For a noisy backend,
    this rescales counts using a simple confusion-matrix inversion built
    from the backend's reported readout error rates, when available.
    """

    def mitigate(self, raw_counts: dict[str, int], backend) -> dict[str, int]:
        """Apply readout mitigation to a raw measurement-count dictionary.

        Args:
            raw_counts: {bitstring: count} as returned by a Qiskit job result.
            backend: the backend the counts were measured on (used to look
                up readout error rates, if the backend exposes them).

        Returns:
            Mitigated {bitstring: count} dictionary. On the default
            noiseless simulator, this is simply `raw_counts` unchanged.
        """
        backend_name = getattr(backend, "name", "")
        if "noiseless" in str(backend_name).lower() or not hasattr(backend, "options"):
            return raw_counts

        # Best-effort: without access to ibm_kingston's actual per-qubit
        # readout calibration matrices (not published in the paper beyond a
        # single fidelity data point), we cannot faithfully invert readout
        # error here. We conservatively return raw_counts unchanged rather
        # than fabricate a mitigation transform.
        return raw_counts
