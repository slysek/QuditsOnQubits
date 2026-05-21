from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from .bell_builders import BellTerm, bell_terms, num_qutrits_for_candidate
from .encoding import (
    embed_projector_E,
    infer_num_qutrits_from_state,
    kron_all,
    leakage_probability,
    projector_E,
    statevector_data,
)
from .estimator_backend import BellResult


def bell_value_sampler(
    state_or_circuit: Statevector | QuantumCircuit | np.ndarray,
    candidate: str,
    *,
    E: np.ndarray,
    shots: int | None = None,
    seed: int | None = 1234,
    postselect: bool = True,
    sampler: Any | None = None,
) -> BellResult:
    """Evaluate the Bell expression via explicit projective measurement sums.

    The implementation uses qutrit projectors embedded by E P_a E^dagger. This
    is the safe path for a general isometry E, including encodings with support
    on |11>. A qiskit sampler object can be passed for API symmetry, but the
    actual decoding remains projector-based to avoid computational-basis
    assumptions.
    """
    del sampler
    state = _as_statevector(state_or_circuit)
    num_qutrits = num_qutrits_for_candidate(candidate)
    if infer_num_qutrits_from_state(state) != num_qutrits:
        raise ValueError(f"state dimension does not match candidate {candidate!r}")

    rng = np.random.default_rng(seed)
    value = 0.0 + 0.0j
    for term in bell_terms(candidate):
        correlator = _term_correlator(
            state,
            term,
            E,
            num_qutrits,
            shots=shots,
            rng=rng,
            postselect=postselect,
        )
        value += term.coefficient * correlator

    leak = leakage_probability(state, E, num_qutrits)
    return BellResult(
        value=complex(value),
        leakage_probability=leak,
        backend="ProjectorSampler",
        shots=shots,
    )


def _as_statevector(state_or_circuit: Statevector | QuantumCircuit | np.ndarray) -> Statevector:
    if isinstance(state_or_circuit, Statevector):
        return state_or_circuit
    if isinstance(state_or_circuit, QuantumCircuit):
        return Statevector.from_instruction(state_or_circuit)
    return Statevector(statevector_data(state_or_circuit))


def _term_correlator(
    state: Statevector,
    term: BellTerm,
    E: np.ndarray,
    num_qutrits: int,
    *,
    shots: int | None,
    rng: np.random.Generator,
    postselect: bool,
) -> complex:
    vector = np.asarray(state.data, dtype=complex)
    factors = tuple(term.factors)
    measurements = [_embedded_measurement(E, factor.matrix) for factor in factors]
    code_projector = projector_E(E)

    probabilities: list[float] = []
    phases: list[complex] = []
    for outcomes in product(range(3), repeat=len(factors)):
        local = [code_projector for _ in range(num_qutrits)]
        phase = 1.0 + 0.0j
        for factor, outcome, measurement in zip(factors, outcomes, measurements):
            projectors, eigenvalues = measurement
            local[factor.party] = projectors[outcome]
            phase *= eigenvalues[outcome]
        projector = kron_all(local)
        probability = float(np.real_if_close(np.vdot(vector, projector @ vector)))
        probabilities.append(max(0.0, probability))
        phases.append(phase)

    code_probability = sum(probabilities)
    leakage = max(0.0, 1.0 - code_probability)
    if shots is None:
        denominator = code_probability if postselect and code_probability > 0 else 1.0
        return sum(p * phase for p, phase in zip(probabilities, phases)) / denominator

    event_probabilities = np.array(probabilities + [leakage], dtype=float)
    total = event_probabilities.sum()
    if total <= 0:
        return 0.0 + 0.0j
    event_probabilities /= total
    counts = rng.multinomial(shots, event_probabilities)
    nonleak_counts = counts[:-1]
    denominator = nonleak_counts.sum() if postselect else shots
    if denominator == 0:
        return 0.0 + 0.0j
    return sum(c * phase for c, phase in zip(nonleak_counts, phases)) / denominator


def _embedded_measurement(E: np.ndarray, observable: np.ndarray) -> tuple[list[np.ndarray], list[complex]]:
    projectors, eigenvalues = _qutrit_spectral_projectors(observable)
    return [embed_projector_E(E, projector) for projector in projectors], eigenvalues


def _qutrit_spectral_projectors(observable: np.ndarray) -> tuple[list[np.ndarray], list[complex]]:
    eigenvalues, eigenvectors = np.linalg.eig(np.asarray(observable, dtype=complex))
    order = np.argsort(np.angle(eigenvalues))
    projectors: list[np.ndarray] = []
    ordered_eigenvalues: list[complex] = []
    for idx in order:
        eigenvalue = eigenvalues[idx]
        vector = eigenvectors[:, idx]
        vector = vector / np.linalg.norm(vector)
        projectors.append(np.outer(vector, vector.conj()))
        ordered_eigenvalues.append(complex(eigenvalue))
    return projectors, ordered_eigenvalues


def make_aer_sampler() -> Any | None:
    try:
        from qiskit_aer.primitives import SamplerV2 as AerSampler
    except Exception:
        return None
    return AerSampler()
