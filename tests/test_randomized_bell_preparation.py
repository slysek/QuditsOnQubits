from itertools import product
import json

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qudits_on_qubits.bell_measurements import build_sampler_circuits_for_candidate
from qudits_on_qubits.experiments.artifacts import BasisArtifacts
from qudits_on_qubits.experiments.errors import ExperimentValidationError
from qudits_on_qubits.experiments.measurement import RandomizedBlocks
from qudits_on_qubits.experiments.preparation import metadata_summary, prepare_measurements
from qudits_on_qubits.experiments.setting_schedule import generate_schedule, local_settings_for_reference
from qudits_on_qubits.reference_experiments import get_reference_experiment


def tensor(matrices):
    result = np.ones((1, 1), complex)
    for matrix in matrices:
        result = np.kron(result, matrix)
    return result


def full_schedule(reference):
    local = local_settings_for_reference(reference)
    settings = list(product(*local))
    draws = [settings[0]] + settings
    indices = iter(local[i].index(label) for setting in draws for i, label in enumerate(setting))
    return generate_schedule(reference, RandomizedBlocks(len(draws), 2, len(draws)), _randbelow=lambda n: next(indices), _source="test_sequence")


def encoding(kind):
    if kind == "canonical":
        return np.eye(4, 3, dtype=complex)
    if kind == "permuted":
        return np.eye(4, dtype=complex)[:, [2, 0, 3]]
    rng = np.random.default_rng(45)
    matrix, _ = np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))
    return matrix[:, :3]


@pytest.mark.parametrize("state", ["two_qutrit", "ghz3", "ame43"])
@pytest.mark.parametrize("kind", ["canonical", "permuted", "dense"])
def test_full_catalogue_probabilities_match_local_eigenbasis_oracle(state, kind, tmp_path):
    reference = get_reference_experiment(state)
    parties = reference.state.num_parties
    rng = np.random.default_rng(451)
    logical = rng.normal(size=3**parties) + 1j * rng.normal(size=3**parties)
    logical /= np.linalg.norm(logical)
    e = encoding(kind)
    physical = tensor([e] * parties) @ logical
    state_circuit = QuantumCircuit(2 * parties)
    state_circuit.initialize(physical)
    original = state_circuit.copy()
    artifacts = BasisArtifacts(tmp_path, state, state_circuit, e, {}, {})
    schedule = full_schedule(reference)

    prepared = prepare_measurements(artifacts, schedule=schedule)

    assert state_circuit == original
    assert len(prepared.circuits) == len(schedule.blocks) - 1
    assert prepared.metadata["catalog_index_by_block_id"][0] == prepared.metadata["catalog_index_by_block_id"][1] == 0
    assert prepared.metadata["physical_to_logical_outcome_map"] == {0: 0, 1: 1, 2: 2, 3: None}
    accepted_indices = [sum(outcome * 4 ** (parties - i - 1) for i, outcome in enumerate(outcomes)) for outcomes in product(range(3), repeat=parties)]
    for circuit, setting, meta in zip(prepared.circuits, prepared.metadata["setting_by_circuit_index"], prepared.metadata["circuit_metadata"]):
        assert len([item for item in circuit.data if item.operation.name == "measure"]) == 2 * parties
        assert all(gate is not None for gate in meta["local_basis_gates"])
        measured = circuit.remove_final_measurements(inplace=False)
        probabilities = Statevector.from_instruction(measured).probabilities()
        basis = tensor([reference.observable(label).ordered_eigenbasis()[0] for label in setting])
        expected = np.abs(basis.conj().T @ logical) ** 2
        np.testing.assert_allclose(probabilities[accepted_indices], expected, atol=1e-10, rtol=0)
        assert sum(probabilities[accepted_indices]) == pytest.approx(1)
        # Invert only the appended local rotations: preparation must be intact.
        restored = measured.copy()
        for gate in reversed(meta["local_basis_gates"]):
            restored.unitary(gate["unitary"].conj().T, gate["qubits"])
        np.testing.assert_allclose(Statevector.from_instruction(restored).data, physical, atol=1e-10, rtol=0)
    summary = metadata_summary(prepared.metadata)
    assert summary["catalog_index_by_block_id"] == dict(prepared.metadata["catalog_index_by_block_id"])
    json.dumps(summary)


def test_explicit_settings_deduplicate_in_first_occurrence_order_and_keep_terms():
    explicit = [("A2", "B2", "C1"), ("A0", "B0", "C0"), ("A2", "B2", "C1")]
    circuit = QuantumCircuit(6)
    _, legacy = build_sampler_circuits_for_candidate("ghz3", circuit, np.eye(4, 3))
    circuits, metadata = build_sampler_circuits_for_candidate("ghz3", circuit, np.eye(4, 3), explicit_settings=explicit)
    assert len(circuits) == 2
    assert metadata["setting_by_circuit_index"] == explicit[:2]
    assert metadata["terms"] == legacy["terms"]
    assert metadata["encoding_outcome_map"] == legacy["encoding_outcome_map"]


@pytest.mark.parametrize("settings", [[], [("A0", "B0")], [("A0", "B0", None)], [("B0", "A0", "C0")], [("A0", "B0", "C2")], ["A0B0C0"]])
def test_explicit_settings_reject_invalid_labels_before_inplace_mutation(settings):
    circuit = QuantumCircuit(6)
    with pytest.raises(ValueError, match="explicit_settings"):
        build_sampler_circuits_for_candidate("ghz3", circuit, np.eye(4, 3), explicit_settings=settings, inplace=True)
    assert circuit.size() == 0


def test_preparation_requires_complete_matching_schedule(tmp_path):
    reference = get_reference_experiment("ghz3")
    incomplete = generate_schedule(reference, RandomizedBlocks(1, 1, 1), _randbelow=lambda n: 0)
    artifacts = BasisArtifacts(tmp_path, "ghz3", QuantumCircuit(6), np.eye(4, 3), {}, {})
    with pytest.raises(ExperimentValidationError, match="complete"):
        prepare_measurements(artifacts, schedule=incomplete)
    with pytest.raises(ExperimentValidationError, match="state"):
        prepare_measurements(artifacts, schedule=full_schedule(get_reference_experiment("two_qutrit")))
