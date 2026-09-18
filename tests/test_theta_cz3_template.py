from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from qiskit import QuantumCircuit, qpy
from qiskit.circuit import Gate, Parameter
from qiskit.circuit.library import U3Gate
from qiskit.quantum_info import Operator

from qudits_on_qubits.benchmarks.direct_basis.math_utils import qutrit_cz
from qudits_on_qubits.benchmarks.direct_basis.optimized_gates import validate_code_space_gate
from qudits_on_qubits.benchmarks.theta_continuation.cz3_template import (
    load_cz3_template,
    parameterize_cz3_circuit,
)


BASELINE = (
    Path(__file__).resolve().parents[1]
    / "experiment_inputs/iqm_randomized_bell/canonical_optimized_20260909/CZ3_W.qpy"
)
BASELINE_SHA256 = "a84ec3bfb10c03e65942fc653c7ba485ffc62a8a85756fa3f62aec8102b5e791"


def _read_baseline():
    with BASELINE.open("rb") as handle:
        return qpy.load(handle)[0]


def _topology(circuit):
    return [
        (item.operation.name, tuple(circuit.find_bit(qubit).index for qubit in item.qubits))
        for item in circuit.data
    ]


def _synthetic():
    circuit = QuantumCircuit(4, global_phase=0.73)
    circuit.metadata = {"nested": {"value": 4}}
    for index in range(6):
        if index % 2:
            circuit.append(U3Gate(index + 0.1, index + 0.2, index + 0.3), [index % 4])
        else:
            circuit.u(index + 0.1, index + 0.2, index + 0.3, index % 4)
        circuit.cz(index % 4, (index + 1) % 4)
    return circuit


def _write_baseline(directory, circuit=None, encoding=None):
    path = directory / "CZ3_W.qpy"
    with path.open("wb") as handle:
        qpy.dump(_read_baseline() if circuit is None else circuit, handle)
    np.save(directory / "E.npy", np.eye(4, 3) if encoding is None else encoding)
    return path


def test_pinned_baseline_has_exact_existing_slots_and_correct_code_space():
    template = load_cz3_template(BASELINE)
    bound = template.bind(template.initial_parameters)
    original = _read_baseline()

    assert template.baseline_sha256 == BASELINE_SHA256
    assert template.initial_parameters.shape == (33,)
    assert len(template.circuit.parameters) == 33
    assert len(template.parameter_metadata) == 33
    assert _topology(template.circuit) == _topology(original)
    assert _topology(bound) == _topology(original)
    assert bound.count_ops() == {"cz": 6, "u3": 11}
    assert bound.depth() == 11
    np.testing.assert_allclose(Operator(bound).data, Operator(original).data, atol=1e-14)
    encoding = np.kron(np.eye(4, 3), np.eye(4, 3))
    validation = validate_code_space_gate(bound, encoding, qutrit_cz())
    assert validation.E_norm == pytest.approx(1.4566641475869985e-6, abs=1e-13)
    assert validation.L_norm == pytest.approx(6.417337310175103e-7, abs=1e-13)


def test_binding_uses_numeric_instruction_order_and_preserves_phase():
    source = _synthetic()
    template = parameterize_cz3_circuit(source)
    parameters = np.linspace(-1.5, 2.6, 18)
    actual = template.bind(parameters)
    expected = QuantumCircuit(4, global_phase=0.73)
    for index in range(6):
        angles = parameters[3 * index:3 * index + 3]
        if index % 2:
            expected.append(U3Gate(*angles), [index % 4])
        else:
            expected.u(*angles, index % 4)
        expected.cz(index % 4, (index + 1) % 4)

    assert _topology(actual) == _topology(source)
    assert actual.global_phase == source.global_phase
    assert len(actual.parameters) == 0
    np.testing.assert_allclose(Operator(actual).data, Operator(expected).data, atol=1e-14)
    np.testing.assert_allclose(
        Operator(template.bind(template.initial_parameters)).data,
        Operator(source).data,
        atol=1e-14,
    )
    for index, record in enumerate(template.parameter_metadata):
        assert record == {
            "index": index,
            "instruction_index": 2 * (index // 3),
            "qubit": (index // 3) % 4,
            "angle": ("theta", "phi", "lambda")[index % 3],
        }


def test_parameterization_copies_source_and_binding_does_not_alias():
    source = _synthetic()
    original_operator = Operator(source).data.copy()
    template = parameterize_cz3_circuit(source)
    source.x(0)
    source.global_phase = 0.21
    source.metadata["nested"]["value"] = 99

    bound = template.bind(template.initial_parameters)
    bound.x(3)
    bound.metadata["nested"]["value"] = 55
    np.testing.assert_allclose(Operator(template.original_circuit).data, original_operator)
    np.testing.assert_allclose(Operator(template.bind(template.initial_parameters)).data, original_operator)
    assert template.original_circuit.metadata["nested"]["value"] == 4
    assert template.circuit.metadata["nested"]["value"] == 4


def test_template_identity_is_deterministic_and_phase_sensitive():
    source = _synthetic()
    first = parameterize_cz3_circuit(source)
    second = parameterize_cz3_circuit(source.copy())
    assert first.template_id == second.template_id
    assert len(first.template_id) == 64
    int(first.template_id, 16)
    source.global_phase += 0.4
    assert parameterize_cz3_circuit(source).template_id != first.template_id
    document = first.to_dict()
    assert document["template_id"] == first.template_id
    assert document["global_phase"] == first.circuit.global_phase
    assert len(document["operations"]) == len(first.circuit.data)
    assert document["parameter_metadata"] == list(first.parameter_metadata)
    json.dumps(document, allow_nan=False)
    document["parameter_metadata"][0]["qubit"] = 99
    assert first.parameter_metadata[0]["qubit"] == 0


@pytest.mark.parametrize("parameters", [
    [0.0] * 17,
    [0.0] * 19,
    [[0.0] * 18],
    [0.0] * 17 + [True],
    np.ones(18, dtype=bool),
    [0.0] * 17 + [np.nan],
    [0.0] * 17 + [np.inf],
    [0.0] * 17 + [1j],
    [0.0] * 17 + ["0.1"],
    None,
])
def test_bind_rejects_invalid_parameter_vectors(parameters):
    template = parameterize_cz3_circuit(_synthetic())
    with pytest.raises(ValueError, match="parameters"):
        template.bind(parameters)


@pytest.mark.parametrize("name", ["x", "cx", "barrier", "reset", "measure", "spoof"])
def test_rejects_unsupported_instructions(name):
    circuit = QuantumCircuit(4, 1) if name == "measure" else QuantumCircuit(4)
    if name == "cx":
        circuit.cx(0, 1)
    elif name == "measure":
        circuit.measure(0, 0)
    elif name == "spoof":
        circuit.append(Gate("u", 1, [0.1, 0.2, 0.3]), [0])
    else:
        getattr(circuit, name)(0)
    with pytest.raises(ValueError):
        parameterize_cz3_circuit(circuit)


@pytest.mark.parametrize("qubits", [2, 3, 5])
def test_rejects_wrong_qubit_count(qubits):
    with pytest.raises(ValueError, match="four"):
        parameterize_cz3_circuit(QuantumCircuit(qubits))


def test_rejects_unused_classical_bits_and_symbolic_source():
    with pytest.raises(ValueError, match="classical"):
        parameterize_cz3_circuit(QuantumCircuit(4, 1))
    circuit = QuantumCircuit(4)
    circuit.u(Parameter("source"), 0.2, 0.3, 0)
    with pytest.raises(ValueError, match="symbolic"):
        parameterize_cz3_circuit(circuit)
    circuit = QuantumCircuit(4, global_phase=Parameter("phase"))
    with pytest.raises(ValueError, match="symbolic"):
        parameterize_cz3_circuit(circuit)


@pytest.mark.parametrize("angle", [np.nan, np.inf, -np.inf])
def test_rejects_nonfinite_source_angles(angle):
    circuit = QuantumCircuit(4)
    circuit.u(angle, 0.2, 0.3, 0)
    with pytest.raises(ValueError, match="finite"):
        parameterize_cz3_circuit(circuit)


@pytest.mark.parametrize("encoding", [np.eye(3), np.eye(4, 3)[:, ::-1], np.eye(4, 3) * 1j, np.full((4, 3), np.nan)])
def test_loader_rejects_noncanonical_encoding(tmp_path, encoding):
    path = _write_baseline(tmp_path, encoding=encoding)
    with pytest.raises(ValueError, match="E.npy|encoding"):
        load_cz3_template(path, expected_sha256=None)


def test_loader_verifies_expected_hash(tmp_path):
    path = _write_baseline(tmp_path)
    with pytest.raises(ValueError, match="SHA256|sha256|hash"):
        load_cz3_template(path, expected_sha256="0" * 64)
    checksum = hashlib.sha256(path.read_bytes()).hexdigest()
    assert load_cz3_template(path, expected_sha256=checksum).baseline_sha256 == checksum


def test_loader_revalidates_code_space_instead_of_trusting_metadata(tmp_path):
    circuit = _read_baseline()
    circuit.u(0.7, 0.0, 0.0, 0)
    circuit.metadata = {"E_norm": 0.0, "L_norm": 0.0}
    path = _write_baseline(tmp_path, circuit)
    with pytest.raises(ValueError, match="code.space"):
        load_cz3_template(path, expected_sha256=None)


def test_loader_accepts_valid_custom_topology_when_hash_is_explicitly_disabled(tmp_path):
    circuit = _read_baseline()
    circuit.u(0.0, 0.0, 0.0, 0)
    path = _write_baseline(tmp_path, circuit)
    template = load_cz3_template(path, expected_sha256=None)
    assert template.initial_parameters.shape == (36,)
    assert _topology(template.circuit) == _topology(circuit)


def test_loader_requires_exactly_one_qpy_circuit(tmp_path):
    path = _write_baseline(tmp_path)
    with path.open("wb") as handle:
        qpy.dump([_read_baseline(), _read_baseline()], handle)
    with pytest.raises(ValueError, match="exactly one"):
        load_cz3_template(path, expected_sha256=None)


@pytest.mark.parametrize("tolerance", [True, 0, -1, np.nan, np.inf, "1e-5"])
def test_loader_rejects_invalid_tolerance(tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        load_cz3_template(BASELINE, tolerance=tolerance)


def test_loader_enforces_requested_tolerance():
    with pytest.raises(ValueError, match="code.space"):
        load_cz3_template(BASELINE, tolerance=1e-8)
