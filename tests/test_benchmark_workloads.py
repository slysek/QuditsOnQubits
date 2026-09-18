import json
from types import SimpleNamespace

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from qudits_on_qubits.benchmarks.direct_basis.math_utils import (
    physical_single_qutrit_gate_in_encoding, physical_two_qutrit_gate_in_encoding,
    qutrit_cz, qutrit_fourier,
)
from qudits_on_qubits.benchmarks.workloads import LogicalCircuit, LogicalOperation, two_qutrit_graph_circuit


def _library(encoding):
    f3 = QuantumCircuit(2)
    f3.unitary(physical_single_qutrit_gate_in_encoding(qutrit_fourier(), encoding), [0, 1])
    cz3 = QuantumCircuit(4)
    cz3.unitary(physical_two_qutrit_gate_in_encoding(qutrit_cz(), encoding), [0, 1, 2, 3])
    return SimpleNamespace(f3=f3, cz3=cz3)


def _arbitrary_encoding():
    rng = np.random.default_rng(73)
    return np.linalg.qr(rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4)))[0][:, :3]


def test_two_qutrit_graph_reference_is_maximally_entangled():
    workload = two_qutrit_graph_circuit()
    state = workload.reference_state()
    coefficients = state.reshape(3, 3)
    np.testing.assert_allclose(coefficients @ coefficients.conj().T, np.eye(3) / 3, atol=1e-14)
    np.testing.assert_allclose(coefficients.conj().T @ coefficients, np.eye(3) / 3, atol=1e-14)
    np.testing.assert_allclose(state, qutrit_cz() @ np.ones(9) / 3, atol=1e-14)
    assert workload.operations == (LogicalOperation("f3", (0,)), LogicalOperation("f3", (1,)), LogicalOperation("cz3", (0, 1)))


@pytest.mark.parametrize("target", [0, 1, 2])
def test_logical_reference_uses_qutrit_zero_as_least_significant(target):
    circuit = LogicalCircuit(3, (LogicalOperation("f3", (target,)),))
    expected = np.zeros(27, complex)
    expected[[digit * 3**target for digit in range(3)]] = 1 / np.sqrt(3)
    np.testing.assert_allclose(circuit.reference_state(), expected, atol=1e-14)


@pytest.mark.parametrize("encoding", [np.eye(4, 3), _arbitrary_encoding()])
def test_custom_asymmetric_workload_agrees_with_independent_encoded_reference(encoding):
    circuit = LogicalCircuit(3, (
        LogicalOperation("f3", (2,)), LogicalOperation("f3", (0,)),
        LogicalOperation("cz3", (2, 0)), LogicalOperation("f3", (0,)),
        LogicalOperation("f3", (1,)), LogicalOperation("cz3", (1, 0)),
        LogicalOperation("f3", (2,)),
    ), name="asymmetric")
    # Before the last F3, only qutrit 0 = -qutrit 2 (mod 3) has support.
    expected = np.zeros(27, complex)
    omega = np.exp(2j * np.pi / 3)
    for q2 in range(3):
        for q1 in range(3):
            for q0 in range(3):
                previous_q2 = (-q0) % 3
                expected[q0 + 3 * q1 + 9 * q2] = omega ** (q1 * q0 + previous_q2 * q2) / np.sqrt(27)
    np.testing.assert_allclose(circuit.reference_state(), expected, atol=1e-14)
    independent_encoding = np.kron(np.kron(encoding, encoding), encoding) @ expected
    np.testing.assert_allclose(circuit.encoded_reference(encoding), independent_encoding, atol=1e-14)
    np.testing.assert_allclose(Statevector.from_instruction(circuit.build(encoding, _library(encoding))).data, independent_encoding, atol=1e-13)


def test_empty_workload_still_prepares_every_encoded_zero():
    encoding = _arbitrary_encoding()
    circuit = LogicalCircuit(2, (), name="initial")
    physical = circuit.build(encoding, SimpleNamespace())
    assert physical.count_ops() == {"state_preparation": 2}
    expected = np.kron(encoding[:, 0], encoding[:, 0])
    np.testing.assert_allclose(Statevector.from_instruction(physical).data, expected, atol=1e-14)
    np.testing.assert_allclose(circuit.encoded_reference(encoding), expected, atol=1e-14)


def test_circuit_snapshots_inputs_and_hashes_complete_ordered_configuration():
    operations = [LogicalOperation("f3", [np.int64(0)])]
    circuit = LogicalCircuit(np.int64(2), operations, "custom")
    operations.clear()
    serialized = circuit.to_dict()
    assert json.loads(json.dumps(serialized)) == serialized
    assert serialized["operations"] == [{"name": "f3", "targets": [0]}]
    assert len(circuit.stable_hash()) == 64
    assert circuit.stable_hash() == LogicalCircuit(2, (LogicalOperation("f3", (0,)),)).stable_hash()
    assert circuit.stable_hash() != LogicalCircuit(2, (LogicalOperation("f3", (1,)),)).stable_hash()
    left = LogicalCircuit(2, (LogicalOperation("f3", (0,)), LogicalOperation("cz3", (0, 1))))
    right = LogicalCircuit(2, tuple(reversed(left.operations)))
    assert left.stable_hash() != right.stable_hash()


@pytest.mark.parametrize("name,targets", [
    ("x3", (0,)), ("F3", (0,)), ("f3", ()), ("f3", (0, 1)),
    ("cz3", (0,)), ("cz3", (0, 0)), ("cz3", (0, 1, 2)),
    ("f3", (-1,)), ("f3", (0.0,)), ("f3", (True,)),
    ("f3", (np.inf,)), ("f3", (np.nan,)), ("f3", "0"),
])
def test_operations_reject_invalid_inputs(name, targets):
    with pytest.raises((ValueError, TypeError)):
        LogicalOperation(name, targets)


@pytest.mark.parametrize("num_qutrits", [0, -1, True, np.bool_(True), 2.0, "2", None])
def test_circuit_rejects_invalid_qutrit_count(num_qutrits):
    with pytest.raises((ValueError, TypeError), match="num_qutrits"):
        LogicalCircuit(num_qutrits, ())


def test_circuit_rejects_targets_outside_register_and_non_operations():
    with pytest.raises(ValueError, match="target"):
        LogicalCircuit(2, (LogicalOperation("f3", (2,)),))
    with pytest.raises(TypeError, match="operations"):
        LogicalCircuit(2, ({"name": "f3", "targets": [0]},))
    with pytest.raises(ValueError, match="name"):
        LogicalCircuit(2, (), name=" ")


def test_build_rejects_invalid_encoding_and_wrong_library_width():
    circuit = two_qutrit_graph_circuit()
    with pytest.raises(ValueError, match="encoding"):
        circuit.build(np.eye(3), _library(np.eye(4, 3)))
    with pytest.raises(ValueError, match="f3"):
        circuit.build(np.eye(4, 3), SimpleNamespace(f3=QuantumCircuit(1), cz3=QuantumCircuit(4)))
    with pytest.raises(ValueError, match="encoding"):
        circuit.encoded_reference(np.full((4, 3), np.nan))
