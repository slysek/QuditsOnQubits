"""Check the physical circuits for the qutrit graph states."""

from itertools import combinations, product
from pathlib import Path

import numpy as np
import pytest
from qiskit import QuantumCircuit, qpy, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Statevector

from qudits_on_qubits.evolution.state_workloads import (
    _prepare, build_workload, evaluate_workload,
)
from qudits_on_qubits.reference_experiments import get_reference_experiment

REFERENCE = (Path(__file__).resolve().parents[1] / "experiment_inputs" / "iqm_randomized_bell"
             / "canonical_optimized_20260909")


def _circuit(name):
    with (REFERENCE / name).open("rb") as stream:
        result, = qpy.load(stream)
    return result


def _estar():
    return np.array([[0, 1 / np.sqrt(2), 1 / np.sqrt(2)],
                     [1, 0, 0], [0, 1 / np.sqrt(2), -1 / np.sqrt(2)],
                     [0, 0, 0]], dtype=complex)


@pytest.mark.parametrize("name,count", [("two_qutrit", 10), ("ghz3", 18), ("ame43", 32)])
def test_canonical_gate_by_gate(name, count):
    encoding = np.eye(4, 3)
    circuit = build_workload(name, encoding, _circuit("CZ3_W.qpy"), f3=_circuit("F3_W.qpy"))
    result = evaluate_workload(name, circuit, encoding)
    assert result["is_valid"]
    assert not result["strict_valid"]
    assert result["cz_count"] == count
    assert result["cx_count"] == 0
    assert result["two_qubit_gate_count"] == count
    assert {item.operation.name for item in circuit.data} <= {"u", "u3", "cz"}


@pytest.mark.parametrize("name,count", [("two_qutrit", 8), ("ghz3", 15), ("ame43", 28)])
def test_canonical_direct_preparation(name, count):
    encoding = np.eye(4, 3)
    circuit = build_workload(name, encoding, _circuit("CZ3_W.qpy"))
    assert evaluate_workload(name, circuit, encoding)["is_valid"]
    assert circuit.count_ops().get("cz", 0) == count


def test_estar_direct_preparation_is_product():
    target = _estar() @ np.ones(3) / np.sqrt(3)
    circuit = _prepare(target)
    assert circuit.count_ops().get("cz", 0) == 0
    assert abs(np.vdot(target, Statevector(circuit).data)) == pytest.approx(1.0, abs=1e-14)


def test_estar_encoding_with_fixed_gate():
    encoding = _estar()
    change = np.column_stack((encoding, np.array([0, 0, 0, 1])))
    circuit = QuantumCircuit(2)
    circuit.append(UnitaryGate(change), [0, 1])
    circuit = transpile(circuit, basis_gates=["u", "cz"], optimization_level=0, seed_transpiler=0)
    gate = QuantumCircuit(4)
    gate.compose(circuit.inverse(), [0, 1], inplace=True)
    gate.compose(circuit.inverse(), [2, 3], inplace=True)
    gate.compose(_circuit("CZ3_W.qpy"), [0, 1, 2, 3], inplace=True)
    gate.compose(circuit, [0, 1], inplace=True)
    gate.compose(circuit, [2, 3], inplace=True)
    workload = build_workload("ame43", encoding, gate)
    result = evaluate_workload("ame43", workload, encoding)
    assert result["is_valid"]
    assert result["L"] < 1e-11


def test_ame_graph_has_all_maximally_mixed_two_qutrit_reductions():
    spec = get_reference_experiment("ame43").state
    tensor = np.empty((3, 3, 3, 3), dtype=complex)
    for levels in product(range(3), repeat=4):
        phase = sum(w * levels[a] * levels[b] for a, b, w in spec.weighted_edges)
        tensor[levels] = np.exp(2j * np.pi * phase / 3) / 9
    for pair in combinations(range(4), 2):
        axes = list(pair) + [axis for axis in range(4) if axis not in pair]
        coefficients = tensor.transpose(axes).reshape(9, 9)
        np.testing.assert_allclose(coefficients @ coefficients.conj().T, np.eye(9) / 9, atol=1e-14)


def test_wrong_relative_phase_and_invalid_gates_are_rejected():
    encoding = np.eye(4, 3)
    circuit = build_workload("ame43", encoding, _circuit("CZ3_W.qpy"))
    circuit.u(0, 0, np.pi, 0)
    assert not evaluate_workload("ame43", circuit, encoding)["is_valid"]
    circuit.append(UnitaryGate(np.eye(4)), [0, 1])
    with pytest.raises(ValueError, match="not supported"):
        evaluate_workload("ame43", circuit, encoding)


def test_invalid_encoding_and_wrong_cz_are_rejected():
    with pytest.raises(ValueError, match="isometry"):
        build_workload("ghz3", np.zeros((4, 3)), _circuit("CZ3_W.qpy"))
    with pytest.raises(ValueError, match="CZ3 gate"):
        build_workload("ghz3", np.eye(4, 3), QuantumCircuit(4))


def test_nonfinite_circuit_phase_is_rejected():
    circuit = QuantumCircuit(4, global_phase=np.nan)
    with pytest.raises(ValueError, match="phase must be finite"):
        evaluate_workload("two_qutrit", circuit, np.eye(4, 3))
