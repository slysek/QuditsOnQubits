import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler import CouplingMap

from qudits_on_qubits.benchmarks.validation import state_metrics, circuit_metrics


def test_layout_is_respected_without_simulating_full_backend():
    source = QuantumCircuit(2)
    source.x(0)
    compiled = transpile(source, basis_gates=["u", "cx"], coupling_map=CouplingMap.from_line(40),
                         initial_layout=[38, 39], optimization_level=1, seed_transpiler=0)
    metrics = state_metrics(compiled, np.array([0, 1, 0, 0]), np.eye(4, 3), 1, max_qubits=4)
    assert metrics["fidelity"] == pytest.approx(1)
    assert metrics["leakage"] == pytest.approx(0)
    assert metrics["validation_qubits"] == 2
    assert metrics["output_mapping"] == [38, 39]


def test_routed_ancillas_and_native_entanglers():
    source = QuantumCircuit(2)
    source.h(0)
    source.cx(0, 1)
    encoding = np.eye(4)[:, [0, 3, 1]]
    expected = np.array([1, 0, 0, 1]) / np.sqrt(2)
    compiled = transpile(source, basis_gates=["u", "cx"], coupling_map=CouplingMap.from_line(4),
                         initial_layout=[0, 3], optimization_level=0, seed_transpiler=3)
    metrics = state_metrics(compiled, expected, encoding, 1, max_qubits=4)
    assert metrics["fidelity"] == pytest.approx(1)
    assert metrics["leakage"] == pytest.approx(0, abs=1e-12)
    assert circuit_metrics(compiled)["two_qubit_gate_count"] > 1


def test_leakage_and_invalid_nonunitary_circuits():
    source = QuantumCircuit(2)
    source.x(0)
    source.x(1)
    metrics = state_metrics(source, np.array([1, 0, 0, 0]), np.eye(4, 3), 1)
    assert metrics["fidelity"] == 0
    assert metrics["leakage"] == 1
    source.reset(0)
    with pytest.raises(ValueError, match="unitary"):
        state_metrics(source, np.array([1, 0, 0, 0]), np.eye(4, 3), 1)
