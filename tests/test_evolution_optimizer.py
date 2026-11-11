"""Check the fixed-topology CZ3 fit."""

import json
from pathlib import Path

import numpy as np
import pytest
from qiskit import QuantumCircuit, qpy
from qiskit.quantum_info import Operator

from qudits_on_qubits.evolution.encodings import canonical_encoding
from qudits_on_qubits.evolution.optimizer import fit_fixed_cz3, parameterize_cz3_circuit
from qudits_on_qubits.benchmarks.theta_continuation.encoding import theta_embedding


REFERENCE = (Path(__file__).resolve().parents[1] / "experiment_inputs" / "iqm_randomized_bell"
             / "canonical_optimized_20260909" / "CZ3_W.qpy")


def reference_circuit():
    with REFERENCE.open("rb") as stream:
        circuit, = qpy.load(stream)
    return circuit


def test_parameterization_rebuilds_the_saved_circuit():
    circuit = reference_circuit()
    template = parameterize_cz3_circuit(circuit)
    rebuilt = template.bind(template.initial_parameters)
    np.testing.assert_allclose(Operator(rebuilt).data, Operator(circuit).data, atol=1e-14)
    assert rebuilt.count_ops() == circuit.count_ops()


def test_fit_preserves_valid_source_and_reports_independent_metrics():
    circuit = reference_circuit()
    template = parameterize_cz3_circuit(circuit)
    encoding = canonical_encoding(3)
    result = fit_fixed_cz3(circuit, [encoding, encoding], template.initial_parameters,
                           max_nfev=1, tolerance=1e-5)
    assert result.evaluation.is_valid
    assert result.evaluation.cz_count == 6
    assert result.evaluation.error_norm < 1e-5
    assert result.nfev <= 1
    assert result.evaluations >= result.nfev
    assert result.elapsed_seconds >= 0
    json.dumps(result.to_dict(), allow_nan=False)


def test_fit_accepts_different_encoding_on_low_qudit():
    source = reference_circuit()
    circuit = QuantumCircuit(4)
    circuit.u(np.pi / 2, 0, np.pi, 0)
    circuit.compose(source, inplace=True)
    circuit.u(np.pi / 2, 0, np.pi, 0)
    low_unitary = np.kron(np.eye(2), np.array([[1, 1], [1, -1]]) / np.sqrt(2))
    low = low_unitary @ canonical_encoding(3)
    high = canonical_encoding(3)
    template = parameterize_cz3_circuit(circuit)
    result = fit_fixed_cz3(circuit, [low, high], template.initial_parameters,
                           max_nfev=1, tolerance=1e-5)
    reversed_result = fit_fixed_cz3(circuit, [high, low], template.initial_parameters,
                                    max_nfev=1, tolerance=1e-5)
    assert result.evaluation.is_valid
    assert not reversed_result.evaluation.is_valid
    assert result.evaluation.cz_count == 6


def test_fit_theta_keeps_u3_gates_unitary():
    circuit = reference_circuit()
    template = parameterize_cz3_circuit(circuit)
    encoding = theta_embedding(np.pi / 8)
    result = fit_fixed_cz3(circuit, [encoding, encoding], template.initial_parameters,
                           max_nfev=200, tolerance=1e-10)
    assert np.max(np.abs(result.parameters - template.initial_parameters)) <= 2 * np.pi + 1e-12
    assert np.isfinite(result.evaluation.eps)
    assert result.nfev <= 200


def test_fit_preserves_five_cz_topology_and_global_phase():
    circuit = reference_circuit()
    first_cz = next(index for index, item in enumerate(circuit.data) if item.operation.name == "cz")
    del circuit.data[first_cz]
    circuit.global_phase += 0.7
    template = parameterize_cz3_circuit(circuit)
    encoding = canonical_encoding(3)
    result = fit_fixed_cz3(circuit, [encoding, encoding], template.initial_parameters,
                           max_nfev=2, tolerance=1e-10)
    assert result.circuit.global_phase == pytest.approx(circuit.global_phase)
    original = [(item.operation.name, tuple(circuit.find_bit(q).index for q in item.qubits))
                for item in circuit.data]
    fitted = [(item.operation.name, tuple(result.circuit.find_bit(q).index for q in item.qubits))
              for item in result.circuit.data]
    assert fitted == original
    assert result.evaluation.cz_count == 5
    assert result.to_dict()["solver_config"]["angle_bound_radius"] == pytest.approx(2 * np.pi)


def test_fit_rejects_invalid_inputs():
    source = reference_circuit()
    template = parameterize_cz3_circuit(source)
    encoding = canonical_encoding(3)
    with pytest.raises(ValueError, match="two encodings"):
        fit_fixed_cz3(source, [encoding], template.initial_parameters)
    with pytest.raises(ValueError, match="isometry"):
        fit_fixed_cz3(source, [encoding, np.zeros((4, 3))], template.initial_parameters)
    source.cx(0, 1)
    with pytest.raises(ValueError, match="Unsupported"):
        parameterize_cz3_circuit(source)
