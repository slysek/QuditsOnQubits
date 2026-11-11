"""Universal evaluator contracts and regression checks on stored physical circuits."""

import json
from pathlib import Path

import numpy as np
import pytest
from qiskit import AncillaRegister, QuantumCircuit, QuantumRegister, qpy
from qiskit.circuit import Gate, Parameter
from qiskit.circuit.library import CXGate, UnitaryGate
from qiskit.quantum_info import Clifford, Operator

from qudits_on_qubits.benchmarks.direct_basis.math_utils import qutrit_cz, qutrit_fourier
from qudits_on_qubits.benchmarks.direct_basis.optimized_gates import synthesize_f3
from qudits_on_qubits.benchmarks.theta_continuation.encoding import theta_embedding
from qudits_on_qubits.evolution import (
    EvaluationTask, canonical_encoding, controlled_z_d, evaluate, fourier_d, tensor_encoding,
)

REFERENCE = (Path(__file__).resolve().parents[1] / "experiment_inputs" / "iqm_randomized_bell"
             / "canonical_optimized_20260909")


def identity_task(d=3, n=1):
    return EvaluationTask(d, n, np.eye(d**n))


def test_qutrit_constructors_match_benchmark():
    np.testing.assert_array_equal(canonical_encoding(3), np.load(REFERENCE / "E.npy"))
    np.testing.assert_allclose(fourier_d(3), qutrit_fourier(), atol=1e-15)
    np.testing.assert_allclose(controlled_z_d(3), qutrit_cz(), atol=1e-15)


@pytest.mark.parametrize("name,n,target,cz,cx,one_q,depth,two_q_depth", [
    ("F3", 1, fourier_d(3), 0, 2, 6, 5, 2),
    ("CZ3", 2, controlled_z_d(3), 6, 0, 11, 11, 5),
])
def test_saved_reference(name, n, target, cz, cx, one_q, depth, two_q_depth):
    with (REFERENCE / f"{name}_W.qpy").open("rb") as stream:
        circuit, = qpy.load(stream)
    encoding = np.load(REFERENCE / "E.npy")
    result = evaluate(EvaluationTask(3, n, target), circuit, [encoding] * n)
    dimension = 3**n
    # The pinned QPY stores unnormalized Frobenius norms from the old validator.
    error = circuit.metadata["E_norm"] / np.sqrt(dimension)
    assert result.eps == pytest.approx(error, abs=1e-15)
    assert np.sqrt(dimension * result.L) == pytest.approx(circuit.metadata["L_norm"], abs=1e-15)
    assert result.F == pytest.approx((1 - error**2 / 2)**2, abs=2e-15)
    assert (result.cz_count, result.cx_count, result.two_qubit_gate_count) == (cz, cx, cz + cx)
    assert (result.one_qubit_gate_count, result.depth, result.two_qubit_depth) == (
        one_q, depth, two_q_depth,
    )


def test_f3_with_noncanonical_theta_encoding_and_global_phase():
    encoding = theta_embedding(np.pi / 8)
    circuit = synthesize_f3(encoding)
    circuit.global_phase += 1.234
    result = evaluate(EvaluationTask(3, 1, fourier_d(3)), circuit, [encoding])
    assert result.F == pytest.approx(1.0, abs=2e-15)
    assert result.L < 1e-26
    assert result.eps < 1e-13
    assert result.cx_count == 2


def test_relative_phase_is_detected():
    circuit = QuantumCircuit(2, global_phase=0.37)
    circuit.u(0, 0, np.pi, 0)
    result = evaluate(identity_task(), circuit, [canonical_encoding(3)])
    assert result.F == pytest.approx(1 / 9)
    assert result.L == 0
    assert result.eps == pytest.approx(2 / np.sqrt(3))


def test_leakage_is_averaged_only_at_the_end():
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)  # |01> leaves the code. The other two inputs stay in it.
    encoding = [canonical_encoding(3)]
    leaked = evaluate(identity_task(), circuit, encoding)
    assert leaked.F == pytest.approx(4 / 9)
    assert leaked.L == pytest.approx(1 / 3)
    assert leaked.eps == pytest.approx(np.sqrt(2 / 3))

    circuit.cx(0, 1)
    circuit.cz(0, 1)  # Give the complement a minus sign after returning to the code.
    restored = evaluate(identity_task(), circuit, encoding)
    assert (restored.F, restored.L, restored.eps) == (1.0, 0.0, 0.0)
    assert (restored.cx_count, restored.cz_count, restored.two_qubit_gate_count) == (2, 1, 3)


def test_little_endian_with_distinct_encodings_and_asymmetric_target():
    low = canonical_encoding(3)[[1, 0, 3, 2]]  # logical level 2 maps to |11>
    high = theta_embedding(np.pi / 8)
    encoded = tensor_encoding([low, high])
    np.testing.assert_array_equal(encoded[:, 3 + 2], np.kron(high[:, 1], low[:, 2]))
    circuit = QuantumCircuit(4)
    circuit.cz(0, 1)
    target = np.kron(np.eye(3), np.diag([1, 1, -1]))
    result = evaluate(EvaluationTask(3, 2, target), circuit, [low, high])
    assert result.F == pytest.approx(1.0)
    assert result.eps < 1e-14
    assert result.L < 1e-28


def test_complex_encodings_and_cross_qudit_gates_match_dense_operator():
    low = theta_embedding(0.3)
    low[3, 0] *= 1j
    high = canonical_encoding(3) @ fourier_d(3)
    target = np.kron(fourier_d(3), np.eye(3)) @ controlled_z_d(3)
    circuit = QuantumCircuit(4, global_phase=1.3)
    circuit.u(0.7, 0.2, -0.3, 2)
    circuit.cx(3, 0)
    circuit.cz(2, 1)
    result = evaluate(EvaluationTask(3, 2, target), circuit, [low, high])

    code = np.kron(high, low)
    logical = code.conj().T @ Operator(circuit).data @ code
    amplitude = abs(np.trace(target.conj().T @ logical)) / 9
    assert result.F == pytest.approx(amplitude**2)
    assert result.L == pytest.approx(1 - np.linalg.norm(logical, "fro")**2 / 9)
    assert result.eps == pytest.approx(np.sqrt(2 - 2 * amplitude))


def test_d4_has_no_leakage_even_for_an_incorrect_gate():
    circuit = QuantumCircuit(2)
    circuit.cx(1, 0)
    result = evaluate(identity_task(4), circuit, [fourier_d(4)])
    assert result.F == pytest.approx(0.25)
    assert result.eps == pytest.approx(1)
    assert result.L < 1e-28


@pytest.mark.parametrize("d,n,width", [(2, 3, 3), (5, 1, 3), (6, 2, 6)])
def test_general_dimensions(d, n, width):
    result = evaluate(identity_task(d, n), QuantumCircuit(width), [canonical_encoding(d)] * n)
    assert result.num_qubits == width
    assert (result.F, result.L, result.eps) == (1.0, 0.0, 0.0)
    np.testing.assert_allclose(fourier_d(d).conj().T @ fourier_d(d), np.eye(d), atol=1e-14)
    assert controlled_z_d(d)[d + d - 1, d + d - 1] == pytest.approx(np.exp(-2j * np.pi / d))


def test_zero_overlap_and_deterministic_json(tmp_path):
    circuit = QuantumCircuit(1)
    x_gate = np.array([[0, 1], [1, 0]])
    task = EvaluationTask(np.int64(2), 1, x_gate)
    x_gate[:] = 0
    result = evaluate(task, circuit, [canonical_encoding(2)])
    assert (result.F, result.L) == (0.0, 0.0)
    assert result.eps == pytest.approx(np.sqrt(2))
    path = tmp_path / "result.json"
    result.write_json(path)
    text = result.to_json()
    assert path.read_bytes() == text.encode("utf-8")
    assert json.loads(text) == dict(
        F=0.0, L=0.0, eps=result.eps, error_norm=2.0, is_valid=False,
        evaluation_seconds=result.evaluation_seconds, d=2, n=1, num_qubits=1,
        cz_count=0, cx_count=0, two_qubit_gate_count=0, one_qubit_gate_count=0,
        depth=0, two_qubit_depth=0,
    )


def test_result_uses_both_tolerances_and_reports_frobenius_norm():
    circuit = QuantumCircuit(2)
    circuit.cx(0, 1)
    encoding = [canonical_encoding(3)]
    metrics = evaluate(EvaluationTask(3, 1, np.eye(3), 1.0, 1.0), circuit, encoding)
    assert metrics.is_valid
    assert metrics.error_norm == pytest.approx(metrics.eps * np.sqrt(3))
    assert metrics.evaluation_seconds >= 0
    assert not evaluate(EvaluationTask(3, 1, np.eye(3), 0.5, 1.0), circuit, encoding).is_valid
    assert not evaluate(EvaluationTask(3, 1, np.eye(3), 1.0, 0.1), circuit, encoding).is_valid


@pytest.mark.parametrize("value", [-1, np.nan, np.inf, True, "0.1", 10**400])
@pytest.mark.parametrize("name", ["logical_error_tolerance", "leakage_tolerance"])
def test_invalid_tolerance(name, value):
    with pytest.raises(ValueError, match=name):
        EvaluationTask(3, 1, np.eye(3), **{name: value})


def test_task_gate_set_restricts_circuit_and_rejects_unknown_costs():
    circuit = QuantumCircuit(2)
    circuit.cz(0, 1)
    with pytest.raises(ValueError, match="unsupported gate"):
        evaluate(EvaluationTask(3, 1, np.eye(3), allowed_gates=("u", "cx")),
                 circuit, [canonical_encoding(3)])
    for gates in ((), ("unitary",), "cz"):
        with pytest.raises(ValueError, match="allowed_gates"):
            EvaluationTask(3, 1, np.eye(3), allowed_gates=gates)


@pytest.mark.parametrize("d", [1, 2.5, True])
def test_invalid_d(d):
    for constructor in (fourier_d, controlled_z_d, canonical_encoding):
        with pytest.raises(ValueError, match="d must"):
            constructor(d)
    with pytest.raises(ValueError, match="d must"):
        EvaluationTask(d, 1, np.eye(2))


@pytest.mark.parametrize("n", [0, 1.5, True])
def test_invalid_n(n):
    with pytest.raises(ValueError, match="n must"):
        EvaluationTask(2, n, np.eye(2))


@pytest.mark.parametrize("target,reason", [
    (np.eye(9), "shape"), (np.diag([1, 1, 0.5]), "unitary"), (np.full((3, 3), np.nan), "finite"),
])
def test_invalid_target(target, reason):
    with pytest.raises(ValueError, match=reason):
        EvaluationTask(3, 1, target)


@pytest.mark.parametrize("bad,reason", [
    (np.eye(3), "shape"), (np.eye(4, 2), "shape"), (np.zeros((4, 3)), "isometry"),
    (np.full((4, 3), np.inf), "finite"),
])
def test_invalid_local_encoding(bad, reason):
    with pytest.raises(ValueError, match=reason):
        evaluate(identity_task(n=2), QuantumCircuit(4), [canonical_encoding(3), bad])


def test_invalid_register_and_encoding_counts():
    with pytest.raises(ValueError, match="2 physical qubits"):
        evaluate(identity_task(), QuantumCircuit(3), [canonical_encoding(3)])
    with pytest.raises(ValueError, match="1 local encodings"):
        evaluate(identity_task(), QuantumCircuit(2), [])
    with pytest.raises(ValueError, match="at least one"):
        tensor_encoding([])


@pytest.mark.parametrize("gate", [
    UnitaryGate(np.eye(4)), Gate("opaque", 2, []), Gate("cz", 2, []),
    Clifford(QuantumCircuit(2)), CXGate(ctrl_state=0),
])
def test_reject_opaque_unknown_and_open_control_gates(gate):
    circuit = QuantumCircuit(2)
    circuit.append(gate, [0, 1])
    with pytest.raises(ValueError, match="unsupported gate"):
        evaluate(identity_task(), circuit, [canonical_encoding(3)])


def test_reject_measurement_reset_and_explicit_ancillas():
    measured = QuantumCircuit(2, 1)
    measured.measure(0, 0)
    reset = QuantumCircuit(2)
    reset.reset(1)
    with_ancilla = QuantumCircuit(QuantumRegister(1), AncillaRegister(1))
    for circuit, reason in [
        (measured, "classical bits"), (reset, "reset"), (with_ancilla, "ancillas"),
    ]:
        with pytest.raises(ValueError, match=reason):
            evaluate(identity_task(), circuit, [canonical_encoding(3)])


@pytest.mark.parametrize("angle", [np.nan, np.inf, Parameter("a")])
def test_reject_invalid_gate_parameters(angle):
    circuit = QuantumCircuit(2)
    circuit.u(angle, 0, 0, 0)
    with pytest.raises(ValueError, match="parameters"):
        evaluate(identity_task(), circuit, [canonical_encoding(3)])


def test_reject_nonfinite_global_phase():
    with pytest.raises(ValueError, match="global phase"):
        evaluate(identity_task(), QuantumCircuit(2, global_phase=np.nan), [canonical_encoding(3)])
