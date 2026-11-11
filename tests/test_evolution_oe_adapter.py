"""Check the OpenEvolve candidate contract and scoring."""

from pathlib import Path

import numpy as np
import pytest
from qiskit import QuantumCircuit

from qudits_on_qubits.evolution.oe_candidates import (
    MAX_ANGLE, MAX_AST_NODES, MAX_BYTES, MAX_OPERATIONS, candidate_from_circuit, load_candidate,
)
from qudits_on_qubits.evolution.oe_evaluator import assess_candidate, score_metrics
from qudits_on_qubits.evolution.encodings import canonical_encoding
from qudits_on_qubits.evolution.state_workloads import evaluate_workload


def write_candidate(tmp_path, body):
    path = tmp_path / "candidate.py"
    path.write_text(body, encoding="utf-8")
    return path


def test_circuit_round_trip(tmp_path):
    circuit = QuantumCircuit(6, global_phase=0.71)
    circuit.u(0.3, -0.2, 1.7, 0)
    circuit.cz(0, 5)
    source = candidate_from_circuit(circuit)
    path = write_candidate(tmp_path, "CANDIDATE = " + repr(source))
    actual = load_candidate(path, expected_qubits=6)
    from qiskit.quantum_info import Operator
    np.testing.assert_allclose(Operator(actual).data, Operator(circuit).data, atol=1e-14)


@pytest.mark.parametrize("source", [
    "import os\nCANDIDATE = {'num_qubits': 6, 'operations': []}",
    "CANDIDATE = __import__('os').system('false')",
    "CANDIDATE = {'num_qubits': 6, 'operations': []}\nopen('owned','w')",
    "CANDIDATE = {'num_qubits': 6, 'operations': [['cx', 0, 1]]}",
    "CANDIDATE = {'num_qubits': 6, 'operations': [['cz', 0, 0]]}",
    "CANDIDATE = {'num_qubits': 6, 'operations': [['cz', 0, 6]]}",
    "CANDIDATE = {'num_qubits': 6, 'operations': [['u', True, 0, 0, 0]]}",
    "CANDIDATE = {'num_qubits': 6, 'operations': [['u', 0, 1e999, 0, 0]]}",
    "CANDIDATE = {'num_qubits': 6, 'operations': [], 'secret': 1}",
    "CANDIDATE = {'num_qubits': 8, 'operations': []}",
])
def test_reject_untrusted_programs(tmp_path, source):
    with pytest.raises(ValueError):
        load_candidate(write_candidate(tmp_path, source), expected_qubits=6)
    assert not (tmp_path / "owned").exists()


def test_reject_too_many_operations(tmp_path):
    data = {"num_qubits": 6, "operations": [["u", 0, 0., 0., 0.]] * (MAX_OPERATIONS + 1)}
    with pytest.raises(ValueError):
        load_candidate(write_candidate(tmp_path, "CANDIDATE = " + repr(data)), expected_qubits=6)


@pytest.mark.parametrize("source", [
    "CANDIDATE = {'num_qubits': 6, 'num_qubits': 6, 'operations': []}",
    "CANDIDATE = {**{'num_qubits': 6}, 'operations': []}",
    "CANDIDATE = {'num_qubits': True, 'operations': []}",
    "CANDIDATE = {'num_qubits': 6.0, 'operations': []}",
    "CANDIDATE = {'num_qubits': 6}",
    "CANDIDATE = {'num_qubits': 6, 'operations': ()}",
    "CANDIDATE = {'num_qubits': 6, 'operations': [()]}",
    "CANDIDATE = {'num_qubits': 6, 'operations': [['cz', -1, 0]]}",
    "CANDIDATE = {'num_qubits': 6, 'operations': [['cz', 1.0, 0]]}",
    "CANDIDATE = {'num_qubits': 6, 'operations': [['cz', 0]]}",
    "CANDIDATE = {'num_qubits': 6, 'operations': [['u', 0, 0, 0]]}",
    "CANDIDATE = {'num_qubits': 6, 'operations': [['u', 0, True, 0, 0]]}",
    "CANDIDATE = {'num_qubits': 6, 'operations': [['u', 0, -1e999, 0, 0]]}",
    "CANDIDATE = {'num_qubits': 6, 'operations': [['u', 0, -1000001, 0, 0]]}",
    "CANDIDATE = {'num_qubits': 6, 'operations': [], 'global_phase': 1000001}",
    "CANDIDATE = {'num_qubits': 6, 'operations': []",
])
def test_reject_malformed_literal_candidates(tmp_path, source):
    with pytest.raises(ValueError):
        load_candidate(write_candidate(tmp_path, source), expected_qubits=6)


@pytest.mark.parametrize("width", [True, 6.0, '6', None, 0, 7, 9])
def test_reject_invalid_expected_width(tmp_path, width):
    path = write_candidate(tmp_path, "CANDIDATE = {'num_qubits': 6, 'operations': []}")
    with pytest.raises(ValueError, match="width"):
        load_candidate(path, expected_qubits=width)


def test_literal_limits_accept_boundaries(tmp_path):
    data = {"num_qubits": 6, "operations": [["u", 5, MAX_ANGLE, -MAX_ANGLE, 0.0]] * MAX_OPERATIONS}
    path = write_candidate(tmp_path, "CANDIDATE = " + repr(data))
    assert len(load_candidate(path, expected_qubits=6).data) == MAX_OPERATIONS
    source = "CANDIDATE = {'num_qubits': 6, 'operations': []}\n#"
    path.write_bytes(source.encode() + b'x' * (MAX_BYTES - len(source)))
    assert len(load_candidate(path, expected_qubits=6).data) == 0
    path.write_bytes(path.read_bytes() + b'x')
    with pytest.raises(ValueError, match="large"):
        load_candidate(path, expected_qubits=6)


def test_invalid_encoding_and_embedded_code_are_rejected(tmp_path):
    path = tmp_path / "candidate.py"
    path.write_bytes(b"\xff")
    with pytest.raises(ValueError):
        load_candidate(path, expected_qubits=6)
    target = tmp_path / "executed"
    path.write_text(f"CANDIDATE = {{'num_qubits': 6, 'operations': [open({str(target)!r}, 'w')]}}")
    with pytest.raises(ValueError):
        load_candidate(path, expected_qubits=6)
    assert not target.exists()


def test_ast_node_limit_is_checked_before_literal_validation(tmp_path):
    source = "CANDIDATE = {'num_qubits': 6, 'operations': [], 'extra': [" + ",".join(["0"] * MAX_AST_NODES) + "]}"
    with pytest.raises(ValueError, match="too large or too deep"):
        load_candidate(write_candidate(tmp_path, source), expected_qubits=6)


def test_evaluator_uses_actual_state_metrics(tmp_path):
    source = {"num_qubits": 6, "operations": []}
    path = write_candidate(tmp_path, "CANDIDATE = " + repr(source))
    circuit, metrics = assess_candidate(path, state="ghz3", encoding=canonical_encoding(3))
    expected = evaluate_workload("ghz3", circuit, canonical_encoding(3), tolerance=1e-10)
    assert not metrics["is_valid"]
    assert metrics["F"] == pytest.approx(expected["F"])
    assert metrics["L"] == pytest.approx(expected["L"])
    assert metrics["combined_score"] < 1


def test_score_is_correctness_first_and_lexicographic():
    def row(cz, two_depth, depth, one, valid=True, decoder=0):
        return {"is_valid": valid, "cz_count": cz, "two_qubit_depth": two_depth,
                "depth": depth, "one_qubit_gate_count": one, "F": 1.0, "decoder_cz_count": decoder}
    assert score_metrics(row(MAX_OPERATIONS, MAX_OPERATIONS, MAX_OPERATIONS, MAX_OPERATIONS)) > score_metrics(row(0, 0, 0, 0, False))
    assert score_metrics(row(4, 512, 512, 512)) > score_metrics(row(5, 0, 0, 0))
    assert score_metrics(row(5, 2, 512, 512)) > score_metrics(row(5, 3, 0, 0))
    assert score_metrics(row(5, 3, 5, 512)) > score_metrics(row(5, 3, 6, 0))
    assert score_metrics(row(5, 3, 6, 3)) > score_metrics(row(5, 3, 6, 4))
    # The decoder CZ count only breaks ties of the preparation costs.
    assert score_metrics(row(5, 3, 6, 3, decoder=4)) > score_metrics(row(5, 3, 6, 3, decoder=6))
    assert score_metrics(row(5, 3, 6, 3, decoder=12)) > score_metrics(row(5, 3, 6, 4, decoder=0))


def test_external_configuration_is_required(tmp_path, monkeypatch):
    from qudits_on_qubits.evolution.oe_evaluator import evaluate
    monkeypatch.delenv("QOQ_OE_CONFIG", raising=False)
    with pytest.raises(ValueError, match="QOQ_OE_CONFIG"):
        evaluate(write_candidate(tmp_path, "CANDIDATE = {}"))


@pytest.mark.parametrize("state", ["ghz3", "ame43"])
def test_valid_graph_state_with_and_without_fit(tmp_path, state):
    from qiskit import qpy, transpile
    from qudits_on_qubits.evolution.state_workloads import build_workload
    legacy = Path(__file__).resolve().parents[1] / "src/qudits_on_qubits/quantum_circuits/CZgate3.qpy"
    with legacy.open("rb") as stream:
        gate, = qpy.load(stream)
    gate = transpile(gate, basis_gates=["u", "cz"], optimization_level=0, seed_transpiler=0)
    circuit = build_workload(state, canonical_encoding(3), gate)
    data = candidate_from_circuit(circuit)
    path = write_candidate(tmp_path, "CANDIDATE = " + repr(data))
    _, direct = assess_candidate(path, state=state, encoding=canonical_encoding(3))
    fitted, optimized = assess_candidate(path, state=state, encoding=canonical_encoding(3),
                                         max_nfev=1, max_parameters=3)
    assert direct["is_valid"] and optimized["is_valid"]
    assert direct["combined_score"] >= 1 and optimized["combined_score"] >= 1
    assert optimized["nfev"] <= 1
    assert evaluate_workload(state, fitted, canonical_encoding(3), tolerance=1e-10)["is_valid"]


def test_subprocess_timeout_is_enforced(tmp_path):
    from time import monotonic
    path = write_candidate(tmp_path, "CANDIDATE = {'num_qubits': 6, 'operations': []}")
    started = monotonic()
    with pytest.raises(TimeoutError):
        assess_candidate(path, state="ghz3", encoding=canonical_encoding(3),
                         timeout_seconds=0.001, cpu_seconds=1)
    assert monotonic() - started < 3


def test_numeric_rejection_and_no_code_execution(tmp_path, monkeypatch):
    import json
    from qudits_on_qubits.evolution.oe_evaluator import evaluate
    target = tmp_path / "owned"
    path = write_candidate(tmp_path, f"CANDIDATE = open({str(target)!r}, 'w')")
    encoding = tmp_path / "E.npy"
    np.save(encoding, canonical_encoding(3), allow_pickle=False)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"state": "ghz3", "encoding_path": str(encoding)}))
    monkeypatch.setenv("QOQ_OE_CONFIG", str(config))
    evaluation = evaluate(path)
    assert evaluation.artifacts["evaluation"].startswith("REJECTED")
    result = evaluation.metrics
    assert result["evaluation_failed"] == 1
    assert result["combined_score"] == 0
    assert result["cz_count"] == MAX_OPERATIONS
    assert all(isinstance(value, float) and np.isfinite(value) for value in result.values())
    assert not target.exists()


def test_reject_size_and_nesting_before_evaluation(tmp_path):
    from qudits_on_qubits.evolution.oe_candidates import MAX_BYTES
    with pytest.raises(ValueError, match="large"):
        load_candidate(write_candidate(tmp_path, "#" * (MAX_BYTES + 1)), expected_qubits=6)
    source = "CANDIDATE = {'num_qubits': 6, 'operations': " + "[" * 24 + "]" * 24 + "}"
    with pytest.raises(ValueError, match="deep"):
        load_candidate(write_candidate(tmp_path, source), expected_qubits=6)
