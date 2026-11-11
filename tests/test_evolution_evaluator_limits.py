"""Check evaluator limits and process failures without network access."""

import base64
from io import BytesIO
import json
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
from qiskit import QuantumCircuit, qpy

from qudits_on_qubits.evolution.encodings import canonical_encoding
from qudits_on_qubits.evolution import oe_evaluator


@pytest.mark.parametrize("limits", [
    (0, 2048, 30), (600.001, 2048, 30), (float("nan"), 2048, 30),
    (True, 2048, 30), (1, 255, 30), (1, 4097, 30),
    (1, 256.0, 30), (1, True, 30), (1, 2048, 0),
    (1, 2048, 601), (1, 2048, 1.0), (1, 2048, True),
])
def test_reject_invalid_limits(limits):
    with pytest.raises(ValueError):
        oe_evaluator._limits(*limits)


@pytest.mark.parametrize("limits", [(0.001, 256, 1), (600, 4096, 600)])
def test_accept_limit_boundaries(limits):
    oe_evaluator._limits(*limits)


def _circuit_bytes():
    stream = BytesIO()
    qpy.dump(QuantumCircuit(6), stream)
    return base64.b64encode(stream.getvalue()).decode("ascii")


@pytest.mark.parametrize("response", [
    b"not json", b"[]", b"null", b'{}', b'{"ok": true}',
    b'{"ok": true, "circuit": "!!"}',
    json.dumps({"ok": True, "circuit": base64.b64encode(b"bad").decode()}).encode(),
    json.dumps({"ok": True, "circuit": _circuit_bytes()}).encode(),
    json.dumps({"ok": True, "circuit": _circuit_bytes(), "metrics": []}).encode(),
    json.dumps({"ok": True, "circuit": _circuit_bytes(), "metrics": {"nfev": -1, "evaluations": 0, "jacobian_evaluations": 0}}).encode(),
])
def test_reject_malformed_worker_response(tmp_path, monkeypatch, response):
    monkeypatch.setattr(oe_evaluator.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout=response))
    with pytest.raises(ValueError):
        oe_evaluator.assess_candidate(tmp_path / "candidate.py", state="ghz3", encoding=canonical_encoding(3))


@pytest.mark.parametrize("failure", ["timeout", "exit", "response"])
def test_process_failure_returns_numeric_failure(tmp_path, monkeypatch, failure):
    encoding_path = tmp_path / "encoding.npy"
    np.save(encoding_path, canonical_encoding(3), allow_pickle=False)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"state": "ghz3", "encoding_path": str(encoding_path), "timeout_seconds": 0.01}))
    monkeypatch.setenv("QOQ_OE_CONFIG", str(config_path))
    def run(*args, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])
        return SimpleNamespace(returncode=int(failure == "exit"), stdout=b"[]")
    monkeypatch.setattr(oe_evaluator.subprocess, "run", run)
    result = oe_evaluator.evaluate(tmp_path / "candidate.py")
    assert result.artifacts["evaluation"].startswith("REJECTED")
    metrics = result.metrics
    assert metrics["evaluation_failed"] == 1
    assert metrics["timed_out"] == int(failure == "timeout")
    assert metrics["combined_score"] == 0
    assert all(np.isfinite(value) for value in metrics.values())


def test_repeated_skeleton_is_served_from_the_cache(tmp_path, monkeypatch):
    from qudits_on_qubits.evolution.oe_candidates import skeleton_program
    encoding_path = tmp_path / "encoding.npy"
    np.save(encoding_path, canonical_encoding(3), allow_pickle=False)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"state": "ghz3", "encoding_path": str(encoding_path),
                                       "representation": "skeleton", "encoding_mode": "free"}))
    monkeypatch.setenv("QOQ_OE_CONFIG", str(config_path))
    metrics = {"is_valid": False, "F": 0.25, "cz_count": 5, "two_qubit_depth": 3, "depth": 7,
               "one_qubit_gate_count": 16, "error_norm": 0.9, "leakage_norm": 0.1, "fit_restarts": 8,
               "decoder_cz_count": 0, "evaluation_seconds": 12.0, "combined_score": 0.125}
    calls = []
    monkeypatch.setattr(oe_evaluator, "assess", lambda path, **_: calls.append(path) or SimpleNamespace(metrics=metrics))
    pairs = [[0, 1], [0, 2], [1, 3], [0, 4], [1, 5]]
    first, second, other = (tmp_path / name for name in ("a.py", "b.py", "c.py"))
    first.write_text(skeleton_program(6, pairs))
    second.write_text("# same structure, other layout\nCANDIDATE = " + repr({"num_qubits": 6, "cz": pairs}) + "\n")
    other.write_text(skeleton_program(6, pairs[::-1]))

    fresh = oe_evaluator.evaluate(first)
    repeated = oe_evaluator.evaluate(second)
    changed = oe_evaluator.evaluate(other)

    assert calls == [first, other]
    assert fresh.metrics["cache_hit"] == 0 and changed.metrics["cache_hit"] == 0
    assert repeated.metrics["cache_hit"] == 1 and repeated.metrics["evaluation_seconds"] == 0
    assert repeated.metrics["F"] == fresh.metrics["F"] == 0.25
    assert repeated.artifacts["evaluation"].startswith("DUPLICATE")
    assert "fidelity F=0.2500" in fresh.artifacts["evaluation"] and "Add CZ gates" in fresh.artifacts["evaluation"]
    assert not list((tmp_path / oe_evaluator.CACHE_DIRECTORY).glob("*.tmp"))


def test_failed_evaluations_are_not_cached(tmp_path, monkeypatch):
    encoding_path = tmp_path / "encoding.npy"
    np.save(encoding_path, canonical_encoding(3), allow_pickle=False)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"state": "ghz3", "encoding_path": str(encoding_path)}))
    monkeypatch.setenv("QOQ_OE_CONFIG", str(config_path))
    program = tmp_path / "candidate.py"
    program.write_text("CANDIDATE = {'num_qubits': 6, 'operations': []}\n")
    def fail(*_, **__):
        raise TimeoutError("slow")
    monkeypatch.setattr(oe_evaluator, "assess", fail)
    for _ in range(2):
        assert oe_evaluator.evaluate(program).metrics["timed_out"] == 1
    assert not (tmp_path / oe_evaluator.CACHE_DIRECTORY).exists()
