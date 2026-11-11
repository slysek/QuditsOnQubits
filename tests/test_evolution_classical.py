"""Check the bounded CZ3 runner and its saved records."""
import json
from pathlib import Path

import numpy as np
import pytest

from qudits_on_qubits.benchmarks.theta_continuation.encoding import theta_embedding
from qudits_on_qubits.evolution import classical
from qudits_on_qubits.evolution.baseline import HASHES, INPUT


def make_baseline(tmp_path):
    source = tmp_path / "baseline"
    source.mkdir()
    names = ["CZ3_W.qpy", "E.npy"]
    for name in names:
        (source / name).write_bytes((INPUT / name).read_bytes())
    (source / "baseline_summary.json").write_text(json.dumps({"results": [{
        "gate": "CZ3", "is_valid": True, "method": "saved_reference", "encoding": "canonical",
        "circuit_artifact": "CZ3_W.qpy", "encoding_artifacts": json.dumps(["E.npy", "E.npy"]),
    }]}))
    files = {name: HASHES[name] for name in names}
    files["baseline_summary.json"] = classical._hash((source / "baseline_summary.json").read_bytes())
    (source / "manifest.json").write_text(json.dumps({"files": files}))
    return source


def success(case, parameters, max_nfev, timeout):
    return {"status": "completed", "solver": {"nfev": 1, "evaluations": 3},
            "parameters": parameters, "wall_seconds": 0.0}


def test_resume_reuses_completed_records_and_checks_artifacts(tmp_path, monkeypatch):
    source = make_baseline(tmp_path)
    output = tmp_path / "run"
    calls = []
    def attempt(*args):
        calls.append(1)
        return success(*args)
    monkeypatch.setattr(classical, "_attempt", attempt)
    results = classical.run_classical(source, output, seeds=(7,), max_nfev=1, timeout=10)
    assert len(calls) == len(results) == 2
    expected_settings = {"method": "scipy_least_squares_trf", "angle_bound_radius": 2 * np.pi,
                         "phase_bound_radius": np.pi}
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["configuration"]["solver_config"] == expected_settings
    assert results[0]["solver_config"] == expected_settings
    report = json.loads((output / "report_0001.json").read_text())
    assert report["solver_config"] == expected_settings
    readme = (output / "README.md").read_bytes()
    assert b"--baseline" in readme and b"--resume" in readme
    assert results[0]["metrics"]["is_valid"]
    assert results[0]["artifact_hashes"]
    assert not results[1]["metrics"]["is_valid"]
    again = classical.run_classical(source, output, seeds=(7,), max_nfev=1, timeout=10, resume=True)
    assert len(calls) == 2
    assert results == again
    assert (output / "README.md").read_bytes() == readme
    with pytest.raises(ValueError, match="configuration"):
        classical.run_classical(source, output, seeds=(8,), max_nfev=1, timeout=10, resume=True)
    with pytest.raises(FileExistsError):
        classical.run_classical(source, output, seeds=(7,), max_nfev=1, timeout=10)
    saved = output / "case_001/original/attempt_0001/circuit.qpy"
    saved.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        classical.run_classical(source, output, seeds=(7,), max_nfev=1, timeout=10, resume=True)


def test_timeout_is_kept_and_retried(tmp_path, monkeypatch):
    source = make_baseline(tmp_path)
    output = tmp_path / "run"
    monkeypatch.setattr(classical, "_attempt", lambda *args: {"status": "timeout", "wall_seconds": 1.0})
    classical.run_classical(source, output, seeds=(), max_nfev=1, timeout=1)
    monkeypatch.setattr(classical, "_attempt", success)
    result = classical.run_classical(source, output, seeds=(), max_nfev=1, timeout=1, resume=True)
    assert result[0]["status"] == "completed"
    assert json.loads((output / "case_001/original/attempt_0001/result.json").read_text())["status"] == "timeout"
    assert (output / "case_001/original/attempt_0002/result.json").exists()


def test_interrupted_attempt_is_kept_and_retried(tmp_path, monkeypatch):
    source = make_baseline(tmp_path)
    output = tmp_path / "run"
    monkeypatch.setattr(classical, "_attempt", success)
    classical.run_classical(source, output, seeds=(), max_nfev=1, timeout=1)
    attempt = output / "case_001/original/attempt_0001"
    (attempt / "commit.json").unlink()
    result = classical.run_classical(source, output, seeds=(), max_nfev=1, timeout=1, resume=True)
    assert result[0]["status"] == "completed"
    assert (attempt / "result.json").exists()
    assert (attempt.parent / "attempt_0002/result.json").exists()


def test_result_hash_is_checked(tmp_path, monkeypatch):
    source = make_baseline(tmp_path)
    output = tmp_path / "run"
    monkeypatch.setattr(classical, "_attempt", success)
    classical.run_classical(source, output, seeds=(), max_nfev=1, timeout=1)
    (output / "case_001/original/attempt_0001/result.json").write_text("{}")
    with pytest.raises(ValueError, match="cached result"):
        classical.run_classical(source, output, seeds=(), max_nfev=1, timeout=1, resume=True)


def test_best_results_use_separate_tolerances():
    def result(count, strict, encoding):
        return {"status": "completed", "encoding": encoding, "case": f"case_{count}",
                "source_sha256": ["circuit", encoding, encoding],
                "trial_path": f"case_{count}/attempt_0001", "strict_valid": strict,
                "metrics": {"is_valid": True, "cz_count": count, "cx_count": 0,
                            "two_qubit_depth": 2, "depth": 5, "one_qubit_gate_count": 6,
                            "error_norm": 1e-6 if not strict else 1e-14}}
    records = [result(5, False, "a"), result(6, True, "a"), result(8, True, "b")]
    assert [row["case"] for row in classical._best(records)] == ["case_5", "case_8"]
    assert [row["case"] for row in classical._best(records, strict=True)] == ["case_6", "case_8"]


def test_resume_completes_interrupted_initialization(tmp_path, monkeypatch):
    source = make_baseline(tmp_path)
    output = tmp_path / "run"
    monkeypatch.setattr(classical, "_attempt", success)
    classical.run_classical(source, output, seeds=(), max_nfev=1, timeout=1)
    (output / "manifest.json").unlink()
    saved = output / "inputs/case_001/E_A.npy"
    before = saved.read_bytes()
    missing = output / "inputs/case_001/E_B.npy"
    missing.unlink()
    records = classical.run_classical(source, output, seeds=(), max_nfev=1, timeout=1, resume=True)
    assert records[0]["metrics"]["is_valid"]
    assert before == saved.read_bytes() == missing.read_bytes()
    assert (output / "manifest.json").exists()
    report = json.loads((output / "report_0002.json").read_text())
    best = report["best"][0]
    assert (output / best["circuit_artifact"]).exists()


def test_override_encoding_is_recorded_even_when_initial_circuit_fails(tmp_path, monkeypatch):
    source = make_baseline(tmp_path)
    output = tmp_path / "run"
    encoding = tmp_path / "theta.npy"
    np.save(encoding, theta_embedding(np.pi / 8))
    monkeypatch.setattr(classical, "_attempt", success)
    records = classical.run_classical(source, output, seeds=(), max_nfev=1, timeout=1,
                                      encoding_paths=(encoding, encoding))
    assert not records[0]["metrics"]["is_valid"]
    np.testing.assert_array_equal(np.load(output / "inputs/case_001/E_A.npy"), np.load(encoding))
    assert not (output / "case_001/original/attempt_0001/circuit.qpy").exists()


def test_timeout_stops_the_process(tmp_path):
    cases = classical._read_baseline(make_baseline(tmp_path))
    result = classical._attempt(cases[0], cases[0]["angles"], 1, 0.001)
    assert result["status"] == "timeout"


def test_real_fit_with_saved_angles(tmp_path):
    output = tmp_path / "run"
    records = classical.run_classical(make_baseline(tmp_path), output, seeds=(), max_nfev=1, timeout=30)
    assert records[0]["status"] == "completed", records[0]
    assert records[0]["metrics"]["is_valid"]
    assert records[0]["solver"]["nfev"] >= 1
    assert records[0]["solver"]["evaluations"] >= records[0]["solver"]["nfev"]
    assert (output / "case_001/original/attempt_0001/circuit.qpy").exists()
