"""Check saved circuits and the baseline report."""

import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
from qiskit import QuantumCircuit, qpy

from qudits_on_qubits.evolution import EvaluationTask, evaluate
from qudits_on_qubits.evolution.artifacts import file_hash, load_circuit, load_encoding
from qudits_on_qubits.evolution.baseline import HASHES, INPUT, run_baseline


def test_baseline_outputs_and_refuses_overwrite(tmp_path):
    output = tmp_path / "run"
    rows = run_baseline(output)
    assert [(r["gate"], r["method"], r["cz_count"], r["cx_count"]) for r in rows] == [
        ("F3", "saved_reference", 0, 2), ("CZ3", "saved_reference", 6, 0),
        ("F3", "legacy_transpile", 0, 6), ("CZ3", "legacy_transpile", 8, 0),
    ]
    assert all(row["is_valid"] for row in rows)
    assert [r["strict_valid"] for r in rows] == [True, False, True, True]
    assert all(abs(r["error_frobenius"] - r["previous_error_frobenius"]) < 1e-13 for r in rows)
    with (output / "baseline_results.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == 4
    summary = json.loads((output / "baseline_summary.json").read_text())
    assert len(summary["results"]) == 4
    manifest = json.loads((output / "manifest.json").read_text())
    assert "scipy" in manifest["version"]
    assert "src/qudits_on_qubits/benchmarks/validation.py" in manifest["code_sha256"]
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((output / name).read_bytes()).hexdigest() == expected
    for row in rows:
        with (output / row["circuit_artifact"]).open("rb") as stream:
            circuit, = qpy.load(stream)
        assert circuit.num_qubits == row["num_qubits"]
        assert circuit.count_ops().get("cz", 0) == row["cz_count"]
        assert circuit.count_ops().get("cx", 0) == row["cx_count"]
        target = np.load(output / row["target_artifact"], allow_pickle=False)
        encodings = [np.load(output / name, allow_pickle=False)
                     for name in json.loads(row["encoding_artifacts"])]
        task = EvaluationTask(3, row["n"], target,
                              logical_error_tolerance=row["logical_error_tolerance"],
                              leakage_tolerance=row["leakage_tolerance"])
        result = evaluate(task, circuit, encodings)
        assert result.is_valid
        assert result.F == pytest.approx(row["F"], abs=1e-15)
        assert result.L == pytest.approx(row["L"], abs=1e-25)
        assert result.error_norm == pytest.approx(row["error_frobenius"], abs=1e-13)
        assert hashlib.sha256((output / row["circuit_artifact"]).read_bytes()).hexdigest() == row["artifact_sha256"]
    before = (output / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_baseline(output)
    assert (output / "manifest.json").read_bytes() == before


def test_loader_rejects_hash_mismatch_before_output(tmp_path):
    copied = tmp_path / "inputs"
    shutil.copytree(INPUT, copied)
    (copied / "F3_W.qpy").write_bytes(b"invalid")
    with pytest.raises(ValueError, match="SHA-256"):
        run_baseline(tmp_path / "run", input_dir=copied)
    assert not (tmp_path / "run").exists()


def test_loader_checks_circuit_and_encoding_dimensions(tmp_path):
    circuit_path = tmp_path / "wrong.qpy"
    with circuit_path.open("wb") as stream:
        qpy.dump(QuantumCircuit(1), stream)
    with pytest.raises(ValueError, match="2 qubits"):
        load_circuit(circuit_path, file_hash(circuit_path.read_bytes()), 2)
    encoding_path = tmp_path / "wrong.npy"
    np.save(encoding_path, np.eye(4, 2))
    with pytest.raises(ValueError, match="shape"):
        load_encoding(encoding_path, file_hash(encoding_path.read_bytes()), 3)
    assert set(HASHES) == {"E.npy", "F3_W.qpy", "CZ3_W.qpy"}


def test_cli_writes_report(tmp_path):
    output = tmp_path / "cli"
    result = subprocess.run(
        [sys.executable, "-m", "qudits_on_qubits.evolution.baseline", "--output", str(output)],
        check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    assert "F3: CZ=0 CX=2" in result.stdout
    assert "CZ3: CZ=6 CX=0" in result.stdout
    assert (output / "manifest.json").is_file()


def _candidate(path, circuit):
    path.mkdir()
    with (path / "circuit.qpy").open("wb") as stream:
        qpy.dump(circuit, stream)
    encoding = np.load(INPUT / "E.npy", allow_pickle=False)
    for part in ("A", "B"):
        for direction in ("in", "out"):
            np.save(path / f"E_{direction}_{part}.npy", encoding, allow_pickle=False)
    (path / "meta.json").write_text(json.dumps({"target": "CZ3"}))


def test_multiple_candidates_save_only_valid_circuits(tmp_path):
    with (INPUT / "CZ3_W.qpy").open("rb") as stream:
        reference, = qpy.load(stream)
    good = tmp_path / "good"
    bad = tmp_path / "bad"
    _candidate(good, reference)
    _candidate(bad, QuantumCircuit(4))
    output = tmp_path / "run"
    rows = run_baseline(output, candidate_dirs=(good, bad))
    valid, invalid = rows[-2:]
    assert valid["is_valid"] and not invalid["is_valid"]
    assert valid["circuit_artifact"] == "candidate_001_CZ3.qpy"
    assert invalid["circuit_artifact"] is None
    assert not (output / "candidate_002_CZ3.qpy").exists()
    assert len(json.loads(valid["encoding_artifacts"])) == 2
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["candidate_inputs"]) == 2
    assert manifest["candidate_inputs"][0]["hash_check"] == "recorded_only"
    assert manifest["candidate_inputs"][0]["sha256"]["circuit.qpy"] == file_hash((good / "circuit.qpy").read_bytes())
    with (output / valid["circuit_artifact"]).open("rb") as stream:
        circuit, = qpy.load(stream)
    encodings = [np.load(output / name, allow_pickle=False) for name in json.loads(valid["encoding_artifacts"])]
    target = np.load(output / valid["target_artifact"], allow_pickle=False)
    result = evaluate(EvaluationTask(3, 2, target, logical_error_tolerance=1e-5 / 3,
                                     leakage_tolerance=1e-10 / 9), circuit, encodings)
    assert result.is_valid and result.cz_count == 6


def test_candidate_requires_identical_input_and_output(tmp_path):
    directory = tmp_path / "candidate"
    _candidate(directory, QuantumCircuit(4))
    np.save(directory / "E_out_A.npy", np.eye(4, 3)[[1, 0, 2, 3]])
    with pytest.raises(ValueError, match="input and output encodings"):
        run_baseline(tmp_path / "run", candidate_dirs=(directory,))
    assert not (tmp_path / "run").exists()


def test_candidate_requires_cz3_target(tmp_path):
    directory = tmp_path / "candidate"
    _candidate(directory, QuantumCircuit(4))
    (directory / "meta.json").write_text(json.dumps({"target": "CSUM"}))
    with pytest.raises(ValueError, match="target must be CZ3"):
        run_baseline(tmp_path / "run", candidate_dirs=(directory,))
