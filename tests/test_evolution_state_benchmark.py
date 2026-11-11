"""Check the state benchmark and its saved circuits."""

import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pytest
from qiskit import QuantumCircuit, qpy, transpile

from qudits_on_qubits.evolution.encodings import canonical_encoding
from qudits_on_qubits.evolution.state_benchmark import run_state_benchmark
from qudits_on_qubits.evolution.state_workloads import evaluate_workload

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "experiment_inputs/iqm_randomized_bell/canonical_optimized_20260909"
LEGACY = ROOT / "src/qudits_on_qubits/quantum_circuits/CZgate3.qpy"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def inputs(tmp_path):
    m1 = tmp_path / "m1"
    m1.mkdir()
    f3 = (REFERENCE / "F3_W.qpy").read_bytes()
    (m1 / "F3_W.qpy").write_bytes(f3)
    with LEGACY.open("rb") as stream:
        original, = qpy.load(stream)
    circuit = transpile(original, basis_gates=["u", "cz"], optimization_level=0, seed_transpiler=0)
    for index in reversed(range(len(circuit.data))):
        if circuit.data[index].operation.name == "barrier":
            del circuit.data[index]
    canonical = canonical_encoding(3)
    s = 1 / np.sqrt(2)
    estar = np.array([[0, s, s], [1, 0, 0], [0, s, -s], [0, 0, 0]], dtype=complex)
    unitary = np.column_stack((estar, [0, 0, 0, 1]))
    star_circuit = QuantumCircuit(4)
    star_circuit.unitary(unitary.conj().T, [0, 1])
    star_circuit.unitary(unitary.conj().T, [2, 3])
    star_circuit.compose(circuit, inplace=True)
    star_circuit.unitary(unitary, [0, 1])
    star_circuit.unitary(unitary, [2, 3])
    star_circuit = transpile(star_circuit, basis_gates=["u", "cz"], optimization_level=0,
                             seed_transpiler=0)
    star_encoding = BytesIO()
    np.save(star_encoding, estar, allow_pickle=False)
    star_bytes = star_encoding.getvalue()
    (m1 / "estar.npy").write_bytes(star_bytes)
    (m1 / "manifest.json").write_text(json.dumps({"files": {
        "F3_W.qpy": digest(f3), "estar.npy": digest(star_bytes),
    }}))
    (m1 / "baseline_summary.json").write_text(json.dumps({"results": [{
        "encoding": "CZ3_sym_k5_Estar", "is_valid": True, "strict_valid": True,
        "encoding_artifacts": json.dumps(["estar.npy", "estar.npy"]),
    }]}))
    m2 = tmp_path / "m2"
    m2.mkdir()
    best = []
    for index, (name, encoded, selected) in enumerate((
        ("canonical", canonical, circuit), ("CZ3_sym_k5_Estar", estar, star_circuit),
    )):
        stream = BytesIO()
        qpy.dump(selected, stream)
        encoding = BytesIO()
        np.save(encoding, encoded, allow_pickle=False)
        raw = {"circuit.qpy": stream.getvalue(), "E_A.npy": encoding.getvalue(),
               "E_B.npy": encoding.getvalue()}
        relative = Path(f"case_{index}/attempt_0001")
        directory = m2 / relative
        directory.mkdir(parents=True)
        for filename, data in raw.items():
            (directory / filename).write_bytes(data)
        result = {"status": "completed", "strict_valid": True, "trial_path": str(relative),
                  "metrics": {"is_valid": True},
                  "artifact_hashes": {filename: digest(data) for filename, data in raw.items()}}
        result_bytes = json.dumps(result).encode()
        (directory / "result.json").write_bytes(result_bytes)
        (directory / "commit.json").write_text(json.dumps({"result_sha256": digest(result_bytes)}))
        best.append({"encoding": name, "circuit_artifact": str(relative / "circuit.qpy")})
    (m2 / "report_0001.json").write_text(json.dumps({"best_strict": best}))
    return m1, m2


def test_saved_circuits_and_manifest(tmp_path):
    m1, m2 = inputs(tmp_path)
    output = tmp_path / "result"
    rows = run_state_benchmark(m1, m2, output)
    assert len(rows) == 9
    assert {row["workload"] for row in rows} == {"two_qutrit", "ghz3", "ame43"}
    assert {row["method"] for row in rows} == {"canonical_f3", "canonical_direct", "estar_direct"}
    assert all(row["is_valid"] for row in rows)
    assert all(row["cx_count"] == 0 for row in rows)
    for row in rows:
        with (output / row["circuit_artifact"]).open("rb") as stream:
            circuit, = qpy.load(stream)
        encoding = np.load(output / row["encoding_artifact"], allow_pickle=False)
        metrics = evaluate_workload(row["workload"], circuit, encoding, tolerance=1e-10)
        assert metrics["is_valid"]
        assert metrics["cz_count"] == row["cz_count"]
    manifest = json.loads((output / "manifest.json").read_bytes())
    for name, expected in manifest["files"].items():
        assert digest((output / name).read_bytes()) == expected
    before = (output / "summary.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_state_benchmark(m1, m2, output)
    assert (output / "summary.json").read_bytes() == before


def test_input_tampering_is_rejected(tmp_path):
    m1, m2 = inputs(tmp_path)
    target = m2 / "case_0/attempt_0001/circuit.qpy"
    target.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="hash"):
        run_state_benchmark(m1, m2, tmp_path / "result")
    assert not (tmp_path / "result").exists()


def test_rejects_path_outside_run(tmp_path):
    m1, m2 = inputs(tmp_path)
    report = json.loads((m2 / "report_0001.json").read_text())
    report["best_strict"][0]["circuit_artifact"] = "../circuit.qpy"
    (m2 / "report_0001.json").write_text(json.dumps(report))
    with pytest.raises(ValueError, match="path"):
        run_state_benchmark(m1, m2, tmp_path / "result")
    assert not (tmp_path / "result").exists()


def test_rejects_mismatched_estar_encoding(tmp_path):
    m1, m2 = inputs(tmp_path)
    stream = BytesIO()
    np.save(stream, canonical_encoding(3), allow_pickle=False)
    raw = stream.getvalue()
    (m1 / "estar.npy").write_bytes(raw)
    manifest = json.loads((m1 / "manifest.json").read_text())
    manifest["files"]["estar.npy"] = digest(raw)
    (m1 / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="M2 E-star encoding"):
        run_state_benchmark(m1, m2, tmp_path / "result")
    assert not (tmp_path / "result").exists()
