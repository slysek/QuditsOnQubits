"""Check the published circuits, comparison inputs, and AME(4,6) tensors."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from qiskit import qpy

from qudits_on_qubits.evolution.ame46 import import_ame46
from qudits_on_qubits.evolution.oe_run import _source
from qudits_on_qubits.evolution.records import read
from qudits_on_qubits.evolution.state_workloads import evaluate_workload

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "benchmarks/state_preparation"
INPUTS = ROOT / "experiment_inputs/ame46"


@pytest.fixture(autouse=True)
def public_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QOQ_AME46_TARGETS", str(INPUTS / "targets.npz"))


def test_published_files_match_the_manifest() -> None:
    manifest = json.loads((RELEASE / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name


def test_public_graph_baselines_support_the_default_method() -> None:
    from qudits_on_qubits.evolution.harness import BASELINES

    for state in ("two_qutrit", "ghz3", "ame43"):
        assert (BASELINES[state] / "manifest.json").is_file()
        metrics = _source(ROOT / "experiment_inputs/state_preparation", state, "estar_direct")[3]
        assert metrics["is_valid"], state


@pytest.mark.parametrize("state", ["two_qutrit", "ghz3", "ame43"])
def test_published_comparison_checks_preparation_and_decoder(state: str) -> None:
    summary = json.loads((RELEASE / "summary.json").read_text())
    row = next(row for row in summary["comparisons"] if row["state"] == state)
    with (ROOT / row["baseline_circuit"]).open("rb") as stream:
        baseline = qpy.load(stream)[0]
    encoding = np.load(ROOT / row["baseline_encoding"], allow_pickle=False)
    metrics = evaluate_workload(state, baseline, encoding, tolerance=1e-10)
    record = read(ROOT / row["record"], state)
    assert metrics["is_valid"] and record.metrics["is_valid"]
    assert metrics["cz_count"] == row["canonical_cz"]
    assert record.metrics["cz_count"] == row["preparation_cz"]
    assert record.metrics["decoder_cz_count"] == row["decoder_cz"]
    assert row["preparation_cz"] + row["decoder_cz"] < row["canonical_cz"]


def test_published_records_prepare_their_states() -> None:
    manifest = json.loads((RELEASE / "manifest.json").read_text())
    records = manifest["records"]
    assert records
    for name in records:
        directory = ROOT / name
        data = json.loads((directory / "results.json").read_text())
        record = read(directory, data["state"])
        assert record.metrics["is_valid"], name
        assert record.metrics["cz_count"] == data["logical"]["cz_count"], name
        if "native" in data:
            with (directory / "circuit_native.qpy").open("rb") as stream:
                native = qpy.load(stream)[0]
            count = sum(item.operation.num_qubits == 2 for item in native.data)
            assert count == data["native"]["two_qubit_count"], name


def test_public_ame46_inputs_import_and_have_maximally_mixed_pairs(tmp_path: Path) -> None:
    rows = import_ame46(INPUTS / "targets.npz", INPUTS / "circuits", tmp_path / "imported")
    assert {row["workload"] for row in rows} == {"Bruzda60", "Pandya60prime", "HZ96", "Casas36"}
    assert all(row["is_valid"] for row in rows)
    with np.load(INPUTS / "targets.npz", allow_pickle=False) as data:
        for name in data.files:
            tensor = data[name] / np.linalg.norm(data[name])
            for pair in ((0, 1), (0, 2), (0, 3)):
                order = pair + tuple(k for k in range(4) if k not in pair)
                matrix = tensor.transpose(order).reshape(36, 36)
                np.testing.assert_allclose(matrix @ matrix.conj().T, np.eye(36) / 36, atol=1e-10, rtol=0)
