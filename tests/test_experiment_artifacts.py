from __future__ import annotations

import hashlib

import numpy as np
import pytest
from qiskit import QuantumCircuit, qpy

from qudits_on_qubits.experiments.artifacts import (
    load_basis_artifacts,
    resolve_basis_directory,
)
from qudits_on_qubits.experiments.errors import ExperimentValidationError
from qudits_on_qubits.experiments.models import BenchmarkBasis, PathBasis


def _write_artifacts(directory, state="two_qutrit", circuit=None, encoding=None):
    directory.mkdir(parents=True, exist_ok=True)
    if circuit is None:
        circuit = QuantumCircuit({"two_qutrit": 4, "ghz3": 6, "ame43": 8}[state])
    with (directory / "graph_state_direct_basis.qpy").open("wb") as handle:
        qpy.dump(circuit, handle)
    np.save(directory / "E.npy", np.eye(4, 3, dtype=complex) if encoding is None else encoding)


def test_path_basis_loads_raw_circuit_and_hashes_original_bytes(tmp_path):
    raw = tmp_path / "basis"
    _write_artifacts(raw)
    with (raw / "graph_state_direct_basis_transpiled.qpy").open("wb") as handle:
        qpy.dump(QuantumCircuit(99), handle)

    artifacts = load_basis_artifacts(PathBasis(raw), "two_qutrit")

    assert artifacts.state_circuit.num_qubits == 4
    assert artifacts.directory == raw.resolve()
    assert artifacts.source_paths["state"] == raw / "graph_state_direct_basis.qpy"
    assert artifacts.source_hashes["state"] == hashlib.sha256((raw / "graph_state_direct_basis.qpy").read_bytes()).hexdigest()
    assert artifacts.source_hashes["encoding"] == hashlib.sha256((raw / "E.npy").read_bytes()).hexdigest()


def test_benchmark_basis_resolves_candidate_and_rank(tmp_path):
    root = tmp_path / "repo"
    base = root / "artifacts" / "direct_basis_runs" / "selected_best" / "ghz3" / "run-7" / "top"
    candidate = base / "candidate-a"
    rank = base / "rank02_fitness"
    candidate.mkdir(parents=True)
    rank.mkdir()

    assert resolve_basis_directory(BenchmarkBasis("direct_basis_runs", "run-7", "top", candidate="candidate-a"), "ghz3", root) == candidate
    assert resolve_basis_directory(BenchmarkBasis("direct_basis_runs", "run-7", "top", rank=2), "ghz3", root) == rank


@pytest.mark.parametrize(
    ("basis", "message"),
    [
        (BenchmarkBasis("direct_basis_runs", "run-7", "top", candidate="missing"), "candidate"),
        (BenchmarkBasis("direct_basis_runs", "run-7", "top", rank=2), "count=2"),
        (BenchmarkBasis("direct_basis_runs", "run-7", "top", candidate="../escape"), "candidate"),
    ],
)
def test_benchmark_resolution_rejects_missing_ambiguous_and_traversal(tmp_path, basis, message):
    root = tmp_path / "repo"
    base = root / "artifacts" / "direct_basis_runs" / "selected_best" / "ghz3" / "run-7" / "top"
    (base / "rank02_first").mkdir(parents=True)
    (base / "rank02_second").mkdir()

    with pytest.raises(ExperimentValidationError, match=message):
        resolve_basis_directory(basis, "ghz3", root)


@pytest.mark.parametrize("missing_name", ["graph_state_direct_basis.qpy", "E.npy"])
def test_artifact_loading_rejects_missing_required_files(tmp_path, missing_name):
    directory = tmp_path / "basis"
    _write_artifacts(directory)
    (directory / missing_name).unlink()

    with pytest.raises(ExperimentValidationError, match=missing_name):
        load_basis_artifacts(PathBasis(directory), "two_qutrit")


def test_artifact_loading_rejects_multiple_qpy_circuits(tmp_path):
    directory = tmp_path / "basis"
    directory.mkdir()
    with (directory / "graph_state_direct_basis.qpy").open("wb") as handle:
        qpy.dump([QuantumCircuit(4), QuantumCircuit(4)], handle)
    np.save(directory / "E.npy", np.eye(4, 3))

    with pytest.raises(ExperimentValidationError, match="exactly one"):
        load_basis_artifacts(PathBasis(directory), "two_qutrit")


@pytest.mark.parametrize(
    ("circuit", "encoding", "message"),
    [
        (QuantumCircuit(3), None, "4 qubits"),
        (QuantumCircuit(4, 4).measure_all(inplace=False), None, "measurements"),
        (None, np.ones((3, 3)), "shape"),
        (None, np.full((4, 3), np.nan), "finite"),
    ],
)
def test_artifact_loading_validates_circuit_and_encoding(tmp_path, circuit, encoding, message):
    directory = tmp_path / "basis"
    _write_artifacts(directory, circuit=circuit, encoding=encoding)

    with pytest.raises(ExperimentValidationError, match=message):
        load_basis_artifacts(PathBasis(directory), "two_qutrit")
