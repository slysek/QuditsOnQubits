from __future__ import annotations

import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
from qiskit import QuantumCircuit, qpy
from qiskit.quantum_info import Operator

from qudits_on_qubits.benchmarks.theta_continuation import artifacts


@pytest.fixture
def store(tmp_path):
    return artifacts.RunStore.create(tmp_path / "run", {"fingerprint": "test-run", "config": {"seed": 3}})


def test_fingerprint_canonical_strict_json():
    assert artifacts.fingerprint({"b": 2, "a": [1, None]}) == hashlib.sha256(b'{"a":[1,null],"b":2}').hexdigest()
    assert artifacts.fingerprint({"b": 2, "a": [1, None]}) == artifacts.fingerprint({"a": [1, None], "b": 2})
    for value in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError):
            artifacts.fingerprint({"value": value})
    with pytest.raises((ValueError, TypeError)):
        artifacts.fingerprint({1: "not a JSON object key"})


def test_atomic_json_preserves_previous_file_on_interruption(tmp_path, monkeypatch):
    path = tmp_path / "value.json"
    artifacts.atomic_write_json(path, {"value": 1})

    def interrupted(source, target):
        assert Path(source).parent == Path(target).parent
        raise OSError("simulated interruption")

    monkeypatch.setattr(artifacts.os, "replace", interrupted)
    with pytest.raises(OSError, match="simulated interruption"):
        artifacts.atomic_write_json(path, {"value": 2})
    assert json.loads(path.read_text()) == {"value": 1}
    assert list(tmp_path.iterdir()) == [path]


def test_create_open_manifest_and_alias_safety(tmp_path):
    manifest = {"fingerprint": "initial", "nested": {"value": 1}}
    store = artifacts.RunStore.create(tmp_path / "run", manifest)
    manifest["nested"]["value"] = 2
    assert store.manifest["nested"]["value"] == 1
    opened = artifacts.RunStore.open(store.root, expected_fingerprint="initial")
    assert opened.manifest == store.manifest
    with pytest.raises(ValueError, match="fingerprint"):
        artifacts.RunStore.open(store.root, expected_fingerprint="changed")
    with pytest.raises(FileExistsError):
        artifacts.RunStore.create(store.root, {"fingerprint": "changed"})
    assert artifacts.RunStore.open(store.root).manifest == store.manifest
    empty = tmp_path / "empty"
    empty.mkdir()
    assert artifacts.RunStore.create(empty, {"fingerprint": "empty"}).root == empty.resolve()


@pytest.mark.parametrize("manifest", [{}, {"fingerprint": 123}, {"fingerprint": ""}, []])
def test_open_rejects_invalid_manifest(tmp_path, manifest):
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises((ValueError, TypeError)):
        artifacts.RunStore.open(tmp_path)


def test_roundtrip_bundle_with_complex_arrays_and_circuit(store):
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cz(0, 1)
    circuit.global_phase = 0.29
    circuit.metadata = {"seed": 3}
    array = np.array([[1 + 2j, 3 - 4j], [0, 1j]])
    metadata = {"theta": 0.25, "metrics": {"failed_error": None}}
    complete = store.write_bundle(
        "points/00000", metadata=metadata, circuits={"f3.qpy": circuit},
        arrays={"state.npy": array, "empty.npy": np.array([], dtype=float)},
        documents={"metrics.json": {"cx_count": 3}},
    )
    restored = store.read_bundle("points/00000")
    assert restored["metadata"] == metadata
    assert restored["complete"] == complete
    assert restored["documents"] == {"metrics.json": {"cx_count": 3}}
    np.testing.assert_array_equal(restored["arrays"]["state.npy"], array)
    assert restored["arrays"]["empty.npy"].size == 0
    np.testing.assert_allclose(Operator(restored["circuits"]["f3.qpy"]).data, Operator(circuit).data)
    assert restored["circuits"]["f3.qpy"].metadata == circuit.metadata
    assert store.completed_point_indices() == [0]
    restored["metadata"]["theta"] = 10
    restored["arrays"]["state.npy"][0, 0] = 100
    assert store.read_bundle("points/00000")["metadata"]["theta"] == 0.25
    np.testing.assert_array_equal(store.read_bundle("points/00000")["arrays"]["state.npy"], array)
    with pytest.raises(FileExistsError):
        store.write_bundle("points/00000", metadata={"theta": 1})


def test_incomplete_bundle_can_be_repaired_and_is_not_discovered(store, monkeypatch):
    original = artifacts.atomic_write_json

    def interrupted(path, payload):
        if Path(path).name == "complete.json":
            raise OSError("interrupted completion")
        original(path, payload)

    monkeypatch.setattr(artifacts, "atomic_write_json", interrupted)
    with pytest.raises(OSError, match="interrupted completion"):
        store.write_bundle("points/00002", metadata={"theta": 2}, arrays={"state.npy": np.ones(2)})
    assert store.completed_point_indices() == []
    with pytest.raises((ValueError, FileNotFoundError)):
        store.read_bundle("points/00002")
    unrelated = store.root / "points/00002/keep.txt"
    unrelated.write_text("untracked")
    monkeypatch.setattr(artifacts, "atomic_write_json", original)
    store.write_bundle("points/00002", metadata={"theta": 3}, arrays={"state.npy": np.zeros(2)})
    assert store.read_bundle("points/00002")["metadata"] == {"theta": 3}
    assert unrelated.read_text() == "untracked"
    store.write_bundle("points/00001", metadata={})
    assert store.completed_point_indices() == [1, 2]


def test_bundle_corruption_and_missing_files_fail(store):
    store.write_bundle("baseline", metadata={}, arrays={"state.npy": np.ones(2)})
    path = store.root / "baseline/state.npy"
    path.write_bytes(path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="hash|digest|checksum"):
        store.read_bundle("baseline")
    path.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        store.read_bundle("baseline")


def test_corrupt_complete_never_reused_or_overwritten(store):
    store.write_bundle("points/00000", metadata={})
    (store.root / "points/00000/complete.json").write_text("{")
    assert store.completed_point_indices() == [0]
    with pytest.raises(ValueError):
        store.read_bundle("points/00000")
    with pytest.raises(FileExistsError):
        store.write_bundle("points/00000", metadata={})


@pytest.mark.parametrize("relative", ["../escape", "/absolute", "C:/escape", "points/../escape", "points\\escape", ".", "", "points//00000", "points/00000."])
def test_rejects_unsafe_bundle_paths(store, relative):
    with pytest.raises(ValueError):
        store.write_bundle(relative, metadata={})
    with pytest.raises(ValueError):
        store.read_bundle(relative)


@pytest.mark.parametrize("name", ["../escaped.npy", "dir/state.npy", "state.json", "C:state.npy", "state.npy."])
def test_rejects_unsafe_or_wrong_array_filenames(store, name):
    with pytest.raises(ValueError):
        store.write_bundle("baseline", metadata={}, arrays={name: np.ones(2)})
    assert not (store.root / "baseline/complete.json").exists()


@pytest.mark.parametrize("name", ["metadata.json", "complete.json", "METADATA.json"])
def test_rejects_reserved_document_names(store, name):
    with pytest.raises(ValueError):
        store.write_bundle("baseline", metadata={}, documents={name: {}})


@pytest.mark.parametrize("array", [np.array([np.nan]), np.array([np.inf]), np.array([1j * np.inf]), np.array([object()]), np.array(["text"])])
def test_rejects_invalid_arrays(store, array):
    with pytest.raises((ValueError, TypeError)):
        store.write_bundle("baseline", metadata={}, arrays={"state.npy": array})


def test_rejects_nonfinite_json_without_publishing_complete(store):
    with pytest.raises(ValueError):
        store.write_bundle("baseline", metadata={"error": float("nan")})
    with pytest.raises(ValueError):
        store.write_bundle("baseline", metadata={}, documents={"metrics.json": {"error": float("inf")}})
    assert not (store.root / "baseline/complete.json").exists()


@pytest.mark.parametrize("filename", ["../manifest.json", "../../escape.json", "/absolute.json", "sub/file.json", "C:escape.json", "metadata.json."])
def test_tampered_complete_paths_rejected(store, filename):
    complete = store.write_bundle("baseline", metadata={})
    complete["files"][filename] = "0" * 64
    artifacts.atomic_write_json(store.root / "baseline/complete.json", complete)
    with pytest.raises(ValueError):
        store.read_bundle("baseline")


def _replace_tracked_file(store, relative, name, content):
    path = store.root / relative / name
    path.write_bytes(content)
    complete_path = path.parent / "complete.json"
    complete = json.loads(complete_path.read_text())
    complete["files"][name] = hashlib.sha256(content).hexdigest()
    artifacts.atomic_write_json(complete_path, complete)


def test_rehashed_invalid_payloads_still_fail_validation(store):
    store.write_bundle("arrays", metadata={}, arrays={"state.npy": np.ones(2)})
    buffer = io.BytesIO()
    np.save(buffer, np.array([{"unsafe": True}], dtype=object), allow_pickle=True)
    _replace_tracked_file(store, "arrays", "state.npy", buffer.getvalue())
    with pytest.raises(ValueError):
        store.read_bundle("arrays")
    store.write_bundle("circuits", metadata={}, circuits={"circuit.qpy": QuantumCircuit(1)})
    buffer = io.BytesIO()
    qpy.dump([QuantumCircuit(1), QuantumCircuit(1)], buffer)
    _replace_tracked_file(store, "circuits", "circuit.qpy", buffer.getvalue())
    with pytest.raises(ValueError, match="one|single"):
        store.read_bundle("circuits")
    store.write_bundle("json", metadata={})
    _replace_tracked_file(store, "json", "metadata.json", b'{"value":NaN}')
    with pytest.raises(ValueError):
        store.read_bundle("json")


def _directory_link(link, target):
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            pytest.skip("Directory symlinks unavailable")
        # Junctions exercise resolved-path containment without Windows symlink
        # privileges. This command only creates the test's temporary junction.
        result = subprocess.run(
            ["cmd.exe", "/d", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True, text=True, check=False,
        )
        if result.returncode:
            pytest.skip("Directory symlinks and Windows junctions unavailable")


def test_symlink_escape_rejected_for_bundles_and_checkpoint(store, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    link = store.root / "escaped"
    _directory_link(link, outside)
    with pytest.raises(ValueError):
        store.write_bundle("escaped/point", metadata={})
    with pytest.raises(ValueError):
        store.read_bundle("escaped/point")
    checkpoint_link = store.root / "checkpoints"
    _directory_link(checkpoint_link, outside)
    with pytest.raises(ValueError):
        store.save_checkpoint({"index": 1})
    with pytest.raises(ValueError):
        store.load_checkpoint()
    assert not list(outside.iterdir())


def test_internal_directory_alias_cannot_overwrite_completed_bundle(store):
    store.write_bundle("baseline", metadata={"original": True})
    _directory_link(store.root / "alias", store.root / "baseline")
    assert store.read_bundle("alias")["metadata"] == {"original": True}
    with pytest.raises(FileExistsError):
        store.write_bundle("alias", metadata={"original": False})
    assert store.read_bundle("baseline")["metadata"] == {"original": True}


def test_manifest_symlink_escape_rejected(store, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (store.root / "manifest.json").unlink()
    (outside / "manifest.json").write_text('{"fingerprint":"outside"}')
    try:
        (store.root / "manifest.json").symlink_to(outside / "manifest.json")
    except OSError:
        pytest.skip("File symlinks require Windows developer mode or privilege")
    with pytest.raises(ValueError):
        artifacts.RunStore.open(store.root)


def test_checkpoint_roundtrip_tamper_and_interruption(store, monkeypatch):
    assert store.load_checkpoint() is None
    payload = {"index": 2, "parameters": [0.2, 0.3], "failure": None}
    store.save_checkpoint(payload)
    assert store.load_checkpoint() == payload
    checkpoint = store.load_checkpoint()
    checkpoint["parameters"][0] = 99
    assert store.load_checkpoint() == payload

    def interrupted(source, target):
        raise OSError("checkpoint interruption")

    monkeypatch.setattr(artifacts.os, "replace", interrupted)
    with pytest.raises(OSError):
        store.save_checkpoint({"index": 3})
    assert store.load_checkpoint() == payload
    checkpoint_path = store.root / "checkpoints/latest.json"
    envelope = json.loads(checkpoint_path.read_text())
    envelope["payload"]["index"] = 99
    checkpoint_path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="hash|digest|checksum"):
        store.load_checkpoint()


def test_runtime_provenance_contains_versions_and_relative_source_hashes():
    provenance = artifacts.runtime_provenance()
    assert set(provenance["versions"]) >= {"python", "numpy", "scipy", "qiskit", "bqskit"}
    hashes = provenance["source_hashes"]
    assert any("theta_continuation/artifacts.py" in path for path in hashes)
    assert any("direct_basis/benchmark.py" in path for path in hashes)
    assert any("direct_basis/circuits.py" in path for path in hashes)
    for path, digest in hashes.items():
        assert not Path(path).is_absolute()
        assert ".." not in Path(path).parts
        assert len(digest) == 64


@pytest.mark.parametrize("dependency", [
    "encoding_search/states.py", "core/graph_states.py", "reference_experiments.py",
])
def test_graph_definition_bytes_change_runtime_fingerprint(tmp_path, monkeypatch, dependency):
    package_root = Path(artifacts.__file__).resolve().parents[2]
    dependency_path = package_root / dependency
    before = artifacts.runtime_provenance()
    changed_source = tmp_path / "changed_definition.py"
    changed_source.write_bytes(dependency_path.read_bytes() + b"\n# Changed graph target definition.\n")
    original_read_bytes = Path.read_bytes

    def read_with_changed_definition(path):
        if path == dependency_path:
            return original_read_bytes(changed_source)
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_with_changed_definition)
    after = artifacts.runtime_provenance()
    assert artifacts.fingerprint(after) != artifacts.fingerprint(before)
    key = f"qudits_on_qubits/{dependency}"
    assert after["source_hashes"][key] == hashlib.sha256(changed_source.read_bytes()).hexdigest()
    assert after["versions"] == before["versions"]
    assert after["source_hashes"].keys() == before["source_hashes"].keys()
    assert {
        path: digest for path, digest in before["source_hashes"].items() if path != key
    } == {
        path: digest for path, digest in after["source_hashes"].items() if path != key
    }


@pytest.mark.parametrize("state_name", ["two_qutrit", "ghz3", "ame43"])
def test_all_fixed_graph_target_paths_are_covered_by_provenance(state_name):
    from qudits_on_qubits.benchmarks.direct_basis.circuits import (
        build_direct_basis_graph_state_circuit,
        build_optimized_direct_basis_graph_state_circuit,
        resolve_direct_state,
    )
    from qudits_on_qubits.benchmarks.direct_basis.math_utils import encoding_embedding
    from qudits_on_qubits.core.graph_states import resolve_graph_state
    from qudits_on_qubits.encoding_search.states import resolve_benchmark_state
    from qudits_on_qubits.reference_experiments import LogicalStateSpec, get_reference_experiment

    state = resolve_direct_state(state_name)
    reference = get_reference_experiment(state_name).state
    assert tuple(state.edges) == reference.legacy_edges()
    assert state.num_qutrits == reference.num_parties
    source_root = Path(artifacts.__file__).resolve().parents[3]
    hashes = artifacts.runtime_provenance()["source_hashes"]
    for dependency in (
        build_direct_basis_graph_state_circuit,
        build_optimized_direct_basis_graph_state_circuit,
        encoding_embedding, resolve_direct_state, resolve_benchmark_state,
        resolve_graph_state, get_reference_experiment, LogicalStateSpec.legacy_edges,
    ):
        path = Path(inspect.getsourcefile(dependency)).resolve()
        key = path.relative_to(source_root).as_posix()
        assert hashes.get(key) == hashlib.sha256(path.read_bytes()).hexdigest(), key
