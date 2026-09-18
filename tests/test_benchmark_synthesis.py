"""Synthesis policy and immutable saved-artifact replay contracts."""

from dataclasses import replace
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from qiskit import QuantumCircuit, qpy, transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import Operator


def _synthesis():
    from qudits_on_qubits.benchmarks import synthesis
    return synthesis


def _candidate(theta=0.0, *, family="schmidt_theta", grid_index=0):
    from qudits_on_qubits.benchmarks.theta_continuation.encoding import theta_embedding
    return SimpleNamespace(candidate_id=f"theta_{grid_index:05d}", family=family,
                           encoding=theta_embedding(theta),
                           parameters={"theta": theta, "grid_index": grid_index})


def _dump(path, circuits):
    with path.open("wb") as stream:
        qpy.dump(circuits, stream)


@pytest.fixture
def saved_directory(tmp_path):
    root = tmp_path / "export"
    root.mkdir()
    np.save(root / "E.npy", np.eye(4, 3, dtype=complex))
    f3, cz3 = QuantumCircuit(2), QuantumCircuit(4)
    f3.h(0)
    cz3.cz(0, 2)
    graph = QuantumCircuit(4)
    graph.h(0)
    graph.cz(0, 2)
    for name, circuit in (("F3_W.qpy", f3), ("CZ3_W.qpy", cz3),
                          ("graph_state_direct_basis.qpy", graph)):
        _dump(root / name, circuit)
    return root


def test_optimized_delegates_candidate_and_records_settings(monkeypatch, tmp_path):
    synthesis = _synthesis()
    candidate = _candidate()
    calls = []
    result = object()
    monkeypatch.setattr(synthesis.optimized_gates, "optimized_gate_library",
                        lambda encoding, **kwargs: (calls.append((encoding, kwargs)), result)[1])
    strategy = synthesis.OptimizedSynthesis(cache_dir=tmp_path / "cache")
    identity = strategy.to_dict()
    assert identity["strategy"] == "optimized"
    assert identity["seed"] == synthesis.optimized_gates.SYNTHESIS_SEED
    assert identity["epsilon"] == synthesis.optimized_gates.SYNTHESIS_EPSILON
    assert identity["provenance"]["versions"]["qiskit"]
    strategy.prepare([candidate], tmp_path)
    assert strategy.build(candidate) is result
    assert np.array_equal(calls[0][0], candidate.encoding)
    assert calls[0][1]["cache_dir"] == tmp_path / "cache"
    identity["strategy"] = "changed"
    assert strategy.to_dict()["strategy"] == "optimized"


def test_exact_strategy_implements_logical_gates_without_optimized_synthesis(monkeypatch):
    synthesis = _synthesis()
    from qudits_on_qubits.benchmarks.direct_basis.math_utils import qutrit_cz, qutrit_fourier
    monkeypatch.setattr(synthesis.optimized_gates, "optimized_gate_library",
                        lambda *args, **kwargs: pytest.fail("unexpected optimized synthesis"))
    candidate = _candidate(0.23)
    strategy = synthesis.ExactSynthesis()
    gates = strategy.build(candidate)
    for circuit, encoding, logical in (
        (gates.f3, candidate.encoding, qutrit_fourier()),
        (gates.cz3, np.kron(candidate.encoding, candidate.encoding), qutrit_cz()),
    ):
        assert np.linalg.norm(Operator(circuit).data @ encoding - encoding @ logical) < 1e-12
    assert strategy.to_dict()["strategy"] == "exact"


def test_saved_library_replays_original_qpy_and_explicit_workload_binding(saved_directory):
    synthesis = _synthesis()
    workload_hash = "a" * 64
    strategy = synthesis.SavedGateSynthesis(saved_directory, workload_hash=workload_hash)
    candidate = synthesis.saved_candidate(saved_directory, candidate_id="best_saved")
    assert candidate.family == "saved"
    assert candidate.candidate_id == "best_saved"
    strategy.prepare([candidate], saved_directory.parent / "unused")
    library = strategy.build(candidate)
    assert library.f3 == QuantumCircuit.from_qasm_str(
        'OPENQASM 2.0; include "qelib1.inc"; qreg q[2]; h q[0];')
    assert library.source_circuit.count_ops() == {"h": 1, "cz": 1}
    assert library.workload_hash == workload_hash
    identity = strategy.to_dict()
    assert identity["files"]["CZ3_W.qpy"] == hashlib.sha256(
        (saved_directory / "CZ3_W.qpy").read_bytes()).hexdigest()
    assert identity["workload_hash"] == workload_hash
    library.f3.x(1)
    assert "x" not in strategy.build(candidate).f3.count_ops()
    assert synthesis.SavedGateSynthesis(saved_directory).build(candidate).source_circuit is None


def test_saved_rejects_changed_files_and_wrong_encoding(saved_directory):
    synthesis = _synthesis()
    strategy = synthesis.SavedGateSynthesis(saved_directory)
    with pytest.raises(ValueError, match="encoding"):
        strategy.build(_candidate(0.1))
    _dump(saved_directory / "CZ3_W.qpy", QuantumCircuit(4))
    with pytest.raises(ValueError, match="changed|hash"):
        strategy.build(_candidate())


@pytest.mark.parametrize("problem", ["many", "width", "measure", "reset", "parameter"])
def test_saved_rejects_invalid_qpy(saved_directory, problem):
    synthesis = _synthesis()
    circuit = QuantumCircuit(3 if problem == "width" else 2)
    if problem == "measure":
        circuit.measure_all()
    elif problem == "reset":
        circuit.reset(0)
    elif problem == "parameter":
        circuit.rx(Parameter("unbound"), 0)
    _dump(saved_directory / "F3_W.qpy", [circuit, circuit] if problem == "many" else circuit)
    with pytest.raises(ValueError):
        strategy = synthesis.SavedGateSynthesis(saved_directory)
        strategy.build(_candidate())


def test_saved_rejects_invalid_encoding_and_binding(saved_directory):
    synthesis = _synthesis()
    with pytest.raises(ValueError, match="workload_hash"):
        synthesis.SavedGateSynthesis(saved_directory, workload_hash="not-a-hash")
    np.save(saved_directory / "E.npy", np.ones((4, 3)))
    with pytest.raises(ValueError, match="encoding|isometr"):
        synthesis.saved_candidate(saved_directory)


@pytest.fixture(scope="module")
def theta_run(tmp_path_factory):
    from qudits_on_qubits.benchmarks.theta_continuation.artifacts import RunStore, fingerprint
    from qudits_on_qubits.benchmarks.theta_continuation.compilation import validate_cz3_candidate
    from qudits_on_qubits.benchmarks.theta_continuation.f3 import synthesize_theta_f3
    from qudits_on_qubits.benchmarks.theta_continuation.models import ThetaBenchmarkConfig
    config = ThetaBenchmarkConfig(limit_points=1, transpiler_seeds=(0,), states=("two_qutrit",))
    root = tmp_path_factory.mktemp("theta_replay") / "run"
    manifest = {"format_version": 1, "benchmark": "theta_continuation_v1", "config": config.to_dict(),
                "baseline_sha256": config.baseline_sha256, "template_id": "test-template"}
    store = RunStore.create(root, {**manifest, "fingerprint": fingerprint(manifest)})
    encoding = _candidate().encoding
    f3 = synthesize_theta_f3(encoding, transpiler_seeds=(0,))
    cz3 = transpile(_synthesis().ExactSynthesis().build(_candidate()).cz3,
                    basis_gates=["u", "cz"], optimization_level=1, seed_transpiler=0)
    metrics = validate_cz3_candidate(cz3, encoding, compiled=True)
    store.write_bundle("points/00000", metadata={"index": 0, "theta": 0.0,
                       "template_id": "test-template", "correct": True, "selected_method": "baseline",
                       "f3_alpha": f3.alpha, "f3_metrics": f3.optimal_metrics, "cz3_metrics": metrics},
                       circuits={"f3_optimal.qpy": f3.optimal, "cz3_selected.qpy": cz3},
                       arrays={"E.npy": encoding})
    return config, root


def test_theta_reuses_validated_source_without_synthesis(theta_run, monkeypatch, tmp_path):
    synthesis = _synthesis()
    config, root = theta_run
    monkeypatch.setattr(synthesis, "run_theta_benchmark",
                        lambda *args, **kwargs: pytest.fail("source replay launched synthesis"))
    strategy = synthesis.ThetaContinuationSynthesis(config, source_run=root)
    identity = strategy.to_dict()
    canonical = _candidate(family="canonical")
    strategy.prepare([canonical, _candidate()], tmp_path)
    library = strategy.build(_candidate())
    assert library.f3.num_qubits == 2
    assert library.cz3.num_qubits == 4
    assert library.selected_method == "baseline"
    assert strategy.build(canonical).cz3 == library.cz3
    assert strategy.to_dict() == identity
    assert identity["parameter_source"] == "saved_theta_run"
    assert identity["source"]["points"]["00000"]["cz3_selected.qpy"]


def test_theta_prepare_runs_continuation_once(theta_run, monkeypatch, tmp_path):
    synthesis = _synthesis()
    config, root = theta_run
    calls = []
    import shutil

    def run(config_arg, destination, **kwargs):
        calls.append((config_arg, destination, kwargs))
        shutil.copytree(root, destination)
        return {"output_dir": str(destination)}

    monkeypatch.setattr(synthesis, "run_theta_benchmark", run)
    strategy = synthesis.ThetaContinuationSynthesis(config)
    identity = strategy.to_dict()
    strategy.prepare([_candidate()], tmp_path)
    strategy.prepare([_candidate()], tmp_path)
    assert len(calls) == 1
    assert calls[0][1] == tmp_path / "theta_synthesis"
    assert calls[0][2]["mode"] == "gates"
    assert strategy.build(_candidate()).selected_method == "baseline"
    assert strategy.to_dict() == identity


@pytest.mark.parametrize("candidate", [_candidate(family="local_su2"), _candidate(0.2),
                                      _candidate(0.0, grid_index=1)])
def test_theta_rejects_candidate_mismatch_before_start(candidate, monkeypatch, tmp_path):
    synthesis = _synthesis()
    from qudits_on_qubits.benchmarks.theta_continuation.models import ThetaBenchmarkConfig
    monkeypatch.setattr(synthesis, "run_theta_benchmark",
                        lambda *args, **kwargs: pytest.fail("invalid cohort started synthesis"))
    strategy = synthesis.ThetaContinuationSynthesis(ThetaBenchmarkConfig(limit_points=1))
    with pytest.raises(ValueError, match="family|grid|theta"):
        strategy.prepare([candidate], tmp_path)


def test_theta_rejects_modified_manifest_and_config(theta_run, tmp_path):
    synthesis = _synthesis()
    config, root = theta_run
    with pytest.raises(ValueError, match="config"):
        synthesis.ThetaContinuationSynthesis(replace(config, cz3_tolerance=5e-4), source_run=root)
    import shutil
    source = tmp_path / "copy"
    shutil.copytree(root, source)
    path = source / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["template_id"] = "changed"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="fingerprint"):
        synthesis.ThetaContinuationSynthesis(config, source_run=source)


def test_theta_rejects_changed_artifacts_after_identity(theta_run, tmp_path):
    synthesis = _synthesis()
    config, root = theta_run
    import shutil
    source = tmp_path / "copy"
    shutil.copytree(root, source)
    strategy = synthesis.ThetaContinuationSynthesis(config, source_run=source)
    (source / "points/00000/cz3_selected.qpy").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        strategy.build(_candidate())


def test_theta_requires_prepare_for_new_run():
    synthesis = _synthesis()
    from qudits_on_qubits.benchmarks.theta_continuation.models import ThetaBenchmarkConfig
    strategy = synthesis.ThetaContinuationSynthesis(ThetaBenchmarkConfig(limit_points=1))
    with pytest.raises(RuntimeError, match="prepare"):
        strategy.build(_candidate())



def test_saved_corruption_has_distinct_fatal_exception(saved_directory):
    synthesis = _synthesis()
    strategy = synthesis.SavedGateSynthesis(saved_directory)
    (saved_directory / "CZ3_W.qpy").write_bytes(b"corrupt")
    with pytest.raises(synthesis.SynthesisArtifactError, match="hash"):
        strategy.build(_candidate())


def test_theta_corruption_is_fatal_but_failed_fitting_remains_scientific(theta_run, tmp_path):
    synthesis = _synthesis()
    from qudits_on_qubits.benchmarks.theta_continuation.artifacts import RunStore
    config, root = theta_run
    original = RunStore.open(root)
    bundle = original.read_bundle("points/00000")
    failed = RunStore.create(tmp_path / "failed", original.manifest)
    failed.write_bundle("points/00000", metadata={**bundle["metadata"], "correct": False},
                        arrays=bundle["arrays"], circuits=bundle["circuits"])
    strategy = synthesis.ThetaContinuationSynthesis(config, source_run=failed.root)
    with pytest.raises(ValueError, match="correct") as caught:
        strategy.build(_candidate())
    assert not isinstance(caught.value, synthesis.SynthesisArtifactError)
    (failed.root / "points/00000/E.npy").write_bytes(b"corrupt")
    with pytest.raises(synthesis.SynthesisArtifactError, match="hash"):
        strategy.build(_candidate())



def test_theta_reassessment_replays_saved_gates_and_records_history(theta_run, tmp_path, monkeypatch):
    synthesis = _synthesis()
    from qudits_on_qubits.benchmarks.theta_continuation.artifacts import RunStore, fingerprint
    config, root = theta_run
    original = RunStore.open(root)
    bundle = original.read_bundle("points/00000")
    relaxed = replace(config, cz3_tolerance=5e-4)
    policy = "reassess_saved_attempts_preserve_historical_branch"
    manifest = {**original.manifest, "benchmark": "theta_threshold_reassessment_v1",
                "config": relaxed.to_dict(), "procedure_version": 1,
                "source_run": str(root), "source_fingerprint": original.manifest["fingerprint"],
                "source_inventory_fingerprint": fingerprint(bundle["complete"]),
                "source_cz3_tolerance": config.cz3_tolerance, "continuation_policy": policy,
                "recovery_protocol": {"target": "StateSystem_9_columns", "seed": 0}}
    del manifest["fingerprint"]
    store = RunStore.create(tmp_path / "reassessed", {**manifest, "fingerprint": fingerprint(manifest)})
    details = {"source_point_fingerprint": fingerprint(bundle["complete"]),
               "cz3_tolerance": relaxed.cz3_tolerance, "source_cz3_tolerance": config.cz3_tolerance,
               "continuation_policy": policy, "origin": "source_saved"}
    store.write_bundle("points/00000", metadata={**bundle["metadata"], "reassessment": details},
                       circuits=bundle["circuits"], arrays=bundle["arrays"],
                       documents={"reassessment.json": details})
    monkeypatch.setattr(synthesis, "run_theta_benchmark",
                        lambda *args, **kwargs: pytest.fail("reassessment replay started fitting"))
    strategy = synthesis.ThetaContinuationSynthesis(relaxed, source_run=store.root)
    strategy.prepare([_candidate()], tmp_path)
    library = strategy.build(_candidate())
    assert library.cz3 == bundle["circuits"]["cz3_selected.qpy"]
    assert strategy.to_dict()["parameter_source"] == "saved_theta_reassessment"
    recorded = strategy.to_dict()["source"]["manifest_data"]
    assert recorded["source_fingerprint"] == original.manifest["fingerprint"]
    assert library.provenance["reassessment"] == details
    with pytest.raises(synthesis.SynthesisArtifactError, match="config"):
        synthesis.ThetaContinuationSynthesis(config, source_run=store.root)


def test_default_optimized_cache_avoids_installation_root(monkeypatch, tmp_path, exact_cz3_synthesis):
    synthesis = _synthesis()
    install_root = tmp_path / "read_only_installation"
    install_root.write_text("not a writable cache directory", encoding="utf-8")
    monkeypatch.setattr(synthesis.optimized_gates, "repo_path", lambda *args: str(install_root))
    user_home = tmp_path / "user"
    monkeypatch.setattr(synthesis.Path, "home", lambda: user_home)
    strategy = synthesis.OptimizedSynthesis()
    library = strategy.build(_candidate())
    assert library.cz3.num_qubits == 4
    assert strategy.cache_dir.is_relative_to(user_home)
    assert list(strategy.cache_dir.rglob("*.qpy"))
    assert strategy.to_dict() == synthesis.OptimizedSynthesis(cache_dir=tmp_path / "other").to_dict()

@pytest.mark.parametrize("round_saved", [False, True])
def test_theta_replay_accepts_one_ulp_roundoff(theta_run, tmp_path, round_saved):
    synthesis = _synthesis()
    from qudits_on_qubits.benchmarks.theta_continuation.artifacts import RunStore
    config, root = theta_run
    source = RunStore.open(root)
    original = source.read_bundle("points/00000")
    candidate = _candidate()
    encoding = original["arrays"]["E.npy"].copy()
    encoding[0, 0] = np.nextafter(encoding[0, 0].real, 0.0)
    if round_saved:
        source = RunStore.create(tmp_path / "rounded", source.manifest)
        source.write_bundle("points/00000", metadata=original["metadata"],
                            arrays={"E.npy": encoding}, circuits=original["circuits"])
    else:
        candidate.encoding = encoding
    strategy = synthesis.ThetaContinuationSynthesis(config, source_run=source.root)
    strategy.prepare([candidate], tmp_path)
    library = strategy.build(candidate)
    assert library.f3 == original["circuits"]["f3_optimal.qpy"]
    assert library.cz3 == original["circuits"]["cz3_selected.qpy"]


@pytest.mark.parametrize("encoding", [
    np.eye(4, 3) * (1 + 1e-8), np.zeros((4, 1)),
])
def test_theta_replay_rejects_wrong_encoding_with_valid_hashes(theta_run, tmp_path, encoding):
    synthesis = _synthesis()
    from qudits_on_qubits.benchmarks.theta_continuation.artifacts import RunStore
    config, root = theta_run
    source = RunStore.open(root)
    bundle = source.read_bundle("points/00000")
    wrong = RunStore.create(tmp_path / "wrong", source.manifest)
    wrong.write_bundle("points/00000", metadata=bundle["metadata"],
                       arrays={"E.npy": encoding}, circuits=bundle["circuits"])
    with pytest.raises(synthesis.SynthesisArtifactError, match="encoding"):
        synthesis.ThetaContinuationSynthesis(config, source_run=wrong.root)


def test_theta_replay_still_checks_hash_for_one_ulp_file_change(theta_run, tmp_path):
    synthesis = _synthesis()
    import shutil
    config, root = theta_run
    copy = tmp_path / "changed"
    shutil.copytree(root, copy)
    strategy = synthesis.ThetaContinuationSynthesis(config, source_run=copy)
    path = copy / "points/00000/E.npy"
    encoding = np.load(path, allow_pickle=False)
    encoding[0, 0] = np.nextafter(encoding[0, 0].real, 0.0)
    np.save(path, encoding, allow_pickle=False)
    with pytest.raises(synthesis.SynthesisArtifactError, match="hash"):
        strategy.build(_candidate())