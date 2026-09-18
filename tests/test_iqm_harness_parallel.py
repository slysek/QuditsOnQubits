from dataclasses import replace
from concurrent.futures import Future, ProcessPoolExecutor
import json
import os
from pathlib import Path
import pickle
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, qpy, transpile

from qudits_on_qubits.benchmarks.direct_basis import iqm_transpiler_harness as harness
from qudits_on_qubits.benchmarks.direct_basis import optimized_gates as gates
from qudits_on_qubits.benchmarks.direct_basis.candidates import DirectBasisCandidate
from qudits_on_qubits.benchmarks.direct_basis.math_utils import physical_two_qutrit_gate_in_encoding, qutrit_cz


def _test_synthesis(embedding):
    circuit = QuantumCircuit(4)
    circuit.unitary(physical_two_qutrit_gate_in_encoding(qutrit_cz(), embedding), range(4))
    return transpile(circuit, basis_gates=["u", "cz"], seed_transpiler=0, optimization_level=1)


def _initialize_test_worker(config, metadata, runner, progress_events, cache, rendezvous):
    harness._initialize_candidate_worker(config, metadata, runner, progress_events)
    gates.repo_path = lambda *args: cache

    def synthesis(embedding):
        # Both different encodings must enter synthesis before either continues.
        marker = Path(rendezvous) / str(os.getpid())
        marker.touch()
        deadline = time.monotonic() + 30
        while len(list(Path(rendezvous).iterdir())) < 2:
            if time.monotonic() > deadline:
                raise TimeoutError("Two candidate syntheses did not overlap")
            time.sleep(0.02)
        return _test_synthesis(embedding)

    gates._compile_cz3 = synthesis


def _test_runner(strategy, circuit, *, backend, seed_transpiler, optimization_level):
    if strategy == "failed":
        raise ValueError("expected trial failure")
    return SimpleNamespace(
        strategy_name=strategy, seed_transpiler=seed_transpiler, success=True,
        circuit=circuit, compile_time_seconds=0.0,
    )


def config(**kwargs):
    return harness.IqmTranspilerHarnessConfig(
        state_name="two_qutrit", n_qutrits=2, backend=object(),
        iqm_backend_name="offline", iqm_use_metrics=False, candidates=[], **kwargs,
    )


@pytest.mark.parametrize("jobs", [0, -1, 1.5, True, "2", None])
def test_jobs_validated_before_backend_or_candidate_work(jobs):
    with patch.object(harness, "_backend_metadata") as metadata:
        with pytest.raises(ValueError, match="jobs"):
            harness.run_iqm_transpiler_harness(config(jobs=jobs))
        metadata.assert_not_called()


def test_empty_parallel_run_does_not_start_workers():
    with patch.object(harness, "ProcessPoolExecutor") as executor:
        trials, best, summary = harness.run_iqm_transpiler_harness(config(jobs=4))
    assert trials.empty and best.empty
    assert summary["trial_count"] == 0
    executor.assert_not_called()


def test_parallel_spawn_matches_serial_and_writes_full_reports(tmp_path, monkeypatch, capsys):
    candidates = [
        DirectBasisCandidate("I", "test", np.eye(3)),
        DirectBasisCandidate("phase", "test", np.diag([1, 1j, -1])),
        DirectBasisCandidate("duplicate", "test", -np.eye(3)),
        DirectBasisCandidate("unsupported", "test", None, error_message="unsupported"),
    ]
    cache = tmp_path / "cache"
    rendezvous = tmp_path / "rendezvous"
    rendezvous.mkdir()
    original = config(jobs=2)
    parallel = replace(
        original, candidates=iter(candidates), n_transpile_runs=2,
        strategy_names=("ok", "failed"), quantum_circuits_dir=tmp_path / "circuits",
    )

    def executor(**kwargs):
        assert kwargs["max_workers"] == 2
        kwargs["initializer"] = _initialize_test_worker
        kwargs["initargs"] += (str(cache), str(rendezvous))
        return ProcessPoolExecutor(**kwargs)

    with patch.object(harness, "ProcessPoolExecutor", side_effect=executor):
        trials, best, summary = harness.run_iqm_transpiler_harness(parallel, strategy_runner=_test_runner)
    progress_output = capsys.readouterr().err
    assert "test/I" in progress_output and "test/phase" in progress_output
    assert "trials 4/4" in progress_output
    assert "unsupported_candidate" in progress_output
    assert len(list(rendezvous.iterdir())) == 2
    assert summary["successful_trial_count"] == 4
    assert summary["failed_trial_count"] == 4
    assert summary["unsupported_candidate_count"] == 1
    assert summary["global_phase_duplicate_count"] == 1
    assert summary["jobs"] == 2
    assert len(trials) == 9
    paths = trials.loc[trials.success, "graph_state_transpiled_qpy"].tolist()
    assert len(set(paths)) == 4
    for path in paths:
        with open(path, "rb") as handle:
            assert qpy.load(handle)[0].num_qubits == 4

    monkeypatch.setattr(gates, "repo_path", lambda *args: str(cache))
    monkeypatch.setattr(gates, "_compile_cz3", _test_synthesis)
    serial = replace(parallel, jobs=1, candidates=candidates)
    serial_trials, serial_best, serial_summary = harness.run_iqm_transpiler_harness(serial, strategy_runner=_test_runner)
    columns = [column for column in trials if column not in ("gate_cache_hit", "compile_time_seconds")]
    pd.testing.assert_frame_equal(trials[columns], serial_trials[columns])
    pd.testing.assert_frame_equal(best[columns], serial_best[columns])
    assert {**summary, "jobs": 1} == serial_summary
    parallel_paths = harness.write_iqm_transpiler_harness_outputs(
        tmp_path / "parallel", all_trials=trials, best_by_candidate=best, summary=summary,
    )
    serial_paths = harness.write_iqm_transpiler_harness_outputs(
        tmp_path / "serial", all_trials=serial_trials, best_by_candidate=serial_best, summary=serial_summary,
    )
    for key in ("pareto_ranked_csv", "state_equivalence_groups_csv", "recommended_circuits_csv"):
        pd.testing.assert_frame_equal(pd.read_csv(parallel_paths[key]), pd.read_csv(serial_paths[key]))
    assert json.loads(Path(parallel_paths["summary_json"]).read_text())["jobs"] == 2


def test_nonserializable_runner_fails_before_starting_workers():
    candidate = DirectBasisCandidate("I", "test", np.eye(3))
    with patch.object(harness, "ProcessPoolExecutor") as executor:
        with pytest.raises(ValueError, match="module-level"):
            harness.run_iqm_transpiler_harness(
                replace(config(jobs=2), candidates=[candidate]), strategy_runner=lambda *args: None,
            )
    executor.assert_not_called()


def test_iqm_snapshot_preserves_targets_and_compiles_offline():
    from iqm.qiskit_iqm.fake_backends import IQMFakeAdonis
    from qudits_on_qubits.benchmarks.direct_basis.iqm_compilation_backend import snapshot_compilation_backend
    from qudits_on_qubits.benchmarks.direct_basis.iqm_transpiler_strategies import run_iqm_transpiler_strategy

    original = IQMFakeAdonis()
    snapshot = pickle.loads(pickle.dumps(snapshot_compilation_backend(original)))
    assert snapshot.architecture == original.architecture
    assert snapshot.metrics == original.metrics
    assert snapshot.physical_qubits == original.physical_qubits
    assert set(snapshot.target.operation_names) == set(original.target.operation_names)
    assert set(snapshot.coupling_map.get_edges()) == set(original.coupling_map.get_edges())
    assert not hasattr(snapshot, "client")
    circuit = QuantumCircuit(4)
    circuit.h(0)
    circuit.cx(0, 3)
    for strategy in ("preset_exact", "transpile_to_iqm_exact"):
        result = run_iqm_transpiler_strategy(strategy, circuit, backend=snapshot, seed_transpiler=0)
        assert result.success, result.error_message
    with pytest.raises(RuntimeError, match="cannot submit"):
        snapshot.run(circuit)


def test_parallel_rejects_colliding_artifact_directories_before_workers(tmp_path):
    candidates = [
        DirectBasisCandidate("a:b", "test", np.eye(3)),
        DirectBasisCandidate("a?b", "test", np.diag([1, 1j, -1])),
    ]
    with patch.object(harness, "ProcessPoolExecutor", side_effect=AssertionError("workers started")):
        with pytest.raises(ValueError, match="artifact.*collision"):
            harness.run_iqm_transpiler_harness(
                replace(config(jobs=2), candidates=candidates, quantum_circuits_dir=tmp_path),
            )


@pytest.mark.parametrize("failure_type", [RuntimeError, KeyboardInterrupt])
def test_worker_failure_does_not_drain_remaining_candidates(failure_type):
    futures = []

    class Executor:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, *args):
            future = Future()
            if not futures:
                future.set_exception(failure_type("worker broke"))
            futures.append(future)
            return future

        def shutdown(self, *, wait, cancel_futures):
            assert wait and cancel_futures

    candidates = [DirectBasisCandidate(str(i), "test", np.diag([1, np.exp(0.1j * i), 1])) for i in range(5)]
    with patch.object(harness, "ProcessPoolExecutor", Executor):
        with pytest.raises(failure_type, match="worker broke"):
            harness.run_iqm_transpiler_harness(replace(config(jobs=2), candidates=candidates))
    assert len(futures) == 2
    assert futures[1].cancelled()
