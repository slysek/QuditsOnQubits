"""Replay checkpoints retain both original identity and additional shot requests."""

import itertools
import json
from pathlib import Path

import pytest

from qudits_on_qubits.experiments import ExperimentPersistenceError
from qudits_on_qubits.experiments.backends.base import ExecutionResult, SubmittedJob
from qudits_on_qubits.experiments.store import ExperimentStore
import qudits_on_qubits.experiments.block_runner as runner
from test_randomized_bell_runner import RecordingAdapter, all_settings_draw, make_spec


@pytest.fixture
def replay_adapter_type():
    class LocalReplayAdapter(RecordingAdapter):
        ids = itertools.count()
        submitted_ids = []

        def __init__(self, spec):
            super().__init__()

        def submit(self, circuits, shots, options=None):
            job_id = f"replay-{next(self.ids)}"
            self.submitted_ids.append(job_id)
            return SubmittedJob(job_id, object(), self.identity, len(circuits), shots)

        def result(self, job, timeout=None):
            return ExecutionResult(tuple({"0000": job.shots} for _ in range(job.circuit_count)),
                                   job.job_id, self.identity)

    return LocalReplayAdapter


def test_recovery_write_before_manifest_retains_original_id_and_replay_cost(
    tmp_path, monkeypatch, replay_adapter_type,
):
    adapter = RecordingAdapter(fail_result=0)
    first = runner.run_randomized_experiment(make_spec(tmp_path), adapter=adapter,
                                            _randbelow=all_settings_draw())
    original = (first.artifact_dir / "batches/0000/submission.json").read_bytes()
    write = ExperimentStore.write_plain_json

    def interrupt_after_recovery(store, run, filename, value):
        result = write(store, run, filename, value)
        if str(filename).replace("\\", "/") == "batches/0000/recovery.json":
            raise ExperimentPersistenceError("injected crash after durable replay recovery")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(runner, "AerAdapter", replay_adapter_type)
        patch.setattr(ExperimentStore, "write_plain_json", interrupt_after_recovery)
        with pytest.raises(ExperimentPersistenceError):
            runner.replay_randomized_aer_batch(first.artifact_dir, batch_index=0)
    resumed = runner.resume_randomized_experiment(first.artifact_dir, adapter=adapter)
    assert resumed.job_ids == ("job-0", "replay-0")
    assert resumed.values["execution"]["local_replays"][0]["additional_requested_shots"] == 24
    assert (first.artifact_dir / "batches/0000/submission.json").read_bytes() == original
    again = runner.resume_randomized_experiment(first.artifact_dir, adapter=adapter)
    assert again.job_ids == resumed.job_ids
    assert again.values["execution"]["local_replays"] == resumed.values["execution"]["local_replays"]


def test_stale_original_manifest_reads_previous_job_id_from_saved_submission(
    tmp_path, monkeypatch, replay_adapter_type,
):
    spec = make_spec(tmp_path)
    adapter = RecordingAdapter()
    write = ExperimentStore.write_plain_json

    def interrupt_after_original_submission(store, run, filename, value):
        result = write(store, run, filename, value)
        if str(filename).replace("\\", "/") == "batches/0000/submission.json":
            raise ExperimentPersistenceError("injected crash before original job ID manifest update")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(ExperimentStore, "write_plain_json", interrupt_after_original_submission)
        with pytest.raises(ExperimentPersistenceError):
            runner.run_randomized_experiment(spec, adapter=adapter, _randbelow=all_settings_draw())
    run = next(Path(spec.output_root).rglob("experiment.json")).parent
    manifest = json.loads((run / "experiment.json").read_text(encoding="utf-8"))
    assert manifest["batches"][0]["job_id"] is None
    original = (run / "batches/0000/submission.json").read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(runner, "AerAdapter", replay_adapter_type)
        result = runner.replay_randomized_aer_batch(run, batch_index=0)
    assert result.status.value == "completed"
    assert result.job_ids[:2] == ("job-0", "replay-0")
    assert (run / "batches/0000/submission.json").read_bytes() == original
    marker = json.loads((run / "batches/0000/local-replay-attempt.json").read_text(encoding="utf-8"))
    assert marker["previous_job_id"] == "job-0"
    assert result.values["execution"]["local_replays"][0]["additional_requested_shots"] == 24
    assert runner.resume_randomized_experiment(run, adapter=object()).job_ids == result.job_ids


def test_marker_only_replay_stays_unknown_without_following_original_job(
    tmp_path, monkeypatch, replay_adapter_type,
):
    adapter = RecordingAdapter(fail_result=0)
    first = runner.run_randomized_experiment(make_spec(tmp_path), adapter=adapter,
                                            _randbelow=all_settings_draw())
    exclusive = ExperimentStore.write_exclusive_json

    def interrupt_after_marker(store, run, filename, value):
        result = exclusive(store, run, filename, value)
        if str(filename).replace("\\", "/") == "batches/0000/local-replay-attempt.json":
            raise ExperimentPersistenceError("injected crash after replay marker")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(runner, "AerAdapter", replay_adapter_type)
        patch.setattr(ExperimentStore, "write_exclusive_json", interrupt_after_marker)
        with pytest.raises(ExperimentPersistenceError):
            runner.replay_randomized_aer_batch(first.artifact_dir, batch_index=0)
    resumed = runner.resume_randomized_experiment(first.artifact_dir, adapter=adapter)
    assert resumed.status.value == "submission_unknown"
    assert resumed.job_ids == ("job-0",)
    assert resumed.values["execution"]["local_replays"][0]["additional_requested_shots"] == 24
    assert adapter.restores == []
    assert replay_adapter_type.submitted_ids == []
