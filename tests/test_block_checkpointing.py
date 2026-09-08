"""Fault-injection regressions at schema 4 artifact boundaries."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from qudits_on_qubits.experiments import ExperimentPersistenceError
from qudits_on_qubits.experiments.block_runner import (
    resume_randomized_experiment,
    run_randomized_experiment,
)
from qudits_on_qubits.experiments.store import ExperimentStore
from test_randomized_bell_runner import RecordingAdapter, all_settings_draw, make_spec


def only_run(spec):
    return next(Path(spec.output_root).rglob("experiment.json")).parent


@pytest.mark.parametrize("boundary,expected_submissions,expected_restores", [
    ("batches/0000/submission.json", 4, ["job-0"]),
    ("batches/0000/counts.json", 4, []),
    ("batches/0000/receipt.json", 4, []),
])
def test_committed_raw_and_job_id_survive_next_manifest_write_failure(
    tmp_path, monkeypatch, boundary, expected_submissions, expected_restores,
):
    spec = make_spec(tmp_path)
    adapter = RecordingAdapter()
    write = ExperimentStore.write_plain_json
    failed = False

    def fail_after_write(store, run, filename, value):
        nonlocal failed
        result = write(store, run, filename, value)
        if str(filename).replace("\\", "/") == boundary and not failed:
            failed = True
            raise ExperimentPersistenceError("injected failure after durable artifact")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(ExperimentStore, "write_plain_json", fail_after_write)
        with pytest.raises(ExperimentPersistenceError):
            run_randomized_experiment(spec, adapter=adapter, _randbelow=all_settings_draw())
    run = only_run(spec)
    assert (run / boundary).is_file()
    result = resume_randomized_experiment(run, adapter=adapter)
    assert result.status.value == "completed"
    assert len(adapter.submissions) == expected_submissions
    assert adapter.restores == expected_restores
    assert result.values["budget"]["completed_shots"] == 96
    assert resume_randomized_experiment(run, adapter=object()).status.value == "completed"


def test_failure_after_attempt_marker_does_not_submit_or_retry(tmp_path, monkeypatch):
    spec = make_spec(tmp_path)
    adapter = RecordingAdapter()
    exclusive = ExperimentStore.write_exclusive_json

    def fail_after_marker(store, run, filename, value):
        result = exclusive(store, run, filename, value)
        assert result
        raise ExperimentPersistenceError("injected failure after attempt marker")

    with monkeypatch.context() as patch:
        patch.setattr(ExperimentStore, "write_exclusive_json", fail_after_marker)
        with pytest.raises(ExperimentPersistenceError):
            run_randomized_experiment(spec, adapter=adapter, _randbelow=all_settings_draw())
    result = resume_randomized_experiment(only_run(spec), adapter=adapter)
    assert result.status.value == "submission_unknown"
    assert adapter.submissions == []


def test_runtime_preflight_failure_preserves_preparation_and_completed_raw(tmp_path):
    class InterruptedAdapter(RecordingAdapter):
        interrupted = True

        def preflight(self, circuits, shots):
            super().preflight(circuits, shots)
            if self.interrupted and len(self.submissions) == 1:
                raise RuntimeError("transient backend unavailable before second batch")

    spec = make_spec(tmp_path)
    adapter = InterruptedAdapter()
    partial = run_randomized_experiment(spec, adapter=adapter, _randbelow=all_settings_draw())
    assert (partial.artifact_dir / "batches/0000/counts.json").is_file()
    assert partial.status.value == "failed"
    assert partial.values["preparation"]["status"] == "complete"
    adapter.interrupted = False
    completed = resume_randomized_experiment(partial.artifact_dir, adapter=adapter)
    assert completed.status.value == "completed"
    assert len(adapter.submissions) == 4


def test_reanalysis_commit_failure_remains_recoverable_from_raw_offline(tmp_path, monkeypatch):
    spec = make_spec(tmp_path)
    adapter = RecordingAdapter(fail_result=1)
    partial = run_randomized_experiment(spec, adapter=adapter, _randbelow=all_settings_draw())
    assert partial.values["analysis"]["status"] == "partial"
    assert partial.values["budget"]["completed_shots"] == 24
    adapter.fail_result = None
    write = ExperimentStore.write_plain_json
    failed = False

    def fail_after_analysis(store, run, filename, value):
        nonlocal failed
        result = write(store, run, filename, value)
        if str(filename).replace("\\", "/").startswith("analysis/") and not failed:
            failed = True
            raise ExperimentPersistenceError("injected interruption after newer analysis artifact")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(ExperimentStore, "write_plain_json", fail_after_analysis)
        with pytest.raises(ExperimentPersistenceError):
            resume_randomized_experiment(partial.artifact_dir, adapter=adapter)
    assert len(adapter.submissions) == 4
    completed = resume_randomized_experiment(partial.artifact_dir, adapter=object())
    assert completed.status.value == "completed"
    assert completed.values["budget"]["completed_shots"] == 96


def test_analysis_exception_keeps_completed_raw_for_offline_resume(tmp_path, monkeypatch):
    import qudits_on_qubits.experiments.block_runner as runner

    adapter = RecordingAdapter()

    def fail_analysis(*args, **kwargs):
        raise RuntimeError("injected numerical analysis exception")

    with monkeypatch.context() as patch:
        patch.setattr(runner, "evaluate_blocks", fail_analysis)
        partial = run_randomized_experiment(make_spec(tmp_path), adapter=adapter,
                                            _randbelow=all_settings_draw())
    assert partial.status.value == "postprocessing"
    assert len(adapter.submissions) == 4
    result = resume_randomized_experiment(partial.artifact_dir, adapter=object())
    assert result.status.value == "completed"
    assert result.values["budget"]["completed_shots"] == 96


def test_completed_state_cannot_hide_a_missing_raw_batch(tmp_path):
    adapter = RecordingAdapter()
    result = run_randomized_experiment(make_spec(tmp_path), adapter=adapter,
                                       _randbelow=all_settings_draw())
    path = result.artifact_dir / "experiment.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    # Exercise schema consistency, not just a bad hash. Losing an index entry
    # must not make completed raw evidence optional to the validator.
    document["artifacts"].pop("batches/0000/counts.json")
    ExperimentStore(result.artifact_dir.parent).write_plain_json(result.artifact_dir, "experiment.json", document)
    (result.artifact_dir / "batches/0000/counts.json").unlink()
    with pytest.raises(ExperimentPersistenceError):
        resume_randomized_experiment(result.artifact_dir, adapter=object())


def test_mismatched_submit_descriptor_cannot_be_hidden_by_restore_arguments(tmp_path):
    class MismatchedDescriptorAdapter(RecordingAdapter):
        def submit(self, circuits, shots, options=None):
            # Some restore APIs fill circuit_count/shots from the caller's
            # request. The saved original descriptor must retain this mismatch.
            original = super().submit(circuits, shots, options)
            return replace(original, shots=shots+1)

    adapter = MismatchedDescriptorAdapter()
    initial = run_randomized_experiment(make_spec(tmp_path), adapter=adapter,
                                         _randbelow=all_settings_draw())
    assert initial.status.value == "submission_unknown"
    saved = json.loads((initial.artifact_dir / "batches/0000/submission.json").read_text(encoding="utf-8"))
    assert saved["job_id"] == "job-0"
    assert '"shots": 9' in json.dumps(saved)
    resumed = resume_randomized_experiment(initial.artifact_dir, adapter=adapter)
    assert resumed.status.value != "completed"
    assert len(adapter.submissions) == 1
    assert not (initial.artifact_dir / "batches/0000/counts.json").exists()


@pytest.mark.parametrize("where", ["submitted_job", "result_identity"])
def test_unsafe_provider_metadata_cannot_discard_known_job_or_raw_counts(tmp_path, where):
    class ExtraMetadataAdapter(RecordingAdapter):
        def submit(self, circuits, shots, options=None):
            job = super().submit(circuits, shots, options)
            if where == "submitted_job":
                return replace(job, metadata={"api_token": "synthetic-secret-never-persist"})
            return job

        def result(self, job, timeout=None):
            result = super().result(job, timeout)
            if where == "result_identity":
                identity = replace(result.target_identity,
                                   metadata={"api_token": "synthetic-secret-never-persist"})
                return replace(result, target_identity=identity)
            return result

    adapter = ExtraMetadataAdapter()
    result = run_randomized_experiment(make_spec(tmp_path), adapter=adapter,
                                       _randbelow=all_settings_draw())
    assert result.status.value == "completed"
    assert result.values["budget"]["completed_shots"] == 96
    assert len(adapter.submissions) == 4
    for path in result.artifact_dir.rglob("*.json"):
        assert "synthetic-secret-never-persist" not in path.read_text(encoding="utf-8")
