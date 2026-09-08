import json
from dataclasses import replace

import numpy as np
import pytest
from qiskit import QuantumCircuit, qpy

from qudits_on_qubits.experiments import ExperimentPersistenceError, ExperimentValidationError, run_experiment, resume_experiment
from qudits_on_qubits.experiments.block_runner import run_randomized_experiment, recover_randomized_job
from qudits_on_qubits.reference_experiments import get_reference_experiment
from test_randomized_bell_runner import RecordingAdapter, make_spec, all_settings_draw


def test_timeout_resumes_known_job_without_submitting_it_again(tmp_path):
    adapter = RecordingAdapter(fail_result=1)
    first = run_randomized_experiment(make_spec(tmp_path), adapter=adapter, _randbelow=all_settings_draw())
    assert first.status.value == "failed"
    first_raw = (first.artifact_dir / "batches/0000/counts.json").read_bytes()
    assert len(adapter.submissions) == 2
    adapter.fail_result = None
    completed = resume_experiment(first.artifact_dir, adapter=adapter)
    assert completed.status.value == "completed"
    assert adapter.restores == ["job-1"]
    assert len(adapter.submissions) == 4
    assert (first.artifact_dir / "batches/0000/counts.json").read_bytes() == first_raw
    assert resume_experiment(first.artifact_dir, adapter=object()).to_safe_dict() == completed.to_safe_dict()


def test_unknown_submission_is_never_retried_and_prior_raw_stays(tmp_path):
    adapter = RecordingAdapter(fail_submit=1)
    first = run_randomized_experiment(make_spec(tmp_path), adapter=adapter, _randbelow=all_settings_draw())
    assert first.status.value == "submission_unknown"
    raw = (first.artifact_dir / "batches/0000/counts.json").read_bytes()
    again = resume_experiment(first.artifact_dir, adapter=object())
    assert again.status.value == "submission_unknown"
    assert len(adapter.submissions) == 2
    assert (first.artifact_dir / "batches/0000/counts.json").read_bytes() == raw


def test_changed_schedule_blocks_resume_before_backend(tmp_path):
    adapter = RecordingAdapter(fail_submit=0)
    first = run_randomized_experiment(make_spec(tmp_path), adapter=adapter, _randbelow=all_settings_draw())
    file = first.artifact_dir / "schedule.json"
    value = json.loads(file.read_text())
    value["blocks"][0]["settings"] = ["A1", "B2"]
    file.write_text(json.dumps(value))
    with pytest.raises(ExperimentPersistenceError, match="hash"):
        resume_experiment(first.artifact_dir, adapter=object())


def test_new_dispatch_rejects_old_injected_evaluator(tmp_path):
    with pytest.raises(ExperimentValidationError, match="injection"):
        run_experiment(make_spec(tmp_path), adapter=object(), _evaluator=lambda x: 0)


@pytest.mark.parametrize("state", ["two_qutrit", "ghz3", "ame43"])
@pytest.mark.parametrize("encoding", ["canonical", "permuted"])
def test_real_aer_reference_state_all_scenarios_and_encodings(tmp_path, state, encoding):
    n = {"two_qutrit":9, "ghz3":18, "ame43":36}[state]
    spec = make_spec(tmp_path, state, draws=n, cap=n, shots=512, limit=7)
    ref = get_reference_experiment(state)
    e = np.eye(4, dtype=complex)[:, [0,1,2] if encoding == "canonical" else [2,0,3]]
    embedding = np.ones((1,1), complex)
    for _ in range(ref.state.num_parties): embedding = np.kron(embedding, e)
    physical = embedding @ ref.state.statevector()
    circuit = QuantumCircuit(ref.state.num_parties*2)
    circuit.initialize(physical)
    with (spec.basis.directory / "graph_state_direct_basis.qpy").open("wb") as handle: qpy.dump(circuit, handle)
    np.save(spec.basis.directory / "E.npy", e)
    result = run_randomized_experiment(spec, _randbelow=all_settings_draw(state))
    assert result.status.value == "completed", result.values
    expected = np.vdot(ref.state.statevector(), ref.logical_bell_operator() @ ref.state.statevector())
    value = result.values["raw"]
    assert abs(complex(value["real"], value["imag"]) - expected) < 0.65
    assert result.values["conditional"] == result.values["raw"]
    assert result.values["budget"]["completed_shots"] == n*512
    assert resume_experiment(result.artifact_dir, adapter=object()).to_safe_dict() == result.to_safe_dict()
    request_files = sorted(result.artifact_dir.glob("batches/*/request.json"))
    seeds = [json.loads(p.read_text())["seed_simulator"] for p in request_files]
    assert len(set(seeds)) == len(seeds)


def test_unknown_recovery_requires_provider_proof_and_never_submits(tmp_path):
    class VerifyingAdapter(RecordingAdapter):
        def restore_job(self, job_id, *, circuit_count=None, shots=None, circuits=None):
            from qudits_on_qubits.experiments.backends.base import SubmittedJob
            job = SubmittedJob(job_id, object(), self.identity, circuit_count, shots)
            self.jobs[job_id] = (job, circuits)
            return job
        def verify_restored_job(self, job, *, circuits, shots):
            assert job.job_id == "job-5"
            assert len(circuits) == 3 and shots == 8
    adapter = VerifyingAdapter(fail_submit=0)
    first = run_randomized_experiment(make_spec(tmp_path), adapter=adapter, _randbelow=all_settings_draw())
    attached = recover_randomized_job(first.artifact_dir, batch_index=0, job_id="job-5", adapter=adapter)
    assert attached.job_ids == ("job-5",)
    assert len(adapter.submissions) == 1
    completed = resume_experiment(first.artifact_dir, adapter=adapter)
    assert completed.status.value == "completed"
    assert len(adapter.submissions) == 4


def test_explicit_aer_replay_preserves_original_attempt_and_saved_schedule(tmp_path, monkeypatch):
    from qudits_on_qubits.experiments.backends.aer import AerAdapter
    from qudits_on_qubits.experiments import replay_randomized_aer_batch
    def lost_result(self, submitted, timeout=None): raise TimeoutError("simulated lost local handle")
    with monkeypatch.context() as patch:
        patch.setattr(AerAdapter, "result", lost_result)
        first = run_randomized_experiment(make_spec(tmp_path), _randbelow=all_settings_draw())
    assert first.status.value == "failed"
    original = (first.artifact_dir / "batches/0000/submission.json").read_bytes()
    schedule = (first.artifact_dir / "schedule.json").read_bytes()
    result = replay_randomized_aer_batch(first.artifact_dir, batch_index=0)
    assert result.status.value == "completed"
    assert (first.artifact_dir / "batches/0000/submission.json").read_bytes() == original
    assert (first.artifact_dir / "schedule.json").read_bytes() == schedule
    assert result.job_ids[0] == first.job_ids[0]
    assert result.values["execution"]["local_replays"][0]["additional_requested_shots"] == 24
    assert resume_experiment(first.artifact_dir, adapter=object()).to_safe_dict() == result.to_safe_dict()
    with pytest.raises(ExperimentValidationError, match="lost original"):
        replay_randomized_aer_batch(first.artifact_dir, batch_index=0)
