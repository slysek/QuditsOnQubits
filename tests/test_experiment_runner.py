from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
import traceback

import pytest
from qiskit import QuantumCircuit

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
sys.modules.pop("qudits_on_qubits", None)

from qudits_on_qubits.experiments.backends import (
    Availability,
    BackendCapabilities,
    BackendIdentity,
    CompiledBatch,
    ExecutionResult,
    SubmittedJob,
)
from qudits_on_qubits.experiments.models import (
    BootstrapConfig,
    CustomBackend,
    ExperimentSpec,
    ExperimentStatus,
    PathBasis,
    RetryConfig,
    MitigationConfig,
)
from qudits_on_qubits.experiments.errors import JobResultError, JobSubmissionError
from qudits_on_qubits.experiments.errors import (
    BackendCompatibilityError,
    BackendUnavailableError,
)


class RecordingAdapter:
    def __init__(self, *, result_errors=()):
        self.identity = BackendIdentity("custom", "target")
        self.calls = []
        self.submit_calls = 0
        self.restore_calls = 0
        self._result_errors = list(result_errors)

    def resolve(self):
        self.calls.append("resolve")
        return self.identity

    def capabilities(self):
        self.calls.append("capabilities")
        return BackendCapabilities(local=False, supports_resume=True)

    def metadata(self):
        self.calls.append("metadata")
        return {"safe": "metadata"}

    def availability(self):
        self.calls.append("availability")
        return Availability(True)

    def preflight(self, circuits, shots):
        self.calls.append(("preflight", circuits, shots))

    def compile(self, circuits, config):
        self.calls.append(("compile", circuits, config))
        return CompiledBatch(tuple(circuits), self.identity)

    def submit(self, circuits, shots, options=None):
        self.submit_calls += 1
        self.calls.append(("submit", circuits, shots, options))
        return SubmittedJob(
            f"job-{self.submit_calls}", object(), self.identity, len(circuits), shots
        )

    def result(self, submitted, timeout=None):
        self.calls.append(("result", submitted, timeout))
        if self._result_errors:
            raise self._result_errors.pop(0)
        return ExecutionResult(
            tuple({"0": submitted.shots} for _ in range(submitted.circuit_count)),
            submitted.job_id,
            self.identity,
            status="done",
        )

    def restore_job(self, job_id, *, circuit_count=None, shots=None):
        self.restore_calls += 1
        self.calls.append(("restore", job_id, circuit_count, shots))
        return SubmittedJob(job_id, object(), self.identity, circuit_count, shots)


@pytest.fixture
def prepared_run(monkeypatch):
    source = QuantumCircuit(1, name="source")
    logical = QuantumCircuit(1, 1, name="logical")
    logical.measure(0, 0)
    settings = [("A0",)]
    metadata = {
        "setting_by_circuit_index": settings,
        "terms": [],
        "qutrit_bit_indices_by_setting": {settings[0]: [(0, 0)]},
        "physical_to_logical_outcome_map": {"0": 0, "1": 1},
        "d": 3,
        "qutrit_qubits": [(0, 0)],
        "candidate": "two_qutrit",
    }

    class Artifacts:
        state_circuit = source
        encoding = [[1.0]]
        source_hashes = {"state": "source-hash", "encoding": "encoding-hash"}
        provenance = {"kind": "test"}

    import qudits_on_qubits.experiments.runner as runner

    monkeypatch.setattr(runner, "load_basis_artifacts", lambda *_args, **_kwargs: Artifacts())
    monkeypatch.setattr(
        runner,
        "prepare_measurements",
        lambda _artifacts: SimpleNamespace(circuits=(logical,), metadata=metadata),
    )
    return source, logical


def make_spec(tmp_path, **kwargs):
    values = {
        "state": "two_qutrit",
        "basis": PathBasis(tmp_path / "unused-basis"),
        "backend": CustomBackend(object(), identity="target", supports_resume=True),
        "shots": 10,
        "bootstrap": BootstrapConfig(samples=2),
        "retry": RetryConfig(max_attempts=3, initial_delay=0.01, max_delay=0.04),
        "output_root": tmp_path / "runs",
    }
    values.update(kwargs)
    return ExperimentSpec(**values)


def test_run_experiment_persists_durable_state_and_exact_compiled_batch(
    tmp_path, prepared_run
):
    from qudits_on_qubits.experiments.runner import run_experiment

    _, logical = prepared_run
    adapter = RecordingAdapter()
    backend_call_counts_during_bootstrap = []

    def evaluator(counts):
        backend_call_counts_during_bootstrap.append(len(adapter.calls))
        return complex(sum(next(iter(counts.values())).values()))

    result = run_experiment(
        make_spec(tmp_path),
        adapter=adapter,
        _clock=lambda: datetime(2026, 8, 17, 12, tzinfo=timezone.utc),
        _sleep=lambda _delay: None,
        _evaluator=evaluator,
    )

    document = __import__("json").loads(
        (result.artifact_dir / "experiment.json").read_text(encoding="utf-8")
    )
    submit = next(call for call in adapter.calls if isinstance(call, tuple) and call[0] == "submit")

    assert result.status is ExperimentStatus.COMPLETED
    assert result.job_ids == ("job-1",)
    assert [entry["status"] for entry in document["status_history"]] == [
        "created",
        "validated",
        "compiled",
        "submission_unknown",
        "submitted",
        "running",
        "postprocessing",
        "completed",
    ]
    assert all(entry["timestamp"].endswith("Z") for entry in document["status_history"])
    assert document["counts"]["1"]["artifact"] == "counts-factor-1.json"
    assert document["result"]["raw"]["estimate"] == {"imag": 0.0, "real": 10.0}
    assert (result.artifact_dir / "compiled-factor-1.qpy").exists()
    assert submit[1][0] is not logical
    assert submit[1] == (logical,)
    assert "handle" not in repr(document)
    assert len(set(backend_call_counts_during_bootstrap)) == 1


def test_experiments_package_exports_runner_functions():
    from qudits_on_qubits.experiments import resume_experiment, run_experiment, run_experiments

    assert callable(run_experiment)
    assert callable(resume_experiment)
    assert callable(run_experiments)


def test_ambiguous_submission_is_never_retried_and_persists_sanitized_unknown(
    tmp_path, prepared_run
):
    from qudits_on_qubits.experiments.runner import run_experiment

    class AmbiguousAdapter(RecordingAdapter):
        def submit(self, circuits, shots, options=None):
            self.submit_calls += 1
            raise RuntimeError

    adapter = AmbiguousAdapter()
    with pytest.raises(JobSubmissionError) as caught:
        run_experiment(
            make_spec(tmp_path),
            adapter=adapter,
            _sleep=lambda _delay: None,
            _evaluator=lambda _counts: 1 + 0j,
        )
    document = __import__("json").loads(
        (caught.value.__qoq_artifact_dir__ / "experiment.json").read_text(encoding="utf-8")
    )

    assert adapter.submit_calls == 1
    assert document["status"] == "submission_unknown"
    assert document["failure"]["stage"] == "submission"
    assert document["failure"]["exception_type"] == "RuntimeError"
    rendered = "".join(traceback.format_exception(caught.type, caught.value, caught.tb))
    assert caught.value.__cause__ is None
    assert "RuntimeError" in rendered


def test_result_retries_use_same_submitted_job_and_exact_exponential_delays(
    tmp_path, prepared_run
):
    from qudits_on_qubits.experiments.runner import run_experiment

    adapter = RecordingAdapter(
        result_errors=(JobResultError("temporary-1"), JobResultError("temporary-2"))
    )
    delays = []
    result = run_experiment(
        make_spec(tmp_path),
        adapter=adapter,
        _sleep=delays.append,
        _evaluator=lambda _counts: 1 + 0j,
    )
    result_calls = [call for call in adapter.calls if isinstance(call, tuple) and call[0] == "result"]
    document = __import__("json").loads(
        (result.artifact_dir / "experiment.json").read_text(encoding="utf-8")
    )

    assert delays == [0.01, 0.02]
    assert len(result_calls) == 3
    assert result_calls[0][1] is result_calls[1][1] is result_calls[2][1]
    assert adapter.submit_calls == 1
    assert [
        attempt["outcome"] for attempt in document["attempts"] if attempt["operation"] == "result"
    ] == ["failed", "failed", "succeeded"]


def test_readout_calibration_is_checkpointed_with_raw_evidence_before_measurements(
    tmp_path, prepared_run
):
    from qudits_on_qubits.experiments.runner import run_experiment

    class ReadoutAdapter(RecordingAdapter):
        def __init__(self, root):
            super().__init__()
            self.root = root
            self.compile_calls = 0
            self.calibration_checkpoint_seen = False

        def compile(self, circuits, config):
            self.compile_calls += 1
            return CompiledBatch(tuple(circuits), self.identity)

        def result(self, submitted, timeout=None):
            if submitted.circuit_count == 2:
                document_path = next(self.root.rglob("experiment.json"))
                document = __import__("json").loads(document_path.read_text(encoding="utf-8"))
                self.calibration_checkpoint_seen = (
                    document["status"] == "submitted"
                    and document["jobs"]["calibration"]["job_id"] == submitted.job_id
                    and (document_path.parent / "readout-calibration-circuits.qpy").exists()
                    and (document_path.parent / "compiled-factor-1.qpy").exists()
                    and (document_path.parent / "compiled-factor-3.qpy").exists()
                )
                return ExecutionResult(
                    ({"0": submitted.shots}, {"1": submitted.shots}),
                    submitted.job_id,
                    self.identity,
                )
            return super().result(submitted, timeout)

    class PureReadout:
        def build_context(self, calibration):
            return calibration.assignment_matrices

        def resample_calibration(self, calibration, rng):
            return calibration.assignment_matrices

        def apply(self, counts_by_setting, context):
            return {
                setting: {
                    outcome: count / sum(counts.values())
                    for outcome, count in counts.items()
                }
                for setting, counts in counts_by_setting.items()
            }

    class PureZNE:
        def extrapolate(self, factors, values):
            return values[0]

    adapter = ReadoutAdapter(tmp_path / "runs")
    result = run_experiment(
        make_spec(
            tmp_path,
            mitigation=MitigationConfig(readout=True, zne=True, zne_factors=(3, 1)),
        ),
        adapter=adapter,
        _sleep=lambda _delay: None,
        _readout_strategy=PureReadout(),
        _zne_strategy=PureZNE(),
        _evaluator=lambda counts: complex(sum(next(iter(counts.values())).values())),
    )
    document = __import__("json").loads(
        (result.artifact_dir / "experiment.json").read_text(encoding="utf-8")
    )
    evidence = __import__("json").loads(
        (result.artifact_dir / "readout-calibration.json").read_text(encoding="utf-8")
    )

    assert adapter.compile_calls == 2
    assert adapter.calibration_checkpoint_seen
    assert evidence["raw_counts"] == [{"0": 10, "1": 0}, {"0": 0, "1": 10}]
    assert evidence["assignment_matrices"] == [[[1.0, 0.0], [0.0, 1.0]]]
    assert document["calibration"]["status"] == "completed"
    assert set(document["result"]) == {
        "raw",
        "readout_mitigated",
        "zne",
        "zne_readout_mitigated",
        "config",
        "diagnostics",
    }


def test_exhausted_known_calibration_result_remains_recoverable(tmp_path, prepared_run):
    from qudits_on_qubits.experiments.runner import run_experiment

    class FailingCalibrationAdapter(RecordingAdapter):
        def __init__(self, root):
            super().__init__()
            self.root = root
            self.unknown_checkpoint_seen = False

        def compile(self, circuits, config):
            return CompiledBatch(tuple(circuits), self.identity)

        def submit(self, circuits, shots, options=None):
            if len(circuits) == 2:
                document_path = next(self.root.rglob("experiment.json"))
                document = __import__("json").loads(document_path.read_text(encoding="utf-8"))
                self.unknown_checkpoint_seen = (
                    document["status"] == "submission_unknown"
                    and document["jobs"]["calibration"]["status"] == "submission_unknown"
                    and document["jobs"]["calibration"]["job_id"] is None
                )
            return super().submit(circuits, shots, options)

        def result(self, submitted, timeout=None):
            if submitted.circuit_count == 2:
                raise JobResultError("calibration still running")
            return super().result(submitted, timeout)

    adapter = FailingCalibrationAdapter(tmp_path / "runs")
    with pytest.raises(JobResultError) as caught:
        run_experiment(
            make_spec(tmp_path, mitigation=MitigationConfig(readout=True)),
            adapter=adapter,
            _sleep=lambda _delay: None,
            _readout_strategy=object(),
            _evaluator=lambda _counts: 1 + 0j,
        )
    document = __import__("json").loads(
        (caught.value.__qoq_artifact_dir__ / "experiment.json").read_text(encoding="utf-8")
    )

    assert document["status"] in {"submitted", "running"}
    assert document["status"] != "failed"
    assert document["jobs"]["calibration"]["job_id"] == "job-1"
    assert adapter.submit_calls == 1
    assert adapter.unknown_checkpoint_seen


def test_submitted_identity_mismatch_keeps_provable_job_and_fails_terminally(
    tmp_path, prepared_run
):
    from qudits_on_qubits.experiments.runner import run_experiment

    class WrongTargetAdapter(RecordingAdapter):
        def submit(self, circuits, shots, options=None):
            self.submit_calls += 1
            return SubmittedJob(
                "wrong-target-job",
                object(),
                BackendIdentity("custom", "other-target"),
                len(circuits),
                shots,
            )

    adapter = WrongTargetAdapter()
    with pytest.raises(BackendCompatibilityError) as caught:
        run_experiment(
            make_spec(tmp_path),
            adapter=adapter,
            _sleep=lambda _delay: None,
            _evaluator=lambda _counts: 1 + 0j,
        )
    document = __import__("json").loads(
        (caught.value.__qoq_artifact_dir__ / "experiment.json").read_text(encoding="utf-8")
    )

    assert adapter.submit_calls == 1
    assert document["status"] == "failed"
    assert document["jobs"]["1"]["job_id"] == "wrong-target-job"
    assert document["jobs"]["1"]["status"] == "incompatible"


def test_availability_and_preflight_retry_with_bounded_independent_backoff(
    tmp_path, prepared_run
):
    from qudits_on_qubits.experiments.runner import run_experiment

    class TransientAdapter(RecordingAdapter):
        def __init__(self):
            super().__init__()
            self.availability_calls = 0
            self.preflight_calls = 0

        def availability(self):
            self.availability_calls += 1
            if self.availability_calls < 3:
                return Availability(False, "queue unavailable")
            return Availability(True)

        def preflight(self, circuits, shots):
            self.preflight_calls += 1
            if self.preflight_calls < 3:
                raise BackendUnavailableError("preflight unavailable")

    adapter = TransientAdapter()
    delays = []
    result = run_experiment(
        make_spec(tmp_path),
        adapter=adapter,
        _sleep=delays.append,
        _evaluator=lambda _counts: 1 + 0j,
    )
    document = __import__("json").loads(
        (result.artifact_dir / "experiment.json").read_text(encoding="utf-8")
    )

    assert delays == [0.01, 0.02, 0.01, 0.02]
    assert adapter.availability_calls == 4
    assert adapter.preflight_calls == 3
    assert [
        attempt["outcome"] for attempt in document["attempts"] if attempt["operation"] == "availability"
    ] == ["failed", "failed", "succeeded"]
    assert [
        attempt["outcome"]
        for attempt in document["attempts"]
        if attempt["operation"] == "preflight-factor-1"
    ] == ["failed", "failed", "succeeded"]


def test_remote_submit_is_pessimistically_unknown_before_provider_call_and_resume_refuses(
    tmp_path, prepared_run
):
    from qudits_on_qubits.experiments.runner import resume_experiment, run_experiment

    class InterruptInsideSubmit(RecordingAdapter):
        def __init__(self, root):
            super().__init__()
            self.root = root
            self.checkpoint_seen = False

        def submit(self, circuits, shots, options=None):
            self.submit_calls += 1
            document_path = next(self.root.rglob("experiment.json"))
            document = __import__("json").loads(document_path.read_text(encoding="utf-8"))
            self.checkpoint_seen = (
                document["status"] == "submission_unknown"
                and document["jobs"]["1"]["status"] == "submission_unknown"
                and document["jobs"]["1"]["job_id"] is None
            )
            raise KeyboardInterrupt

    spec = make_spec(tmp_path)
    adapter = InterruptInsideSubmit(tmp_path / "runs")
    with pytest.raises(KeyboardInterrupt) as caught:
        run_experiment(
            spec,
            adapter=adapter,
            _sleep=lambda _delay: None,
            _evaluator=lambda _counts: 1 + 0j,
        )
    run = caught.value.__qoq_artifact_dir__

    assert adapter.checkpoint_seen
    with pytest.raises(JobSubmissionError, match="unknown"):
        resume_experiment(run, adapter=object(), spec=spec)


def test_remote_nonresumable_backend_fails_before_submit(tmp_path, prepared_run):
    from qudits_on_qubits.experiments.runner import run_experiment

    class UnsafeAdapter(RecordingAdapter):
        def capabilities(self):
            return BackendCapabilities(local=False, supports_resume=False)

    adapter = UnsafeAdapter()
    with pytest.raises(BackendCompatibilityError, match="resume") as caught:
        run_experiment(
            make_spec(tmp_path),
            adapter=adapter,
            _sleep=lambda _delay: None,
            _evaluator=lambda _counts: 1 + 0j,
        )
    document = __import__("json").loads(
        (caught.value.__qoq_artifact_dir__ / "experiment.json").read_text(encoding="utf-8")
    )

    assert adapter.submit_calls == 0
    assert document["status"] == "failed"


@pytest.mark.parametrize(
    "unsafe_metadata",
    [
        "https://guest:placeholder@example.invalid/path",
        "Authorization: Bearer placeholder",
        "api-key placeholder",
        "unsafe\x07control",
    ],
)
def test_unsafe_adapter_metadata_is_rejected_without_echo_or_cause(
    tmp_path, prepared_run, unsafe_metadata
):
    from qudits_on_qubits.experiments.runner import run_experiment

    class UnsafeMetadataAdapter(RecordingAdapter):
        def metadata(self):
            return {"nested": [{"provider": unsafe_metadata}]}

    adapter = UnsafeMetadataAdapter()
    with pytest.raises(BackendCompatibilityError) as caught:
        run_experiment(
            make_spec(tmp_path),
            adapter=adapter,
            _sleep=lambda _delay: None,
            _evaluator=lambda _counts: 1 + 0j,
        )
    document_bytes = (caught.value.__qoq_artifact_dir__ / "experiment.json").read_bytes()
    rendered = "".join(traceback.format_exception(caught.type, caught.value, caught.tb))

    assert adapter.submit_calls == 0
    assert unsafe_metadata.encode() not in document_bytes
    assert unsafe_metadata not in rendered
    assert caught.value.__cause__ is None


@pytest.mark.parametrize("field_name", ["api_key", "access_token"])
def test_credential_named_adapter_metadata_field_is_rejected_before_persistence(
    tmp_path, prepared_run, field_name
):
    from qudits_on_qubits.experiments.runner import run_experiment

    class CredentialFieldAdapter(RecordingAdapter):
        def metadata(self):
            return {"nested": {field_name: "placeholder-value"}}

    adapter = CredentialFieldAdapter()
    with pytest.raises(BackendCompatibilityError) as caught:
        run_experiment(
            make_spec(tmp_path),
            adapter=adapter,
            _sleep=lambda _delay: None,
            _evaluator=lambda _counts: 1 + 0j,
        )
    document_bytes = (caught.value.__qoq_artifact_dir__ / "experiment.json").read_bytes()

    assert adapter.submit_calls == 0
    assert b"placeholder-value" not in document_bytes
    assert caught.value.__cause__ is None


@pytest.mark.parametrize(
    ("unsafe_payload", "sensitive_text"),
    [
        (
            {"provider": "https://user:super-secret@example.invalid/path"},
            "https://user:super-secret@example.invalid/path",
        ),
        ({"provider": "unsafe\x07control"}, "unsafe\x07control"),
        ({"access_token": "super-secret"}, "super-secret"),
    ],
)
def test_initial_document_is_validated_before_first_persistent_write(
    tmp_path, monkeypatch, unsafe_payload, sensitive_text
):
    from qudits_on_qubits.experiments.runner import run_experiment

    original = ExperimentSpec.to_safe_dict
    def unsafe_safe_dict(self):
        payload = original(self)
        payload["nested"] = unsafe_payload
        return payload

    monkeypatch.setattr(ExperimentSpec, "to_safe_dict", unsafe_safe_dict)

    with pytest.raises(BackendCompatibilityError) as caught:
        run_experiment(make_spec(tmp_path), adapter=RecordingAdapter())

    assert list((tmp_path / "runs").rglob("experiment.json")) == []
    assert sensitive_text not in str(caught.value)
    assert caught.value.__cause__ is None


def test_preflight_mutation_cannot_cross_persisted_qpy_submit_boundary(
    tmp_path, prepared_run
):
    from qudits_on_qubits.experiments.runner import run_experiment
    from qudits_on_qubits.experiments.store import ExperimentStore

    _, logical = prepared_run
    logical_before_preflight = logical.copy()

    class MutatingPreflightAdapter(RecordingAdapter):
        def __init__(self):
            super().__init__()
            self.preflight_circuit = None
            self.submitted_circuit = None

        def preflight(self, circuits, shots):
            self.preflight_circuit = circuits[0]
            circuits[0].x(0)

        def submit(self, circuits, shots, options=None):
            self.submitted_circuit = circuits[0]
            return super().submit(circuits, shots, options)

    adapter = MutatingPreflightAdapter()
    result = run_experiment(
        make_spec(tmp_path),
        adapter=adapter,
        _sleep=lambda _delay: None,
        _evaluator=lambda _counts: 1 + 0j,
    )
    document = __import__("json").loads(
        (result.artifact_dir / "experiment.json").read_text(encoding="utf-8")
    )
    persisted = ExperimentStore(tmp_path / "runs").read_circuits(
        result.artifact_dir, "compiled-factor-1.qpy"
    )
    digest = hashlib.sha256((result.artifact_dir / "compiled-factor-1.qpy").read_bytes()).hexdigest()

    assert adapter.submitted_circuit is not adapter.preflight_circuit
    assert adapter.submitted_circuit == persisted[0]
    assert adapter.submitted_circuit == logical_before_preflight
    assert document["circuits"]["factors"]["1"]["sha256"] == digest
