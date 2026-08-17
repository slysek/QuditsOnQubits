from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

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
from qudits_on_qubits.experiments.store import ExperimentStore
from qudits_on_qubits.experiments.errors import (
    BackendCompatibilityError,
    ExperimentPersistenceError,
    ExperimentValidationError,
    JobSubmissionError,
)


class InterruptThenRestoreAdapter:
    def __init__(self, *, interrupt=False):
        self.identity = BackendIdentity("custom", "resume-target")
        self.interrupt = interrupt
        self.submit_calls = 0
        self.restore_calls = 0
        self.restored = None

    def resolve(self):
        return self.identity

    def capabilities(self):
        return BackendCapabilities(False, True)

    def metadata(self):
        return {"identity": self.identity.to_safe_dict()}

    def availability(self):
        return Availability(True)

    def preflight(self, circuits, shots):
        return None

    def compile(self, circuits, config):
        return CompiledBatch(tuple(circuits), self.identity)

    def submit(self, circuits, shots, options=None):
        self.submit_calls += 1
        return SubmittedJob("durable-job", object(), self.identity, len(circuits), shots)

    def result(self, submitted, timeout=None):
        if self.interrupt:
            raise KeyboardInterrupt
        return ExecutionResult(
            tuple({"0": submitted.shots} for _ in range(submitted.circuit_count)),
            submitted.job_id,
            self.identity,
        )

    def restore_job(self, job_id, *, circuit_count=None, shots=None):
        self.restore_calls += 1
        self.restored = (job_id, circuit_count, shots)
        return SubmittedJob(job_id, object(), self.identity, circuit_count, shots)


def _spec(tmp_path):
    return ExperimentSpec(
        state="two_qutrit",
        basis=PathBasis(tmp_path / "unused"),
        backend=CustomBackend(object(), identity="resume-target", supports_resume=True),
        shots=7,
        bootstrap=BootstrapConfig(samples=2),
        retry=RetryConfig(max_attempts=2, initial_delay=0.01, max_delay=0.02),
        output_root=tmp_path / "runs",
    )


def _patch_preparation(monkeypatch):
    source = QuantumCircuit(1)
    logical = QuantumCircuit(1, 1)
    logical.measure(0, 0)
    setting = ("A0",)
    metadata = {
        "setting_by_circuit_index": [setting],
        "terms": [],
        "qutrit_bit_indices_by_setting": {setting: [(0, 0)]},
        "physical_to_logical_outcome_map": {"0": 0, "1": 1},
        "d": 3,
    }
    artifacts = SimpleNamespace(
        state_circuit=source,
        encoding=[[1.0]],
        source_hashes={"state": "a", "encoding": "b"},
        provenance={"kind": "test"},
    )
    prepared = SimpleNamespace(circuits=(logical,), metadata=metadata)
    import qudits_on_qubits.experiments.runner as runner

    monkeypatch.setattr(runner, "load_basis_artifacts", lambda *_args, **_kwargs: artifacts)
    monkeypatch.setattr(runner, "prepare_measurements", lambda _artifacts: prepared)


def test_interrupt_after_job_id_leaves_submitted_and_resume_restores_without_submit(
    tmp_path, monkeypatch
):
    from qudits_on_qubits.experiments.runner import resume_experiment, run_experiment

    _patch_preparation(monkeypatch)
    spec = _spec(tmp_path)
    first = InterruptThenRestoreAdapter(interrupt=True)
    with pytest.raises(KeyboardInterrupt) as caught:
        run_experiment(
            spec,
            adapter=first,
            _clock=lambda: datetime(2026, 8, 17, tzinfo=timezone.utc),
            _sleep=lambda _delay: None,
            _evaluator=lambda _counts: 1 + 0j,
        )
    run = caught.value.__qoq_artifact_dir__
    interrupted = json.loads((run / "experiment.json").read_text(encoding="utf-8"))

    assert interrupted["status"] == "submitted"
    assert interrupted["jobs"]["1"]["job_id"] == "durable-job"

    resumed_adapter = InterruptThenRestoreAdapter()
    result = resume_experiment(
        run,
        adapter=resumed_adapter,
        spec=spec,
        _clock=lambda: datetime(2026, 8, 17, 1, tzinfo=timezone.utc),
        _sleep=lambda _delay: None,
        _evaluator=lambda _counts: 1 + 0j,
    )

    assert result.status is ExperimentStatus.COMPLETED
    assert resumed_adapter.restore_calls == 1
    assert resumed_adapter.submit_calls == 0
    assert resumed_adapter.restored == ("durable-job", 1, 7)
    assert result.job_ids == ("durable-job",)


def test_completed_resume_is_idempotent_without_adapter_calls(tmp_path, monkeypatch):
    from qudits_on_qubits.experiments.runner import resume_experiment, run_experiment

    _patch_preparation(monkeypatch)
    spec = _spec(tmp_path)
    adapter = InterruptThenRestoreAdapter()
    completed = run_experiment(
        spec,
        adapter=adapter,
        _sleep=lambda _delay: None,
        _evaluator=lambda _counts: 1 + 0j,
    )

    result = resume_experiment(completed.artifact_dir, adapter=object())

    assert result.status is ExperimentStatus.COMPLETED
    assert result.to_safe_dict() == completed.to_safe_dict()


def test_resume_identity_mismatch_is_terminal_and_never_restores_or_submits(
    tmp_path, monkeypatch
):
    from qudits_on_qubits.experiments.runner import resume_experiment, run_experiment

    _patch_preparation(monkeypatch)
    spec = _spec(tmp_path)
    first = InterruptThenRestoreAdapter(interrupt=True)
    with pytest.raises(KeyboardInterrupt) as caught:
        run_experiment(
            spec,
            adapter=first,
            _sleep=lambda _delay: None,
            _evaluator=lambda _counts: 1 + 0j,
        )
    run = caught.value.__qoq_artifact_dir__
    mismatch = InterruptThenRestoreAdapter()
    mismatch.identity = BackendIdentity("custom", "different-target")

    with pytest.raises(BackendCompatibilityError, match="identity"):
        resume_experiment(run, adapter=mismatch, spec=spec)

    document = json.loads((run / "experiment.json").read_text(encoding="utf-8"))
    assert document["status"] == "failed"
    assert document["failure"]["stage"] == "resume"
    assert mismatch.restore_calls == 0
    assert mismatch.submit_calls == 0


def test_completed_resume_rejects_corrupt_source_encoding_before_adapter_calls(
    tmp_path, monkeypatch
):
    from qudits_on_qubits.experiments.runner import resume_experiment, run_experiment

    _patch_preparation(monkeypatch)
    completed = run_experiment(
        _spec(tmp_path),
        adapter=InterruptThenRestoreAdapter(),
        _sleep=lambda _delay: None,
        _evaluator=lambda _counts: 1 + 0j,
    )
    (completed.artifact_dir / "source-encoding.json").write_text(
        '{"corrupt":true}\n', encoding="utf-8"
    )

    with pytest.raises(ExperimentPersistenceError, match="encoding.*hash"):
        resume_experiment(completed.artifact_dir, adapter=object())


def test_resume_wraps_missing_run_path_as_typed_persistence_error(tmp_path):
    from qudits_on_qubits.experiments.runner import resume_experiment

    with pytest.raises(ExperimentPersistenceError, match="run directory"):
        resume_experiment(tmp_path / "missing" / "run")


def test_resume_mixed_saved_known_and_missing_zne_factors_uses_each_safe_path_once(
    tmp_path, monkeypatch
):
    from qudits_on_qubits.experiments.runner import resume_experiment, run_experiment

    _patch_preparation(monkeypatch)
    base = _spec(tmp_path)
    spec = replace(base, mitigation=MitigationConfig(zne=True, zne_factors=(5, 1, 3)))

    class InterruptSecondFactor(InterruptThenRestoreAdapter):
        def result(self, submitted, timeout=None):
            if submitted.job_id == "durable-job-2":
                raise KeyboardInterrupt
            return ExecutionResult(
                tuple({"0": submitted.shots} for _ in range(submitted.circuit_count)),
                submitted.job_id,
                self.identity,
            )

        def submit(self, circuits, shots, options=None):
            self.submit_calls += 1
            return SubmittedJob(
                f"durable-job-{self.submit_calls}",
                object(),
                self.identity,
                len(circuits),
                shots,
            )

    class PureZNE:
        def extrapolate(self, factors, values):
            return values[0]

    first = InterruptSecondFactor()
    with pytest.raises(KeyboardInterrupt) as caught:
        run_experiment(
            spec,
            adapter=first,
            _sleep=lambda _delay: None,
            _zne_strategy=PureZNE(),
            _evaluator=lambda _counts: 1 + 0j,
        )
    run = caught.value.__qoq_artifact_dir__
    interrupted = json.loads((run / "experiment.json").read_text(encoding="utf-8"))
    assert set(interrupted["counts"]) == {"1"}
    assert interrupted["jobs"]["3"]["job_id"] == "durable-job-2"
    assert interrupted["jobs"]["5"]["job_id"] is None

    resumed = InterruptSecondFactor()
    resumed.result = lambda submitted, timeout=None: ExecutionResult(
        tuple({"0": submitted.shots} for _ in range(submitted.circuit_count)),
        submitted.job_id,
        resumed.identity,
    )
    result = resume_experiment(
        run,
        adapter=resumed,
        spec=spec,
        _sleep=lambda _delay: None,
        _zne_strategy=PureZNE(),
        _evaluator=lambda _counts: 1 + 0j,
    )
    final = json.loads((run / "experiment.json").read_text(encoding="utf-8"))

    assert result.status is ExperimentStatus.COMPLETED
    assert resumed.restore_calls == 1
    assert resumed.submit_calls == 1
    assert set(final["counts"]) == {"1", "3", "5"}


def test_resume_missing_factor_ambiguous_submit_remains_submission_unknown(
    tmp_path, monkeypatch
):
    from qudits_on_qubits.experiments.runner import resume_experiment, run_experiment

    _patch_preparation(monkeypatch)
    spec = _spec(tmp_path)

    class InterruptPreflight(InterruptThenRestoreAdapter):
        def preflight(self, circuits, shots):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt) as caught:
        run_experiment(
            spec,
            adapter=InterruptPreflight(),
            _sleep=lambda _delay: None,
            _evaluator=lambda _counts: 1 + 0j,
        )
    run = caught.value.__qoq_artifact_dir__

    class AmbiguousResume(InterruptThenRestoreAdapter):
        def __init__(self, run):
            super().__init__()
            self.run = run
            self.checkpoint_seen = False

        def submit(self, circuits, shots, options=None):
            self.submit_calls += 1
            document = json.loads((self.run / "experiment.json").read_text(encoding="utf-8"))
            self.checkpoint_seen = (
                document["status"] == "submission_unknown"
                and document["jobs"]["1"]["status"] == "submission_unknown"
                and document["jobs"]["1"]["job_id"] is None
            )
            raise RuntimeError("resume-provider-sensitive-detail")

    resumed = AmbiguousResume(run)
    with pytest.raises(JobSubmissionError):
        resume_experiment(
            run,
            adapter=resumed,
            spec=spec,
            _sleep=lambda _delay: None,
            _evaluator=lambda _counts: 1 + 0j,
        )
    document = json.loads((run / "experiment.json").read_text(encoding="utf-8"))

    assert resumed.submit_calls == 1
    assert resumed.checkpoint_seen
    assert document["status"] == "submission_unknown"
    assert "resume-provider" not in repr(document)


def test_resume_known_calibration_restores_once_without_duplicate_submit(
    tmp_path, monkeypatch
):
    from qudits_on_qubits.experiments.runner import resume_experiment, run_experiment

    _patch_preparation(monkeypatch)
    base = _spec(tmp_path)
    spec = replace(base, mitigation=MitigationConfig(readout=True))

    class CalibrationInterrupt(InterruptThenRestoreAdapter):
        def result(self, submitted, timeout=None):
            if submitted.circuit_count == 2:
                raise KeyboardInterrupt
            return super().result(submitted, timeout)

    class CalibrationResume(InterruptThenRestoreAdapter):
        def result(self, submitted, timeout=None):
            if submitted.circuit_count == 2:
                return ExecutionResult(
                    ({"0": submitted.shots}, {"1": submitted.shots}),
                    submitted.job_id,
                    self.identity,
                )
            return super().result(submitted, timeout)

    class PureReadout:
        def build_context(self, calibration):
            return None

        def resample_calibration(self, calibration, rng):
            return None

        def apply(self, counts_by_setting, context):
            return {
                setting: {outcome: count / sum(counts.values()) for outcome, count in counts.items()}
                for setting, counts in counts_by_setting.items()
            }

    with pytest.raises(KeyboardInterrupt) as caught:
        run_experiment(
            spec,
            adapter=CalibrationInterrupt(),
            _sleep=lambda _delay: None,
            _readout_strategy=PureReadout(),
            _evaluator=lambda _counts: 1 + 0j,
        )
    run = caught.value.__qoq_artifact_dir__
    store = ExperimentStore(spec.output_root)
    interrupted = store.read_experiment(run)
    assert interrupted["jobs"]["calibration"]["job_id"] == "durable-job"
    assert "job_id" not in interrupted["calibration"]

    counts_path = store.write_counts(run, 1, {("A0",): {"0": spec.shots}})
    interrupted["counts"]["1"] = {
        "artifact": counts_path.name,
        "sha256": hashlib.sha256(counts_path.read_bytes()).hexdigest(),
        "settings": [("A0",)],
    }
    interrupted["jobs"]["1"]["status"] = "completed"
    store.write_experiment(run, interrupted)
    resumed = CalibrationResume()
    result = resume_experiment(
        run,
        adapter=resumed,
        spec=spec,
        _sleep=lambda _delay: None,
        _readout_strategy=PureReadout(),
        _evaluator=lambda _counts: 1 + 0j,
    )
    document = json.loads((run / "experiment.json").read_text(encoding="utf-8"))

    assert result.status is ExperimentStatus.COMPLETED
    assert resumed.restore_calls == 1
    assert resumed.submit_calls == 0
    assert document["calibration"]["status"] == "completed"
    assert document["counts"]["1"]["artifact"] == "counts-factor-1.json"


def test_resume_rejects_disagreeing_legacy_calibration_job_id_without_backend_job_calls(
    tmp_path, monkeypatch
):
    from qudits_on_qubits.experiments.runner import resume_experiment, run_experiment

    _patch_preparation(monkeypatch)
    base = _spec(tmp_path)
    spec = replace(base, mitigation=MitigationConfig(readout=True))

    class CalibrationInterrupt(InterruptThenRestoreAdapter):
        def result(self, submitted, timeout=None):
            if submitted.circuit_count == 2:
                raise KeyboardInterrupt
            return super().result(submitted, timeout)

    class PureReadout:
        def build_context(self, calibration):
            return None

        def resample_calibration(self, calibration, rng):
            return None

        def apply(self, counts_by_setting, context):
            return counts_by_setting

    with pytest.raises(KeyboardInterrupt) as caught:
        run_experiment(
            spec,
            adapter=CalibrationInterrupt(),
            _sleep=lambda _delay: None,
            _readout_strategy=PureReadout(),
            _evaluator=lambda _counts: 1 + 0j,
        )
    run = caught.value.__qoq_artifact_dir__
    store = ExperimentStore(spec.output_root)
    interrupted = store.read_experiment(run)
    interrupted["calibration"]["job_id"] = "disagreeing-legacy-job"
    store.write_experiment(run, interrupted)
    resumed = InterruptThenRestoreAdapter()

    with pytest.raises(ExperimentPersistenceError, match="calibration job ID"):
        resume_experiment(
            run,
            adapter=resumed,
            spec=spec,
            _sleep=lambda _delay: None,
            _readout_strategy=PureReadout(),
            _evaluator=lambda _counts: 1 + 0j,
        )

    assert resumed.restore_calls == 0
    assert resumed.submit_calls == 0


@pytest.mark.parametrize("timeout", [0, -1, True, float("inf")])
def test_completed_resume_still_validates_timeout(tmp_path, monkeypatch, timeout):
    from qudits_on_qubits.experiments.runner import resume_experiment, run_experiment

    _patch_preparation(monkeypatch)
    completed = run_experiment(
        _spec(tmp_path),
        adapter=InterruptThenRestoreAdapter(),
        _sleep=lambda _delay: None,
        _evaluator=lambda _counts: 1 + 0j,
    )

    with pytest.raises(ExperimentValidationError, match="timeout"):
        resume_experiment(completed.artifact_dir, timeout=timeout)
