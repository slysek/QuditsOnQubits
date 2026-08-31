from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from types import MappingProxyType, SimpleNamespace
import traceback
from unittest.mock import Mock

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
    AerIdeal,
    BootstrapConfig,
    CustomBackend,
    ExperimentSpec,
    ExperimentStatus,
    IQMHardware,
    IQMQubitSelectorConfig,
    NoisySimulator,
    PathBasis,
    PiastQHardware,
    RetryConfig,
    MitigationConfig,
    TranspilationConfig,
    WorkloadOptimizationConfig,
)
from qudits_on_qubits.experiments.errors import JobResultError, JobSubmissionError
from qudits_on_qubits.experiments.errors import (
    BackendCompatibilityError,
    ExperimentDurabilityError,
    ExperimentPersistenceError,
    ExperimentValidationError,
    OptionalDependencyError,
)
from qudits_on_qubits.experiments.execution import ExecutionMode
from qudits_on_qubits.experiments.mitigation import ReadoutCalibration


class RecordingAdapter:
    def __init__(self, *, result_errors=()):
        self.identity = BackendIdentity("custom", "target")
        self.calls = []
        self.submit_calls = 0
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
        source_paths = {
            "state": Path("source-state.qpy"),
            "encoding": Path("encoding.npy"),
        }
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
        "backend": CustomBackend(
            object(),
            identity="target",
            supports_resume=True,
            execution_mode=ExecutionMode.HARDWARE,
        ),
        "shots": 10,
        "bootstrap": BootstrapConfig(samples=2),
        "retry": RetryConfig(max_attempts=3, initial_delay=0.01, max_delay=0.04),
        "output_root": tmp_path / "runs",
    }
    values.update(kwargs)
    return ExperimentSpec(**values)


def _install_two_setting_workload(monkeypatch):
    import qudits_on_qubits.experiments.runner as runner

    circuits = []
    for index in range(2):
        circuit = QuantumCircuit(2, 2, name=f"logical-{index}")
        circuit.measure((0, 1), (0, 1))
        circuits.append(circuit)
    settings = (("A0",), ("A1",))
    metadata = {
        "setting_by_circuit_index": settings,
        "terms": [],
        "qutrit_bit_indices_by_setting": {
            setting: [(0, 1)] for setting in settings
        },
        "physical_to_logical_outcome_map": {
            "0": 0,
            "1": 1,
            "2": 2,
            "3": None,
        },
        "d": 3,
        "qutrit_qubits": [(0, 1)],
        "candidate": "two_qutrit",
    }
    logical = tuple(circuits)
    monkeypatch.setattr(
        runner,
        "prepare_measurements",
        lambda _artifacts: SimpleNamespace(circuits=logical, metadata=metadata),
    )
    return logical, settings


def _physical_measurement_circuit(
    layout,
    *,
    name,
    cz_count=0,
):
    width = max(layout) + 1
    circuit = QuantumCircuit(width, len(layout), name=name)
    if len(layout) >= 2:
        for _ in range(cz_count):
            circuit.cz(layout[0], layout[1])
    circuit.measure(layout, tuple(range(len(layout))))
    return circuit


class _CandidateAdapter(RecordingAdapter):
    def __init__(self, compiler):
        super().__init__()
        self.compiler = compiler
        self.compile_calls = []

    def compile(self, circuits, config):
        batch = tuple(circuits)
        self.compile_calls.append((batch, config))
        return self.compiler(config, self.identity)


class _SelectorCandidateAdapter(_CandidateAdapter):
    def __init__(self, compiler, selector_result):
        super().__init__(compiler)
        self.identity = BackendIdentity(
            "iqm",
            "garnet",
            provider="iqm",
            version="35",
            metadata={"calibration_set_id": "cal-17"},
        )
        self.selector_result = selector_result
        self.selector_calls = []

    def suggest_layouts(self, circuit, config):
        self.selector_calls.append((circuit, config))
        if isinstance(self.selector_result, BaseException):
            raise self.selector_result
        return self.selector_result


def _selector_payload(config, *, layouts=((2, 3), (4, 5)), costs=(0.01, 0.02)):
    return {
        "provider": "iqm-qubit-selector",
        "version": "1.2.3",
        "configuration": config.to_safe_dict(),
        "layouts": layouts,
        "costs": costs,
    }


def test_run_experiment_submits_compiler_outputs_without_qpy_or_preflight(
    tmp_path, prepared_run, monkeypatch
):
    import qudits_on_qubits.experiments.runner as runner
    from qudits_on_qubits.experiments.store import ExperimentStore

    compiled = QuantumCircuit(1, 1, name="compiled")
    compiled.measure(0, 0)
    adapter = RecordingAdapter()
    adapter.compile = lambda _circuits, config: CompiledBatch(
        (compiled,), adapter.identity, {"transpilation": config.to_safe_dict()}
    )
    adapter.preflight = Mock(side_effect=AssertionError("preflight called"))
    adapter.availability = Mock(side_effect=AssertionError("availability called"))
    adapter.capabilities = Mock(side_effect=AssertionError("capabilities called"))
    adapter.metadata = Mock(side_effect=AssertionError("metadata called"))
    forbidden = Mock(side_effect=AssertionError("durable helper called"))
    for name in (
        "_sha256",
        "_transition",
        "_retry",
        "_persist_prepared",
        "_persist_factor_batches",
        "_execute_measurement_factor",
    ):
        monkeypatch.setattr(runner, name, forbidden)
    monkeypatch.setattr(
        ExperimentStore,
        "write_circuits",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("QPY write")),
    )
    monkeypatch.setattr(
        ExperimentStore,
        "read_circuits",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("QPY read")),
    )

    result = runner.run_experiment(
        make_spec(tmp_path),
        adapter=adapter,
        _evaluator=lambda counts: complex(sum(next(iter(counts.values())).values())),
    )

    submit = next(call for call in adapter.calls if call[0] == "submit")
    assert submit[1][0] is compiled
    adapter.preflight.assert_not_called()
    adapter.availability.assert_not_called()
    adapter.capabilities.assert_not_called()
    adapter.metadata.assert_not_called()
    forbidden.assert_not_called()
    assert result.status is ExperimentStatus.COMPLETED


def test_run_experiment_persists_explicit_bell_semantics_from_default_evaluator(
    tmp_path, prepared_run, monkeypatch
):
    import qudits_on_qubits.experiments.runner as runner

    setting = ("A0",)
    logical = QuantumCircuit(2, 2, name="logical-invalid-codeword")
    logical.measure((0, 1), (0, 1))
    metadata = {
        "setting_by_circuit_index": (setting,),
        "terms": ({"coeff": 1.0, "settings": setting, "powers": (0,)},),
        "qutrit_bit_indices_by_setting": {setting: ((0, 1),)},
        "physical_to_logical_outcome_map": {
            0: 0,
            1: 1,
            2: 2,
            3: None,
        },
        "d": 3,
    }
    monkeypatch.setattr(
        runner,
        "prepare_measurements",
        lambda _artifacts: SimpleNamespace(circuits=(logical,), metadata=metadata),
    )

    class InvalidCodewordAdapter(RecordingAdapter):
        def result(self, submitted, timeout=None):
            self.calls.append(("result", submitted, timeout))
            return ExecutionResult(
                ({"00": 8, "11": 2},),
                submitted.job_id,
                self.identity,
                status="done",
            )

    real_bootstrap = runner.bootstrap_bell_results
    bootstrap = Mock(wraps=real_bootstrap)
    monkeypatch.setattr(runner, "bootstrap_bell_results", bootstrap)

    result = runner.run_experiment(make_spec(tmp_path), adapter=InvalidCodewordAdapter())

    document = json.loads(
        (result.artifact_dir / "experiment.json").read_text(encoding="utf-8")
    )
    expected_keys = {
        "raw",
        "raw_conditional",
        "raw_unconditional",
        "raw_invalid_codeword_rate",
        "raw_invalid_codeword_shots",
        "config",
        "diagnostics",
    }
    assert expected_keys <= set(result.values)
    assert document["result"] == dict(result.values)
    assert result.values["raw"] == result.values["raw_conditional"]
    assert result.values["raw_invalid_codeword_rate"]["estimate"] == pytest.approx(
        0.2
    )
    assert result.values["raw_invalid_codeword_shots"] == {
        "total_shots": 10,
        "accepted_shots": 8,
        "invalid_shots": 2,
    }
    bootstrap.assert_called_once()
    assert bootstrap.call_args.kwargs["_evaluator"] is None
    assert sorted(path.name for path in result.artifact_dir.iterdir()) == [
        "experiment.json"
    ]


def test_interrupted_postprocessing_resumes_exact_direct_result_from_saved_counts(
    tmp_path, prepared_run, monkeypatch
):
    import qudits_on_qubits.experiments.runner as runner

    setting = ("A0",)
    logical = QuantumCircuit(2, 2, name="logical-invalid-codeword")
    logical.measure((0, 1), (0, 1))
    metadata = {
        "setting_by_circuit_index": (setting,),
        "terms": (
            {
                "coeff": 1.0 + 0.0j,
                "settings": setting,
                "powers": (0,),
                "source": "resume-parity",
            },
        ),
        "qutrit_bit_indices_by_setting": {setting: ((0, 1),)},
        "physical_to_logical_outcome_map": {
            0: 0,
            1: 1,
            2: 2,
            3: None,
        },
        "d": 3,
    }
    monkeypatch.setattr(
        runner,
        "prepare_measurements",
        lambda _artifacts: SimpleNamespace(circuits=(logical,), metadata=metadata),
    )

    class InvalidCodewordAdapter(RecordingAdapter):
        def compile(self, _circuits, config):
            compiled = _physical_measurement_circuit(
                config.initial_layout,
                name=f"candidate-{config.seed_transpiler}",
            )
            return CompiledBatch(
                (compiled,),
                self.identity,
                {"transpilation": config.to_safe_dict()},
            )

        def result(self, submitted, timeout=None):
            self.calls.append(("result", submitted, timeout))
            return ExecutionResult(
                ({"00": 8, "11": 2},),
                submitted.job_id,
                self.identity,
                status="done",
            )

    fixed_now = datetime(2026, 8, 27, 12, tzinfo=timezone.utc)
    spec = make_spec(
        tmp_path,
        mitigation=MitigationConfig(
            zne=True,
            zne_factors=(1, 3, 11),
        ),
        workload_optimization=WorkloadOptimizationConfig(
            initial_layouts=((0, 1),),
            seed_transpilers=(3,),
        ),
    )
    direct = runner.run_experiment(
        spec,
        adapter=InvalidCodewordAdapter(),
        _clock=lambda: fixed_now,
    )

    real_bootstrap = runner.bootstrap_bell_results

    def interrupt_after_checkpoint(*_args, **_kwargs):
        checkpoints = []
        for artifact in spec.output_root.glob("**/experiment.json"):
            document = json.loads(artifact.read_text(encoding="utf-8"))
            if document.get("status") == "postprocessing":
                checkpoints.append((artifact.parent, document))
        assert len(checkpoints) == 1
        _run, checkpoint = checkpoints[0]
        assert checkpoint["workload_optimization"]["selected_layout"] == [0, 1]
        assert [job["role"] for job in checkpoint["jobs"]] == [
            "execution",
            "execution",
            "execution",
        ]
        assert [job["factor"] for job in checkpoint["jobs"]] == [1, 3, 11]
        assert checkpoint["jobs"][0]["job_id"] == "job-1"
        assert checkpoint["counts_by_factor"]["1"][0]["counts"] == {
            "00": 8,
            "11": 2,
        }
        assert checkpoint["counts_by_factor"]["11"][0]["counts"] == {
            "00": 8,
            "11": 2,
        }
        assert checkpoint["calibration"] is None
        assert checkpoint["postprocessing_checkpoint"]["version"] == 1
        assert checkpoint["postprocessing_checkpoint"]["readout_strategy_mode"] == "unused"
        assert checkpoint["postprocessing_checkpoint"]["zne_strategy_mode"] == "default"
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "bootstrap_bell_results", interrupt_after_checkpoint)
    with pytest.raises(KeyboardInterrupt):
        runner.run_experiment(
            spec,
            adapter=InvalidCodewordAdapter(),
            _clock=lambda: fixed_now,
        )

    monkeypatch.setattr(runner, "bootstrap_bell_results", real_bootstrap)
    interrupted = next(
        artifact.parent
        for artifact in spec.output_root.glob("**/experiment.json")
        if json.loads(artifact.read_text(encoding="utf-8")).get("status")
        == "postprocessing"
    )
    checkpoint_path = interrupted / "experiment.json"
    original_checkpoint = checkpoint_path.read_text(encoding="utf-8")
    tampered_counts = json.loads(original_checkpoint)
    tampered_counts["counts_by_factor"]["1"][0]["counts"]["00"] = 7
    checkpoint_path.write_text(
        json.dumps(tampered_counts, allow_nan=False),
        encoding="utf-8",
    )
    with pytest.raises(
        ExperimentPersistenceError,
        match="shot totals are inconsistent with the spec",
    ):
        runner.resume_experiment(
            interrupted,
            spec=spec,
            _clock=lambda: fixed_now,
        )
    checkpoint_path.write_text(original_checkpoint, encoding="utf-8")

    missing_jobs = json.loads(original_checkpoint)
    missing_jobs["jobs"] = []
    missing_jobs["job_ids"] = []
    checkpoint_path.write_text(
        json.dumps(missing_jobs, allow_nan=False),
        encoding="utf-8",
    )
    with pytest.raises(
        ExperimentPersistenceError,
        match="jobs are inconsistent",
    ):
        runner.resume_experiment(
            interrupted,
            spec=spec,
            _clock=lambda: fixed_now,
        )
    checkpoint_path.write_text(original_checkpoint, encoding="utf-8")

    wrong_job_factor = json.loads(original_checkpoint)
    wrong_job_factor["jobs"][0]["factor"] = 3
    checkpoint_path.write_text(
        json.dumps(wrong_job_factor, allow_nan=False),
        encoding="utf-8",
    )
    with pytest.raises(
        ExperimentPersistenceError,
        match="job record is invalid",
    ):
        runner.resume_experiment(
            interrupted,
            spec=spec,
            _clock=lambda: fixed_now,
        )
    checkpoint_path.write_text(original_checkpoint, encoding="utf-8")

    wrong_job_shots = json.loads(original_checkpoint)
    wrong_job_shots["jobs"][0]["shots"] = spec.shots - 1
    checkpoint_path.write_text(
        json.dumps(wrong_job_shots, allow_nan=False),
        encoding="utf-8",
    )
    with pytest.raises(
        ExperimentPersistenceError,
        match="job record is invalid",
    ):
        runner.resume_experiment(
            interrupted,
            spec=spec,
            _clock=lambda: fixed_now,
        )
    checkpoint_path.write_text(original_checkpoint, encoding="utf-8")

    wrong_job_circuit_count = json.loads(original_checkpoint)
    wrong_job_circuit_count["jobs"][0]["circuit_count"] = 2
    checkpoint_path.write_text(
        json.dumps(wrong_job_circuit_count, allow_nan=False),
        encoding="utf-8",
    )
    with pytest.raises(
        ExperimentPersistenceError,
        match="job record is invalid",
    ):
        runner.resume_experiment(
            interrupted,
            spec=spec,
            _clock=lambda: fixed_now,
        )
    checkpoint_path.write_text(original_checkpoint, encoding="utf-8")

    with pytest.raises(
        ExperimentValidationError,
        match="injected postprocessing seams",
    ):
        runner.resume_experiment(
            interrupted,
            spec=spec,
            _clock=lambda: fixed_now,
            _zne_strategy=object(),
        )
    with pytest.raises(
        ExperimentValidationError,
        match="injected postprocessing seams",
    ):
        runner.resume_experiment(
            interrupted,
            spec=spec,
            _clock=lambda: fixed_now,
            _evaluator=lambda _counts: 0j,
        )
    resumed = runner.resume_experiment(
        interrupted,
        spec=spec,
        _clock=lambda: fixed_now,
    )

    assert resumed.values.keys() == direct.values.keys()
    assert dict(resumed.values) == dict(direct.values)
    assert resumed.values["raw"] == resumed.values["raw_conditional"]
    assert resumed.values["raw_unconditional"] == direct.values[
        "raw_unconditional"
    ]
    assert resumed.values["raw_invalid_codeword_rate"] == direct.values[
        "raw_invalid_codeword_rate"
    ]
    final_document = json.loads(
        (resumed.artifact_dir / "experiment.json").read_text(encoding="utf-8")
    )
    assert final_document["status"] == "completed"
    assert final_document["result"] == dict(resumed.values)
    assert sorted(path.name for path in resumed.artifact_dir.iterdir()) == [
        "experiment.json"
    ]


def test_injected_strategy_checkpoint_preserves_calibration_but_is_not_resumable(
    tmp_path, prepared_run, monkeypatch
):
    import qudits_on_qubits.experiments.runner as runner

    class PureZNE:
        def extrapolate(self, _factors, values):
            return values[0]

    class CalibrationAdapter(RecordingAdapter):
        def compile_physical(self, circuits, _config):
            return CompiledBatch(tuple(circuits), self.identity)

        def result(self, submitted, timeout=None):
            self.calls.append(("result", submitted, timeout))
            counts = (
                ({"0": submitted.shots}, {"1": submitted.shots})
                if submitted.circuit_count == 2
                else tuple(
                    {"0": submitted.shots}
                    for _ in range(submitted.circuit_count)
                )
            )
            return ExecutionResult(
                counts,
                submitted.job_id,
                self.identity,
                status="done",
            )

    fixed_now = datetime(2026, 8, 27, 12, tzinfo=timezone.utc)
    spec = make_spec(
        tmp_path,
        mitigation=MitigationConfig(readout=True, zne=True, zne_factors=(1, 3)),
    )
    monkeypatch.setattr(
        runner,
        "bootstrap_bell_results",
        Mock(side_effect=KeyboardInterrupt),
    )

    with pytest.raises(KeyboardInterrupt) as interrupted:
        runner.run_experiment(
            spec,
            adapter=CalibrationAdapter(),
            _clock=lambda: fixed_now,
            _readout_strategy=_PureReadout(),
            _zne_strategy=PureZNE(),
        )

    artifact_dir = interrupted.value.__qoq_artifact_dir__
    checkpoint = json.loads(
        (artifact_dir / "experiment.json").read_text(encoding="utf-8")
    )
    assert checkpoint["calibration"]["qubit_mapping"] == [0]
    assert [job["role"] for job in checkpoint["jobs"]] == [
        "calibration",
        "execution",
        "execution",
    ]
    assert [job["factor"] for job in checkpoint["jobs"]] == [None, 1, 3]
    assert checkpoint["postprocessing_checkpoint"]["readout_strategy_mode"] == "injected"
    assert checkpoint["postprocessing_checkpoint"]["zne_strategy_mode"] == "injected"

    checkpoint_path = artifact_dir / "experiment.json"
    original_checkpoint = checkpoint_path.read_text(encoding="utf-8")
    tampered_calibration = json.loads(original_checkpoint)
    tampered_calibration["calibration"]["backend_identity"] = "custom:other"
    checkpoint_path.write_text(
        json.dumps(tampered_calibration, allow_nan=False),
        encoding="utf-8",
    )
    with pytest.raises(
        ExperimentPersistenceError,
        match="calibration does not match the saved backend",
    ):
        runner.resume_experiment(
            artifact_dir,
            spec=spec,
            _readout_strategy=_PureReadout(),
            _zne_strategy=PureZNE(),
        )
    checkpoint_path.write_text(original_checkpoint, encoding="utf-8")

    malformed_calibration = json.loads(original_checkpoint)
    malformed_calibration["calibration"].pop("backend_identity")
    checkpoint_path.write_text(
        json.dumps(malformed_calibration, allow_nan=False),
        encoding="utf-8",
    )
    with pytest.raises(
        ExperimentPersistenceError,
        match="calibration is invalid",
    ):
        runner.resume_experiment(
            artifact_dir,
            spec=spec,
            _readout_strategy=_PureReadout(),
            _zne_strategy=PureZNE(),
        )
    checkpoint_path.write_text(original_checkpoint, encoding="utf-8")

    with pytest.raises(
        ExperimentValidationError,
        match="injected postprocessing seams",
    ):
        runner.resume_experiment(
            artifact_dir,
            spec=spec,
            _readout_strategy=_PureReadout(),
            _zne_strategy=PureZNE(),
        )


def test_run_experiment_keeps_direct_readout_and_zne_in_memory(
    tmp_path, prepared_run, monkeypatch
):
    import qudits_on_qubits.experiments.runner as runner
    from qudits_on_qubits.experiments.store import ExperimentStore

    compiled = QuantumCircuit(1, 1, name="compiled")
    compiled.measure(0, 0)

    class DirectMitigationAdapter(RecordingAdapter):
        def compile(self, circuits, config):
            self.calls.append(("compile", circuits, config))
            return CompiledBatch((compiled,), self.identity)

        def compile_physical(self, circuits, config):
            batch = tuple(circuits)
            self.calls.append(("compile_physical", batch, config))
            return CompiledBatch(batch, self.identity)

        def result(self, submitted, timeout=None):
            self.calls.append(("result", submitted, timeout))
            if submitted.circuit_count == 2:
                counts = ({"0": submitted.shots}, {"1": submitted.shots})
            else:
                counts = ({"0": submitted.shots},)
            return ExecutionResult(
                counts,
                submitted.job_id,
                self.identity,
                status="done",
            )

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

    adapter = DirectMitigationAdapter()
    adapter.preflight = Mock(side_effect=AssertionError("preflight called"))
    monkeypatch.setattr(
        ExperimentStore,
        "write_circuits",
        Mock(side_effect=AssertionError("QPY write")),
    )
    monkeypatch.setattr(
        ExperimentStore,
        "read_circuits",
        Mock(side_effect=AssertionError("QPY read")),
    )

    result = runner.run_experiment(
        make_spec(
            tmp_path,
            mitigation=MitigationConfig(
                readout=True,
                zne=True,
                zne_factors=(1, 3),
            ),
        ),
        adapter=adapter,
        _clock=lambda: datetime(2026, 8, 27, 12, tzinfo=timezone.utc),
        _readout_strategy=PureReadout(),
        _zne_strategy=PureZNE(),
        _evaluator=lambda counts: complex(sum(next(iter(counts.values())).values())),
    )

    submissions = [
        call for call in adapter.calls if isinstance(call, tuple) and call[0] == "submit"
    ]
    assert [len(call[1]) for call in submissions] == [2, 1, 1]
    assert submissions[1][1][0] is compiled
    assert submissions[2][1][0] is not compiled
    assert [call[2] for call in submissions] == [10, 10, 10]
    adapter.preflight.assert_not_called()
    ExperimentStore.write_circuits.assert_not_called()
    ExperimentStore.read_circuits.assert_not_called()
    assert set(result.values) == {
        "raw",
        "readout_mitigated",
        "zne",
        "zne_readout_mitigated",
        "config",
        "diagnostics",
    }


@pytest.mark.parametrize(
    ("backend", "shots", "instances", "message"),
    [
        (
            CustomBackend(
                object(),
                identity="target",
                execution_mode=ExecutionMode.HARDWARE,
            ),
            10,
            2,
            "IQMHardware",
        ),
        (IQMHardware("garnet"), 10, 3, "divisible"),
    ],
)
def test_twirling_validation_fails_before_compile_or_submit(
    tmp_path, prepared_run, backend, shots, instances, message
):
    from qudits_on_qubits.experiments.errors import ExperimentValidationError
    from qudits_on_qubits.experiments.runner import run_experiment

    adapter = RecordingAdapter()
    if isinstance(backend, IQMHardware):
        adapter.identity = BackendIdentity("iqm", backend.device)
    adapter.compile = Mock(side_effect=AssertionError("compile called"))

    with pytest.raises(ExperimentValidationError, match=message):
        run_experiment(
            make_spec(
                tmp_path,
                backend=backend,
                shots=shots,
                mitigation=MitigationConfig(
                    circuit_twirling=True,
                    twirling_instances=instances,
                ),
            ),
            adapter=adapter,
            _evaluator=lambda _counts: 1 + 0j,
        )

    adapter.compile.assert_not_called()
    assert adapter.submit_calls == 0


def test_twirling_transform_propagates_memory_error_before_submit(
    tmp_path, prepared_run
):
    from qudits_on_qubits.experiments.runner import run_experiment

    adapter = RecordingAdapter()
    adapter.identity = BackendIdentity("iqm", "garnet")

    def transform(_circuits, *, instances, seed):
        assert (instances, seed) == (2, None)
        raise MemoryError("twirling exhausted")

    with pytest.raises(MemoryError, match="twirling exhausted"):
        run_experiment(
            make_spec(
                tmp_path,
                backend=IQMHardware("garnet"),
                mitigation=MitigationConfig(
                    circuit_twirling=True,
                    twirling_instances=2,
                ),
            ),
            adapter=adapter,
            _twirling_transform=transform,
            _evaluator=lambda _counts: 1 + 0j,
        )

    assert adapter.submit_calls == 0


def test_twirling_compiles_once_reuses_ensemble_for_zne_and_aggregates_counts(
    tmp_path, prepared_run
):
    from qudits_on_qubits.experiments.mitigation import TwirledBatch
    from qudits_on_qubits.experiments.runner import run_experiment

    compiled = QuantumCircuit(2, 1, name="compiled")
    compiled.cz(0, 1)
    compiled.measure(0, 0)
    variants = []
    for index in range(2):
        variant = compiled.copy(name=f"twirled-{index}")
        variants.append(variant)

    class IQMRecordingAdapter(RecordingAdapter):
        def __init__(self):
            super().__init__()
            self.identity = BackendIdentity("iqm", "garnet")
            self.compile_calls = 0

        def compile(self, circuits, config):
            self.compile_calls += 1
            self.calls.append(("compile", tuple(circuits), config))
            return CompiledBatch(
                (compiled,),
                self.identity,
                {"transpilation": config.to_safe_dict()},
            )

    transformed = []

    def transform(circuits, *, instances, seed):
        transformed.append((tuple(circuits), instances, seed))
        return TwirledBatch(
            circuits=tuple(variants),
            original_indices=(0, 0),
            instance_indices=(0, 1),
            metadata={
                "provider": "iqm-error-reduction-tools",
                "method": "circuit_twirling",
                "readout_strategy": "NONE",
                "instances_per_circuit": 2,
                "seed": 123,
            },
        )

    adapter = IQMRecordingAdapter()
    result = run_experiment(
        make_spec(
            tmp_path,
            backend=IQMHardware("garnet"),
            shots=10,
            mitigation=MitigationConfig(
                circuit_twirling=True,
                twirling_instances=2,
                twirling_seed=123,
                zne=True,
                zne_factors=(1, 3),
            ),
        ),
        adapter=adapter,
        _twirling_transform=transform,
        _evaluator=lambda counts: complex(
            sum(next(iter(counts.values())).values())
        ),
    )

    submissions = [call for call in adapter.calls if call[0] == "submit"]
    assert adapter.compile_calls == 1
    assert transformed == [((compiled,), 2, 123)]
    assert [call[2] for call in submissions] == [5, 5]
    assert submissions[0][1][0] is variants[0]
    assert submissions[0][1][1] is variants[1]
    assert [circuit.count_ops().get("cz", 0) for circuit in submissions[1][1]] == [
        3,
        3,
    ]
    document = json.loads(
        (result.artifact_dir / "experiment.json").read_text(encoding="utf-8")
    )
    assert document["counts_by_factor"] == {
        "1": [{"setting": ["A0"], "counts": {"0": 10}}],
        "3": [{"setting": ["A0"], "counts": {"0": 10}}],
    }


def test_twirling_rejects_inconsistent_variant_mappings_before_submission(
    tmp_path, prepared_run
):
    from qudits_on_qubits.experiments.mitigation import TwirledBatch
    from qudits_on_qubits.experiments.runner import run_experiment

    compiled = QuantumCircuit(2, 1, name="compiled")
    compiled.measure(0, 0)
    variant_zero = compiled.copy(name="twirled-0")
    variant_one = QuantumCircuit(2, 1, name="twirled-1")
    variant_one.measure(1, 0)

    adapter = RecordingAdapter()
    adapter.identity = BackendIdentity("iqm", "garnet")
    adapter.compile = lambda _circuits, _config: CompiledBatch(
        (compiled,), adapter.identity
    )

    def transform(_circuits, *, instances, seed):
        assert (instances, seed) == (2, None)
        return TwirledBatch(
            circuits=(variant_zero, variant_one),
            original_indices=(0, 0),
            instance_indices=(0, 1),
            metadata={
                "provider": "iqm-error-reduction-tools",
                "method": "circuit_twirling",
                "readout_strategy": "NONE",
                "instances_per_circuit": 2,
                "seed": None,
            },
        )

    with pytest.raises(BackendCompatibilityError, match="twirled.*mapping"):
        run_experiment(
            make_spec(
                tmp_path,
                backend=IQMHardware("garnet"),
                mitigation=MitigationConfig(
                    circuit_twirling=True,
                    twirling_instances=2,
                ),
            ),
            adapter=adapter,
            _twirling_transform=transform,
            _evaluator=lambda _counts: 1 + 0j,
        )

    assert adapter.submit_calls == 0
    assert not (tmp_path / "runs").exists()


def test_twirling_readout_and_zne_share_direct_pipeline_and_persist_metadata(
    tmp_path, prepared_run
):
    from qudits_on_qubits.experiments.mitigation import TwirledBatch
    from qudits_on_qubits.experiments.runner import run_experiment

    compiled = QuantumCircuit(2, 1, name="compiled")
    compiled.cz(0, 1)
    compiled.measure(0, 0)
    variants = (
        compiled.copy(name="twirled-0"),
        compiled.copy(name="twirled-1"),
    )

    class CombinedAdapter(RecordingAdapter):
        def __init__(self):
            super().__init__()
            self.identity = BackendIdentity(
                "iqm",
                "garnet",
                metadata={"calibration_set_id": "cal-123"},
            )
            self.compile_calls = 0
            self.compile_physical_calls = 0

        def compile(self, circuits, config):
            self.compile_calls += 1
            return CompiledBatch(
                (compiled,),
                self.identity,
                {"transpilation": config.to_safe_dict()},
            )

        def compile_physical(self, circuits, config):
            self.compile_physical_calls += 1
            batch = tuple(circuits)
            return CompiledBatch(batch, self.identity)

        def result(self, submitted, timeout=None):
            self.calls.append(("result", submitted, timeout))
            if submitted.job_id == "job-1":
                counts = ({"0": submitted.shots}, {"1": submitted.shots})
            else:
                counts = tuple(
                    {"0": submitted.shots}
                    for _ in range(submitted.circuit_count)
                )
            return ExecutionResult(
                counts,
                submitted.job_id,
                self.identity,
                status="done",
            )

    class PureReadout:
        def build_context(self, calibration):
            return calibration.assignment_matrices

        def resample_calibration(self, calibration, rng):
            return calibration.assignment_matrices

        def apply(self, counts_by_setting, _context):
            return {
                setting: {
                    outcome: count / sum(counts.values())
                    for outcome, count in counts.items()
                }
                for setting, counts in counts_by_setting.items()
            }

    class PureZNE:
        def extrapolate(self, _factors, values):
            return values[0]

    transform_calls = []

    def transform(circuits, *, instances, seed):
        transform_calls.append((tuple(circuits), instances, seed))
        return TwirledBatch(
            circuits=variants,
            original_indices=(0, 0),
            instance_indices=(0, 1),
            metadata={
                "provider": "iqm-error-reduction-tools",
                "method": "circuit_twirling",
                "readout_strategy": "NONE",
                "instances_per_circuit": 2,
                "seed": 99,
            },
        )

    adapter = CombinedAdapter()
    result = run_experiment(
        make_spec(
            tmp_path,
            backend=IQMHardware("garnet"),
            shots=10,
            mitigation=MitigationConfig(
                readout=True,
                zne=True,
                zne_factors=(1, 3),
                circuit_twirling=True,
                twirling_instances=2,
                twirling_seed=99,
            ),
        ),
        adapter=adapter,
        _clock=lambda: datetime(2026, 8, 30, 12, tzinfo=timezone.utc),
        _twirling_transform=transform,
        _readout_strategy=PureReadout(),
        _zne_strategy=PureZNE(),
        _evaluator=lambda _counts: 1 + 0j,
    )

    submissions = [call for call in adapter.calls if call[0] == "submit"]
    document = json.loads(
        (result.artifact_dir / "experiment.json").read_text(encoding="utf-8")
    )
    assert adapter.compile_calls == 1
    assert adapter.compile_physical_calls == 1
    assert transform_calls == [((compiled,), 2, 99)]
    assert [call[2] for call in submissions] == [10, 5, 5]
    assert set(result.values) == {
        "raw",
        "readout_mitigated",
        "zne",
        "zne_readout_mitigated",
        "config",
        "diagnostics",
    }
    assert document["twirling"] == {
        "provider": "iqm-error-reduction-tools",
        "method": "circuit_twirling",
        "readout_strategy": "NONE",
        "instances_per_circuit": 2,
        "seed": 99,
        "shots_per_instance": 5,
        "total_shots_per_circuit": 10,
    }
    assert document["counts_by_factor"]["1"][0]["counts"] == {"0": 10}
    assert document["counts_by_factor"]["3"][0]["counts"] == {"0": 10}
    assert sorted(path.name for path in result.artifact_dir.iterdir()) == [
        "experiment.json"
    ]


def test_run_experiment_writes_one_final_json_only_after_success(
    tmp_path, prepared_run
):
    from qudits_on_qubits.experiments.runner import run_experiment

    result = run_experiment(
        make_spec(tmp_path),
        adapter=RecordingAdapter(),
        _clock=lambda: datetime(2026, 8, 27, 12, tzinfo=timezone.utc),
        _evaluator=lambda counts: complex(
            sum(next(iter(counts.values())).values())
        ),
    )

    files = sorted(path.name for path in result.artifact_dir.iterdir())
    document = json.loads(
        (result.artifact_dir / "experiment.json").read_text(encoding="utf-8")
    )

    assert files == ["experiment.json"]
    assert document["schema_version"] == 3
    assert "sha256" not in json.dumps(document)
    assert document["status"] == "completed"
    assert document["counts_by_factor"]["1"][0]["setting"] == ["A0"]
    assert document["twirling"] is None
    assert "workload_optimization" not in document
    assert "workload_optimization" not in document["spec"]


def test_run_experiment_serializes_nested_frozen_provenance_as_plain_json(
    tmp_path, prepared_run, monkeypatch
):
    import qudits_on_qubits.experiments.runner as runner

    artifacts = runner.load_basis_artifacts(None, None, None)
    artifacts.provenance = MappingProxyType(
        {
            "state": "two_qutrit",
            "basis": MappingProxyType(
                {"kind": "path", "directory": str(tmp_path / "basis")}
            ),
        }
    )
    monkeypatch.setattr(
        runner, "load_basis_artifacts", lambda *_args, **_kwargs: artifacts
    )

    result = runner.run_experiment(
        make_spec(tmp_path),
        adapter=RecordingAdapter(),
        _evaluator=lambda _counts: 1 + 0j,
    )
    document = json.loads(
        (result.artifact_dir / "experiment.json").read_text(encoding="utf-8")
    )

    assert document["source"]["provenance"]["basis"] == {
        "kind": "path",
        "directory": str(tmp_path / "basis"),
    }


def test_run_experiment_serializes_transpilation_tuples_as_plain_json_arrays(
    tmp_path, prepared_run
):
    from qudits_on_qubits.experiments.runner import run_experiment

    adapter = RecordingAdapter()

    def compile_with_metadata(circuits, config):
        return CompiledBatch(
            tuple(circuits),
            adapter.identity,
            {"transpilation": config.to_safe_dict()},
        )

    adapter.compile = compile_with_metadata
    result = run_experiment(
        make_spec(
            tmp_path,
            transpilation=TranspilationConfig(initial_layout=(0,)),
        ),
        adapter=adapter,
        _evaluator=lambda _counts: 1 + 0j,
    )
    document = json.loads(
        (result.artifact_dir / "experiment.json").read_text(encoding="utf-8")
    )

    assert document["transpilation"]["initial_layout"] == [0]


def test_run_experiment_rejects_reserved_json_tag_before_creating_run(
    tmp_path, prepared_run
):
    from qudits_on_qubits.experiments.runner import run_experiment

    adapter = RecordingAdapter()
    with pytest.raises(ExperimentPersistenceError, match="reserved JSON key"):
        run_experiment(
            make_spec(tmp_path, tags={"__qoq_type__": "tuple"}),
            adapter=adapter,
            _evaluator=lambda _counts: 1 + 0j,
        )

    assert adapter.submit_calls == 0
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize(
    "transpilation_metadata",
    [
        {"api_key": "do-not-persist"},
        {"__qoq_type__": "tuple"},
    ],
    ids=("credential", "reserved-key"),
)
def test_run_experiment_rejects_invalid_compiled_metadata_before_submit(
    tmp_path, prepared_run, transpilation_metadata
):
    from qudits_on_qubits.experiments.runner import run_experiment

    adapter = RecordingAdapter()

    def compile_invalid_metadata(circuits, _config):
        return CompiledBatch(
            tuple(circuits),
            adapter.identity,
            {"transpilation": transpilation_metadata},
        )

    adapter.compile = compile_invalid_metadata
    with pytest.raises(ExperimentPersistenceError):
        run_experiment(
            make_spec(tmp_path),
            adapter=adapter,
            _evaluator=lambda _counts: 1 + 0j,
        )

    assert adapter.submit_calls == 0
    assert not (tmp_path / "runs").exists()


def test_compile_failure_is_sanitized_without_secret_cause_or_artifact(
    tmp_path, prepared_run
):
    from qudits_on_qubits.experiments.runner import run_experiment

    sensitive_message = "token" + "=compile-secret"

    class FailingCompileAdapter(RecordingAdapter):
        def compile(self, circuits, config):
            raise RuntimeError(sensitive_message)

    with pytest.raises(BackendCompatibilityError) as caught:
        run_experiment(
            make_spec(tmp_path),
            adapter=FailingCompileAdapter(),
            _evaluator=lambda _counts: 1 + 0j,
        )

    assert str(caught.value) == "adapter compilation failed (RuntimeError)"
    assert caught.value.__cause__ is None
    assert sensitive_message not in "".join(traceback.format_exception(caught.value))
    assert not (tmp_path / "runs").exists()


def test_compile_physical_failure_is_sanitized_without_secret_or_artifact(
    tmp_path, prepared_run
):
    from qudits_on_qubits.experiments.runner import run_experiment

    sensitive_message = "token" + "=physical-compile-secret"

    class FailingPhysicalCompileAdapter(RecordingAdapter):
        def compile_physical(self, circuits, config):
            raise RuntimeError(sensitive_message)

    with pytest.raises(BackendCompatibilityError) as caught:
        run_experiment(
            make_spec(tmp_path, mitigation=MitigationConfig(readout=True)),
            adapter=FailingPhysicalCompileAdapter(),
            _readout_strategy=object(),
            _evaluator=lambda _counts: 1 + 0j,
        )

    assert str(caught.value) == "adapter physical compilation failed (RuntimeError)"
    assert caught.value.__cause__ is None
    assert sensitive_message not in "".join(traceback.format_exception(caught.value))
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("signal", [KeyboardInterrupt, SystemExit, MemoryError])
def test_compile_preserves_process_control_exceptions(
    tmp_path, prepared_run, signal
):
    from qudits_on_qubits.experiments.runner import run_experiment

    class InterruptingCompileAdapter(RecordingAdapter):
        def compile(self, circuits, config):
            raise signal()

    with pytest.raises(signal):
        run_experiment(
            make_spec(tmp_path),
            adapter=InterruptingCompileAdapter(),
            _evaluator=lambda _counts: 1 + 0j,
        )

    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize(
    ("stage", "error_type", "message"),
    [
        (
            "compile",
            BackendCompatibilityError,
            "adapter compilation failed (PoisonedProviderError)",
        ),
        (
            "submit",
            JobSubmissionError,
            "job submission failed (PoisonedProviderError)",
        ),
        (
            "result",
            JobResultError,
            "job result retrieval failed (PoisonedProviderError)",
        ),
    ],
)
def test_provider_exception_type_property_failure_is_sanitized(
    tmp_path, prepared_run, stage, error_type, message
):
    from qudits_on_qubits.experiments.runner import run_experiment

    sensitive_message = "token" + "=provider-property-secret"

    class PoisonedProviderError(RuntimeError):
        @property
        def provider_exception_type(self):
            raise RuntimeError(sensitive_message)

    class PoisonedAdapter(RecordingAdapter):
        def compile(self, circuits, config):
            if stage == "compile":
                raise PoisonedProviderError("provider failed")
            return super().compile(circuits, config)

        def submit(self, circuits, shots, options=None):
            if stage == "submit":
                raise PoisonedProviderError("provider failed")
            return super().submit(circuits, shots, options)

        def result(self, submitted, timeout=None):
            if stage == "result":
                raise PoisonedProviderError("provider failed")
            return super().result(submitted, timeout)

    with pytest.raises(error_type) as caught:
        run_experiment(
            make_spec(tmp_path),
            adapter=PoisonedAdapter(),
            _evaluator=lambda _counts: 1 + 0j,
        )

    assert str(caught.value) == message
    assert caught.value.__cause__ is None
    assert sensitive_message not in "".join(traceback.format_exception(caught.value))
    assert not (tmp_path / "runs").exists()


def test_oversized_final_document_leaves_resumable_postprocessing_checkpoint(
    tmp_path, prepared_run, monkeypatch
):
    import qudits_on_qubits.experiments.runner as runner
    import qudits_on_qubits.experiments.store as store_module

    class OversizedResult:
        def to_safe_dict(self):
            return {
                "raw": {"padding": "x" * (store_module._MAX_PLAIN_JSON_STRING_LENGTH + 1)}
            }

    monkeypatch.setattr(
        runner,
        "bootstrap_bell_results",
        lambda *_args, **_kwargs: OversizedResult(),
    )

    with pytest.raises(ExperimentPersistenceError, match="JSON complexity limit"):
        runner.run_experiment(
            make_spec(tmp_path),
            adapter=RecordingAdapter(),
            _evaluator=lambda _counts: 1 + 0j,
        )

    artifacts = list((tmp_path / "runs").glob("**/experiment.json"))
    assert len(artifacts) == 1
    checkpoint = json.loads(artifacts[0].read_text(encoding="utf-8"))
    assert checkpoint["status"] == "postprocessing"
    assert checkpoint["counts_by_factor"]["1"][0]["counts"] == {"0": 10}


def test_submit_failure_writes_no_final_experiment_json(tmp_path, prepared_run):
    from qudits_on_qubits.experiments.runner import run_experiment

    class FailingSubmitAdapter(RecordingAdapter):
        def submit(self, circuits, shots, options=None):
            raise RuntimeError("submit failed")

    with pytest.raises(JobSubmissionError, match="job submission failed"):
        run_experiment(
            make_spec(tmp_path),
            adapter=FailingSubmitAdapter(),
            _evaluator=lambda _counts: 1 + 0j,
        )

    assert not list((tmp_path / "runs").glob("**/experiment.json"))
    assert not (tmp_path / "runs").exists()


def test_result_failure_writes_no_final_experiment_json(tmp_path, prepared_run):
    from qudits_on_qubits.experiments.runner import run_experiment

    class FailingResultAdapter(RecordingAdapter):
        def result(self, submitted, timeout=None):
            raise RuntimeError("result failed")

    with pytest.raises(JobResultError, match="job result retrieval failed"):
        run_experiment(
            make_spec(tmp_path),
            adapter=FailingResultAdapter(),
            _evaluator=lambda _counts: 1 + 0j,
        )

    assert not list((tmp_path / "runs").glob("**/experiment.json"))
    assert not (tmp_path / "runs").exists()


def test_evaluator_failure_leaves_unresumable_postprocessing_checkpoint(
    tmp_path, prepared_run
):
    import qudits_on_qubits.experiments.runner as runner

    def fail_evaluator(_counts):
        raise RuntimeError("evaluator failed")

    spec = make_spec(tmp_path)
    with pytest.raises(JobResultError, match="bootstrap Bell evaluation failed"):
        runner.run_experiment(
            spec,
            adapter=RecordingAdapter(),
            _evaluator=fail_evaluator,
        )

    artifacts = list((tmp_path / "runs").glob("**/experiment.json"))
    assert len(artifacts) == 1
    checkpoint = json.loads(artifacts[0].read_text(encoding="utf-8"))
    assert checkpoint["status"] == "postprocessing"
    assert checkpoint["postprocessing_checkpoint"]["evaluator_mode"] == "injected"
    with pytest.raises(
        ExperimentValidationError,
        match="injected postprocessing seams",
    ):
        runner.resume_experiment(
            artifacts[0].parent,
            spec=spec,
            _evaluator=fail_evaluator,
        )


def test_bootstrap_failure_leaves_resumable_postprocessing_checkpoint(
    tmp_path, prepared_run, monkeypatch
):
    import qudits_on_qubits.experiments.runner as runner

    monkeypatch.setattr(
        runner,
        "bootstrap_bell_results",
        Mock(side_effect=RuntimeError("bootstrap failed")),
    )

    with pytest.raises(RuntimeError, match="bootstrap failed"):
        runner.run_experiment(
            make_spec(tmp_path),
            adapter=RecordingAdapter(),
            _evaluator=lambda _counts: 1 + 0j,
        )

    artifacts = list((tmp_path / "runs").glob("**/experiment.json"))
    assert len(artifacts) == 1
    checkpoint = json.loads(artifacts[0].read_text(encoding="utf-8"))
    assert checkpoint["status"] == "postprocessing"
    assert checkpoint["counts_by_factor"]["1"][0]["setting"] == ["A0"]


def _readout_calibration(
    *,
    backend_identity="custom:target",
    calibration_id="target",
    qubit_mapping=(0,),
    timestamp=datetime(2026, 8, 27, 12, tzinfo=timezone.utc),
):
    raw_counts = tuple(
        counts
        for _qubit in qubit_mapping
        for counts in ({"0": 10, "1": 0}, {"0": 0, "1": 10})
    )
    assignment_matrices = tuple(
        ((1.0, 0.0), (0.0, 1.0)) for _qubit in qubit_mapping
    )
    return ReadoutCalibration(
        backend_identity=backend_identity,
        calibration_id=calibration_id,
        qubit_mapping=qubit_mapping,
        timestamp=timestamp,
        shots=10,
        raw_counts=raw_counts,
        assignment_matrices=assignment_matrices,
    )


class _PureReadout:
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


@pytest.mark.parametrize(
    "calibration",
    [
        _readout_calibration(backend_identity="custom:other"),
        _readout_calibration(calibration_id="other-set"),
        _readout_calibration(qubit_mapping=(0, 1)),
        _readout_calibration(
            timestamp=datetime(2026, 8, 25, 12, tzinfo=timezone.utc)
        ),
    ],
    ids=("backend", "calibration-set", "qubit-union", "stale"),
)
def test_run_experiment_rejects_incompatible_injected_readout_calibration(
    tmp_path, prepared_run, calibration
):
    from qudits_on_qubits.experiments.runner import run_experiment

    adapter = RecordingAdapter()
    with pytest.raises(BackendCompatibilityError, match="not valid for this target"):
        run_experiment(
            make_spec(tmp_path, mitigation=MitigationConfig(readout=True)),
            adapter=adapter,
            readout_calibration=calibration,
            _clock=lambda: datetime(2026, 8, 27, 12, tzinfo=timezone.utc),
            _readout_strategy=_PureReadout(),
            _evaluator=lambda _counts: 1 + 0j,
        )

    assert adapter.submit_calls == 0

def test_run_experiment_force_recalibration_ignores_injected_calibration(
    tmp_path, prepared_run
):
    from qudits_on_qubits.experiments.runner import run_experiment

    class ForceCalibrationAdapter(RecordingAdapter):
        def compile_physical(self, circuits, config):
            return CompiledBatch(tuple(circuits), self.identity)

        def result(self, submitted, timeout=None):
            self.calls.append(("result", submitted, timeout))
            counts = (
                ({"0": submitted.shots}, {"1": submitted.shots})
                if submitted.circuit_count == 2
                else ({"0": submitted.shots},)
            )
            return ExecutionResult(
                counts,
                submitted.job_id,
                self.identity,
                status="done",
            )

    adapter = ForceCalibrationAdapter()
    run_experiment(
        make_spec(
            tmp_path,
            mitigation=MitigationConfig(readout=True, force_recalibration=True),
        ),
        adapter=adapter,
        readout_calibration=_readout_calibration(),
        _clock=lambda: datetime(2026, 8, 27, 12, tzinfo=timezone.utc),
        _readout_strategy=_PureReadout(),
        _evaluator=lambda _counts: 1 + 0j,
    )

    submissions = [
        call for call in adapter.calls if isinstance(call, tuple) and call[0] == "submit"
    ]
    assert [len(call[1]) for call in submissions] == [2, 1]


def test_fresh_readout_calibration_falls_back_to_adapter_compile_contract(
    tmp_path, prepared_run
):
    from qudits_on_qubits.experiments.runner import run_experiment

    class CompileOnlyAdapter(RecordingAdapter):
        def result(self, submitted, timeout=None):
            self.calls.append(("result", submitted, timeout))
            counts = (
                ({"0": submitted.shots}, {"1": submitted.shots})
                if submitted.circuit_count == 2
                else ({"0": submitted.shots},)
            )
            return ExecutionResult(
                counts,
                submitted.job_id,
                self.identity,
                status="done",
            )

    adapter = CompileOnlyAdapter()
    result = run_experiment(
        make_spec(tmp_path, mitigation=MitigationConfig(readout=True)),
        adapter=adapter,
        _readout_strategy=_PureReadout(),
        _evaluator=lambda _counts: 1 + 0j,
    )

    compile_calls = [
        call for call in adapter.calls if isinstance(call, tuple) and call[0] == "compile"
    ]
    assert [len(call[1]) for call in compile_calls] == [1, 2]
    assert result.status is ExperimentStatus.COMPLETED


def test_fresh_readout_compile_fallback_matches_real_non_iqm_adapter_contracts(
    monkeypatch
):
    from qudits_on_qubits.experiments.backends import (
        AerAdapter,
        CustomBackendAdapter,
        NoisyAerAdapter,
        PiastQAdapter,
    )
    import qudits_on_qubits.experiments.backends.aer as aer_module
    import qudits_on_qubits.experiments.backends.custom as custom_module
    import qudits_on_qubits.experiments.backends.piastq as piastq_module
    from qudits_on_qubits.experiments.models import AerIdeal, PiastQHardware
    from qudits_on_qubits.experiments.runner import _compile_with_adapter

    class Backend:
        name = "contract-target"
        local = True

        def run(self, *_args, **_kwargs):
            return object()

    backend = Backend()

    class Client:
        def __init__(self, **_kwargs):
            self.backend = backend

    class Sampler:
        def __init__(self, *_args, **_kwargs):
            pass

    identity = BackendIdentity("noisy", "contract-noise")
    adapters = (
        AerAdapter(AerIdeal(), simulator=backend),
        NoisyAerAdapter(
            NoisySimulator(
                noise_model=object(),
                target_backend=backend,
                identity="contract-noise",
            ),
            simulator=backend,
            target_backend=backend,
            identity=identity,
        ),
        PiastQAdapter(
            PiastQHardware("managed", "team"),
            client_type=Client,
            sampler_type=Sampler,
            env_loader=lambda _path: {},
        ),
        CustomBackendAdapter(
            CustomBackend(
                backend,
                identity="contract-target",
                execution_mode=ExecutionMode.HARDWARE,
            )
        ),
    )
    layouts = []

    def identity_transpile(circuits, **options):
        layouts.append(options.get("initial_layout"))
        return list(circuits)

    monkeypatch.setattr(
        aer_module,
        "transpile",
        identity_transpile,
    )
    monkeypatch.setattr(
        custom_module,
        "transpile",
        identity_transpile,
    )
    monkeypatch.setattr(
        piastq_module,
        "transpile",
        identity_transpile,
    )
    source = (QuantumCircuit(20), QuantumCircuit(20))

    for adapter in adapters:
        assert not callable(getattr(adapter, "compile_physical", None))
        compiled = _compile_with_adapter(
            adapter,
            source,
            TranspilationConfig(),
            physical=True,
        )
        assert compiled.circuits == source
        assert compiled.target_identity == adapter.resolve()

    assert layouts == [list(range(20))] * 4


@pytest.mark.parametrize(
    "failure_point",
    ("temporary", "write", "fsync", "file-replace", "directory-publish"),
)
def test_final_publication_failures_leave_no_run_or_staging_artifact(
    tmp_path, prepared_run, monkeypatch, failure_point
):
    import qudits_on_qubits.experiments.store as store_module
    from qudits_on_qubits.experiments.runner import run_experiment
    from qudits_on_qubits.experiments.store import ExperimentStore

    def injected(*_args, **_kwargs):
        raise OSError(f"injected {failure_point} failure")

    if failure_point == "temporary":
        monkeypatch.setattr(store_module.tempfile, "NamedTemporaryFile", injected)
    elif failure_point == "write":
        monkeypatch.setattr(ExperimentStore, "write_plain_json", injected)
    elif failure_point == "fsync":
        monkeypatch.setattr(store_module.os, "fsync", injected)
    elif failure_point == "file-replace":
        monkeypatch.setattr(store_module, "_replace_with_directory_handle", injected)
    else:
        monkeypatch.setattr(store_module, "_publish_directory", injected, raising=False)

    with pytest.raises((ExperimentPersistenceError, OSError)):
        run_experiment(
            make_spec(tmp_path),
            adapter=RecordingAdapter(),
            _evaluator=lambda _counts: 1 + 0j,
        )

    output_root = tmp_path / "runs"
    if output_root.exists():
        for date_directory in output_root.iterdir():
            assert date_directory.is_dir()
            assert list(date_directory.iterdir()) == []


def test_post_publication_directory_fsync_failure_reports_durability_uncertain_run(
    tmp_path, prepared_run, monkeypatch
):
    import qudits_on_qubits.experiments.store as store_module
    from qudits_on_qubits.experiments.runner import run_experiment

    real_fsync_directory = store_module._fsync_directory

    def fail_only_after_publish(directory):
        if directory.parent == tmp_path / "runs" and any(
            not child.name.startswith(".staging-") for child in directory.iterdir()
        ):
            raise ExperimentPersistenceError("injected post-publish fsync failure")
        real_fsync_directory(directory)

    monkeypatch.setattr(store_module, "_fsync_directory", fail_only_after_publish)

    with pytest.raises(ExperimentDurabilityError) as caught:
        run_experiment(
            make_spec(tmp_path),
            adapter=RecordingAdapter(),
            _evaluator=lambda _counts: 1 + 0j,
        )

    assert caught.value.published is True
    assert (caught.value.artifact_dir / "experiment.json").is_file()
    assert not any(
        path.name.startswith(".staging-")
        for path in (tmp_path / "runs").rglob("*")
    )


def test_directory_publish_rename_then_raise_returns_existing_completed_run(
    tmp_path, prepared_run, monkeypatch
):
    import qudits_on_qubits.experiments.store as store_module
    from qudits_on_qubits.experiments.runner import run_experiment

    real_publish = store_module._publish_directory

    def rename_then_raise(staging, destination):
        real_publish(staging, destination)
        raise OSError("injected error after rename")

    monkeypatch.setattr(store_module, "_publish_directory", rename_then_raise)

    result = run_experiment(
        make_spec(tmp_path),
        adapter=RecordingAdapter(),
        _evaluator=lambda _counts: 1 + 0j,
    )

    assert result.status is ExperimentStatus.COMPLETED
    assert (result.artifact_dir / "experiment.json").is_file()
    assert not any(
        path.name.startswith(".staging-")
        for path in (tmp_path / "runs").rglob("*")
    )


def test_direct_result_retries_same_submitted_job_without_resubmission(
    tmp_path, prepared_run
):
    from qudits_on_qubits.experiments.runner import run_experiment

    adapter = RecordingAdapter(result_errors=(JobResultError("transient"),))
    delays = []

    result = run_experiment(
        make_spec(tmp_path),
        adapter=adapter,
        _sleep=delays.append,
        _evaluator=lambda _counts: 1 + 0j,
    )

    result_calls = [
        call for call in adapter.calls if isinstance(call, tuple) and call[0] == "result"
    ]
    assert result.status is ExperimentStatus.COMPLETED
    assert adapter.submit_calls == 1
    assert len(result_calls) == 2
    assert result_calls[0][1] is result_calls[1][1]
    assert delays == [0.01]


def test_direct_result_does_not_retry_or_wrap_deterministic_compatibility_error(
    tmp_path, prepared_run
):
    from qudits_on_qubits.experiments.runner import run_experiment

    adapter = RecordingAdapter(
        result_errors=(BackendCompatibilityError("deterministic mismatch"),)
    )
    delays = []

    with pytest.raises(BackendCompatibilityError, match="deterministic mismatch"):
        run_experiment(
            make_spec(tmp_path),
            adapter=adapter,
            _sleep=delays.append,
            _evaluator=lambda _counts: 1 + 0j,
        )

    result_calls = [
        call for call in adapter.calls if isinstance(call, tuple) and call[0] == "result"
    ]
    assert len(result_calls) == 1
    assert delays == []


@pytest.mark.parametrize(
    "signal",
    (KeyboardInterrupt, SystemExit, GeneratorExit, MemoryError),
)
def test_direct_result_process_and_memory_signals_propagate_without_retry(
    tmp_path, prepared_run, signal
):
    from qudits_on_qubits.experiments.runner import run_experiment

    adapter = RecordingAdapter(result_errors=(signal("stop"),))
    delays = []

    with pytest.raises(signal, match="stop"):
        run_experiment(
            make_spec(tmp_path),
            adapter=adapter,
            _sleep=delays.append,
            _evaluator=lambda _counts: 1 + 0j,
        )

    result_calls = [
        call for call in adapter.calls if isinstance(call, tuple) and call[0] == "result"
    ]
    assert len(result_calls) == 1
    assert delays == []


def test_run_experiment_sanitizes_direct_submission_failure(
    tmp_path, prepared_run
):
    from qudits_on_qubits.experiments.runner import run_experiment

    sensitive_text = "token=do-not-leak"

    class ProviderRejectingAdapter(RecordingAdapter):
        def submit(self, circuits, shots, options=None):
            error = JobSubmissionError(sensitive_text)
            error.provider_exception_type = "HTTPError"
            raise error

    with pytest.raises(JobSubmissionError) as caught:
        run_experiment(
            make_spec(tmp_path),
            adapter=ProviderRejectingAdapter(),
            _evaluator=lambda _counts: 1 + 0j,
        )

    assert sensitive_text not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.provider_exception_type == "HTTPError"


def test_run_experiment_sanitizes_direct_result_failure(tmp_path, prepared_run):
    from qudits_on_qubits.experiments.runner import run_experiment

    sensitive_text = "Authorization: Bearer do-not-leak"

    class ProviderFailingAdapter(RecordingAdapter):
        def result(self, submitted, timeout=None):
            error = JobResultError(sensitive_text)
            error.provider_exception_type = "HTTPError"
            raise error

    with pytest.raises(JobResultError) as caught:
        run_experiment(
            make_spec(tmp_path),
            adapter=ProviderFailingAdapter(),
            _evaluator=lambda _counts: 1 + 0j,
        )

    assert sensitive_text not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.provider_exception_type == "HTTPError"


def test_run_experiment_rejects_unsafe_resolved_identity_before_execution(
    tmp_path, prepared_run
):
    from qudits_on_qubits.experiments.runner import run_experiment

    sensitive_text = "do-not-persist"
    adapter = RecordingAdapter()
    adapter.identity = BackendIdentity(
        "custom", "target", metadata={"access_token": sensitive_text}
    )

    with pytest.raises(BackendCompatibilityError) as caught:
        run_experiment(
            make_spec(tmp_path),
            adapter=adapter,
            _evaluator=lambda _counts: 1 + 0j,
        )

    assert sensitive_text not in str(caught.value)
    assert adapter.submit_calls == 0
    assert not list((tmp_path / "runs").glob("**/experiment.json"))


def test_run_experiment_rejects_unsafe_job_id_before_persistence(
    tmp_path, prepared_run
):
    from qudits_on_qubits.experiments.runner import run_experiment

    sensitive_job_id = "token:do-not-persist"

    class UnsafeJobIdAdapter(RecordingAdapter):
        def submit(self, circuits, shots, options=None):
            self.submit_calls += 1
            return SubmittedJob(
                sensitive_job_id,
                object(),
                self.identity,
                len(circuits),
                shots,
            )

    adapter = UnsafeJobIdAdapter()
    with pytest.raises(BackendCompatibilityError) as caught:
        run_experiment(
            make_spec(tmp_path),
            adapter=adapter,
            _evaluator=lambda _counts: 1 + 0j,
        )

    assert sensitive_job_id not in str(caught.value)
    assert adapter.submit_calls == 1
    assert not list((tmp_path / "runs").glob("**/experiment.json"))


@pytest.mark.parametrize(
    ("backend", "resolved_identity"),
    [
        (IQMHardware("garnet"), BackendIdentity("iqm", "crystal")),
        (
            CustomBackend(
                object(),
                identity="expected-custom",
                execution_mode=ExecutionMode.HARDWARE,
            ),
            BackendIdentity("custom", "wrong-custom"),
        ),
        (
            NoisySimulator(source=object(), identity="expected-noisy"),
            BackendIdentity("noisy", "wrong-noisy"),
        ),
    ],
)
def test_run_experiment_rejects_resolved_backend_name_mismatch_before_submit(
    tmp_path, prepared_run, backend, resolved_identity
):
    from qudits_on_qubits.experiments.runner import run_experiment

    adapter = RecordingAdapter()
    adapter.identity = resolved_identity

    with pytest.raises(
        BackendCompatibilityError,
        match="resolved adapter identity does not match configured backend",
    ):
        run_experiment(
            make_spec(tmp_path, backend=backend),
            adapter=adapter,
            _evaluator=lambda _counts: 1 + 0j,
        )

    assert adapter.submit_calls == 0
    assert not list((tmp_path / "runs").glob("**/experiment.json"))


def test_run_experiment_rejects_compiled_circuit_count_mismatch_before_submit(
    tmp_path, prepared_run
):
    from qudits_on_qubits.experiments.runner import run_experiment

    adapter = RecordingAdapter()

    def compile_extra_circuit(circuits, _config):
        circuit = circuits[0]
        return CompiledBatch((circuit, circuit.copy()), adapter.identity)

    adapter.compile = compile_extra_circuit

    with pytest.raises(
        BackendCompatibilityError,
        match="compiled circuit count does not match measurement settings",
    ):
        run_experiment(
            make_spec(tmp_path),
            adapter=adapter,
            _evaluator=lambda _counts: 1 + 0j,
        )

    assert adapter.submit_calls == 0
    assert not list((tmp_path / "runs").glob("**/experiment.json"))


@pytest.mark.parametrize("mismatch_stage", ["submit", "result"])
def test_run_experiment_rejects_direct_execution_target_mismatch(
    tmp_path, prepared_run, mismatch_stage
):
    from qudits_on_qubits.experiments.runner import run_experiment

    class WrongTargetAdapter(RecordingAdapter):
        def submit(self, circuits, shots, options=None):
            submitted = super().submit(circuits, shots, options)
            if mismatch_stage == "submit":
                return SubmittedJob(
                    submitted.job_id,
                    object(),
                    BackendIdentity("custom", "wrong-target"),
                    submitted.circuit_count,
                    submitted.shots,
                )
            return submitted

        def result(self, submitted, timeout=None):
            execution = super().result(submitted, timeout)
            if mismatch_stage == "result":
                return ExecutionResult(
                    execution.counts,
                    execution.job_id,
                    BackendIdentity("custom", "wrong-target"),
                    status=execution.status,
                )
            return execution

    with pytest.raises(BackendCompatibilityError, match="target"):
        run_experiment(
            make_spec(tmp_path),
            adapter=WrongTargetAdapter(),
            _evaluator=lambda _counts: 1 + 0j,
        )


def test_experiments_package_exports_runner_functions():
    from qudits_on_qubits.experiments import resume_experiment, run_experiment, run_experiments

    assert callable(run_experiment)
    assert callable(resume_experiment)
    assert callable(run_experiments)


def test_readout_calibration_compile_rejects_changed_physical_target():
    from qudits_on_qubits.experiments.runner import (
        _validate_physical_calibration_compile,
    )

    source = QuantumCircuit(3, 1)
    source.metadata = {"physical_qubit": 2}
    source.measure(2, 0)
    changed = QuantumCircuit(3, 1)
    changed.measure(0, 0)

    with pytest.raises(BackendCompatibilityError, match="changed its physical qubit"):
        _validate_physical_calibration_compile((source,), (changed,))


def test_readout_calibration_uses_layout_physical_index_with_leading_ancillas():
    from types import SimpleNamespace

    from qiskit import ClassicalRegister, QuantumRegister
    from qiskit.transpiler import Layout

    from qudits_on_qubits.experiments.runner import (
        _physical_qubit_mappings,
        _validate_physical_calibration_compile,
    )

    source = QuantumCircuit(16, 1)
    source.metadata = {"physical_qubit": 0}
    source.measure(0, 0)

    ancilla = QuantumRegister(4, "ancilla")
    qubits = QuantumRegister(16, "q")
    classical = ClassicalRegister(1, "c")
    compiled = QuantumCircuit(ancilla, qubits, classical)
    compiled.measure(qubits[0], classical[0])
    physical_layout = Layout(
        {
            **{qubits[index]: index for index in range(16)},
            **{ancilla[index]: index + 16 for index in range(4)},
        }
    )
    physical_layout.add_register(ancilla)
    physical_layout.add_register(qubits)
    compiled._layout = SimpleNamespace(
        initial_layout=physical_layout
    )

    assert compiled.find_bit(qubits[0]).index == 4
    assert physical_layout.get_registers() == set(compiled.qregs)
    assert _physical_qubit_mappings((compiled,)) == ((0,),)
    _validate_physical_calibration_compile((source,), (compiled,))


def test_physical_mapping_falls_back_to_output_wire_for_standard_transpile_layout():
    from qiskit import transpile
    from qiskit.providers.fake_provider import GenericBackendV2

    from qudits_on_qubits.experiments.runner import _physical_qubit_mappings

    source = QuantumCircuit(2, 2)
    source.cx(0, 1)
    source.measure((0, 1), (0, 1))
    compiled = transpile(
        source,
        GenericBackendV2(5),
        initial_layout=(1, 2),
        optimization_level=0,
    )

    assert compiled.layout is not None
    assert compiled.layout.final_layout is None
    assert _physical_qubit_mappings((compiled,)) == ((1, 2),)


def test_physical_mapping_rejects_out_of_range_coherent_layout():
    from types import SimpleNamespace

    from qiskit import ClassicalRegister, QuantumRegister
    from qiskit.transpiler import Layout

    from qudits_on_qubits.experiments.runner import _physical_qubit_mappings

    qubits = QuantumRegister(1, "q")
    classical = ClassicalRegister(1, "c")
    compiled = QuantumCircuit(qubits, classical)
    compiled.measure(qubits[0], classical[0])
    invalid_layout = Layout({qubits[0]: 1_000_000})
    invalid_layout.add_register(qubits)
    compiled._layout = SimpleNamespace(initial_layout=invalid_layout)

    with pytest.raises(BackendCompatibilityError, match="physical layout is invalid"):
        _physical_qubit_mappings((compiled,))


def test_readout_accepts_per_setting_physical_mapping_and_calibrates_union(
    tmp_path, prepared_run, monkeypatch
):
    from qudits_on_qubits.experiments.runner import run_experiment
    import qudits_on_qubits.experiments.runner as runner

    mappings = ((10, 15, 12, 11), (15, 12, 11, 10))

    def measured_circuit(mapping):
        circuit = QuantumCircuit(17, 4)
        for classical, physical in enumerate(mapping):
            circuit.measure(physical, classical)
        return circuit

    logical = tuple(measured_circuit((0, 1, 2, 3)) for _ in mappings)
    settings = (("A0",), ("A1",))
    metadata = {
        "setting_by_circuit_index": settings,
        "terms": [],
        "qutrit_bit_indices_by_setting": {
            settings[0]: [(0, 1), (2, 3)],
            settings[1]: [(0, 1), (2, 3)],
        },
        "physical_to_logical_outcome_map": {"0": 0, "1": 1},
        "d": 3,
        "qutrit_qubits": [(0, 1), (2, 3)],
        "candidate": "two_qutrit",
    }
    monkeypatch.setattr(
        runner,
        "prepare_measurements",
        lambda _artifacts: SimpleNamespace(circuits=logical, metadata=metadata),
    )

    class PerSettingAdapter(RecordingAdapter):
        def __init__(self):
            super().__init__()
            self.measurement_compile_calls = 0
            self.physical_compile_calls = 0
            self.physical_compile_widths = set()

        def compile(self, circuits, config):
            self.measurement_compile_calls += 1
            compiled = tuple(measured_circuit(mapping) for mapping in mappings)
            return CompiledBatch(compiled, self.identity)

        def compile_physical(self, circuits, config):
            self.physical_compile_calls += 1
            self.physical_compile_widths = {
                circuit.num_qubits for circuit in circuits
            }
            return CompiledBatch(tuple(circuits), self.identity)

        def result(self, submitted, timeout=None):
            if submitted.circuit_count == 8:
                counts = tuple(
                    {"0": submitted.shots} if index % 2 == 0 else {"1": submitted.shots}
                    for index in range(8)
                )
            else:
                counts = tuple(
                    {"0000": submitted.shots}
                    for _ in range(submitted.circuit_count)
                )
            return ExecutionResult(
                counts,
                submitted.job_id,
                self.identity,
                status="done",
            )

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

    adapter = PerSettingAdapter()
    result = run_experiment(
        make_spec(tmp_path, mitigation=MitigationConfig(readout=True)),
        adapter=adapter,
        _sleep=lambda _delay: None,
        _readout_strategy=PureReadout(),
        _evaluator=lambda _counts: 1 + 0j,
    )
    document = __import__("json").loads(
        (result.artifact_dir / "experiment.json").read_text(encoding="utf-8")
    )

    assert result.status is ExperimentStatus.COMPLETED
    assert adapter.measurement_compile_calls == 1
    assert adapter.physical_compile_calls == 1
    assert adapter.physical_compile_widths == {16}
    assert document["calibration"]["qubit_mapping"] == [10, 11, 12, 15]

def test_runner_rejects_adapter_identity_that_disagrees_with_backend(
    tmp_path, prepared_run
):
    from qudits_on_qubits.experiments.runner import run_experiment

    adapter = RecordingAdapter()
    spec = make_spec(tmp_path, backend=IQMHardware("garnet"))

    with pytest.raises(
        BackendCompatibilityError,
        match="resolved adapter identity does not match configured backend",
    ):
        run_experiment(
            spec,
            adapter=adapter,
            _sleep=lambda _: None,
            _evaluator=lambda _: 1.0 + 0.0j,
        )

    assert adapter.submit_calls == 0


def test_selector_representative_ranks_two_qubit_work_then_depth_then_index():
    from qudits_on_qubits.experiments.runner import _representative_circuit_index

    first = QuantumCircuit(3, name="first")
    first.cz(0, 1)
    first.x(2)
    first.barrier(0, 1)
    second = QuantumCircuit(3, name="second")
    second.cz(0, 1)
    third = QuantumCircuit(3, name="third")
    third.cz(0, 1)
    third.cz(1, 2)

    assert _representative_circuit_index((first, second, third)) == 2

    tied = (QuantumCircuit(2, name="tie-0"), QuantumCircuit(2, name="tie-1"))
    assert _representative_circuit_index(tied) == 0


def test_selector_representative_rejects_empty_workload():
    from qudits_on_qubits.experiments.runner import _representative_circuit_index

    with pytest.raises(ExperimentValidationError):
        _representative_circuit_index(())


def test_validated_iqm_selector_result_normalizes_layouts_and_costs():
    from qudits_on_qubits.experiments.runner import _validated_iqm_selector_result

    config = IQMQubitSelectorConfig(top_k=2, remove_qubits=(9,))

    assert _validated_iqm_selector_result(
        _selector_payload(config),
        config,
        2,
    ) == ("1.2.3", ((2, 3), (4, 5)), (0.01, 0.02))


@pytest.mark.parametrize(
    "changes",
    [
        {"provider": "other-selector"},
        {"version": ""},
        {"version": "Authorization " + "Bearer runner-selector-test-value"},
        {"configuration": {"top_k": 2}},
        {"layouts": (), "costs": ()},
        {"layouts": "2,3"},
        {"costs": "0.01"},
        {"layouts": ((2,),), "costs": (0.01,)},
        {"layouts": ((True, 3),), "costs": (0.01,)},
        {"layouts": ((2, 2),), "costs": (0.01,)},
        {"layouts": ((2, 9),), "costs": (0.01,)},
        {"layouts": ((2, 3), (2, 3))},
        {"layouts": ((0, 1), (2, 3), (4, 5)), "costs": (0.0, 0.1, 0.2)},
        {"costs": (False, 0.02)},
        {"costs": (-0.01, 0.02)},
        {"costs": (0.01, float("inf"))},
        {"costs": (0.02, 0.01)},
        {"unexpected": "field"},
    ],
)
def test_validated_iqm_selector_result_fails_closed_for_malformed_output(changes):
    from qudits_on_qubits.experiments.runner import _validated_iqm_selector_result

    config = IQMQubitSelectorConfig(top_k=2, remove_qubits=(9,))
    value = _selector_payload(config)
    value.update(changes)

    with pytest.raises(BackendCompatibilityError) as caught:
        _validated_iqm_selector_result(value, config, 2)

    assert str(caught.value) == "IQM qubit selector output is invalid"
    assert caught.value.__cause__ is None


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit, MemoryError])
def test_validated_iqm_selector_result_propagates_critical_base_exceptions(error_type):
    from collections.abc import Mapping

    from qudits_on_qubits.experiments.runner import _validated_iqm_selector_result

    class ExplodingMapping(Mapping):
        def __getitem__(self, _key):
            raise AssertionError("unexpected item lookup")

        def __iter__(self):
            raise error_type("selector validation interrupted")

        def __len__(self):
            return 5

    with pytest.raises(error_type, match="selector validation interrupted"):
        _validated_iqm_selector_result(
            ExplodingMapping(),
            IQMQubitSelectorConfig(),
            2,
        )


def test_iqm_selector_generated_first_merge_deduplicates_and_ranks_full_workload(
    tmp_path, prepared_run, monkeypatch
):
    from qudits_on_qubits.experiments.runner import run_experiment

    logical, _settings = _install_two_setting_workload(monkeypatch)
    selector = IQMQubitSelectorConfig(top_k=2, num_trials=17)
    selector_result = _selector_payload(selector)
    cz_counts = {
        ((2, 3), 3): (0, 4),
        ((2, 3), 7): (0, 5),
        ((4, 5), 3): (1, 1),
        ((4, 5), 7): (1, 2),
        ((0, 1), 3): (3, 3),
        ((0, 1), 7): (3, 4),
    }
    batches = {}

    def compiler(config, identity):
        key = (config.initial_layout, config.seed_transpiler)
        circuits = tuple(
            _physical_measurement_circuit(
                key[0],
                name=f"selector-candidate-{key[0]}-{key[1]}-{index}",
                cz_count=cz_counts[key][index],
            )
            for index in range(2)
        )
        batch = CompiledBatch(circuits, identity)
        batches[key] = batch
        return batch

    adapter = _SelectorCandidateAdapter(compiler, selector_result)
    result = run_experiment(
        make_spec(
            tmp_path,
            backend=IQMHardware("garnet"),
            workload_optimization=WorkloadOptimizationConfig(
                initial_layouts=((4, 5), (0, 1)),
                seed_transpilers=(3, 7),
                prefer_calibration_metrics=False,
                iqm_qubit_selector=selector,
            ),
        ),
        adapter=adapter,
        _evaluator=lambda _counts: 1 + 0j,
    )

    assert adapter.selector_calls == [(logical[0], selector)]
    assert [
        (call[1].initial_layout, call[1].seed_transpiler)
        for call in adapter.compile_calls
    ] == [
        ((2, 3), 3),
        ((2, 3), 7),
        ((4, 5), 3),
        ((4, 5), 7),
        ((0, 1), 3),
        ((0, 1), 7),
    ]
    assert all(call[0] == logical for call in adapter.compile_calls)
    submissions = [call for call in adapter.calls if call[0] == "submit"]
    assert len(submissions) == 1
    assert all(
        submitted is compiled
        for submitted, compiled in zip(
            submissions[0][1],
            batches[((4, 5), 3)].circuits,
            strict=True,
        )
    )

    document = json.loads(
        (result.artifact_dir / "experiment.json").read_text(encoding="utf-8")
    )
    metadata = document["workload_optimization"]
    assert metadata["selected_layout"] == [4, 5]
    assert metadata["selected_seed_transpiler"] == 3
    assert [
        (row["layout_source"], row["selector_cost"])
        for row in metadata["candidates"]
    ] == [
        ("iqm_qubit_selector", 0.01),
        ("iqm_qubit_selector", 0.01),
        ("iqm_qubit_selector", 0.02),
        ("iqm_qubit_selector", 0.02),
        ("explicit", None),
        ("explicit", None),
    ]
    assert metadata["selector"] == {
        "provider": "iqm-qubit-selector",
        "version": "1.2.3",
        "calibration_set_id": "cal-17",
        "configuration": selector.to_safe_dict(),
        "representative_circuit_index": 0,
        "representative_circuit_name": "logical-0",
        "generated_layouts": [[2, 3], [4, 5]],
        "generated_costs": [0.01, 0.02],
        "explicit_layouts": [[4, 5], [0, 1]],
        "merged_layouts": [[2, 3], [4, 5], [0, 1]],
    }


def test_iqm_selector_failure_is_redacted_before_compile_submit_or_artifact(
    tmp_path, prepared_run, monkeypatch
):
    from qudits_on_qubits.experiments.runner import run_experiment

    _install_two_setting_workload(monkeypatch)
    selector = IQMQubitSelectorConfig(top_k=1)
    sensitive_message = "Authorization " + "Bearer runner-selector-test-value"
    adapter = _SelectorCandidateAdapter(
        Mock(side_effect=AssertionError("compile called")),
        RuntimeError(sensitive_message),
    )

    with pytest.raises(BackendCompatibilityError) as caught:
        run_experiment(
            make_spec(
                tmp_path,
                backend=IQMHardware("garnet"),
                workload_optimization=WorkloadOptimizationConfig(
                    initial_layouts=((0, 1),),
                    iqm_qubit_selector=selector,
                ),
            ),
            adapter=adapter,
            _evaluator=lambda _counts: 1 + 0j,
        )

    rendered = "".join(traceback.format_exception(caught.value))
    assert sensitive_message not in str(caught.value)
    assert sensitive_message not in rendered
    assert caught.value.__cause__ is None
    assert adapter.selector_calls and len(adapter.selector_calls) == 1
    assert adapter.compile_calls == []
    assert adapter.submit_calls == 0
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("error_type", [MemoryError, OptionalDependencyError])
def test_iqm_selector_initial_resolution_propagates_critical_and_dependency_errors(
    tmp_path, prepared_run, monkeypatch, error_type
):
    from qudits_on_qubits.experiments.runner import run_experiment

    _install_two_setting_workload(monkeypatch)
    error = error_type("selector resolution interrupted")
    adapter = _SelectorCandidateAdapter(
        Mock(side_effect=AssertionError("compile called")),
        _selector_payload(IQMQubitSelectorConfig(top_k=1)),
    )
    adapter.resolve = Mock(side_effect=error)

    with pytest.raises(error_type) as caught:
        run_experiment(
            make_spec(
                tmp_path,
                backend=IQMHardware("garnet"),
                workload_optimization=WorkloadOptimizationConfig(
                    initial_layouts=(),
                    iqm_qubit_selector=IQMQubitSelectorConfig(top_k=1),
                ),
            ),
            adapter=adapter,
            _evaluator=lambda _counts: 1 + 0j,
        )

    assert caught.value is error
    assert adapter.compile_calls == []
    assert adapter.submit_calls == 0
    assert not (tmp_path / "runs").exists()


def test_iqm_selector_fallback_resolution_propagates_optional_dependency(tmp_path):
    from qudits_on_qubits.experiments.runner import _workload_layout_candidates

    logical = QuantumCircuit(1, 1, name="logical")
    logical.measure(0, 0)
    error = OptionalDependencyError("selector resolution dependency unavailable")
    adapter = SimpleNamespace(resolve=Mock(side_effect=error))
    spec = make_spec(
        tmp_path,
        backend=IQMHardware("garnet"),
        workload_optimization=WorkloadOptimizationConfig(
            initial_layouts=(),
            iqm_qubit_selector=IQMQubitSelectorConfig(top_k=1),
        ),
    )

    with pytest.raises(OptionalDependencyError) as caught:
        _workload_layout_candidates(adapter, (logical,), spec, 1, None)

    assert caught.value is error


def test_iqm_selector_getter_propagates_optional_dependency(tmp_path):
    from qudits_on_qubits.experiments.runner import _workload_layout_candidates

    class GetterFailingAdapter:
        @property
        def suggest_layouts(self):
            raise error

    logical = QuantumCircuit(1, 1, name="logical")
    logical.measure(0, 0)
    error = OptionalDependencyError("selector getter dependency unavailable")
    spec = make_spec(
        tmp_path,
        backend=IQMHardware("garnet"),
        workload_optimization=WorkloadOptimizationConfig(
            initial_layouts=(),
            iqm_qubit_selector=IQMQubitSelectorConfig(top_k=1),
        ),
    )
    identity = BackendIdentity(
        "iqm",
        "garnet",
        metadata={"calibration_set_id": "cal-17"},
    )

    with pytest.raises(OptionalDependencyError) as caught:
        _workload_layout_candidates(
            GetterFailingAdapter(),
            (logical,),
            spec,
            1,
            identity,
        )

    assert caught.value is error


@pytest.mark.parametrize("error_type", [KeyboardInterrupt, SystemExit, MemoryError])
def test_iqm_selector_propagates_critical_exception_before_execution(
    tmp_path, prepared_run, monkeypatch, error_type
):
    from qudits_on_qubits.experiments.runner import run_experiment

    _install_two_setting_workload(monkeypatch)
    selector = IQMQubitSelectorConfig(top_k=1)
    adapter = _SelectorCandidateAdapter(
        Mock(side_effect=AssertionError("compile called")),
        error_type("selector interrupted"),
    )

    with pytest.raises(error_type, match="selector interrupted"):
        run_experiment(
            make_spec(
                tmp_path,
                backend=IQMHardware("garnet"),
                workload_optimization=WorkloadOptimizationConfig(
                    initial_layouts=((0, 1),),
                    iqm_qubit_selector=selector,
                ),
            ),
            adapter=adapter,
            _evaluator=lambda _counts: 1 + 0j,
        )

    assert adapter.compile_calls == []
    assert adapter.submit_calls == 0
    assert not (tmp_path / "runs").exists()


def test_iqm_selector_optional_dependency_error_propagates_before_compile(
    tmp_path, prepared_run, monkeypatch
):
    from qudits_on_qubits.experiments.runner import run_experiment

    _install_two_setting_workload(monkeypatch)
    selector = IQMQubitSelectorConfig(top_k=1)
    adapter = _SelectorCandidateAdapter(
        Mock(side_effect=AssertionError("compile called")),
        OptionalDependencyError("selector dependency unavailable"),
    )

    with pytest.raises(OptionalDependencyError, match="selector dependency unavailable"):
        run_experiment(
            make_spec(
                tmp_path,
                backend=IQMHardware("garnet"),
                workload_optimization=WorkloadOptimizationConfig(
                    initial_layouts=((0, 1),),
                    iqm_qubit_selector=selector,
                ),
            ),
            adapter=adapter,
            _evaluator=lambda _counts: 1 + 0j,
        )

    assert adapter.compile_calls == []
    assert adapter.submit_calls == 0


def test_iqm_selector_empty_output_fails_closed_without_explicit_fallback(
    tmp_path, prepared_run, monkeypatch
):
    from qudits_on_qubits.experiments.runner import run_experiment

    _install_two_setting_workload(monkeypatch)
    selector = IQMQubitSelectorConfig(top_k=1)
    adapter = _SelectorCandidateAdapter(
        Mock(side_effect=AssertionError("compile called")),
        _selector_payload(selector, layouts=(), costs=()),
    )

    with pytest.raises(BackendCompatibilityError) as caught:
        run_experiment(
            make_spec(
                tmp_path,
                backend=IQMHardware("garnet"),
                workload_optimization=WorkloadOptimizationConfig(
                    initial_layouts=((0, 1),),
                    iqm_qubit_selector=selector,
                ),
            ),
            adapter=adapter,
            _evaluator=lambda _counts: 1 + 0j,
        )

    assert str(caught.value) == "IQM qubit selector output is invalid"
    assert caught.value.__cause__ is None
    assert adapter.compile_calls == []
    assert adapter.submit_calls == 0
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize(
    ("backend", "identity"),
    [
        (AerIdeal(), BackendIdentity("aer_ideal", "aer")),
        (PiastQHardware(), BackendIdentity("piastq", "piastq")),
    ],
)
def test_iqm_selector_config_rejects_non_iqm_backend_before_compile(
    tmp_path, prepared_run, monkeypatch, backend, identity
):
    from qudits_on_qubits.experiments.runner import run_experiment

    _install_two_setting_workload(monkeypatch)
    adapter = _CandidateAdapter(Mock(side_effect=AssertionError("compile called")))
    adapter.identity = identity
    adapter.suggest_layouts = Mock(side_effect=AssertionError("selector called"))

    with pytest.raises(
        ExperimentValidationError,
        match="IQMHardware",
    ):
        run_experiment(
            make_spec(
                tmp_path,
                backend=backend,
                workload_optimization=WorkloadOptimizationConfig(
                    initial_layouts=((0, 1),),
                    iqm_qubit_selector=IQMQubitSelectorConfig(top_k=1),
                ),
            ),
            adapter=adapter,
            _evaluator=lambda _counts: 1 + 0j,
        )

    adapter.suggest_layouts.assert_not_called()
    assert adapter.compile_calls == []
    assert adapter.submit_calls == 0


@pytest.mark.parametrize("selector_attribute", ["missing", None])
def test_iqm_selector_requires_callable_adapter_method_before_compile(
    tmp_path, prepared_run, monkeypatch, selector_attribute
):
    from qudits_on_qubits.experiments.runner import run_experiment

    _install_two_setting_workload(monkeypatch)
    adapter = _CandidateAdapter(Mock(side_effect=AssertionError("compile called")))
    adapter.identity = BackendIdentity(
        "iqm",
        "garnet",
        metadata={"calibration_set_id": "cal-17"},
    )
    if selector_attribute is not None:
        assert not hasattr(adapter, "suggest_layouts")
    else:
        adapter.suggest_layouts = None

    with pytest.raises(BackendCompatibilityError, match="suggest_layouts"):
        run_experiment(
            make_spec(
                tmp_path,
                backend=IQMHardware("garnet"),
                workload_optimization=WorkloadOptimizationConfig(
                    initial_layouts=((0, 1),),
                    iqm_qubit_selector=IQMQubitSelectorConfig(top_k=1),
                ),
            ),
            adapter=adapter,
            _evaluator=lambda _counts: 1 + 0j,
        )

    assert adapter.compile_calls == []
    assert adapter.submit_calls == 0


def test_iqm_selector_unsafe_version_never_reaches_artifact(
    tmp_path, prepared_run, monkeypatch
):
    from qudits_on_qubits.experiments.runner import run_experiment

    _install_two_setting_workload(monkeypatch)
    selector = IQMQubitSelectorConfig(top_k=1)
    sensitive_message = "Authorization " + "Bearer runner-selector-test-value"
    selector_result = _selector_payload(
        selector,
        layouts=((2, 3),),
        costs=(0.01,),
    )
    selector_result["version"] = sensitive_message
    adapter = _SelectorCandidateAdapter(
        Mock(side_effect=AssertionError("compile called")),
        selector_result,
    )

    with pytest.raises(BackendCompatibilityError) as caught:
        run_experiment(
            make_spec(
                tmp_path,
                backend=IQMHardware("garnet"),
                workload_optimization=WorkloadOptimizationConfig(
                    initial_layouts=(),
                    iqm_qubit_selector=selector,
                ),
            ),
            adapter=adapter,
            _evaluator=lambda _counts: 1 + 0j,
        )

    assert str(caught.value) == "IQM qubit selector output is invalid"
    assert caught.value.__cause__ is None
    assert sensitive_message not in "".join(
        traceback.format_exception(caught.value)
    )
    assert adapter.compile_calls == []
    assert adapter.submit_calls == 0
    assert not (tmp_path / "runs").exists()


def test_iqm_selector_layout_escape_is_rejected_without_submission(
    tmp_path, prepared_run, monkeypatch
):
    from qudits_on_qubits.experiments.runner import run_experiment

    _install_two_setting_workload(monkeypatch)
    selector = IQMQubitSelectorConfig(top_k=1)

    def compiler(_config, identity):
        return CompiledBatch(
            tuple(
                _physical_measurement_circuit((0, 1), name=f"escaped-{index}")
                for index in range(2)
            ),
            identity,
        )

    adapter = _SelectorCandidateAdapter(
        compiler,
        _selector_payload(selector, layouts=((2, 3),), costs=(0.01,)),
    )

    with pytest.raises(BackendCompatibilityError, match="no workload candidate"):
        run_experiment(
            make_spec(
                tmp_path,
                backend=IQMHardware("garnet"),
                workload_optimization=WorkloadOptimizationConfig(
                    initial_layouts=(),
                    seed_transpilers=(3,),
                    iqm_qubit_selector=selector,
                ),
            ),
            adapter=adapter,
            _evaluator=lambda _counts: 1 + 0j,
        )

    assert len(adapter.compile_calls) == 1
    assert adapter.submit_calls == 0
    assert not (tmp_path / "runs").exists()


def test_workload_candidate_search_compiles_cartesian_product_and_reuses_winner(
    tmp_path, prepared_run, monkeypatch
):
    from qudits_on_qubits.experiments.runner import run_experiment

    logical, _settings = _install_two_setting_workload(monkeypatch)
    batches = {}
    cz_counts = {
        ((0, 1), 3): 3,
        ((0, 1), 7): 0,
        ((2, 3), 3): 1,
        ((2, 3), 7): 2,
    }

    def compiler(config, identity):
        key = (config.initial_layout, config.seed_transpiler)
        circuits = tuple(
            _physical_measurement_circuit(
                key[0],
                name=f"candidate-{key[0]}-{key[1]}-{index}",
                cz_count=cz_counts[key],
            )
            for index in range(2)
        )
        batch = CompiledBatch(
            circuits,
            identity,
            {"transpilation": config.to_safe_dict()},
        )
        batches[key] = batch
        return batch

    adapter = _CandidateAdapter(compiler)
    search = WorkloadOptimizationConfig(
        initial_layouts=((0, 1), (2, 3)),
        seed_transpilers=(3, 7),
    )
    result = run_experiment(
        make_spec(tmp_path, workload_optimization=search),
        adapter=adapter,
        _evaluator=lambda _counts: 1 + 0j,
    )

    assert len(adapter.compile_calls) == 4
    assert [
        (call[1].initial_layout, call[1].seed_transpiler)
        for call in adapter.compile_calls
    ] == [
        ((0, 1), 3),
        ((0, 1), 7),
        ((2, 3), 3),
        ((2, 3), 7),
    ]
    assert all(call[0] == logical and len(call[0]) == 2 for call in adapter.compile_calls)
    submissions = [call for call in adapter.calls if call[0] == "submit"]
    assert len(submissions) == 1
    winner = batches[((0, 1), 7)]
    assert len(submissions[0][1]) == 2
    assert all(
        submitted is compiled
        for submitted, compiled in zip(
            submissions[0][1], winner.circuits, strict=True
        )
    )

    document = json.loads(
        (result.artifact_dir / "experiment.json").read_text(encoding="utf-8")
    )
    metadata = document["workload_optimization"]
    assert metadata["ranking_basis"] == "structural"
    assert metadata["selected_candidate_index"] == 1
    assert metadata["selected_layout"] == [0, 1]
    assert metadata["selected_seed_transpiler"] == 7
    assert len(metadata["candidates"]) == 4
    assert metadata["candidates"][0]["metrics"]["circuit_count"] == 2
    assert metadata["selected_workload"]["aggregate"]["circuit_count"] == 2
    assert set(result.values) == {"raw", "config", "diagnostics"}


def test_workload_candidate_failure_does_not_discard_accepted_candidate_or_leak(
    tmp_path, prepared_run, monkeypatch
):
    from qudits_on_qubits.experiments.runner import run_experiment

    _install_two_setting_workload(monkeypatch)
    sensitive_message = "Authorization: " + "Bearer candidate-secret"

    def compiler(config, identity):
        if config.seed_transpiler == 3:
            raise RuntimeError(sensitive_message)
        circuits = tuple(
            _physical_measurement_circuit(
                config.initial_layout,
                name=f"accepted-{index}",
            )
            for index in range(2)
        )
        return CompiledBatch(circuits, identity)

    adapter = _CandidateAdapter(compiler)
    result = run_experiment(
        make_spec(
            tmp_path,
            workload_optimization=WorkloadOptimizationConfig(
                initial_layouts=((0, 1),),
                seed_transpilers=(3, 7),
            ),
        ),
        adapter=adapter,
        _evaluator=lambda _counts: 1 + 0j,
    )
    document = json.loads(
        (result.artifact_dir / "experiment.json").read_text(encoding="utf-8")
    )
    candidate_rows = document["workload_optimization"]["candidates"]

    assert len(adapter.compile_calls) == 2
    assert adapter.submit_calls == 1
    assert candidate_rows[0] == {
        "status": "rejected",
        "candidate_index": 0,
        "layout": [0, 1],
        "seed_transpiler": 3,
        "category": "BackendCompatibilityError",
    }
    assert candidate_rows[1]["status"] == "accepted"
    assert sensitive_message not in json.dumps(document)


def test_iqm_workload_candidate_search_propagates_transpiler_memory_error(
    tmp_path, prepared_run, monkeypatch
):
    from qudits_on_qubits.experiments.backends import IQMAdapter
    from qudits_on_qubits.experiments.runner import run_experiment

    _install_two_setting_workload(monkeypatch)

    class Backend:
        name = "garnet"
        num_qubits = 20
        calibration_set_id = "cal-1"
        backend_version = "1"
        target = None

        def run(self, *_args, **_kwargs):
            raise AssertionError("submit called")

    def exhausted_transpiler(*_args, **_kwargs):
        raise MemoryError("IQM transpiler exhausted")

    adapter = IQMAdapter(
        IQMHardware("garnet"),
        backend=Backend(),
        transpiler=exhausted_transpiler,
    )
    with pytest.raises(MemoryError, match="IQM transpiler exhausted"):
        run_experiment(
            make_spec(
                tmp_path,
                backend=IQMHardware("garnet"),
                workload_optimization=WorkloadOptimizationConfig(
                    initial_layouts=((0, 1),),
                    seed_transpilers=(3,),
                ),
            ),
            adapter=adapter,
            _evaluator=lambda _counts: 1 + 0j,
        )

    assert not (tmp_path / "runs").exists()


def test_workload_candidate_inspection_failure_does_not_abort_later_candidate(
    tmp_path, prepared_run, monkeypatch
):
    from qudits_on_qubits.experiments.runner import run_experiment

    _install_two_setting_workload(monkeypatch)
    sensitive_message = "Authorization: " + "Bearer inspection-secret"

    class InspectionFailingCircuit(QuantumCircuit):
        def depth(self, *args, **kwargs):
            raise RuntimeError(sensitive_message)

    def compiler(config, identity):
        if config.seed_transpiler == 3:
            circuits = []
            for index in range(2):
                circuit = InspectionFailingCircuit(
                    2,
                    2,
                    name=f"inspection-failure-{index}",
                )
                circuit.measure((0, 1), (0, 1))
                circuits.append(circuit)
            return CompiledBatch(tuple(circuits), identity)
        circuits = tuple(
            _physical_measurement_circuit(
                config.initial_layout,
                name=f"inspection-accepted-{index}",
            )
            for index in range(2)
        )
        return CompiledBatch(circuits, identity)

    adapter = _CandidateAdapter(compiler)
    result = run_experiment(
        make_spec(
            tmp_path,
            workload_optimization=WorkloadOptimizationConfig(
                initial_layouts=((0, 1),),
                seed_transpilers=(3, 7),
            ),
        ),
        adapter=adapter,
        _evaluator=lambda _counts: 1 + 0j,
    )
    document = json.loads(
        (result.artifact_dir / "experiment.json").read_text(encoding="utf-8")
    )
    candidates = document["workload_optimization"]["candidates"]

    assert len(adapter.compile_calls) == 2
    assert adapter.submit_calls == 1
    assert candidates[0]["status"] == "rejected"
    assert candidates[0]["category"] == "RuntimeError"
    assert candidates[1]["status"] == "accepted"
    assert sensitive_message not in json.dumps(document)


def test_workload_candidate_all_fail_raises_stable_safe_error_before_submit(
    tmp_path, prepared_run, monkeypatch
):
    from qudits_on_qubits.experiments.runner import run_experiment

    _install_two_setting_workload(monkeypatch)
    sensitive_message = "token" + "=all-candidates-secret"

    def compiler(_config, _identity):
        raise RuntimeError(sensitive_message)

    adapter = _CandidateAdapter(compiler)
    with pytest.raises(
        BackendCompatibilityError,
        match=r"^no workload candidate",
    ) as caught:
        run_experiment(
            make_spec(
                tmp_path,
                workload_optimization=WorkloadOptimizationConfig(
                    initial_layouts=((0, 1), (2, 3)),
                    seed_transpilers=(3, 7),
                ),
            ),
            adapter=adapter,
            _evaluator=lambda _counts: 1 + 0j,
        )

    assert len(adapter.compile_calls) == 4
    assert adapter.submit_calls == 0
    assert sensitive_message not in str(caught.value)
    assert caught.value.__cause__ is None


def test_workload_candidate_rejects_unavailable_measurement_mapping(
    tmp_path, prepared_run, monkeypatch
):
    from qudits_on_qubits.experiments.runner import run_experiment

    _install_two_setting_workload(monkeypatch)

    def compiler(_config, identity):
        circuits = tuple(QuantumCircuit(2, 2) for _ in range(2))
        return CompiledBatch(circuits, identity)

    adapter = _CandidateAdapter(compiler)
    with pytest.raises(BackendCompatibilityError, match="no workload candidate"):
        run_experiment(
            make_spec(
                tmp_path,
                workload_optimization=WorkloadOptimizationConfig(
                    initial_layouts=((0, 1),),
                    seed_transpilers=(3,),
                ),
            ),
            adapter=adapter,
            _evaluator=lambda _counts: 1 + 0j,
        )

    assert len(adapter.compile_calls) == 1
    assert adapter.submit_calls == 0


def test_workload_candidate_rejects_partial_mapping_even_when_not_exact(
    tmp_path, prepared_run, monkeypatch
):
    from qudits_on_qubits.experiments.runner import run_experiment

    _install_two_setting_workload(monkeypatch)

    def compiler(_config, identity):
        circuits = []
        for index in range(2):
            circuit = QuantumCircuit(2, 2, name=f"partial-{index}")
            circuit.measure(0, 0)
            circuits.append(circuit)
        return CompiledBatch(tuple(circuits), identity)

    adapter = _CandidateAdapter(compiler)
    with pytest.raises(BackendCompatibilityError, match="no workload candidate"):
        run_experiment(
            make_spec(
                tmp_path,
                workload_optimization=WorkloadOptimizationConfig(
                    initial_layouts=((0, 1),),
                    seed_transpilers=(3,),
                    require_exact_physical_qubit_set=False,
                ),
            ),
            adapter=adapter,
            _evaluator=lambda _counts: 1 + 0j,
        )

    assert len(adapter.compile_calls) == 1
    assert adapter.submit_calls == 0


def test_workload_candidate_layout_width_mismatch_fails_before_compile(
    tmp_path, prepared_run, monkeypatch
):
    from qudits_on_qubits.experiments.runner import run_experiment

    _install_two_setting_workload(monkeypatch)
    adapter = RecordingAdapter()
    adapter.compile = Mock(side_effect=AssertionError("compile called"))

    with pytest.raises(ExperimentValidationError, match="layout width"):
        run_experiment(
            make_spec(
                tmp_path,
                workload_optimization=WorkloadOptimizationConfig(
                    initial_layouts=((0,),),
                    seed_transpilers=(3,),
                ),
            ),
            adapter=adapter,
            _evaluator=lambda _counts: 1 + 0j,
        )

    adapter.compile.assert_not_called()
    assert adapter.submit_calls == 0


@pytest.mark.parametrize(
    ("prefer_calibration", "expected_layout", "expected_basis"),
    [
        (True, [2, 3], "calibration_error_duration"),
        (False, [0, 1], "structural"),
    ],
)
def test_workload_candidate_ranking_uses_safe_backend_target_only_when_preferred(
    tmp_path,
    prepared_run,
    monkeypatch,
    prefer_calibration,
    expected_layout,
    expected_basis,
):
    from qudits_on_qubits.experiments.runner import run_experiment

    _install_two_setting_workload(monkeypatch)
    target = {
        "measure": {
            (0,): SimpleNamespace(error=0.2, duration=2.0),
            (1,): SimpleNamespace(error=0.2, duration=2.0),
            (2,): SimpleNamespace(error=0.01, duration=1.0),
            (3,): SimpleNamespace(error=0.01, duration=1.0),
        }
    }

    def compiler(config, identity):
        circuits = tuple(
            _physical_measurement_circuit(
                config.initial_layout,
                name=f"calibrated-{index}",
            )
            for index in range(2)
        )
        return CompiledBatch(circuits, identity)

    adapter = _CandidateAdapter(compiler)
    adapter.backend = SimpleNamespace(target=target)
    result = run_experiment(
        make_spec(
            tmp_path,
            workload_optimization=WorkloadOptimizationConfig(
                initial_layouts=((0, 1), (2, 3)),
                seed_transpilers=(3,),
                prefer_calibration_metrics=prefer_calibration,
            ),
        ),
        adapter=adapter,
        _evaluator=lambda _counts: 1 + 0j,
    )
    document = json.loads(
        (result.artifact_dir / "experiment.json").read_text(encoding="utf-8")
    )

    metadata = document["workload_optimization"]
    assert metadata["selected_layout"] == expected_layout
    assert metadata["ranking_basis"] == expected_basis


def test_workload_candidate_target_lookup_degrades_ordinary_failure_to_structural(
    tmp_path, prepared_run, monkeypatch
):
    from qudits_on_qubits.experiments.runner import run_experiment

    _install_two_setting_workload(monkeypatch)

    class FailingBackend:
        @property
        def target(self):
            raise RuntimeError("provider target secret")

    def compiler(config, identity):
        circuits = tuple(
            _physical_measurement_circuit(
                config.initial_layout,
                name=f"fallback-{index}",
            )
            for index in range(2)
        )
        return CompiledBatch(circuits, identity)

    adapter = _CandidateAdapter(compiler)
    adapter.backend = FailingBackend()
    result = run_experiment(
        make_spec(
            tmp_path,
            workload_optimization=WorkloadOptimizationConfig(
                initial_layouts=((0, 1),),
                seed_transpilers=(3,),
            ),
        ),
        adapter=adapter,
        _evaluator=lambda _counts: 1 + 0j,
    )
    document = json.loads(
        (result.artifact_dir / "experiment.json").read_text(encoding="utf-8")
    )

    assert document["workload_optimization"]["ranking_basis"] == "structural"
    assert "provider target secret" not in json.dumps(document)


def test_workload_candidate_target_lookup_propagates_memory_error(
    tmp_path, prepared_run, monkeypatch
):
    from qudits_on_qubits.experiments.runner import run_experiment

    _install_two_setting_workload(monkeypatch)

    class FailingBackend:
        @property
        def target(self):
            raise MemoryError("target exhausted")

    adapter = RecordingAdapter()
    adapter.backend = FailingBackend()
    adapter.compile = Mock(side_effect=AssertionError("compile called"))
    with pytest.raises(MemoryError, match="target exhausted"):
        run_experiment(
            make_spec(
                tmp_path,
                workload_optimization=WorkloadOptimizationConfig(
                    initial_layouts=((0, 1),),
                    seed_transpilers=(3,),
                ),
            ),
            adapter=adapter,
            _evaluator=lambda _counts: 1 + 0j,
        )

    adapter.compile.assert_not_called()
    assert adapter.submit_calls == 0


def test_workload_selected_layout_rejects_twirling_escape_before_submit(
    tmp_path, prepared_run
):
    from qudits_on_qubits.experiments.mitigation import TwirledBatch
    from qudits_on_qubits.experiments.runner import run_experiment

    compiled = _physical_measurement_circuit(
        (0,),
        name="selected",
    )

    def compiler(_config, identity):
        return CompiledBatch((compiled,), identity)

    adapter = _CandidateAdapter(compiler)
    adapter.identity = BackendIdentity("iqm", "garnet")
    escaped = tuple(
        _physical_measurement_circuit((1,), name=f"escaped-{index}")
        for index in range(2)
    )

    def transform(_circuits, *, instances, seed):
        assert (instances, seed) == (2, None)
        return TwirledBatch(
            circuits=escaped,
            original_indices=(0, 0),
            instance_indices=(0, 1),
            metadata={
                "provider": "iqm-error-reduction-tools",
                "method": "circuit_twirling",
                "readout_strategy": "NONE",
                "instances_per_circuit": 2,
                "seed": None,
            },
        )

    with pytest.raises(
        BackendCompatibilityError,
        match="twirled.*selected physical",
    ):
        run_experiment(
            make_spec(
                tmp_path,
                backend=IQMHardware("garnet"),
                mitigation=MitigationConfig(
                    circuit_twirling=True,
                    twirling_instances=2,
                ),
                workload_optimization=WorkloadOptimizationConfig(
                    initial_layouts=((0,),),
                    seed_transpilers=(3,),
                ),
            ),
            adapter=adapter,
            _twirling_transform=transform,
            _evaluator=lambda _counts: 1 + 0j,
        )

    assert adapter.submit_calls == 0


def test_workload_selected_layout_rejects_zne_escape_before_any_submit(
    tmp_path, prepared_run, monkeypatch
):
    import qudits_on_qubits.experiments.runner as runner

    compiled = QuantumCircuit(2, 1, name="selected")
    compiled.cz(0, 1)
    compiled.measure(0, 0)

    def compiler(_config, identity):
        return CompiledBatch((compiled,), identity)

    def escaping_fold(_circuits, factor):
        assert factor == 3
        escaped = QuantumCircuit(2, 1, name="factor-3-escaped")
        escaped.measure(1, 0)
        return (escaped,)

    monkeypatch.setattr(runner, "fold_cz_batch", escaping_fold)
    adapter = _CandidateAdapter(compiler)
    with pytest.raises(
        BackendCompatibilityError,
        match="ZNE factor 3.*selected physical",
    ):
        runner.run_experiment(
            make_spec(
                tmp_path,
                mitigation=MitigationConfig(zne=True, zne_factors=(1, 3)),
                workload_optimization=WorkloadOptimizationConfig(
                    initial_layouts=((0,),),
                    seed_transpilers=(3,),
                ),
            ),
            adapter=adapter,
            _evaluator=lambda _counts: 1 + 0j,
        )

    assert adapter.submit_calls == 0


def test_workload_readout_calibrates_only_selected_physical_union(
    tmp_path, prepared_run, monkeypatch
):
    import qudits_on_qubits.experiments.runner as runner

    _install_two_setting_workload(monkeypatch)
    target = {
        "measure": {
            (0,): SimpleNamespace(error=0.2, duration=2.0),
            (1,): SimpleNamespace(error=0.2, duration=2.0),
            (2,): SimpleNamespace(error=0.01, duration=1.0),
            (3,): SimpleNamespace(error=0.01, duration=1.0),
        }
    }

    def compiler(config, identity):
        circuits = tuple(
            _physical_measurement_circuit(
                config.initial_layout,
                name=f"readout-candidate-{index}",
            )
            for index in range(2)
        )
        return CompiledBatch(circuits, identity)

    class ReadoutCandidateAdapter(_CandidateAdapter):
        def compile_physical(self, circuits, _config):
            return CompiledBatch(tuple(circuits), self.identity)

        def result(self, submitted, timeout=None):
            self.calls.append(("result", submitted, timeout))
            if submitted.circuit_count == 4:
                counts = tuple(
                    {"0": submitted.shots}
                    if index % 2 == 0
                    else {"1": submitted.shots}
                    for index in range(4)
                )
            else:
                counts = tuple(
                    {"00": submitted.shots}
                    for _ in range(submitted.circuit_count)
                )
            return ExecutionResult(
                counts,
                submitted.job_id,
                self.identity,
                status="done",
            )

    class PureReadout:
        def build_context(self, calibration):
            return calibration.assignment_matrices

        def resample_calibration(self, calibration, _rng):
            return calibration.assignment_matrices

        def apply(self, counts_by_setting, _context):
            return {
                setting: {
                    outcome: count / sum(counts.values())
                    for outcome, count in counts.items()
                }
                for setting, counts in counts_by_setting.items()
            }

    selected_unions = []
    real_builder = runner.build_readout_calibration_circuits

    def recording_builder(physical_qubits):
        selected_unions.append(tuple(physical_qubits))
        return real_builder(physical_qubits)

    monkeypatch.setattr(
        runner,
        "build_readout_calibration_circuits",
        recording_builder,
    )
    adapter = ReadoutCandidateAdapter(compiler)
    adapter.backend = SimpleNamespace(target=target)
    result = runner.run_experiment(
        make_spec(
            tmp_path,
            mitigation=MitigationConfig(readout=True),
            workload_optimization=WorkloadOptimizationConfig(
                initial_layouts=((0, 1), (2, 3)),
                seed_transpilers=(3,),
            ),
        ),
        adapter=adapter,
        _readout_strategy=PureReadout(),
        _evaluator=lambda _counts: 1 + 0j,
    )
    document = json.loads(
        (result.artifact_dir / "experiment.json").read_text(encoding="utf-8")
    )

    assert selected_unions == [(2, 3)]
    assert document["calibration"]["qubit_mapping"] == [2, 3]
    assert adapter.submit_calls == 2
