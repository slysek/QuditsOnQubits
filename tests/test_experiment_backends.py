from __future__ import annotations

from dataclasses import FrozenInstanceError
from enum import Enum
from importlib import import_module
from types import SimpleNamespace
import traceback

import pytest
from qiskit import QuantumCircuit

from qudits_on_qubits.experiments.errors import (
    BackendCompatibilityError,
    ExperimentValidationError,
    JobResultError,
    JobSubmissionError,
    OptionalDependencyError,
)
from qudits_on_qubits.experiments.execution import ExecutionMode
from qudits_on_qubits.experiments.models import AerIdeal, CustomBackend, TranspilationConfig


class _Job:
    def __init__(self, result=None, job_id="job-1"):
        self._result = result
        self._job_id = job_id

    def job_id(self):
        return self._job_id

    def result(self, **_kwargs):
        return self._result


class _Backend:
    local = True

    def __init__(self, job=None):
        self.job = job or _Job()
        self.calls = []

    def run(self, circuits, **options):
        self.calls.append((circuits, options))
        return self.job


def _custom_backend(instance, **kwargs):
    return CustomBackend(
        instance,
        execution_mode=ExecutionMode.IDEAL_SIMULATOR,
        **kwargs,
    )


def _assert_sanitized_error(caught, sensitive_text):
    rendered = "".join(traceback.format_exception(caught.type, caught.value, caught.tb))
    assert caught.value.__cause__ is None
    assert sensitive_text not in str(caught.value)
    assert sensitive_text not in repr(caught.value)
    assert sensitive_text not in rendered


def _compiled(adapter, count=2):
    from qudits_on_qubits.experiments.backends import CompiledBatch

    circuits = tuple(QuantumCircuit(1, 1) for _ in range(count))
    return CompiledBatch(circuits, adapter.resolve())


def test_records_are_frozen_and_deeply_safe():
    from qudits_on_qubits.experiments.backends import (
        BackendCapabilities,
        BackendIdentity,
        CompiledBatch,
        ExecutionResult,
        SubmittedJob,
    )

    nested = {"items": [{"label": "original"}]}
    identity = BackendIdentity("custom", "local", metadata=nested)
    capabilities = BackendCapabilities(True, False, metadata=nested)
    compiled = CompiledBatch((object(),), identity, metadata=nested)
    submitted = SubmittedJob("job-1", object(), identity, 1, 10, metadata=nested)
    result = ExecutionResult(({"0": 10},), "job-1", identity, timing=nested, metadata=nested)
    nested["items"][0]["label"] = "changed"

    assert identity.metadata["items"][0]["label"] == "original"
    assert capabilities.metadata["items"][0]["label"] == "original"
    assert compiled.metadata["items"][0]["label"] == "original"
    assert submitted.metadata["items"][0]["label"] == "original"
    assert result.timing["items"][0]["label"] == "original"
    assert result.metadata["items"][0]["label"] == "original"
    with pytest.raises(TypeError):
        result.counts[0]["0"] = 9
    with pytest.raises(TypeError):
        identity.metadata["items"][0]["label"] = "changed"
    with pytest.raises(FrozenInstanceError):
        submitted.job_id = "changed"
    assert "handle" not in submitted.to_safe_dict()


def test_custom_compile_uses_same_backend_and_non_none_transpile_options(monkeypatch):
    from qudits_on_qubits.experiments.backends import CustomBackendAdapter

    backend = _Backend()
    adapter = CustomBackendAdapter(_custom_backend(backend, identity="local"))
    source = (QuantumCircuit(1), QuantumCircuit(1))
    compiled_circuits = [QuantumCircuit(1), QuantumCircuit(1)]
    seen = {}

    def fake_transpile(circuits, *, backend, **options):
        seen.update(circuits=circuits, backend=backend, options=options)
        return compiled_circuits

    monkeypatch.setattr(
        import_module("qudits_on_qubits.experiments.backends.custom"),
        "transpile",
        fake_transpile,
    )
    config = TranspilationConfig(optimization_level=2, seed_transpiler=7, layout_method="dense")
    compiled = adapter.compile(source, config)

    assert tuple(seen["circuits"]) == source
    assert all(actual is expected for actual, expected in zip(seen["circuits"], source))
    assert seen["backend"] is backend
    assert seen["options"] == {"optimization_level": 2, "seed_transpiler": 7, "layout_method": "dense"}
    assert compiled.circuits == tuple(compiled_circuits)
    assert compiled.target_identity == adapter.resolve()


def test_custom_resolve_preserves_valid_human_readable_identity():
    from qudits_on_qubits.experiments.backends import CustomBackendAdapter

    adapter = CustomBackendAdapter(_custom_backend(_Backend(), identity="lab backend / slot 1"))
    assert adapter.resolve().name == "lab backend / slot 1"


def test_custom_submit_preserves_circuit_objects_and_options_exactly_once():
    from qudits_on_qubits.experiments.backends import CustomBackendAdapter

    backend = _Backend()
    adapter = CustomBackendAdapter(_custom_backend(backend, identity="local"))
    compiled = _compiled(adapter)
    submitted = adapter.submit(compiled.circuits, 25, {"memory": True})

    sent, options = backend.calls[0]
    assert sent is compiled.circuits
    assert sent[0] is compiled.circuits[0]
    assert options == {"memory": True, "shots": 25}
    assert submitted.circuit_count == 2
    assert submitted.shots == 25


@pytest.mark.parametrize("options", [{"shots": 10}, {"shots": None}])
def test_submit_rejects_duplicate_shots(options):
    from qudits_on_qubits.experiments.backends import CustomBackendAdapter

    backend = _Backend()
    adapter = CustomBackendAdapter(_custom_backend(backend))
    with pytest.raises(BackendCompatibilityError, match="shots"):
        adapter.submit(_compiled(adapter).circuits, 10, options)
    assert not backend.calls


def test_submit_wraps_backend_exception_without_leaking_options():
    from qudits_on_qubits.experiments.backends import CustomBackendAdapter

    sensitive_text = "token=do-not-leak"

    class FailingBackend(_Backend):
        def run(self, circuits, **options):
            raise RuntimeError(sensitive_text)

    adapter = CustomBackendAdapter(_custom_backend(FailingBackend(), identity="safe-name"))
    with pytest.raises(JobSubmissionError, match="safe-name") as caught:
        adapter.submit(_compiled(adapter, 1).circuits, 10, {"api_key": "secret"})
    _assert_sanitized_error(caught, sensitive_text)


def test_compile_wraps_transpiler_exception_without_leaking_details(monkeypatch):
    from qudits_on_qubits.experiments.backends import CustomBackendAdapter

    adapter = CustomBackendAdapter(_custom_backend(_Backend(), identity="safe-name"))
    sensitive_text = "token=do-not-leak"
    monkeypatch.setattr(
        import_module("qudits_on_qubits.experiments.backends.custom"),
        "transpile",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError(sensitive_text)),
    )
    with pytest.raises(BackendCompatibilityError, match="safe-name") as caught:
        adapter.compile((QuantumCircuit(1),), TranspilationConfig())
    _assert_sanitized_error(caught, sensitive_text)


@pytest.mark.parametrize(
    ("job", "expected"),
    [
        (_Job(job_id="callable-id"), "callable-id"),
        (SimpleNamespace(job_id="attribute-id", result=lambda: None), "attribute-id"),
        (SimpleNamespace(result=lambda: None), "local"),
    ],
)
def test_job_id_callable_attribute_and_local_fallback(job, expected):
    from qudits_on_qubits.experiments.backends import CustomBackendAdapter

    adapter = CustomBackendAdapter(_custom_backend(_Backend(job), identity="local"))
    job_id = adapter.submit(_compiled(adapter, 1).circuits, 1).job_id
    if expected == "local":
        assert job_id.startswith("local-")
    else:
        assert job_id == expected


def test_invalid_local_job_id_uses_local_fallback():
    from qudits_on_qubits.experiments.backends import CustomBackendAdapter

    adapter = CustomBackendAdapter(_custom_backend(_Backend(_Job(job_id="../../unsafe"))))
    assert adapter.submit(_compiled(adapter, 1).circuits, 1).job_id.startswith("local-")


@pytest.mark.parametrize("local", [False, None])
def test_remote_or_unknown_custom_job_requires_job_id(local):
    from qudits_on_qubits.experiments.backends import CustomBackendAdapter

    backend = _Backend(SimpleNamespace(result=lambda: None))
    backend.local = local
    adapter = CustomBackendAdapter(_custom_backend(backend, identity="remote"))
    with pytest.raises(JobSubmissionError, match="job ID"):
        adapter.submit(_compiled(adapter, 1).circuits, 1)


def test_result_extracts_indexed_backend_counts_in_input_order():
    from qudits_on_qubits.experiments.backends import CustomBackendAdapter, SubmittedJob

    class IndexedResult:
        def get_counts(self, index):
            return ({"00": 7, "11": 3}, {"01": 4, "10": 6})[index]

    adapter = CustomBackendAdapter(_custom_backend(_Backend(), identity="local"))
    submitted = SubmittedJob("job-1", _Job(IndexedResult()), adapter.resolve(), 2, 10)
    result = adapter.result(submitted)

    assert result.counts == ({"00": 7, "11": 3}, {"01": 4, "10": 6})
    assert result.job_id == "job-1"


def test_result_extracts_singleton_get_counts():
    from qudits_on_qubits.experiments.backends import CustomBackendAdapter, SubmittedJob

    raw = SimpleNamespace(get_counts=lambda: {"0": 2, "1": 3})
    adapter = CustomBackendAdapter(_custom_backend(_Backend()))
    result = adapter.result(SubmittedJob("job-1", _Job(raw), adapter.resolve(), 1, 5))
    assert result.counts == ({"0": 2, "1": 3},)


@pytest.mark.parametrize("named", [False, True])
def test_result_extracts_primitive_meas_and_named_data_bins(named):
    from qudits_on_qubits.experiments.backends import CustomBackendAdapter, SubmittedJob

    counts = ({"0": 3}, {"1": 3})
    entries = []
    for item in counts:
        register = SimpleNamespace(get_counts=lambda item=item: item)
        data = SimpleNamespace(aux=register) if named else SimpleNamespace(meas=register)
        entries.append(SimpleNamespace(data=data))
    raw = tuple(entries)
    adapter = CustomBackendAdapter(_custom_backend(_Backend()))
    result = adapter.result(SubmittedJob("job-1", _Job(raw), adapter.resolve(), 2, 3))
    assert result.counts == counts


@pytest.mark.parametrize(
    ("raw", "circuit_count", "shots", "message"),
    [
        (SimpleNamespace(get_counts=lambda: {"0": 2}), 2, 2, "count"),
        (SimpleNamespace(get_counts=lambda: {"2": 2}), 1, 2, "bitstring"),
        (SimpleNamespace(get_counts=lambda: {"0": True, "1": 1}), 1, 2, "integer"),
        (SimpleNamespace(get_counts=lambda: {"0": -1, "1": 3}), 1, 2, "non-negative"),
        (SimpleNamespace(get_counts=lambda: {"0": 1}), 1, 2, "shots"),
    ],
)
def test_result_rejects_mismatched_or_malformed_counts(raw, circuit_count, shots, message):
    from qudits_on_qubits.experiments.backends import CustomBackendAdapter, SubmittedJob

    adapter = CustomBackendAdapter(_custom_backend(_Backend()))
    submitted = SubmittedJob("job-1", _Job(raw), adapter.resolve(), circuit_count, shots)
    with pytest.raises(JobResultError, match=message):
        adapter.result(submitted)


def test_result_wraps_handle_result_exception():
    from qudits_on_qubits.experiments.backends import CustomBackendAdapter, SubmittedJob

    sensitive_text = "password=do-not-leak"

    class FailingJob:
        def result(self, **_kwargs):
            raise RuntimeError(sensitive_text)

    adapter = CustomBackendAdapter(_custom_backend(_Backend(), identity="safe"))
    submitted = SubmittedJob("job-1", FailingJob(), adapter.resolve(), 1, 1)
    with pytest.raises(JobResultError, match="job-1") as caught:
        adapter.result(submitted, timeout=2.0)
    _assert_sanitized_error(caught, sensitive_text)


@pytest.mark.parametrize("exception_type", [RuntimeError, JobResultError])
def test_result_wraps_count_extraction_exception_without_leaking_details(exception_type):
    from qudits_on_qubits.experiments.backends import CustomBackendAdapter, SubmittedJob

    sensitive_text = "api_key=do-not-leak"
    raw = SimpleNamespace(get_counts=lambda: (_ for _ in ()).throw(exception_type(sensitive_text)))
    adapter = CustomBackendAdapter(_custom_backend(_Backend()))
    submitted = SubmittedJob("job-1", _Job(raw), adapter.resolve(), 1, 1)
    with pytest.raises(JobResultError) as caught:
        adapter.result(submitted)
    _assert_sanitized_error(caught, sensitive_text)


def test_custom_restore_job_requires_capability_and_retrieve_method():
    from qudits_on_qubits.experiments.backends import CustomBackendAdapter

    backend = _Backend()
    with pytest.raises(BackendCompatibilityError, match="resume"):
        CustomBackendAdapter(_custom_backend(backend, supports_resume=False)).restore_job("job-1")

    resumable = CustomBackendAdapter(_custom_backend(backend, supports_resume=True))
    with pytest.raises(BackendCompatibilityError, match="retrieve_job"):
        resumable.restore_job("job-1")


@pytest.mark.parametrize(
    "handle",
    [
        _Job(job_id="remote-1"),
        SimpleNamespace(job_id="remote-1", result=lambda: None),
    ],
)
def test_custom_restore_job_retrieves_remote_handle_with_matching_id(handle):
    from qudits_on_qubits.experiments.backends import CustomBackendAdapter

    backend = _Backend()
    backend.local = False
    backend.retrieve_job = lambda job_id: handle if job_id == "remote-1" else None
    adapter = CustomBackendAdapter(_custom_backend(backend, supports_resume=True))
    submitted = adapter.restore_job("remote-1", circuit_count=2, shots=100)
    assert submitted.handle is handle
    assert submitted.circuit_count == 2
    assert submitted.shots == 100


@pytest.mark.parametrize(
    "handle",
    [
        SimpleNamespace(result=lambda: None),
        _Job(job_id="different-job"),
        SimpleNamespace(job_id="different-job", result=lambda: None),
    ],
)
def test_custom_restore_job_rejects_missing_or_mismatched_actual_id(handle):
    from qudits_on_qubits.experiments.backends import CustomBackendAdapter

    backend = _Backend()
    backend.local = False
    backend.retrieve_job = lambda _job_id: handle
    adapter = CustomBackendAdapter(_custom_backend(backend, supports_resume=True))
    with pytest.raises(JobResultError, match="job ID") as caught:
        adapter.restore_job("remote-1")
    assert caught.value.__cause__ is None


def test_custom_restore_job_sanitizes_retrieval_exception():
    from qudits_on_qubits.experiments.backends import CustomBackendAdapter

    sensitive_text = "secret=restore-leak"
    backend = _Backend()
    backend.local = False
    backend.retrieve_job = lambda _job_id: (_ for _ in ()).throw(RuntimeError(sensitive_text))
    adapter = CustomBackendAdapter(_custom_backend(backend, supports_resume=True))
    with pytest.raises(JobResultError, match="restore") as caught:
        adapter.restore_job("remote-1")
    _assert_sanitized_error(caught, sensitive_text)


def test_custom_availability_reports_backend_status():
    from qudits_on_qubits.experiments.backends import CustomBackendAdapter

    backend = _Backend()
    backend.status = lambda: SimpleNamespace(operational=False, status_msg="maintenance")
    availability = CustomBackendAdapter(_custom_backend(backend)).availability()
    assert not availability.available
    assert availability.reason == "maintenance"


def test_registry_builds_aer_and_custom_and_rejects_unknown():
    from qudits_on_qubits.experiments.backends import (
        AerAdapter,
        BackendAdapterRegistry,
        CustomBackendAdapter,
        create_backend_adapter,
    )

    assert isinstance(create_backend_adapter(AerIdeal(), simulator=_Backend()), AerAdapter)
    assert isinstance(create_backend_adapter(_custom_backend(_Backend())), CustomBackendAdapter)
    with pytest.raises(BackendCompatibilityError, match="unsupported"):
        BackendAdapterRegistry().create(object())


def test_aer_uses_one_simulator_for_resolve_compile_and_submit_and_injects_seed(monkeypatch):
    from qudits_on_qubits.experiments.backends import AerAdapter

    simulator = _Backend(_Job(job_id=None))
    simulator.name = "aer_simulator_statevector"
    seen = {}

    def fake_transpile(circuits, *, backend, **options):
        seen["backend"] = backend
        return circuits

    monkeypatch.setattr(
        import_module("qudits_on_qubits.experiments.backends.aer"),
        "transpile",
        fake_transpile,
    )
    adapter = AerAdapter(AerIdeal(seed_simulator=19), simulator=simulator)
    identity = adapter.resolve()
    compiled = adapter.compile((QuantumCircuit(1),), TranspilationConfig())
    submitted = adapter.submit(compiled.circuits, 5)

    assert seen["backend"] is simulator
    assert simulator.calls[0][0] is compiled.circuits
    assert simulator.calls[0][1] == {"seed_simulator": 19, "shots": 5}
    assert identity == compiled.target_identity == submitted.target_identity
    assert submitted.job_id.startswith("local-")


def test_aer_rejects_conflicting_seed_and_cannot_restore():
    from qudits_on_qubits.experiments.backends import AerAdapter

    adapter = AerAdapter(AerIdeal(seed_simulator=19), simulator=_Backend())
    with pytest.raises(BackendCompatibilityError, match="seed_simulator"):
        adapter.submit(_compiled(adapter, 1).circuits, 5, {"seed_simulator": 20})
    with pytest.raises(BackendCompatibilityError, match="restore"):
        adapter.restore_job("local-id")


@pytest.mark.parametrize("options", [{"noise_model": object()}, {"method": "density_matrix"}])
def test_aer_rejects_nonideal_execution_options(options):
    from qudits_on_qubits.experiments.backends import AerAdapter

    adapter = AerAdapter(AerIdeal(), simulator=_Backend())
    with pytest.raises(BackendCompatibilityError, match="ideal"):
        adapter.submit(_compiled(adapter, 1).circuits, 5, options)


def test_aer_optional_dependency_error_has_install_hint(monkeypatch):
    from qudits_on_qubits.experiments.backends import AerAdapter

    adapter = AerAdapter(AerIdeal())
    sensitive_text = "token=dependency-leak"
    monkeypatch.setattr(
        adapter,
        "_load_aer_simulator",
        lambda: (_ for _ in ()).throw(ImportError(sensitive_text)),
    )
    with pytest.raises(OptionalDependencyError, match="pip install qiskit-aer") as caught:
        adapter.resolve()
    _assert_sanitized_error(caught, sensitive_text)


def test_aer_compile_sanitizes_transpiler_exception(monkeypatch):
    from qudits_on_qubits.experiments.backends import AerAdapter

    sensitive_text = "password=aer-compile-leak"
    monkeypatch.setattr(
        import_module("qudits_on_qubits.experiments.backends.aer"),
        "transpile",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError(sensitive_text)),
    )
    adapter = AerAdapter(AerIdeal(), simulator=_Backend())
    with pytest.raises(BackendCompatibilityError, match="compile") as caught:
        adapter.compile((QuantumCircuit(1),), TranspilationConfig())
    _assert_sanitized_error(caught, sensitive_text)


class _ProviderStatus(Enum):
    DONE = "completed"


def test_execution_result_normalizes_safe_enum_status():
    from qudits_on_qubits.experiments.backends import BackendIdentity, ExecutionResult

    result = ExecutionResult(({"0": 1},), "job-1", BackendIdentity("custom", "local"), _ProviderStatus.DONE)
    assert result.status == "done"


@pytest.mark.parametrize("status", ["token=leak", "RUNNING\nsecret", "bad\x01status", "x" * 513])
def test_execution_result_rejects_unsafe_direct_status(status):
    from qudits_on_qubits.experiments.backends import BackendIdentity, ExecutionResult

    with pytest.raises(ExperimentValidationError, match="status"):
        ExecutionResult(({"0": 1},), "job-1", BackendIdentity("custom", "local"), status)


@pytest.mark.parametrize("status", ["token=leak", "RUNNING\nsecret", "bad\x01status", "x" * 513])
def test_result_discards_unsafe_provider_status(status):
    from qudits_on_qubits.experiments.backends import CustomBackendAdapter, SubmittedJob

    handle = _Job(SimpleNamespace(get_counts=lambda: {"0": 1}))
    handle.status = lambda: status
    adapter = CustomBackendAdapter(_custom_backend(_Backend()))
    submitted = SubmittedJob("job-1", handle, adapter.resolve(), 1, 1)
    assert adapter.result(submitted).status is None


def test_result_normalizes_safe_provider_enum_status():
    from qudits_on_qubits.experiments.backends import CustomBackendAdapter, SubmittedJob

    handle = _Job(SimpleNamespace(get_counts=lambda: {"0": 1}))
    handle.status = lambda: _ProviderStatus.DONE
    adapter = CustomBackendAdapter(_custom_backend(_Backend()))
    submitted = SubmittedJob("job-1", handle, adapter.resolve(), 1, 1)
    assert adapter.result(submitted).status == "done"


def test_aer_ideal_real_execution_measured_zero_counts():
    pytest.importorskip("qiskit_aer")
    from qudits_on_qubits.experiments.backends import AerAdapter

    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)
    adapter = AerAdapter(AerIdeal(seed_simulator=11))
    compiled = adapter.compile((circuit,), TranspilationConfig(optimization_level=0))
    submitted = adapter.submit(compiled.circuits, 32)
    result = adapter.result(submitted)
    assert result.counts == ({"0": 32},)
