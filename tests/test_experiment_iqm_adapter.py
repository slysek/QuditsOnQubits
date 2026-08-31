from __future__ import annotations

from types import SimpleNamespace
import traceback

import pytest
from qiskit import QuantumCircuit

from qudits_on_qubits.experiments.errors import (
    BackendCompatibilityError,
    BackendUnavailableError,
    ExperimentValidationError,
    JobResultError,
    OptionalDependencyError,
)
from qudits_on_qubits.experiments.models import IQMHardware, TranspilationConfig


class _Job:
    def __init__(self, job_id="iqm-7", result=None):
        self._job_id = job_id
        self._result = result or SimpleNamespace(get_counts=lambda: {"0": 5})

    def job_id(self):
        return self._job_id

    def result(self, **_kwargs):
        return self._result


class _Backend:
    name = "garnet"
    num_qubits = 20
    calibration_set_id = "cal-17"
    backend_version = "1.2"

    def __init__(self):
        self.run_calls = []
        self.retrieve_calls = []

    def run(self, circuits, **options):
        self.run_calls.append((circuits, options))
        return _Job()

    def retrieve_job(self, job_id):
        self.retrieve_calls.append(job_id)
        return _Job(job_id)

    def status(self):
        return SimpleNamespace(operational=True)


def _assert_sanitized(caught, secret):
    rendered = "".join(traceback.format_exception(caught.type, caught.value, caught.tb))
    assert caught.value.__cause__ is None
    assert secret not in str(caught.value)
    assert secret not in repr(caught.value)
    assert secret not in rendered


def test_iqm_compile_uses_official_transpiler_with_explicit_options():
    from qudits_on_qubits.experiments.backends import IQMAdapter

    backend = _Backend()
    source = QuantumCircuit(4)
    compiled = QuantumCircuit(20)
    calls = []

    def transpiler(circuit, actual_backend, **options):
        calls.append((circuit, actual_backend, options))
        return compiled

    adapter = IQMAdapter(
        IQMHardware("garnet"), backend=backend, transpiler=transpiler
    )
    config = TranspilationConfig(
        optimization_level=3,
        seed_transpiler=9,
        initial_layout=(16, 17, 18, 19),
    )

    result = adapter.compile((source,), config)

    assert result.circuits[0] is compiled
    assert calls == [
        (
            source,
            backend,
            {
                "optimization_level": 3,
                "seed_transpiler": 9,
                "initial_layout": [16, 17, 18, 19],
            },
        )
    ]
    assert result.metadata["transpilation"] == {
        "optimization_level": 3,
        "seed_transpiler": 9,
        "initial_layout": (16, 17, 18, 19),
    }


def test_iqm_reuses_one_backend_for_resolve_compile_and_submit():
    from qudits_on_qubits.experiments.backends import IQMAdapter

    backend = _Backend()
    loader_calls = []
    source = QuantumCircuit(2)
    compiled_object = QuantumCircuit(20)

    def loader(device, use_metrics=False, env_path=None):
        loader_calls.append((device, use_metrics, env_path))
        return backend

    adapter = IQMAdapter(
        IQMHardware("garnet", use_metrics=True),
        backend_loader=loader,
        transpiler=lambda *_args, **_kwargs: compiled_object,
    )
    identity = adapter.resolve()
    compiled = adapter.compile(
        (source,),
        TranspilationConfig(
            optimization_level=2,
            seed_transpiler=11,
            layout_method="dense",
            routing_method="sabre",
            scheduling_method="alap",
        ),
    )
    submitted = adapter.submit(compiled.circuits, 19, {"memory": True})

    assert loader_calls == [("garnet", True, None)]
    assert identity.kind == "iqm"
    assert identity.name == "garnet"
    assert identity.metadata["target"] == "iqm:garnet"
    assert compiled.circuits[0] is compiled_object
    sent, options = backend.run_calls[0]
    assert sent is compiled.circuits
    assert sent[0] is compiled_object
    assert options == {"shots": 19, "memory": True}
    assert submitted.target_identity is identity


def test_iqm_compile_transpiles_each_circuit_and_preserves_output_identity():
    from qudits_on_qubits.experiments.backends import IQMAdapter

    sources = (QuantumCircuit(1), QuantumCircuit(2))
    compiled = (QuantumCircuit(20), QuantumCircuit(20))
    outputs = iter(compiled)
    calls = []

    def transpiler(circuit, backend, **options):
        calls.append((circuit, backend, options))
        return next(outputs)

    backend = _Backend()
    result = IQMAdapter(
        IQMHardware("garnet"), backend=backend, transpiler=transpiler
    ).compile(sources, TranspilationConfig())

    assert [call[0] for call in calls] == list(sources)
    assert all(call[1] is backend for call in calls)
    assert result.circuits[0] is compiled[0]
    assert result.circuits[1] is compiled[1]
    assert calls[0][2] == {"optimization_level": 3}
    assert calls[1][2] == {"optimization_level": 3}


def test_iqm_compile_physical_pins_identity_layout():
    from qudits_on_qubits.experiments.backends import IQMAdapter

    source = QuantumCircuit(4, 1)
    source.measure(3, 0)
    calls = []

    def transpiler(circuit, backend, **options):
        calls.append((circuit, backend, options))
        return circuit

    backend = _Backend()
    result = IQMAdapter(
        IQMHardware("garnet"), backend=backend, transpiler=transpiler
    ).compile_physical((source,), TranspilationConfig())

    assert result.circuits[0] is source
    assert calls == [
        (
            source,
            backend,
            {"optimization_level": 3, "initial_layout": [0, 1, 2, 3]},
        )
    ]


def test_iqm_compile_omits_unset_options():
    from qudits_on_qubits.experiments.backends import IQMAdapter

    calls = []

    def transpiler(circuit, backend, **options):
        calls.append((circuit, backend, options))
        return circuit

    backend = _Backend()
    compiled = IQMAdapter(
        IQMHardware("garnet"), backend=backend, transpiler=transpiler
    ).compile((QuantumCircuit(1),), TranspilationConfig())

    assert calls[0][1] is backend
    assert calls[0][2] == {"optimization_level": 3}
    assert compiled.metadata["transpilation"] == {"optimization_level": 3}


def test_iqm_compile_accepts_all_explicit_options():
    from qudits_on_qubits.experiments.backends import IQMAdapter

    seen = {}

    def transpiler(backend_circuit, backend, **options):
        seen.update(circuit=backend_circuit, backend=backend, options=options)
        return backend_circuit

    backend = _Backend()
    source = QuantumCircuit(2)
    IQMAdapter(
        IQMHardware("garnet"), backend=backend, transpiler=transpiler
    ).compile(
        (source,),
        TranspilationConfig(2, 11, "dense", "sabre", "alap", (3, 4)),
    )

    assert seen == {
        "circuit": source,
        "backend": backend,
        "options": {
            "optimization_level": 2,
            "seed_transpiler": 11,
            "layout_method": "dense",
            "routing_method": "sabre",
            "scheduling_method": "alap",
            "initial_layout": [3, 4],
        },
    }


def test_iqm_injected_backend_never_calls_loader():
    from qudits_on_qubits.experiments.backends import IQMAdapter

    backend = _Backend()
    adapter = IQMAdapter(
        IQMHardware("garnet"),
        backend=backend,
        backend_loader=lambda *_args, **_kwargs: pytest.fail("loader called"),
    )
    assert adapter.resolve().name == "garnet"
    assert adapter.availability().available


def test_iqm_accepts_loader_injection_name():
    from qudits_on_qubits.experiments.backends import IQMAdapter

    backend = _Backend()
    calls = []
    adapter = IQMAdapter(
        IQMHardware("garnet"),
        loader=lambda *args, **kwargs: calls.append((args, kwargs)) or backend,
    )
    assert adapter.backend is backend
    assert calls == [(('garnet',), {"use_metrics": False, "env_path": None})]


@pytest.mark.parametrize(
    ("failure", "error_type", "message"),
    [
        (ModuleNotFoundError("token=dependency-secret"), OptionalDependencyError, "IQM"),
        (RuntimeError("token=network-secret"), BackendUnavailableError, "garnet"),
    ],
)
def test_iqm_loader_errors_are_typed_sanitized_and_available_false(failure, error_type, message):
    from qudits_on_qubits.experiments.backends import IQMAdapter

    adapter = IQMAdapter(
        IQMHardware("garnet"),
        backend_loader=lambda *_args, **_kwargs: (_ for _ in ()).throw(failure),
    )
    availability = adapter.availability()
    assert not availability.available
    assert "secret" not in (availability.reason or "")
    with pytest.raises(error_type, match=message) as caught:
        adapter.resolve()
    _assert_sanitized(caught, str(failure))


def test_iqm_preflight_checks_operational_status_and_qubit_capacity():
    from qudits_on_qubits.experiments.backends import IQMAdapter

    backend = _Backend()
    backend.num_qubits = 2
    adapter = IQMAdapter(IQMHardware("garnet"), backend=backend)
    with pytest.raises(BackendCompatibilityError, match="qubits"):
        adapter.preflight((SimpleNamespace(num_qubits=3),), 10)

    backend.status = lambda: SimpleNamespace(operational=False, status_msg="maintenance")
    availability = adapter.availability()
    assert not availability.available
    assert availability.reason == "maintenance"
    with pytest.raises(BackendUnavailableError, match="maintenance"):
        adapter.preflight((SimpleNamespace(num_qubits=1),), 10)


def test_iqm_restore_retrieves_once_and_validates_reported_id():
    from qudits_on_qubits.experiments.backends import IQMAdapter

    backend = _Backend()
    adapter = IQMAdapter(IQMHardware("garnet"), backend=backend)
    restored = adapter.restore_job("iqm-7", circuit_count=1, shots=5)
    assert backend.retrieve_calls == ["iqm-7"]
    assert restored.handle.job_id() == "iqm-7"

    backend.retrieve_job = lambda _job_id: _Job("iqm-other")
    with pytest.raises(JobResultError, match="does not match"):
        adapter.restore_job("iqm-7")


def test_iqm_restore_and_compile_failures_are_sanitized():
    from qudits_on_qubits.experiments.backends import IQMAdapter

    sensitive_text = "password=iqm-secret"
    backend = _Backend()
    backend.retrieve_job = lambda _job_id: (_ for _ in ()).throw(RuntimeError(sensitive_text))

    def failing_transpiler(*_args, **_kwargs):
        raise RuntimeError(sensitive_text)

    adapter = IQMAdapter(
        IQMHardware("garnet"),
        backend=backend,
        transpiler=failing_transpiler,
    )
    with pytest.raises(JobResultError) as caught:
        adapter.restore_job("iqm-7")
    _assert_sanitized(caught, sensitive_text)
    with pytest.raises(BackendCompatibilityError) as caught:
        adapter.compile((object(),), TranspilationConfig())
    _assert_sanitized(caught, sensitive_text)


def test_iqm_metadata_is_json_safe_and_contains_target_calibration_and_provider():
    from qudits_on_qubits.experiments.backends import IQMAdapter

    adapter = IQMAdapter(IQMHardware("garnet"), backend=_Backend())
    identity = adapter.resolve()
    metadata = adapter.metadata()
    assert identity.metadata["calibration_set_id"] == "cal-17"
    assert identity.metadata["target"] == "iqm:garnet"
    assert identity.metadata["provider"] == "iqm"
    assert metadata["identity"]["metadata"]["calibration_set_id"] == "cal-17"
    assert "token" not in repr(metadata).lower()


def test_iqm_metadata_helper_values_are_normalized_and_credentials_are_dropped(monkeypatch):
    from qudits_on_qubits.experiments.backends import IQMAdapter
    import qudits_on_qubits.experiments.backends.iqm as module

    monkeypatch.setattr(
        module,
        "backend_metadata",
        lambda *_args, **_kwargs: {
            "safe": {"items": (1, 2)},
            "credential": "https://user:password@example.invalid/profile",
            "token_value": "token=metadata-secret",
        },
    )
    metadata = IQMAdapter(IQMHardware("garnet"), backend=_Backend()).resolve().metadata
    assert metadata["safe"] == {"items": (1, 2)}
    assert "password@example" not in repr(metadata)
    assert "metadata-secret" not in repr(metadata)


def test_iqm_spec_rejects_credentialed_device_url_without_echoing_it():
    sensitive_text = "https://user:password@example.invalid/device"
    with pytest.raises(ExperimentValidationError, match="device") as caught:
        IQMHardware(sensitive_text)
    _assert_sanitized(caught, sensitive_text)
