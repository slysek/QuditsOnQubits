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
from qudits_on_qubits.experiments.models import (
    IQMHardware,
    IQMQubitSelectorConfig,
    TranspilationConfig,
)


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


def test_iqm_suggest_layouts_uses_resolved_backend_and_safe_config():
    from qudits_on_qubits.experiments.backends import IQMAdapter

    backend = _Backend()
    circuit = QuantumCircuit(2)
    calls = []

    def selector(actual_backend, actual_circuit, config):
        calls.append((actual_backend, actual_circuit, config))
        return ([[4, 7], [8, 9]], [0.02, 0.03], "1.1.0")

    config = IQMQubitSelectorConfig(
        top_k=2,
        num_trials=500,
        cost_function="cz",
        readout_mode="none",
        remove_qubits=(5,),
    )
    adapter = IQMAdapter(
        IQMHardware("garnet"),
        backend=backend,
        layout_selector=selector,
    )

    result = adapter.suggest_layouts(circuit, config)

    assert calls == [(backend, circuit, config)]
    assert result == {
        "provider": "iqm-qubit-selector",
        "version": "1.1.0",
        "configuration": config.to_safe_dict(),
        "layouts": ((4, 7), (8, 9)),
        "costs": (0.02, 0.03),
    }


def test_iqm_rejects_non_callable_layout_selector():
    from qudits_on_qubits.experiments.backends import IQMAdapter

    with pytest.raises(BackendCompatibilityError, match="layout selector"):
        IQMAdapter(
            IQMHardware("garnet"),
            backend=_Backend(),
            layout_selector=object(),
        )


def test_iqm_preserves_falsey_callable_layout_selector():
    from qudits_on_qubits.experiments.backends import IQMAdapter

    calls = []

    class FalseySelector:
        def __bool__(self):
            return False

        def __call__(self, backend, circuit, config):
            calls.append((backend, circuit, config))
            return (((4, 7),), (0.02,), "1.1.0")

    backend = _Backend()
    circuit = QuantumCircuit(2)
    config = IQMQubitSelectorConfig(top_k=1)
    adapter = IQMAdapter(
        IQMHardware("garnet"),
        backend=backend,
        layout_selector=FalseySelector(),
    )

    result = adapter.suggest_layouts(circuit, config)

    assert calls == [(backend, circuit, config)]
    assert result["layouts"] == ((4, 7),)


@pytest.mark.parametrize(
    ("cost_function", "readout_mode"),
    [
        ("cz", "none"),
        ("cz", "fidelity"),
        ("cz", "qndness"),
        ("clifford", "none"),
        ("clifford", "fidelity"),
        ("clifford", "qndness"),
    ],
)
def test_default_layout_selector_translates_safe_config_to_iqm_api(
    monkeypatch,
    cost_function,
    readout_mode,
):
    import importlib.metadata
    import sys
    from types import ModuleType

    from qudits_on_qubits.experiments.backends.iqm import (
        _default_layout_selector,
    )

    calls = {}
    cost_values = {"cz": object(), "clifford": object()}
    readout_values = {
        "none": object(),
        "fidelity": object(),
        "qndness": object(),
    }

    class FakeEvaluator:
        def __init__(self, **kwargs):
            calls["kwargs"] = kwargs

        def get_top_layouts(self, *, num_layouts):
            calls["num_layouts"] = num_layouts
            return [[4, 7], [8, 9]], [0.02, 0.03]

    module = ModuleType("iqm.qubit_selector.qubit_selector")
    module.CostEvaluator = FakeEvaluator
    module.CostFunction = SimpleNamespace(
        GATE_COST_CZ=cost_values["cz"],
        GATE_COST_CLIFFORD=cost_values["clifford"],
    )
    module.ReadoutMode = SimpleNamespace(
        NONE=readout_values["none"],
        FIDELITY=readout_values["fidelity"],
        QNDNESS=readout_values["qndness"],
    )
    monkeypatch.setitem(
        sys.modules,
        "iqm.qubit_selector.qubit_selector",
        module,
    )
    version_calls = []
    monkeypatch.setattr(
        importlib.metadata,
        "version",
        lambda package: version_calls.append(package) or "1.1.0",
    )
    backend = _Backend()
    circuit = QuantumCircuit(2)
    removed = (5,) if readout_mode != "none" else ()
    config = IQMQubitSelectorConfig(
        top_k=2,
        num_trials=500,
        cost_function=cost_function,
        readout_mode=readout_mode,
        remove_qubits=removed,
    )

    result = _default_layout_selector(backend, circuit, config)

    assert result == ([[4, 7], [8, 9]], [0.02, 0.03], "1.1.0")
    assert calls == {
        "kwargs": {
            "backend": backend,
            "quantum_circuit": circuit,
            "cost_function": cost_values[cost_function],
            "readoutmode": readout_values[readout_mode],
            "remove_qubits": list(removed) if removed else None,
            "num_trials": 500,
        },
        "num_layouts": 2,
    }
    assert version_calls == ["iqm-qubit-selector"]


@pytest.mark.parametrize(
    ("circuit", "config", "message"),
    [
        (object(), IQMQubitSelectorConfig(), "QuantumCircuit"),
        (QuantumCircuit(2), object(), "IQMQubitSelectorConfig"),
    ],
)
def test_iqm_suggest_layouts_rejects_invalid_inputs_without_calling_selector(
    circuit,
    config,
    message,
):
    from qudits_on_qubits.experiments.backends import IQMAdapter

    adapter = IQMAdapter(
        IQMHardware("garnet"),
        backend=_Backend(),
        layout_selector=lambda *_args: pytest.fail("selector called"),
    )

    with pytest.raises(BackendCompatibilityError, match=message):
        adapter.suggest_layouts(circuit, config)


def test_iqm_suggest_layouts_requires_backend_capacity():
    from qudits_on_qubits.experiments.backends import IQMAdapter

    backend = _Backend()
    backend.num_qubits = None
    adapter = IQMAdapter(
        IQMHardware("garnet"),
        backend=backend,
        layout_selector=lambda *_args: pytest.fail("selector called"),
    )

    with pytest.raises(BackendCompatibilityError, match="qubit capacity"):
        adapter.suggest_layouts(QuantumCircuit(2), IQMQubitSelectorConfig())


@pytest.mark.parametrize("failing_attribute", ["num_qubits", "target"])
def test_iqm_suggest_layouts_redacts_capacity_lookup_failure(failing_attribute):
    from qudits_on_qubits.experiments.backends import IQMAdapter

    sensitive_message = "token=selector-test-value"

    class FailingCapacityBackend(_Backend):
        @property
        def num_qubits(self):
            if failing_attribute == "num_qubits":
                raise RuntimeError(sensitive_message)
            return None

        @property
        def target(self):
            raise RuntimeError(sensitive_message)

    adapter = IQMAdapter(
        IQMHardware("garnet"),
        backend=FailingCapacityBackend(),
        layout_selector=lambda *_args: pytest.fail("selector called"),
    )

    with pytest.raises(BackendCompatibilityError) as caught:
        adapter.suggest_layouts(QuantumCircuit(2), IQMQubitSelectorConfig())

    assert str(caught.value) == (
        "IQM qubit selector requires backend qubit capacity"
    )
    _assert_sanitized(caught, "selector-test-value")


def test_iqm_suggest_layouts_propagates_memory_error_from_callable_capacity():
    from qudits_on_qubits.experiments.backends import IQMAdapter

    class ExhaustedCapacityBackend(_Backend):
        def num_qubits(self):
            raise MemoryError("capacity exhausted")

    adapter = IQMAdapter(
        IQMHardware("garnet"),
        backend=ExhaustedCapacityBackend(),
        layout_selector=lambda *_args: pytest.fail("selector called"),
    )

    with pytest.raises(MemoryError, match="capacity exhausted"):
        adapter.suggest_layouts(QuantumCircuit(2), IQMQubitSelectorConfig())


def test_iqm_suggest_layouts_propagates_memory_error_from_backend_loader():
    from qudits_on_qubits.experiments.backends import IQMAdapter

    def exhausted_loader(*_args, **_kwargs):
        raise MemoryError("backend resolution exhausted")

    adapter = IQMAdapter(
        IQMHardware("garnet"),
        backend_loader=exhausted_loader,
        layout_selector=lambda *_args: pytest.fail("selector called"),
    )

    with pytest.raises(MemoryError, match="backend resolution exhausted"):
        adapter.suggest_layouts(QuantumCircuit(2), IQMQubitSelectorConfig())


def test_iqm_suggest_layouts_rejects_removed_qubit_outside_capacity():
    from qudits_on_qubits.experiments.backends import IQMAdapter

    adapter = IQMAdapter(
        IQMHardware("garnet"),
        backend=_Backend(),
        layout_selector=lambda *_args: pytest.fail("selector called"),
    )

    with pytest.raises(BackendCompatibilityError, match="remove_qubits"):
        adapter.suggest_layouts(
            QuantumCircuit(2),
            IQMQubitSelectorConfig(remove_qubits=(20,)),
        )


@pytest.mark.parametrize(
    ("layouts", "costs", "version"),
    [
        ((), (), "1.1.0"),
        (((4, 7), (4, 7)), (0.02, 0.03), "1.1.0"),
        (((4, 4),), (0.02,), "1.1.0"),
        (((4,),), (0.02,), "1.1.0"),
        (((4, 20),), (0.02,), "1.1.0"),
        (((True, 7),), (0.02,), "1.1.0"),
        (((4, 7),), (), "1.1.0"),
        (((4, 7),), (-0.01,), "1.1.0"),
        (((4, 7),), (float("nan"),), "1.1.0"),
        (((4, 7),), (float("inf"),), "1.1.0"),
        (((4, 7),), (True,), "1.1.0"),
        (((4, 7), (8, 9)), (0.03, 0.02), "1.1.0"),
        (((4, 7),), (0.02,), ""),
        (((4, 7),), (0.02,), object()),
        (((4, 7),), (0.02,), "token" + "=selector-test-value"),
        ("47", (0.02,), "1.1.0"),
        (((4, 7),), b"0", "1.1.0"),
        (("47",), (0.02,), "1.1.0"),
        (
            tuple((index, index + 1) for index in range(11)),
            tuple(index / 100 for index in range(11)),
            "1.1.0",
        ),
    ],
)
def test_iqm_suggest_layouts_rejects_malformed_provider_output(
    layouts,
    costs,
    version,
):
    from qudits_on_qubits.experiments.backends import IQMAdapter

    adapter = IQMAdapter(
        IQMHardware("garnet"),
        backend=_Backend(),
        layout_selector=lambda *_args: (layouts, costs, version),
    )

    with pytest.raises(BackendCompatibilityError) as caught:
        adapter.suggest_layouts(QuantumCircuit(2), IQMQubitSelectorConfig())

    assert str(caught.value) == "IQM qubit selector output is invalid"
    assert caught.value.__cause__ is None
    if isinstance(version, str) and "selector-test-value" in version:
        _assert_sanitized(caught, "selector-test-value")


def test_iqm_suggest_layouts_rejects_layout_using_removed_qubit():
    from qudits_on_qubits.experiments.backends import IQMAdapter

    adapter = IQMAdapter(
        IQMHardware("garnet"),
        backend=_Backend(),
        layout_selector=lambda *_args: (((4, 7),), (0.02,), "1.1.0"),
    )

    with pytest.raises(BackendCompatibilityError) as caught:
        adapter.suggest_layouts(
            QuantumCircuit(2),
            IQMQubitSelectorConfig(remove_qubits=(7,)),
        )

    assert str(caught.value) == "IQM qubit selector output is invalid"
    assert caught.value.__cause__ is None


@pytest.mark.parametrize(
    "raw_result",
    [
        object(),
        (),
        ((), ()),
        ((), (), "1.1.0", object()),
        "abc",
    ],
)
def test_iqm_suggest_layouts_rejects_malformed_return_shape(raw_result):
    from qudits_on_qubits.experiments.backends import IQMAdapter

    adapter = IQMAdapter(
        IQMHardware("garnet"),
        backend=_Backend(),
        layout_selector=lambda *_args: raw_result,
    )

    with pytest.raises(BackendCompatibilityError) as caught:
        adapter.suggest_layouts(QuantumCircuit(2), IQMQubitSelectorConfig())

    assert str(caught.value) == "IQM qubit selector output is invalid"
    assert caught.value.__cause__ is None


def test_iqm_suggest_layouts_redacts_provider_failure():
    from qudits_on_qubits.experiments.backends import IQMAdapter

    sensitive_message = "Authorization: " + "Bearer selector-test-value"

    def fail(*_args):
        raise RuntimeError(sensitive_message)

    adapter = IQMAdapter(
        IQMHardware("garnet"),
        backend=_Backend(),
        layout_selector=fail,
    )
    with pytest.raises(BackendCompatibilityError) as caught:
        adapter.suggest_layouts(QuantumCircuit(2), IQMQubitSelectorConfig())

    _assert_sanitized(caught, "selector-test-value")
    assert str(caught.value) == (
        "IQM qubit selector failed for backend iqm:garnet (RuntimeError)"
    )


@pytest.mark.parametrize(
    "failure",
    [
        ModuleNotFoundError("token=selector-test-value"),
        ImportError("token=selector-test-value"),
    ],
)
def test_iqm_suggest_layouts_maps_missing_package_to_optional_dependency(
    failure,
):
    from qudits_on_qubits.experiments.backends import IQMAdapter

    def missing(*_args):
        raise failure

    adapter = IQMAdapter(
        IQMHardware("garnet"),
        backend=_Backend(),
        layout_selector=missing,
    )

    with pytest.raises(OptionalDependencyError) as caught:
        adapter.suggest_layouts(QuantumCircuit(2), IQMQubitSelectorConfig())

    assert str(caught.value) == (
        "IQM automatic layout selection requires iqm-qubit-selector "
        f"({type(failure).__name__})"
    )
    _assert_sanitized(caught, "selector-test-value")


@pytest.mark.parametrize(
    "failure",
    [
        KeyboardInterrupt("selector interrupted"),
        SystemExit("selector stopped"),
        MemoryError("selector exhausted"),
    ],
)
def test_iqm_suggest_layouts_propagates_critical_base_exceptions(failure):
    from qudits_on_qubits.experiments.backends import IQMAdapter

    def fail(*_args):
        raise failure

    adapter = IQMAdapter(
        IQMHardware("garnet"),
        backend=_Backend(),
        layout_selector=fail,
    )

    with pytest.raises(type(failure), match="selector"):
        adapter.suggest_layouts(QuantumCircuit(2), IQMQubitSelectorConfig())
