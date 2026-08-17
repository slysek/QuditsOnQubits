from __future__ import annotations

from types import SimpleNamespace
import traceback

import pytest

from qudits_on_qubits.experiments.errors import (
    BackendCompatibilityError,
    BackendUnavailableError,
)
from qudits_on_qubits.experiments.models import (
    AerIdeal,
    CustomBackend,
    IQMHardware,
    NoisySimulator,
    PiastQHardware,
    TranspilationConfig,
)


class _Job:
    def job_id(self):
        return "noisy-1"

    def result(self, **_kwargs):
        return SimpleNamespace(get_counts=lambda: {"0": 4})


class _Backend:
    name = "source-device"
    num_qubits = 5
    calibration_set_id = "cal-91"
    local = True

    def __init__(self):
        self.calls = []

    def run(self, circuits, **options):
        self.calls.append((circuits, options))
        return _Job()


def _assert_sanitized(caught, secret):
    rendered = "".join(traceback.format_exception(caught.type, caught.value, caught.tb))
    assert caught.value.__cause__ is None
    assert secret not in str(caught.value)
    assert secret not in rendered


def test_noisy_iqm_uses_profile_conversion_once_and_records_identity_calibration(monkeypatch):
    from qudits_on_qubits.experiments.backends import build_noisy_adapter
    import qudits_on_qubits.experiments.backends.aer as module

    iqm_backend = _Backend()
    fake_backend = object()
    simulator = _Backend()
    fake_calls = []
    simulator_calls = []
    transpile_seen = {}

    def fake_factory(source):
        fake_calls.append(source)
        return fake_backend

    def simulator_factory(profile):
        simulator_calls.append(profile)
        return simulator

    def fake_transpile(circuits, *, backend, **options):
        transpile_seen.update(circuits=tuple(circuits), backend=backend, options=options)
        return list(circuits)

    monkeypatch.setattr(module, "transpile", fake_transpile)
    spec = NoisySimulator(source=IQMHardware("garnet"), identity="garnet-noise")
    adapter = build_noisy_adapter(
        spec,
        iqm_backend=iqm_backend,
        fake_backend_factory=fake_factory,
        simulator_factory=simulator_factory,
    )
    identity = adapter.resolve()
    compiled = adapter.compile((object(),), TranspilationConfig(optimization_level=2))
    adapter.submit(compiled.circuits, 4)

    assert fake_calls == [iqm_backend]
    assert simulator_calls == [fake_backend]
    assert identity.kind == "noisy"
    assert identity.name == "garnet-noise"
    assert identity.emulates == "iqm:garnet"
    assert identity.metadata["calibration_set_id"] == "cal-91"
    assert identity.metadata["source_identity"] == "iqm:garnet"
    assert identity.metadata["profile_provenance"] == "iqm-fake-backend"
    assert transpile_seen["backend"] is iqm_backend
    assert simulator.calls[0][0] is compiled.circuits
    assert simulator.calls[0][1] == {"shots": 4}
    assert "error_profile" not in repr(adapter.metadata())


def test_noisy_explicit_model_uses_target_only_for_compile_and_simulator_for_execution(monkeypatch):
    from qudits_on_qubits.experiments.backends import build_noisy_adapter
    import qudits_on_qubits.experiments.backends.aer as module

    target = _Backend()
    simulator = _Backend()
    noise_model = object()
    factory_calls = []
    transpile_targets = []

    def simulator_factory(*, noise_model):
        factory_calls.append(noise_model)
        return simulator

    monkeypatch.setattr(
        module,
        "transpile",
        lambda circuits, *, backend, **_options: transpile_targets.append(backend) or list(circuits),
    )
    adapter = build_noisy_adapter(
        NoisySimulator(noise_model=noise_model, target_backend=target, identity="explicit"),
        simulator_factory=simulator_factory,
    )
    compiled = adapter.compile((object(),), TranspilationConfig())
    adapter.submit(compiled.circuits, 4, {"memory": True})
    assert factory_calls == [noise_model]
    assert transpile_targets == [target]
    assert simulator.calls[0][1] == {"shots": 4, "memory": True}
    assert adapter.resolve().metadata["profile_provenance"] == "explicit-noise-model"


def test_noisy_generic_source_uses_source_target_without_ideal_fallback():
    from qudits_on_qubits.experiments.backends import build_noisy_adapter

    source = _Backend()
    simulator = _Backend()
    calls = []
    adapter = build_noisy_adapter(
        NoisySimulator(source=source),
        simulator_factory=lambda profile: calls.append(profile) or simulator,
    )
    assert calls == [source]
    assert adapter.resolve().kind == "noisy"
    assert adapter.resolve().metadata["source_identity"] == "custom:source-device"


@pytest.mark.parametrize("secret", ["token=profile-secret", "password=simulator-secret"])
def test_noisy_conversion_failure_is_unavailable_sanitized_and_never_falls_back(secret):
    from qudits_on_qubits.experiments.backends import build_noisy_adapter

    source = _Backend()
    calls = []

    def failing_factory(_source):
        calls.append(_source)
        raise RuntimeError(secret)

    with pytest.raises(BackendUnavailableError, match="noise profile") as caught:
        build_noisy_adapter(NoisySimulator(source=source), simulator_factory=failing_factory)
    assert calls == [source]
    _assert_sanitized(caught, secret)


def test_noisy_rejects_wrong_spec_and_unusable_simulator():
    from qudits_on_qubits.experiments.backends import build_noisy_adapter

    with pytest.raises(BackendCompatibilityError, match="NoisySimulator"):
        build_noisy_adapter(AerIdeal())
    with pytest.raises(BackendUnavailableError, match="run"):
        build_noisy_adapter(
            NoisySimulator(source=_Backend()),
            simulator_factory=lambda _source: object(),
        )


def test_noisy_metadata_drops_credentialed_source_url():
    from qudits_on_qubits.experiments.backends import build_noisy_adapter

    source = _Backend()
    source.name = "https://user:password@example.invalid/device"
    adapter = build_noisy_adapter(
        NoisySimulator(source=source),
        simulator_factory=lambda _source: _Backend(),
    )
    assert "password@example" not in repr(adapter.metadata())


def test_singleton_registry_has_all_five_specs_but_fresh_registry_is_empty():
    from qudits_on_qubits.experiments.backends import (
        AerAdapter,
        BackendAdapterRegistry,
        CustomBackendAdapter,
        IQMAdapter,
        NoisyAerAdapter,
        PiastQAdapter,
        create_backend_adapter,
    )

    source_backend = _Backend()
    assert isinstance(create_backend_adapter(AerIdeal(), simulator=source_backend), AerAdapter)
    assert isinstance(create_backend_adapter(CustomBackend(source_backend)), CustomBackendAdapter)
    assert isinstance(create_backend_adapter(IQMHardware("garnet"), backend=source_backend), IQMAdapter)

    class Client:
        def __init__(self, **_kwargs):
            self.backend = source_backend

    assert isinstance(
        create_backend_adapter(
            PiastQHardware(),
            client_type=Client,
            sampler_type=object,
            env_loader=lambda _path: {},
        ),
        PiastQAdapter,
    )
    assert isinstance(
        create_backend_adapter(
            NoisySimulator(source=source_backend),
            simulator_factory=lambda _source: _Backend(),
        ),
        NoisyAerAdapter,
    )
    with pytest.raises(BackendCompatibilityError, match="unsupported"):
        BackendAdapterRegistry().create(IQMHardware("garnet"), backend=source_backend)
