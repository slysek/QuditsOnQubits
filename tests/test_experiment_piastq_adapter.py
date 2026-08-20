from __future__ import annotations

from types import SimpleNamespace
import traceback

import pytest

from qudits_on_qubits.experiments.errors import (
    BackendCompatibilityError,
    BackendUnavailableError,
    ExperimentValidationError,
    JobResultError,
    JobSubmissionError,
    OptionalDependencyError,
)
from qudits_on_qubits.experiments.models import PiastQHardware, TranspilationConfig


class _PiastJob:
    def __init__(self, counts=({"0": 7}, {"1": 7}), job_id="piast-3"):
        self._counts = counts
        self._job_id = job_id
        self.result_calls = []

    def job_id(self):
        return self._job_id

    def result(self, **kwargs):
        self.result_calls.append(kwargs)
        return SimpleNamespace(status="DONE", time_taken=1.25)

    def counts(self):
        return self._counts


class _PiastBackend:
    name = "direct-access"
    num_qubits = 20

    def status(self):
        return SimpleNamespace(operational=True)


def _adapter(*, job=None, env=None, spec=None, client_attrs=None):
    from qudits_on_qubits.experiments.backends import PiastQAdapter

    backend = _PiastBackend()
    client_calls = []
    sampler_calls = []
    jobs = []

    class Client:
        def __init__(self, **kwargs):
            client_calls.append(kwargs)
            self.backend = backend
            for key, value in (client_attrs or {}).items():
                setattr(self, key, value)

    class Sampler:
        def __init__(self, actual_backend, *, options):
            sampler_calls.append((actual_backend, options))

        def run(self, circuits, *, shots):
            sampler_calls.append((circuits, shots))
            actual_job = job or _PiastJob()
            jobs.append(actual_job)
            return actual_job

    adapter = PiastQAdapter(
        spec or PiastQHardware(mode="managed", owner="team"),
        client_type=Client,
        sampler_type=Sampler,
        env_loader=lambda _path: dict(env or {"token": "hidden-token", "dashboard_api_key": "hidden-key"}),
        poll_interval=0.25,
    )
    return adapter, backend, client_calls, sampler_calls, jobs


def _assert_sanitized(caught, secret):
    rendered = "".join(traceback.format_exception(caught.type, caught.value, caught.tb))
    assert caught.value.__cause__ is None
    assert secret not in str(caught.value)
    assert secret not in rendered


def test_piast_constructs_client_with_exact_mode_owner_and_env_without_metadata_secrets():
    adapter, backend, client_calls, _, _ = _adapter()
    identity = adapter.resolve()
    assert client_calls == [
        {
            "token": "hidden-token",
            "dashboard_api_key": "hidden-key",
            "mode": "managed",
            "owner": "team",
        }
    ]
    assert adapter.backend is backend
    assert identity.kind == "piastq"
    assert identity.name == "direct-access"
    assert "hidden-token" not in repr(adapter.metadata())
    assert "hidden-key" not in repr(adapter.metadata())


@pytest.mark.parametrize(
    "unsafe_owner",
    [
        "https://user:password@example.invalid/team",
        "team?dashboard_key=value",
        "team#fragment",
        "team\nops",
        "team owner",
        "x" * 129,
    ],
)
def test_piast_rejects_unsafe_owner_before_environment_or_client(unsafe_owner):
    from qudits_on_qubits.experiments.backends import PiastQAdapter

    calls = []

    class Client:
        def __init__(self, **_kwargs):
            calls.append("client")

    def env_loader(_path):
        calls.append("environment")
        return {}

    expected_error = (
        ExperimentValidationError
        if unsafe_owner in {
            "https://user:password@example.invalid/team",
            "team\nops",
        }
        else BackendCompatibilityError
    )
    with pytest.raises(expected_error, match="owner") as caught:
        specification = PiastQHardware(mode="managed", owner=unsafe_owner)
        PiastQAdapter(
            specification,
            client_type=Client,
            sampler_type=object,
            env_loader=env_loader,
        )
    assert not calls
    _assert_sanitized(caught, unsafe_owner)


def test_piast_passes_managed_mode_exactly():
    adapter, _, client_calls, _, _ = _adapter(
        spec=PiastQHardware(mode="managed", owner="owner")
    )
    adapter.resolve()
    assert client_calls[0]["mode"] == "managed"
    assert client_calls[0]["owner"] == "owner"


def test_piast_compile_targets_client_backend_with_aqt_and_config_options(monkeypatch):
    from qudits_on_qubits.experiments.backends import PiastQAdapter
    import qudits_on_qubits.experiments.backends.piastq as module

    adapter, backend, _, _, _ = _adapter()
    source = (object(), object())
    outputs = [object(), object()]
    seen = {}

    def fake_transpile(circuits, *, backend, **options):
        seen.update(circuits=tuple(circuits), backend=backend, options=options)
        return outputs

    monkeypatch.setattr(module, "transpile", fake_transpile)
    config = TranspilationConfig(1, 9, "dense", "sabre", None)
    compiled = adapter.compile(source, config)
    assert seen == {
        "circuits": source,
        "backend": backend,
        "options": {
            "optimization_level": 1,
            "seed_transpiler": 9,
            "layout_method": "dense",
            "routing_method": "sabre",
            "translation_method": "aqt",
            "scheduling_method": "aqt",
        },
    }
    assert compiled.circuits == tuple(outputs)
    assert compiled.target_identity == adapter.resolve()


def test_piast_submits_one_ordered_sampler_job_and_uses_counts_timeout_path():
    job = _PiastJob()
    adapter, backend, _, sampler_calls, _ = _adapter(job=job)
    circuits = (object(), object())
    submitted = adapter.submit(circuits, 7, {"cft_job_name": "batch"})
    assert sampler_calls[0] == (backend, {"cft_job_name": "batch"})
    assert sampler_calls[1][0] is circuits
    assert sampler_calls[1][0][0] is circuits[0]
    assert sampler_calls[1][1] == 7

    result = adapter.result(submitted, timeout=12.5)
    assert job.result_calls == [{"timeout": 12.5, "poll_interval": 0.25}]
    assert result.counts == ({"0": 7}, {"1": 7})
    assert result.status == "done"
    assert result.timing == {"time_taken": 1.25}


@pytest.mark.parametrize(
    ("counts", "message"),
    [
        (({"0": 7},), "circuit"),
        (({"0": 6}, {"1": 7}), "shots"),
        (({"2": 7}, {"1": 7}), "bitstring"),
    ],
)
def test_piast_result_validates_count_order_length_format_and_shots(counts, message):
    adapter, _, _, _, _ = _adapter(job=_PiastJob(counts=counts))
    submitted = adapter.submit((object(), object()), 7)
    with pytest.raises(JobResultError, match=message):
        adapter.result(submitted, timeout=3)


def test_piast_optional_dependency_and_environment_failures_are_typed_and_sanitized(monkeypatch):
    from qudits_on_qubits.experiments.backends import PiastQAdapter

    dependency_secret = "token=missing-dependency"
    adapter = PiastQAdapter(PiastQHardware(), env_loader=lambda _path: {})
    monkeypatch.setattr(
        adapter,
        "_load_piastq_types",
        lambda: (_ for _ in ()).throw(ModuleNotFoundError(dependency_secret)),
    )
    with pytest.raises(OptionalDependencyError, match="piastq") as caught:
        adapter.resolve()
    assert "install cft-piastq separately in this environment" in str(caught.value)
    _assert_sanitized(caught, dependency_secret)

    environment_secret = "api_key=bad-config"
    adapter = PiastQAdapter(
        PiastQHardware(),
        client_type=lambda **_kwargs: None,
        sampler_type=object,
        env_loader=lambda _path: (_ for _ in ()).throw(RuntimeError(environment_secret)),
    )
    assert not adapter.availability().available
    with pytest.raises(BackendUnavailableError) as caught:
        adapter.resolve()
    _assert_sanitized(caught, environment_secret)


def test_piast_hardware_unavailable_and_submission_failures_are_sanitized():
    adapter, backend, _, _, _ = _adapter()
    backend.status = lambda: SimpleNamespace(operational=False, status_msg="offline")
    assert not adapter.availability().available
    with pytest.raises(BackendUnavailableError, match="offline"):
        adapter.preflight((object(),), 1)

    sensitive_text = "secret=piast-submit"

    class FailingSampler:
        def __init__(self, _backend, *, options):
            pass

        def run(self, _circuits, *, shots):
            raise RuntimeError(sensitive_text)

    adapter._sampler_type = FailingSampler
    backend.status = lambda: SimpleNamespace(operational=True)
    with pytest.raises(JobSubmissionError) as caught:
        adapter.submit((object(),), 1)
    _assert_sanitized(caught, sensitive_text)


def test_piast_resume_is_false_and_restore_never_resubmits_without_provider_api():
    adapter, _, _, sampler_calls, _ = _adapter()
    assert not adapter.capabilities().supports_resume
    with pytest.raises(BackendCompatibilityError, match="resume"):
        adapter.restore_job("piast-3")
    assert not sampler_calls


def test_piast_restore_uses_actual_client_api_and_validates_job_id():
    restored_job = _PiastJob(job_id="piast-3")
    retrieve = lambda job_id: restored_job if job_id == "piast-3" else None
    adapter, _, _, _, _ = _adapter(client_attrs={"retrieve_job": retrieve})
    adapter.resolve()
    assert adapter.capabilities().supports_resume
    restored = adapter.restore_job("piast-3", circuit_count=2, shots=7)
    assert restored.handle is restored_job

    adapter._client.retrieve_job = lambda _job_id: _PiastJob(job_id="other")
    with pytest.raises(JobResultError, match="does not match"):
        adapter.restore_job("piast-3")


def test_piast_rejects_shots_in_sampler_options():
    adapter, _, _, _, _ = _adapter()
    with pytest.raises(BackendCompatibilityError, match="shots"):
        adapter.submit((object(),), 5, {"shots": 5})


def test_piast_drops_credentialed_backend_url_from_identity_and_metadata():
    adapter, backend, _, _, _ = _adapter()
    backend.name = "https://user:password@example.invalid/device"
    assert adapter.resolve().name == "piast"
    assert "password@example" not in repr(adapter.metadata())
