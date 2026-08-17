"""Optional PiastQ hardware adapter."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from dotenv import dotenv_values
from qiskit import transpile

from ..errors import (
    BackendCompatibilityError,
    BackendUnavailableError,
    JobResultError,
    JobSubmissionError,
    OptionalDependencyError,
)
from ..models import PiastQHardware, TranspilationConfig
from .base import (
    Availability,
    BackendCapabilities,
    BackendIdentity,
    BaseBackendAdapter,
    CompiledBatch,
    ExecutionResult,
    SubmittedJob,
    _exception_name,
    _extract_job_id,
    _positive_integer,
    _result_status,
    _result_timing,
    _safe_identifier,
    _validate_counts,
    _validated_circuit_tuple,
    _validated_run_options,
)
from .iqm import _backend_availability, _backend_version, _max_circuits, _num_qubits


_CLIENT_ENVIRONMENT = {
    "PCSS_TOKEN": "token",
    "PCSS_QAPI_TOKEN": "token",
    "CFT_PIASTQ_DASHBOARD_API_URL": "dashboard_api_url",
    "CFT_PIASTQ_DASHBOARD_API_KEY": "dashboard_api_key",
}


def _load_client_environment(env_path: Path | None) -> dict[str, str]:
    values: dict[str, Any] = dict(os.environ)
    if env_path is not None:
        if not env_path.is_file():
            raise RuntimeError("configured PiastQ environment file does not exist")
        values.update(dotenv_values(env_path))
    kwargs: dict[str, str] = {}
    for environment_name, argument_name in _CLIENT_ENVIRONMENT.items():
        value = values.get(environment_name)
        if isinstance(value, str) and value.strip():
            if argument_name == "token" and argument_name in kwargs:
                continue
            kwargs[argument_name] = value.strip()
    return kwargs


class PiastQAdapter(BaseBackendAdapter):
    """PiastQ sampler adapter with optional dependency isolation."""

    def __init__(
        self,
        spec: PiastQHardware,
        client_type: Any = None,
        sampler_type: Any = None,
        env_loader: Any = None,
        poll_interval: float = 5.0,
    ) -> None:
        if not isinstance(spec, PiastQHardware):
            raise BackendCompatibilityError("PiastQAdapter requires a PiastQHardware specification")
        if client_type is not None and not callable(client_type):
            raise BackendCompatibilityError("PiastQ client type must be callable")
        if sampler_type is not None and not callable(sampler_type):
            raise BackendCompatibilityError("PiastQ sampler type must be callable")
        if env_loader is not None and not callable(env_loader):
            raise BackendCompatibilityError("PiastQ environment loader must be callable")
        if (
            isinstance(poll_interval, bool)
            or not isinstance(poll_interval, (int, float))
            or not math.isfinite(poll_interval)
            or poll_interval <= 0
        ):
            raise BackendCompatibilityError("PiastQ poll_interval must be a positive finite number")
        self._spec = spec
        self._client_type = client_type
        self._sampler_type = sampler_type
        self._env_loader = env_loader or _load_client_environment
        self._poll_interval = float(poll_interval)
        self._client: Any = None
        self._backend: Any = None
        self._identity: BackendIdentity | None = None

    def _load_piastq_types(self) -> tuple[Any, Any]:
        from cft_piastq import PiastQClient, PiastQSampler

        return PiastQClient, PiastQSampler

    def _types(self) -> tuple[Any, Any]:
        if self._client_type is not None and self._sampler_type is not None:
            return self._client_type, self._sampler_type
        try:
            client_type, sampler_type = self._load_piastq_types()
        except (ImportError, ModuleNotFoundError) as error:
            raise OptionalDependencyError(
                "PiastQHardware requires cft-piastq; install it with "
                "`pip install -e .[piastq]` "
                f"({_exception_name(error)})"
            ) from None
        self._client_type = self._client_type or client_type
        self._sampler_type = self._sampler_type or sampler_type
        return self._client_type, self._sampler_type

    def _client_instance(self) -> Any:
        if self._client is not None:
            return self._client
        client_type, _ = self._types()
        try:
            loaded = self._env_loader(self._spec.env_path)
            if not isinstance(loaded, Mapping):
                raise TypeError("environment loader returned a non-mapping")
            environment_kwargs = {
                key: value
                for key, value in loaded.items()
                if key in {"token", "dashboard_api_url", "dashboard_api_key"}
                and value is not None
            }
            client = client_type(
                **environment_kwargs,
                mode=self._spec.mode,
                owner=self._spec.owner,
            )
            backend = client.backend
        except (ImportError, ModuleNotFoundError) as error:
            raise OptionalDependencyError(
                "PiastQHardware requires cft-piastq; install it with "
                "`pip install -e .[piastq]` "
                f"({_exception_name(error)})"
            ) from None
        except Exception as error:
            raise BackendUnavailableError(
                f"could not create PiastQ client ({_exception_name(error)})"
            ) from None
        if backend is None:
            raise BackendUnavailableError("PiastQ client did not provide a backend")
        self._client = client
        self._backend = backend
        return client

    @property
    def backend(self) -> Any:
        self._client_instance()
        return self._backend

    def resolve(self) -> BackendIdentity:
        backend = self.backend
        if self._identity is None:
            name = _backend_name(backend)
            self._identity = BackendIdentity(
                kind="piastq",
                name=name,
                provider="cft-piastq",
                version=_backend_version(backend),
                metadata={
                    "target": f"piastq:{name}",
                    "provider": "cft-piastq",
                    "mode": self._spec.mode,
                    "owner": self._spec.owner,
                    "backend_num_qubits": _num_qubits(backend) or 0,
                },
            )
        return self._identity

    def capabilities(self) -> BackendCapabilities:
        client = self._client_instance()
        return BackendCapabilities(
            local=False,
            supports_resume=callable(getattr(client, "retrieve_job", None)),
            max_circuits=_max_circuits(self.backend),
        )

    def availability(self) -> Availability:
        try:
            backend = self.backend
        except OptionalDependencyError as error:
            return Availability(False, f"PiastQ dependency is unavailable ({_exception_name(error)})")
        except BackendUnavailableError as error:
            return Availability(False, f"PiastQ backend is unavailable ({_exception_name(error)})")
        return _backend_availability(backend, "PiastQ")

    def preflight(self, circuits: Sequence[Any], shots: int) -> None:
        batch = _validated_circuit_tuple(circuits)
        super().preflight(batch, shots)
        capacity = _num_qubits(self.backend)
        if capacity is None:
            return
        for circuit in batch:
            required = _num_qubits(circuit)
            if required is not None and required > capacity:
                raise BackendCompatibilityError(
                    f"circuit requires {required} qubits but PiastQ backend provides {capacity} qubits"
                )

    def compile(self, circuits: Sequence[Any], config: TranspilationConfig) -> CompiledBatch:
        batch = _validated_circuit_tuple(circuits)
        if not isinstance(config, TranspilationConfig):
            raise BackendCompatibilityError("compile requires TranspilationConfig")
        options = {
            key: value
            for key, value in config.to_safe_dict().items()
            if value is not None
        }
        options.setdefault("translation_method", "aqt")
        options.setdefault("scheduling_method", "aqt")
        try:
            compiled = transpile(list(batch), backend=self.backend, **options)
        except Exception as error:
            identity = self.resolve()
            raise BackendCompatibilityError(
                f"could not compile circuits for backend {identity.kind}:{identity.name} "
                f"({_exception_name(error)})"
            ) from None
        compiled_batch = tuple(compiled) if isinstance(compiled, (list, tuple)) else (compiled,)
        return CompiledBatch(compiled_batch, self.resolve(), {"transpilation": options})

    def submit(
        self,
        circuits: Sequence[Any],
        shots: int,
        options: Mapping[str, Any] | None = None,
    ) -> SubmittedJob:
        batch = _validated_circuit_tuple(circuits)
        self.preflight(batch, shots)
        sampler_options = _validated_run_options(options)
        _, sampler_type = self._types()
        try:
            sampler = sampler_type(self.backend, options=sampler_options)
            handle = sampler.run(batch, shots=shots)
        except Exception as error:
            identity = self.resolve()
            raise JobSubmissionError(
                f"backend {identity.kind}:{identity.name} rejected job submission "
                f"({_exception_name(error)})"
            ) from None
        try:
            job_id = _extract_job_id(handle, allow_local_fallback=False)
        except Exception as error:
            raise JobSubmissionError(
                f"submitted PiastQ job did not provide a usable job ID ({_exception_name(error)})"
            ) from None
        return SubmittedJob(job_id, handle, self.resolve(), len(batch), shots)

    def result(self, submitted: SubmittedJob, timeout: float | None = None) -> ExecutionResult:
        if not isinstance(submitted, SubmittedJob):
            raise JobResultError("result requires a SubmittedJob")
        if submitted.target_identity != self.resolve():
            raise BackendCompatibilityError("submitted job target does not match this adapter")
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise JobResultError("result timeout must be a positive finite number or None")
        try:
            raw_result = submitted.handle.result(
                timeout=timeout,
                poll_interval=self._poll_interval,
            )
            raw_counts = submitted.handle.counts()
        except Exception as error:
            raise JobResultError(
                f"could not retrieve result for job {submitted.job_id} ({_exception_name(error)})"
            ) from None
        try:
            if isinstance(raw_counts, Mapping):
                count_sets = (raw_counts,)
            else:
                count_sets = tuple(raw_counts)
            validated = tuple(_validate_counts(item) for item in count_sets)
            if submitted.circuit_count is not None and len(validated) != submitted.circuit_count:
                raise JobResultError(
                    f"result circuit count {len(validated)} does not match submitted circuit count "
                    f"{submitted.circuit_count}"
                )
            if submitted.shots is not None and any(
                sum(item.values()) != submitted.shots for item in validated
            ):
                raise JobResultError("result counts do not sum to expected shots")
        except JobResultError:
            raise
        except Exception as error:
            raise JobResultError(
                f"result for job {submitted.job_id} has an unsupported counts format "
                f"({_exception_name(error)})"
            ) from None
        return ExecutionResult(
            validated,
            submitted.job_id,
            submitted.target_identity,
            status=_result_status(submitted.handle, raw_result),
            timing=_result_timing(raw_result),
        )

    def restore_job(
        self,
        job_id: str,
        *,
        circuit_count: int | None = None,
        shots: int | None = None,
    ) -> SubmittedJob:
        job_id = _safe_identifier(job_id, "job_id")
        _positive_integer(circuit_count, "circuit_count", optional=True)
        _positive_integer(shots, "shots", optional=True)
        client = self._client_instance()
        retrieve = getattr(client, "retrieve_job", None)
        if not callable(retrieve):
            raise BackendCompatibilityError("PiastQ provider does not support resume")
        try:
            handle = retrieve(job_id)
        except Exception as error:
            raise JobResultError(
                f"could not restore PiastQ job {job_id} ({_exception_name(error)})"
            ) from None
        if handle is None:
            raise JobResultError(f"could not restore PiastQ job {job_id}")
        try:
            actual_job_id = _extract_job_id(handle, allow_local_fallback=False)
        except Exception as error:
            raise JobResultError(
                f"restored PiastQ job did not provide the requested job ID ({_exception_name(error)})"
            ) from None
        if actual_job_id != job_id:
            raise JobResultError("restored PiastQ job ID does not match requested job ID")
        return SubmittedJob(
            job_id,
            handle,
            self.resolve(),
            circuit_count,
            shots,
            {"restored": True},
        )


def _backend_name(backend: Any) -> str:
    try:
        value = getattr(backend, "name", None)
        value = value() if callable(value) else value
    except Exception:
        value = None
    if not isinstance(value, str) or not value:
        return "piast"
    if any(
        marker in value.lower()
        for marker in ("token=", "api_key=", "password=", "secret=")
    ):
        return "piast"
    if "://" in value and "@" in value.split("://", 1)[1].split("/", 1)[0]:
        return "piast"
    return value


__all__ = ["PiastQAdapter"]
