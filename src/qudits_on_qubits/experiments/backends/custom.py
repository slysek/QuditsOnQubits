"""Adapter for caller-supplied Qiskit-style backends."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from qiskit import transpile

from ..errors import BackendCompatibilityError, JobResultError
from ..models import CustomBackend, TranspilationConfig
from .base import (
    Availability,
    BackendCapabilities,
    BackendIdentity,
    BaseBackendAdapter,
    CompiledBatch,
    SubmittedJob,
    _exception_name,
    _extract_job_id,
    _positive_integer,
    _safe_identifier,
    _validated_circuit_tuple,
)


class CustomBackendAdapter(BaseBackendAdapter):
    def __init__(self, spec: CustomBackend):
        if not isinstance(spec, CustomBackend):
            raise BackendCompatibilityError("CustomBackendAdapter requires a CustomBackend specification")
        self._spec = spec
        self._backend = spec.instance
        if self._backend is None:
            raise BackendCompatibilityError("custom backend instance is required")
        self._identity = BackendIdentity(
            kind="custom",
            name=spec.identity,
            provider=_provider_name(self._backend),
            version=_backend_version(self._backend),
        )

    def resolve(self) -> BackendIdentity:
        return self._identity

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            local=_backend_local(self._backend),
            supports_resume=self._spec.supports_resume,
            max_circuits=_max_circuits(self._backend),
        )

    def availability(self) -> Availability:
        if not callable(getattr(self._backend, "run", None)):
            return Availability(False, "custom backend does not expose run")
        status_method = getattr(self._backend, "status", None)
        if not callable(status_method):
            return Availability(True)
        try:
            status = status_method()
        except Exception:
            return Availability(False, "custom backend status is unavailable")
        operational = getattr(status, "operational", None)
        if operational is False:
            reason = getattr(status, "status_msg", None)
            if not isinstance(reason, str) or not reason:
                reason = "backend is not operational"
            return Availability(False, reason)
        return Availability(True)

    def compile(self, circuits: Sequence[Any], config: TranspilationConfig) -> CompiledBatch:
        batch = _validated_circuit_tuple(circuits)
        if not isinstance(config, TranspilationConfig):
            raise BackendCompatibilityError("compile requires TranspilationConfig")
        options = {
            key: value
            for key, value in config.to_safe_dict().items()
            if value is not None
        }
        try:
            compiled = transpile(list(batch), backend=self._backend, **options)
        except Exception as error:
            raise BackendCompatibilityError(
                f"could not compile circuits for backend {self._identity.kind}:{self._identity.name} "
                f"({_exception_name(error)})"
            ) from None
        compiled_batch = tuple(compiled) if isinstance(compiled, (list, tuple)) else (compiled,)
        return CompiledBatch(compiled_batch, self._identity, {"transpilation": options})

    def submit(
        self,
        circuits: Sequence[Any],
        shots: int,
        options: Mapping[str, Any] | None = None,
    ) -> SubmittedJob:
        return self._submit_to_backend(self._backend, circuits, shots, options)

    def restore_job(
        self,
        job_id: str,
        *,
        circuit_count: int | None = None,
        shots: int | None = None,
    ) -> SubmittedJob:
        job_id = _safe_identifier(job_id, "job_id")
        if not self._spec.supports_resume:
            raise BackendCompatibilityError("custom backend does not support resume")
        retrieve = getattr(self._backend, "retrieve_job", None)
        if not callable(retrieve):
            raise BackendCompatibilityError("resumable custom backend must expose retrieve_job")
        _positive_integer(circuit_count, "circuit_count", optional=True)
        _positive_integer(shots, "shots", optional=True)
        try:
            handle = retrieve(job_id)
        except Exception as error:
            raise JobResultError(
                f"could not restore job {job_id} ({_exception_name(error)})"
            ) from None
        if handle is None:
            raise JobResultError(f"could not restore job {job_id}")
        try:
            actual_job_id = _extract_job_id(handle, allow_local_fallback=False)
        except Exception as error:
            raise JobResultError(
                f"restored job did not provide the requested job ID ({_exception_name(error)})"
            ) from None
        if actual_job_id != job_id:
            raise JobResultError("restored job ID does not match requested job ID")
        return SubmittedJob(job_id, handle, self._identity, circuit_count, shots, {"restored": True})


def _backend_local(backend: Any) -> bool:
    value = getattr(backend, "local", None)
    if type(value) is bool:
        return value
    configuration = getattr(backend, "configuration", None)
    if callable(configuration):
        try:
            value = getattr(configuration(), "local", None)
        except Exception:
            value = None
        if type(value) is bool:
            return value
    return False


def _max_circuits(backend: Any) -> int | None:
    value = getattr(backend, "max_circuits", None)
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    configuration = getattr(backend, "configuration", None)
    if callable(configuration):
        try:
            value = getattr(configuration(), "max_experiments", None)
        except Exception:
            return None
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _provider_name(backend: Any) -> str | None:
    module = type(backend).__module__.split(".", 1)[0]
    if not module or module == "builtins" or module.startswith("_"):
        return None
    return module.replace("_", "-")


def _backend_version(backend: Any) -> str | None:
    value = getattr(backend, "backend_version", None)
    if callable(value):
        try:
            value = value()
        except Exception:
            return None
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        return str(value)
    return None


__all__ = ["CustomBackendAdapter"]
