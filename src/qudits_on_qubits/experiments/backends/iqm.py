"""IQM hardware adapter for the unified experiment runner."""

from __future__ import annotations

from enum import Enum
import math
from typing import Any, Mapping, Sequence

from ...benchmarks.direct_basis.iqm_backend import (
    backend_metadata,
    load_iqm_backend,
)
from ..errors import (
    BackendCompatibilityError,
    BackendUnavailableError,
    JobResultError,
    OptionalDependencyError,
)
from ..models import IQMHardware, TranspilationConfig
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


_SECRET_KEYS = ("token", "password", "secret", "api_key", "dashboard_api_key")


def _default_backend_loader(device: str, use_metrics: bool, env_path: Any) -> Any:
    return load_iqm_backend(device, use_metrics=use_metrics, env_path=env_path)


def _default_transpiler(circuit: Any, backend: Any, **options: Any) -> Any:
    from iqm.qiskit_iqm import transpile_to_IQM

    return transpile_to_IQM(circuit, backend=backend, **options)


def _effective_options(config: TranspilationConfig) -> dict[str, Any]:
    return {
        key: value
        for key, value in config.to_safe_dict().items()
        if value is not None
    }


def _default_backend_metadata(backend: Any, spec: IQMHardware) -> Mapping[str, Any]:
    return backend_metadata(
        backend,
        iqm_backend_name=spec.device,
        iqm_use_metrics=spec.use_metrics,
        optimization_level=3,
        layout_method=None,
        routing_method=None,
        scheduling_method=None,
    )


class IQMAdapter(BaseBackendAdapter):
    """One lazily resolved IQM backend used for compilation and execution."""

    def __init__(
        self,
        spec: IQMHardware,
        backend: Any = None,
        backend_loader: Any = None,
        transpiler: Any = None,
        loader: Any = None,
    ) -> None:
        if not isinstance(spec, IQMHardware):
            raise BackendCompatibilityError("IQMAdapter requires an IQMHardware specification")
        if not _safe_metadata_text(spec.device):
            raise BackendCompatibilityError("IQM device must not contain credential material")
        if backend_loader is not None and loader is not None:
            raise BackendCompatibilityError("provide only one IQM backend loader")
        selected_loader = loader if loader is not None else backend_loader
        if selected_loader is not None and not callable(selected_loader):
            raise BackendCompatibilityError("IQM backend loader must be callable")
        if transpiler is not None and not callable(transpiler):
            raise BackendCompatibilityError("IQM transpiler must be callable")
        self._spec = spec
        self._backend = backend
        self._backend_loader = selected_loader or _default_backend_loader
        self._transpiler = transpiler or _default_transpiler
        self._identity: BackendIdentity | None = None

    @property
    def backend(self) -> Any:
        return self._backend_instance()

    def _backend_instance(self) -> Any:
        if self._backend is not None:
            return self._backend
        try:
            backend = self._backend_loader(
                self._spec.device,
                use_metrics=self._spec.use_metrics,
                env_path=self._spec.env_path,
            )
        except (ImportError, ModuleNotFoundError) as error:
            raise OptionalDependencyError(
                "IQMHardware requires the IQM Qiskit adapter; install project IQM dependencies "
                f"({_exception_name(error)})"
            ) from None
        except Exception as error:
            if "Missing IQM Qiskit adapter" in str(error):
                raise OptionalDependencyError(
                    "IQMHardware requires the IQM Qiskit adapter; install project IQM dependencies "
                    f"({_exception_name(error)})"
                ) from None
            raise BackendUnavailableError(
                f"could not load IQM backend iqm:{self._spec.device} ({_exception_name(error)})"
            ) from None
        if backend is None:
            raise BackendUnavailableError(f"IQM backend iqm:{self._spec.device} is unavailable")
        self._backend = backend
        return backend

    def resolve(self) -> BackendIdentity:
        backend = self._backend_instance()
        if self._identity is None:
            raw_metadata: Mapping[str, Any]
            try:
                raw_metadata = _default_backend_metadata(backend, self._spec)
            except Exception:
                raw_metadata = {}
            calibration_set_id = _calibration_set_id(backend)
            metadata = {
                **_normalise_metadata(raw_metadata),
                "calibration_set_id": calibration_set_id,
                "target": f"iqm:{self._spec.device}",
                "provider": "iqm",
            }
            self._identity = BackendIdentity(
                kind="iqm",
                name=self._spec.device,
                provider="iqm",
                version=_backend_version(backend),
                metadata=metadata,
            )
        return self._identity

    def capabilities(self) -> BackendCapabilities:
        backend = self._backend_instance()
        return BackendCapabilities(
            local=False,
            supports_resume=callable(getattr(backend, "retrieve_job", None)),
            max_circuits=_max_circuits(backend),
        )

    def availability(self) -> Availability:
        try:
            backend = self._backend_instance()
        except OptionalDependencyError as error:
            return Availability(False, f"IQM dependency is unavailable ({_exception_name(error)})")
        except BackendUnavailableError as error:
            return Availability(False, f"IQM backend is unavailable ({_exception_name(error)})")
        if not callable(getattr(backend, "run", None)):
            return Availability(False, "IQM backend does not expose run")
        return _backend_availability(backend, "IQM")

    def preflight(self, circuits: Sequence[Any], shots: int) -> None:
        batch = _validated_circuit_tuple(circuits)
        super().preflight(batch, shots)
        capacity = _num_qubits(self._backend_instance())
        if capacity is None:
            return
        for circuit in batch:
            required = _num_qubits(circuit)
            if required is not None and required > capacity:
                raise BackendCompatibilityError(
                    f"circuit requires {required} qubits but IQM backend provides {capacity} qubits"
                )

    def _transpile_batch(
        self,
        circuits: tuple[Any, ...],
        options: Mapping[str, Any],
        *,
        operation: str,
    ) -> tuple[Any, ...]:
        backend = self._backend_instance()
        try:
            return tuple(
                self._transpiler(circuit, backend, **dict(options))
                for circuit in circuits
            )
        except MemoryError:
            raise
        except (ImportError, ModuleNotFoundError) as error:
            raise OptionalDependencyError(
                "IQM transpilation requires the IQM Qiskit adapter "
                f"({_exception_name(error)})"
            ) from None
        except Exception as error:
            identity = self.resolve()
            raise BackendCompatibilityError(
                f"could not {operation} for backend {identity.kind}:{identity.name} "
                f"({_exception_name(error)})"
            ) from None

    def compile(self, circuits: Sequence[Any], config: TranspilationConfig) -> CompiledBatch:
        batch = _validated_circuit_tuple(circuits)
        if not isinstance(config, TranspilationConfig):
            raise BackendCompatibilityError("compile requires TranspilationConfig")
        options = _effective_options(config)
        compiled = self._transpile_batch(batch, options, operation="compile circuits")
        return CompiledBatch(
            compiled,
            self.resolve(),
            {"transpilation": options},
        )

    def compile_physical(
        self, circuits: Sequence[Any], config: TranspilationConfig
    ) -> CompiledBatch:
        """Compile single-qubit calibration circuits without changing targets."""

        batch = _validated_circuit_tuple(circuits)
        if not isinstance(config, TranspilationConfig):
            raise BackendCompatibilityError(
                "compile_physical requires TranspilationConfig"
            )
        widths = {_num_qubits(circuit) for circuit in batch}
        if None in widths or len(widths) != 1:
            raise BackendCompatibilityError(
                "physical calibration circuits must have one common qubit width"
            )
        width = next(iter(widths))
        assert width is not None
        options = _effective_options(config)
        options["initial_layout"] = list(range(width))
        compiled = self._transpile_batch(
            batch,
            options,
            operation="compile physical calibration circuits",
        )
        return CompiledBatch(
            compiled,
            self.resolve(),
            {"transpilation": options, "physical_layout": True},
        )

    def submit(
        self,
        circuits: Sequence[Any],
        shots: int,
        options: Mapping[str, Any] | None = None,
    ) -> SubmittedJob:
        return self._submit_to_backend(self._backend_instance(), circuits, shots, options)

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
        backend = self._backend_instance()
        retrieve = getattr(backend, "retrieve_job", None)
        if not callable(retrieve):
            raise BackendCompatibilityError("IQM backend does not support resume")
        try:
            handle = retrieve(job_id)
        except Exception as error:
            raise JobResultError(
                f"could not restore IQM job {job_id} ({_exception_name(error)})"
            ) from None
        if handle is None:
            raise JobResultError(f"could not restore IQM job {job_id}")
        try:
            actual_job_id = _extract_job_id(handle, allow_local_fallback=False)
        except Exception as error:
            raise JobResultError(
                f"restored IQM job did not provide the requested job ID ({_exception_name(error)})"
            ) from None
        if actual_job_id != job_id:
            raise JobResultError("restored IQM job ID does not match requested job ID")
        return SubmittedJob(
            job_id,
            handle,
            self.resolve(),
            circuit_count,
            shots,
            {"restored": True},
        )


def _backend_availability(backend: Any, provider: str) -> Availability:
    status_method = getattr(backend, "status", None)
    if not callable(status_method):
        return Availability(True)
    try:
        status = status_method()
    except Exception as error:
        return Availability(False, f"{provider} backend status is unavailable ({_exception_name(error)})")
    if getattr(status, "operational", None) is False:
        return Availability(False, _safe_status_reason(status, provider))
    return Availability(True)


def _safe_status_reason(status: Any, provider: str) -> str:
    try:
        value = getattr(status, "status_msg", None)
    except Exception:
        value = None
    if (
        isinstance(value, str)
        and value
        and len(value) <= 512
        and all(ord(character) >= 32 for character in value)
        and not any(marker in value.lower() for marker in _SECRET_KEYS)
    ):
        return value
    return f"{provider} backend is not operational"


def _num_qubits(value: Any) -> int | None:
    candidate = getattr(value, "num_qubits", None)
    if callable(candidate):
        try:
            candidate = candidate()
        except Exception:
            candidate = None
    if candidate is None:
        target = getattr(value, "target", None)
        candidate = getattr(target, "num_qubits", None)
    if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate > 0:
        return candidate
    return None


def _max_circuits(backend: Any) -> int | None:
    candidate = getattr(backend, "max_circuits", None)
    if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate > 0:
        return candidate
    return None


def _calibration_set_id(backend: Any) -> str:
    for name in ("calibration_set_id", "_calibration_set_id"):
        try:
            value = getattr(backend, name, None)
            value = value() if callable(value) else value
        except Exception:
            continue
        if value is not None:
            rendered = str(value)
            return rendered if _safe_metadata_text(rendered) else ""
    return ""


def _backend_version(backend: Any) -> str | None:
    try:
        value = getattr(backend, "backend_version", None)
        value = value() if callable(value) else value
    except Exception:
        return None
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        return str(value)
    return None


def _normalise_metadata(values: Mapping[str, Any]) -> dict[str, Any]:
    normalised: dict[str, Any] = {}
    for key, value in values.items():
        if not isinstance(key, str) or any(marker in key.lower() for marker in _SECRET_KEYS):
            continue
        safe_value = _normalise_value(value)
        if safe_value is not None or value is None:
            normalised[key] = safe_value
    return normalised


def _normalise_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return _normalise_value(value.value)
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, str):
        return value if _safe_metadata_text(value) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return _normalise_metadata(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_normalise_value(item) for item in value]
    rendered = str(value)
    return rendered if _safe_metadata_text(rendered) else None


def _safe_metadata_text(value: str) -> bool:
    lower = value.lower()
    if any(marker in lower for marker in _SECRET_KEYS):
        return False
    if "://" in value:
        authority = value.split("://", 1)[1].split("/", 1)[0]
        if "@" in authority:
            return False
    return len(value) <= 4096 and all(ord(character) >= 32 for character in value)


__all__ = ["IQMAdapter"]
