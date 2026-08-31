"""Backend-neutral adapter records and validation helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import math
import re
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable
from uuid import uuid4

from ..errors import (
    BackendCompatibilityError,
    BackendUnavailableError,
    ExperimentValidationError,
    JobResultError,
    JobSubmissionError,
)
from ..models import TranspilationConfig


_SAFE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}\Z")
_BITSTRING = re.compile(r"[01]+(?: [01]+)*\Z")
_STATUS_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}\Z")
_EXCEPTION_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}\Z")
_CREDENTIAL_MARKERS = ("token=", "api_key=", "password=", "secret=")


def _safe_identifier(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not _SAFE_IDENTIFIER.fullmatch(value):
        raise ExperimentValidationError(f"{field_name} must be a non-empty safe string")
    return value


def _safe_label(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > 512
        or any(ord(character) < 32 for character in value)
        or any(marker in value.lower() for marker in _CREDENTIAL_MARKERS)
    ):
        raise ExperimentValidationError(f"{field_name} must be a non-empty safe string")
    return value


def _safe_optional_label(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _safe_label(value, field_name)


def _exception_name(error: BaseException) -> str:
    name = type(error).__name__
    return name if _EXCEPTION_NAME.fullmatch(name) else "Exception"


def _normalise_status(value: Any, *, reject_unsafe: bool) -> str | None:
    if value is None:
        return None
    candidates: list[Any] = []
    if isinstance(value, Enum):
        candidates.extend((value.name, value.value))
    elif isinstance(value, str):
        candidates.append(value)
    else:
        try:
            candidates.extend((getattr(value, "name", None), getattr(value, "value", None)))
        except Exception:
            candidates = []
    for candidate in candidates:
        if (
            isinstance(candidate, str)
            and not any(marker in candidate.lower() for marker in _CREDENTIAL_MARKERS)
            and _STATUS_TOKEN.fullmatch(candidate)
        ):
            return candidate.lower()
    if reject_unsafe:
        raise ExperimentValidationError("result status must be a safe status token, enum, or None")
    return None


def _freeze_value(value: Any, field_name: str) -> Any:
    if isinstance(value, Enum):
        return _freeze_value(value.value, field_name)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ExperimentValidationError(f"{field_name} must contain only finite values")
        return value
    if isinstance(value, Mapping):
        return _freeze_mapping(value, field_name)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item, field_name) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_value(item, field_name) for item in value)
    raise ExperimentValidationError(f"{field_name} must contain only safe metadata values")


def _freeze_mapping(values: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    if not isinstance(values, Mapping):
        raise ExperimentValidationError(f"{field_name} must be a mapping")
    frozen: dict[str, Any] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise ExperimentValidationError(f"{field_name} keys must be non-empty strings")
        frozen[key] = _freeze_value(value, field_name)
    return MappingProxyType(frozen)


def _safe_dict_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _safe_dict_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_safe_dict_value(item) for item in value]
    if isinstance(value, frozenset):
        return sorted((_safe_dict_value(item) for item in value), key=repr)
    return value


def _positive_integer(value: object, field_name: str, *, optional: bool = False) -> int | None:
    if optional and value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        suffix = " or None" if optional else ""
        raise ExperimentValidationError(f"{field_name} must be a positive integer{suffix}")
    return value


@dataclass(frozen=True)
class BackendIdentity:
    kind: str
    name: str
    provider: str | None = None
    version: str | None = None
    emulates: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", _safe_identifier(self.kind, "backend kind"))
        object.__setattr__(self, "name", _safe_label(self.name, "backend name"))
        object.__setattr__(self, "provider", _safe_optional_label(self.provider, "backend provider"))
        object.__setattr__(self, "version", _safe_optional_label(self.version, "backend version"))
        object.__setattr__(self, "emulates", _safe_optional_label(self.emulates, "emulated backend"))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, "backend identity metadata"))

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "name": self.name,
            "provider": self.provider,
            "version": self.version,
            "emulates": self.emulates,
            "metadata": _safe_dict_value(self.metadata),
        }


@dataclass(frozen=True)
class BackendCapabilities:
    local: bool
    supports_resume: bool
    max_circuits: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.local) is not bool or type(self.supports_resume) is not bool:
            raise ExperimentValidationError("backend capability flags must be booleans")
        if self.max_circuits is not None:
            _positive_integer(self.max_circuits, "max_circuits")
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, "backend capability metadata"))

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "local": self.local,
            "supports_resume": self.supports_resume,
            "max_circuits": self.max_circuits,
            "metadata": _safe_dict_value(self.metadata),
        }


@dataclass(frozen=True)
class Availability:
    available: bool
    reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if type(self.available) is not bool:
            raise ExperimentValidationError("available must be a boolean")
        if self.reason is not None:
            object.__setattr__(self, "reason", _safe_label(self.reason, "availability reason"))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, "availability metadata"))

    def to_safe_dict(self) -> dict[str, Any]:
        return {"available": self.available, "reason": self.reason, "metadata": _safe_dict_value(self.metadata)}


@dataclass(frozen=True)
class CompiledBatch:
    circuits: tuple[Any, ...]
    target_identity: BackendIdentity
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        circuits = tuple(self.circuits)
        if not circuits:
            raise ExperimentValidationError("compiled circuit batch must not be empty")
        if not isinstance(self.target_identity, BackendIdentity):
            raise ExperimentValidationError("target_identity must be BackendIdentity")
        object.__setattr__(self, "circuits", circuits)
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, "compiled batch metadata"))


@dataclass(frozen=True)
class SubmittedJob:
    job_id: str
    handle: Any = field(repr=False, compare=False)
    target_identity: BackendIdentity
    circuit_count: int | None = None
    shots: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "job_id", _safe_identifier(self.job_id, "job_id"))
        if self.handle is None:
            raise ExperimentValidationError("submitted job handle is required")
        if not isinstance(self.target_identity, BackendIdentity):
            raise ExperimentValidationError("target_identity must be BackendIdentity")
        _positive_integer(self.circuit_count, "circuit_count", optional=True)
        _positive_integer(self.shots, "shots", optional=True)
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, "submitted job metadata"))

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "target_identity": self.target_identity.to_safe_dict(),
            "circuit_count": self.circuit_count,
            "shots": self.shots,
            "metadata": _safe_dict_value(self.metadata),
        }


@dataclass(frozen=True)
class ExecutionResult:
    counts: tuple[Mapping[str, int], ...]
    job_id: str
    target_identity: BackendIdentity
    status: str | Enum | None = None
    timing: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "job_id", _safe_identifier(self.job_id, "job_id"))
        if not isinstance(self.target_identity, BackendIdentity):
            raise ExperimentValidationError("target_identity must be BackendIdentity")
        object.__setattr__(self, "status", _normalise_status(self.status, reject_unsafe=True))
        frozen_counts = tuple(_validate_counts(item) for item in self.counts)
        if not frozen_counts:
            raise ExperimentValidationError("execution result counts must not be empty")
        object.__setattr__(self, "counts", frozen_counts)
        object.__setattr__(self, "timing", _freeze_mapping(self.timing, "execution timing"))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata, "execution result metadata"))

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "counts": [dict(item) for item in self.counts],
            "job_id": self.job_id,
            "target_identity": self.target_identity.to_safe_dict(),
            "status": self.status,
            "timing": _safe_dict_value(self.timing),
            "metadata": _safe_dict_value(self.metadata),
        }


@runtime_checkable
class BackendAdapter(Protocol):
    def resolve(self) -> BackendIdentity: ...
    def capabilities(self) -> BackendCapabilities: ...
    def availability(self) -> Availability: ...
    def preflight(self, circuits: Sequence[Any], shots: int) -> None: ...
    def compile(self, circuits: Sequence[Any], config: TranspilationConfig) -> CompiledBatch: ...
    def submit(
        self, circuits: Sequence[Any], shots: int, options: Mapping[str, Any] | None = None
    ) -> SubmittedJob: ...
    def result(self, submitted: SubmittedJob, timeout: float | None = None) -> ExecutionResult: ...
    def restore_job(
        self, job_id: str, *, circuit_count: int | None = None, shots: int | None = None
    ) -> SubmittedJob: ...
    def metadata(self) -> Mapping[str, Any]: ...


class BaseBackendAdapter(ABC):
    """Common validation for Qiskit-style backend adapters."""

    @abstractmethod
    def resolve(self) -> BackendIdentity:
        raise NotImplementedError

    @abstractmethod
    def capabilities(self) -> BackendCapabilities:
        raise NotImplementedError

    @abstractmethod
    def availability(self) -> Availability:
        raise NotImplementedError

    @abstractmethod
    def compile(self, circuits: Sequence[Any], config: TranspilationConfig) -> CompiledBatch:
        raise NotImplementedError

    @abstractmethod
    def submit(
        self, circuits: Sequence[Any], shots: int, options: Mapping[str, Any] | None = None
    ) -> SubmittedJob:
        raise NotImplementedError

    @abstractmethod
    def restore_job(
        self, job_id: str, *, circuit_count: int | None = None, shots: int | None = None
    ) -> SubmittedJob:
        raise NotImplementedError

    def preflight(self, circuits: Sequence[Any], shots: int) -> None:
        batch = _validated_circuit_tuple(circuits)
        _positive_integer(shots, "shots")
        availability = self.availability()
        if not availability.available:
            reason = availability.reason or "backend is unavailable"
            raise BackendUnavailableError(reason)
        maximum = self.capabilities().max_circuits
        if maximum is not None and len(batch) > maximum:
            raise BackendCompatibilityError(
                f"backend accepts at most {maximum} circuits per submission"
            )

    def metadata(self) -> Mapping[str, Any]:
        return _freeze_mapping(
            {
                "identity": self.resolve().to_safe_dict(),
                "capabilities": self.capabilities().to_safe_dict(),
            },
            "adapter metadata",
        )

    def _submit_to_backend(
        self,
        backend: Any,
        circuits: Sequence[Any],
        shots: int,
        options: Mapping[str, Any] | None,
    ) -> SubmittedJob:
        batch = _validated_circuit_tuple(circuits)
        _positive_integer(shots, "shots")
        run_options = _validated_run_options(options)
        try:
            handle = backend.run(batch, shots=shots, **run_options)
        except Exception as error:
            identity = self.resolve()
            sanitized = JobSubmissionError(
                f"backend {identity.kind}:{identity.name} rejected job submission "
                f"({_exception_name(error)})"
            )
            sanitized.provider_exception_type = _exception_name(error)
            raise sanitized from None
        try:
            job_id = _extract_job_id(handle, allow_local_fallback=self.capabilities().local)
        except ExperimentValidationError as error:
            raise JobSubmissionError(
                f"submitted job did not provide a usable job ID ({_exception_name(error)})"
            ) from None
        return SubmittedJob(
            job_id=job_id,
            handle=handle,
            target_identity=self.resolve(),
            circuit_count=len(batch),
            shots=shots,
        )

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
            if timeout is None:
                raw_result = submitted.handle.result()
            else:
                raw_result = submitted.handle.result(timeout=timeout)
        except MemoryError:
            raise
        except Exception as error:
            raise JobResultError(
                f"could not retrieve result for job {submitted.job_id} ({_exception_name(error)})"
            ) from None
        try:
            counts = _extract_counts(raw_result, submitted.circuit_count)
            validated = tuple(_validate_counts(item) for item in counts)
            if submitted.circuit_count is not None and len(validated) != submitted.circuit_count:
                raise JobResultError(
                    f"result count set count {len(validated)} does not match circuit count {submitted.circuit_count}"
                )
            if submitted.shots is not None:
                for item in validated:
                    if sum(item.values()) != submitted.shots:
                        raise JobResultError("result counts do not sum to expected shots")
        except JobResultError:
            raise
        except MemoryError:
            raise
        except Exception as error:
            raise JobResultError(
                f"result for job {submitted.job_id} has an unsupported format "
                f"({_exception_name(error)})"
            ) from None
        return ExecutionResult(
            counts=validated,
            job_id=submitted.job_id,
            target_identity=submitted.target_identity,
            status=_result_status(submitted.handle, raw_result),
            timing=_result_timing(raw_result),
        )


def _validated_circuit_tuple(circuits: Sequence[Any]) -> tuple[Any, ...]:
    if isinstance(circuits, (str, bytes)):
        raise BackendCompatibilityError("circuits must be a non-empty sequence")
    try:
        batch = circuits if isinstance(circuits, tuple) else tuple(circuits)
    except TypeError as error:
        raise BackendCompatibilityError(
            f"circuits must be a non-empty sequence ({_exception_name(error)})"
        ) from None
    if not batch or any(circuit is None for circuit in batch):
        raise BackendCompatibilityError("circuits must be a non-empty sequence")
    return batch


def _validated_run_options(options: Mapping[str, Any] | None) -> dict[str, Any]:
    if options is None:
        return {}
    if not isinstance(options, Mapping):
        raise BackendCompatibilityError("run options must be a mapping")
    if "shots" in options:
        raise BackendCompatibilityError("shots must be supplied exactly once")
    if any(not isinstance(key, str) or not key for key in options):
        raise BackendCompatibilityError("run option keys must be non-empty strings")
    return dict(options)


def _extract_job_id(handle: Any, *, allow_local_fallback: bool) -> str:
    candidate: Any = None
    try:
        candidate = getattr(handle, "job_id")
    except Exception:
        candidate = None
    if callable(candidate):
        try:
            candidate = candidate()
        except Exception:
            candidate = None
    if candidate is not None:
        try:
            return _safe_identifier(candidate, "job_id")
        except ExperimentValidationError:
            if not allow_local_fallback:
                raise
    if allow_local_fallback:
        return f"local-{uuid4()}"
    raise ExperimentValidationError("job ID is required for remote or unknown jobs")


def _validate_counts(counts: Any) -> Mapping[str, int]:
    if not isinstance(counts, Mapping):
        raise JobResultError("each result count set must be a mapping")
    validated: dict[str, int] = {}
    for bitstring, count in counts.items():
        if not isinstance(bitstring, str) or not _BITSTRING.fullmatch(bitstring):
            raise JobResultError("result count keys must be bitstrings")
        if isinstance(count, bool) or not isinstance(count, int):
            raise JobResultError("result count values must be integers")
        if count < 0:
            raise JobResultError("result count values must be non-negative")
        validated[bitstring] = count
    if not validated:
        raise JobResultError("result count sets must not be empty")
    return MappingProxyType(validated)


def _extract_counts(raw_result: Any, circuit_count: int | None) -> tuple[Mapping[str, int], ...]:
    getter = getattr(raw_result, "get_counts", None)
    if callable(getter):
        if circuit_count == 1:
            value = _call_provider_counts(getter)
            if isinstance(value, (list, tuple)) and len(value) == 1:
                value = value[0]
            return (value,)
        if circuit_count is not None:
            try:
                return tuple(getter(index) for index in range(circuit_count))
            except (TypeError, IndexError):
                value = _call_provider_counts(getter)
                return tuple(value) if isinstance(value, (list, tuple)) else (value,)
            except MemoryError:
                raise
            except Exception as error:
                raise JobResultError(
                    f"provider get_counts failed ({_exception_name(error)})"
                ) from None
        value = _call_provider_counts(getter)
        return tuple(value) if isinstance(value, (list, tuple)) else (value,)

    try:
        entries = tuple(raw_result)
    except MemoryError:
        raise
    except Exception as error:
        raise JobResultError(
            f"result does not expose counts ({_exception_name(error)})"
        ) from None
    if not entries:
        raise JobResultError("result does not contain primitive entries")
    return tuple(_counts_from_primitive_entry(entry) for entry in entries)


def _counts_from_primitive_entry(entry: Any) -> Mapping[str, int]:
    data = getattr(entry, "data", None)
    if data is None:
        raise JobResultError("primitive result entry has no data bin")
    meas = getattr(data, "meas", None)
    getter = getattr(meas, "get_counts", None)
    if callable(getter):
        return _call_provider_counts(getter)

    names: list[str] = []
    keys = getattr(data, "keys", None)
    if callable(keys):
        try:
            names.extend(key for key in keys() if isinstance(key, str))
        except MemoryError:
            raise
        except Exception:
            pass
    values = getattr(data, "__dict__", {})
    if isinstance(values, Mapping):
        names.extend(key for key in values if isinstance(key, str) and not key.startswith("_"))
    for name in dict.fromkeys(names):
        register = getattr(data, name, None)
        getter = getattr(register, "get_counts", None)
        if callable(getter):
            return _call_provider_counts(getter)
    raise JobResultError("primitive data bin does not expose get_counts")


def _call_provider_counts(getter: Any) -> Any:
    try:
        return getter()
    except MemoryError:
        raise
    except Exception as error:
        raise JobResultError(
            f"provider get_counts failed ({_exception_name(error)})"
        ) from None


def _result_status(handle: Any, raw_result: Any) -> str | None:
    for source in (handle, raw_result):
        try:
            value = getattr(source, "status", None)
            value = value() if callable(value) else value
        except Exception:
            continue
        if value is not None:
            status = _normalise_status(value, reject_unsafe=False)
            if status is not None:
                return status
    return None


def _result_timing(raw_result: Any) -> Mapping[str, Any]:
    timing: dict[str, Any] = {}
    try:
        value = getattr(raw_result, "time_taken", None)
    except Exception:
        return timing
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        timing["time_taken"] = value
    return timing


__all__ = [
    "Availability",
    "BackendAdapter",
    "BackendCapabilities",
    "BackendIdentity",
    "BaseBackendAdapter",
    "CompiledBatch",
    "ExecutionResult",
    "SubmittedJob",
]
