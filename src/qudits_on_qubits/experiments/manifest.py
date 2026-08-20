"""Immutable validation boundary for durable experiment manifests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any

from .errors import ExperimentPersistenceError, ExperimentValidationError
from .execution import (
    ExecutionMode,
    expected_backend_identity_kind,
    fixed_execution_mode,
    validate_backend_execution_mode,
)
from .models import ExperimentStatus
from .safety import validate_persisted_strings
from .store import ExperimentStore


MANIFEST_SCHEMA_VERSION = 2
LEGACY_MANIFEST_SCHEMA_VERSION = 1
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_FIELDS = {
    "schema_version", "experiment_id", "spec", "status", "timestamps",
    "status_history", "attempts", "backend", "jobs", "job_ids", "source",
    "circuits", "counts", "postprocessing", "calibration", "result",
    "result_artifact", "failure",
}


def _freeze_json(value: Any, description: str) -> Any:
    if value is None or type(value) in {bool, int, str}:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ExperimentValidationError(
                f"{description} contains a non-finite float"
            ) from None
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ExperimentValidationError(
                    f"{description} mapping keys must be strings"
                ) from None
            frozen[key] = _freeze_json(item, description)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, description) for item in value)
    raise ExperimentValidationError(
        f"{description} contains an unsupported value"
    ) from None


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _mapping(value: Any, description: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExperimentValidationError(f"{description} must be a mapping") from None
    return value


def _sequence(value: Any, description: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ExperimentValidationError(f"{description} must be a sequence") from None
    return value


def _timestamp(value: Any, description: str) -> str:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ExperimentValidationError(f"{description} is invalid") from None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        raise ExperimentValidationError(f"{description} is invalid") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ExperimentValidationError(f"{description} is invalid") from None
    return value


def _artifact_ref(value: Any, description: str) -> None:
    record = _mapping(value, f"{description} artifact")
    artifact = record.get("artifact")
    digest = record.get("sha256")
    if not isinstance(artifact, str):
        raise ExperimentValidationError(
            f"{description} artifact name is invalid"
        ) from None
    relative = Path(artifact)
    if (
        relative.is_absolute()
        or len(relative.parts) != 1
        or relative.name in {"", ".", ".."}
    ):
        raise ExperimentValidationError(
            f"{description} artifact name is unsafe"
        ) from None
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        raise ExperimentValidationError(
            f"{description} artifact hash is invalid"
        ) from None


def _normalize_document(value: Mapping[str, Any]) -> dict[str, Any]:
    validate_persisted_strings(
        value,
        description="manifest",
        error_type=ExperimentValidationError,
    )
    document = _thaw_json(_freeze_json(value, "manifest"))
    version = document.get("schema_version")
    if version == MANIFEST_SCHEMA_VERSION:
        return document
    if version != LEGACY_MANIFEST_SCHEMA_VERSION:
        raise ExperimentPersistenceError(
            "unsupported experiment schema version"
        ) from None
    spec = _mapping(document.get("spec"), "manifest spec")
    backend = dict(_mapping(spec.get("backend"), "manifest backend spec"))
    kind = backend.get("kind")
    if kind == "custom":
        raise ExperimentPersistenceError(
            "schema-v1 custom backend execution mode is ambiguous"
        ) from None
    try:
        mode = fixed_execution_mode(kind)
    except ExperimentValidationError:
        raise ExperimentPersistenceError(
            "schema-v1 backend kind is unsupported"
        ) from None
    if mode is None:
        raise ExperimentPersistenceError(
            "schema-v1 backend execution mode is ambiguous"
        ) from None
    backend["execution_mode"] = mode.value
    mutable_spec = dict(spec)
    mutable_spec["backend"] = backend
    document["spec"] = mutable_spec
    document["schema_version"] = MANIFEST_SCHEMA_VERSION
    document.setdefault("source", None)
    document.setdefault("result_artifact", None)
    return document


def _validate_document(document: Mapping[str, Any]) -> None:
    if set(document) != _FIELDS:
        raise ExperimentValidationError(
            "manifest fields do not match schema v2"
        ) from None
    if document.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ExperimentValidationError(
            "manifest schema version is invalid"
        ) from None
    experiment_id = document.get("experiment_id")
    if not isinstance(experiment_id, str) or not _RUN_ID.fullmatch(experiment_id):
        raise ExperimentValidationError("experiment_id is invalid") from None
    spec = _mapping(document.get("spec"), "manifest spec")
    backend_spec = _mapping(spec.get("backend"), "manifest backend spec")
    backend_kind = backend_spec.get("kind")
    mode = validate_backend_execution_mode(
        backend_kind,
        backend_spec.get("execution_mode"),
    )
    if mode.value != backend_spec.get("execution_mode"):
        raise ExperimentValidationError(
            "execution_mode is not canonical"
        ) from None
    try:
        status = ExperimentStatus(document.get("status"))
    except (TypeError, ValueError):
        raise ExperimentValidationError("experiment status is invalid") from None
    timestamps = _mapping(document.get("timestamps"), "manifest timestamps")
    _timestamp(timestamps.get("created"), "created timestamp")
    _timestamp(timestamps.get("updated"), "updated timestamp")
    history = _sequence(document.get("status_history"), "status_history")
    if not history:
        raise ExperimentValidationError(
            "status_history must not be empty"
        ) from None
    states: list[ExperimentStatus] = []
    for entry in history:
        record = _mapping(entry, "status_history entry")
        try:
            states.append(ExperimentStatus(record.get("status")))
        except (TypeError, ValueError):
            raise ExperimentValidationError(
                "status_history status is invalid"
            ) from None
        _timestamp(record.get("timestamp"), "status_history timestamp")
    if states[0] is not ExperimentStatus.CREATED or states[-1] is not status:
        raise ExperimentValidationError(
            "status_history does not match status"
        ) from None
    for entry in _sequence(document.get("attempts"), "attempts"):
        record = _mapping(entry, "attempt record")
        attempt = record.get("attempt")
        if not isinstance(record.get("operation"), str):
            raise ExperimentValidationError(
                "attempt operation is invalid"
            ) from None
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt <= 0:
            raise ExperimentValidationError("attempt number is invalid") from None
        if record.get("outcome") not in {"failed", "succeeded"}:
            raise ExperimentValidationError("attempt outcome is invalid") from None
        _timestamp(record.get("timestamp"), "attempt timestamp")
    backend = document.get("backend")
    if backend is not None:
        backend_record = _mapping(backend, "resolved backend")
        for name in ("identity", "capabilities", "metadata", "availability"):
            _mapping(backend_record.get(name), f"backend {name}")
        identity = _mapping(backend_record["identity"], "backend identity")
        if identity.get("kind") != expected_backend_identity_kind(backend_kind):
            raise ExperimentValidationError(
                "resolved backend identity does not match spec"
            ) from None
    jobs = _mapping(document.get("jobs"), "jobs")
    recorded_ids: list[str] = []
    for job_key, value in jobs.items():
        if not isinstance(job_key, str):
            raise ExperimentValidationError("job key is invalid") from None
        record = _mapping(value, "job record")
        job_id = record.get("job_id")
        if job_id is not None:
            if not isinstance(job_id, str) or not job_id:
                raise ExperimentValidationError("job_id is invalid") from None
            recorded_ids.append(job_id)
    job_ids = _sequence(document.get("job_ids"), "job_ids")
    if any(not isinstance(item, str) or not item for item in job_ids):
        raise ExperimentValidationError("job_ids are invalid") from None
    if sorted(job_ids) != sorted(recorded_ids):
        raise ExperimentValidationError(
            "job_ids do not match job records"
        ) from None
    circuits = _mapping(document.get("circuits"), "circuits")
    if set(circuits) != {"source", "logical", "factors"}:
        raise ExperimentValidationError(
            "circuit manifest fields are invalid"
        ) from None
    for name in ("source", "logical"):
        if circuits[name] is not None:
            _artifact_ref(circuits[name], f"{name} circuit")
    for factor, record in _mapping(
        circuits["factors"],
        "factor circuits",
    ).items():
        if not isinstance(factor, str) or not factor.isdigit():
            raise ExperimentValidationError(
                "factor circuit key is invalid"
            ) from None
        _artifact_ref(record, f"factor {factor} circuit")
    for factor, record in _mapping(document.get("counts"), "counts").items():
        if not isinstance(factor, str) or not factor.isdigit():
            raise ExperimentValidationError(
                "counts factor key is invalid"
            ) from None
        _artifact_ref(record, f"factor {factor} counts")
    for name in ("postprocessing", "result_artifact"):
        if document.get(name) is not None:
            _artifact_ref(document[name], name)
    for name in ("source", "calibration", "result", "failure"):
        if document.get(name) is not None:
            _mapping(document[name], name)
    if status is ExperimentStatus.COMPLETED:
        if document.get("result") is None or document.get("result_artifact") is None:
            raise ExperimentValidationError(
                "completed manifest requires result artifacts"
            ) from None
        if document.get("failure") is not None:
            raise ExperimentValidationError(
                "completed manifest must not contain failure"
            ) from None
    if status is ExperimentStatus.FAILED and document.get("failure") is None:
        raise ExperimentValidationError(
            "failed manifest requires failure"
        ) from None


@dataclass(frozen=True)
class RunManifest:
    schema_version: int
    experiment_id: str
    spec: Mapping[str, Any]
    status: ExperimentStatus
    timestamps: Mapping[str, Any]
    status_history: tuple[Mapping[str, Any], ...]
    attempts: tuple[Mapping[str, Any], ...]
    backend: Mapping[str, Any] | None
    jobs: Mapping[str, Any]
    job_ids: tuple[str, ...]
    source: Mapping[str, Any] | None
    circuits: Mapping[str, Any]
    counts: Mapping[str, Any]
    postprocessing: Mapping[str, Any] | None
    calibration: Mapping[str, Any] | None
    result: Mapping[str, Any] | None
    result_artifact: Mapping[str, Any] | None
    failure: Mapping[str, Any] | None

    def __post_init__(self) -> None:
        for name in (
            "spec", "timestamps", "status_history", "attempts", "backend", "jobs",
            "job_ids", "source", "circuits", "counts", "postprocessing",
            "calibration", "result", "result_artifact", "failure",
        ):
            object.__setattr__(
                self,
                name,
                _freeze_json(getattr(self, name), f"manifest {name}"),
            )
        _validate_document(self.to_safe_dict())

    @property
    def execution_mode(self) -> ExecutionMode:
        backend = _mapping(self.spec.get("backend"), "manifest backend spec")
        return validate_backend_execution_mode(
            backend.get("kind"),
            backend.get("execution_mode"),
        )

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "spec": _thaw_json(self.spec),
            "status": self.status.value,
            "timestamps": _thaw_json(self.timestamps),
            "status_history": _thaw_json(self.status_history),
            "attempts": _thaw_json(self.attempts),
            "backend": _thaw_json(self.backend),
            "jobs": _thaw_json(self.jobs),
            "job_ids": _thaw_json(self.job_ids),
            "source": _thaw_json(self.source),
            "circuits": _thaw_json(self.circuits),
            "counts": _thaw_json(self.counts),
            "postprocessing": _thaw_json(self.postprocessing),
            "calibration": _thaw_json(self.calibration),
            "result": _thaw_json(self.result),
            "result_artifact": _thaw_json(self.result_artifact),
            "failure": _thaw_json(self.failure),
        }

    @classmethod
    def from_safe_dict(cls, data: Mapping[str, Any]) -> "RunManifest":
        if not isinstance(data, Mapping):
            raise ExperimentValidationError("manifest must be a mapping") from None
        document = _normalize_document(data)
        _validate_document(document)
        return cls(
            schema_version=document["schema_version"],
            experiment_id=document["experiment_id"],
            spec=document["spec"],
            status=ExperimentStatus(document["status"]),
            timestamps=document["timestamps"],
            status_history=tuple(document["status_history"]),
            attempts=tuple(document["attempts"]),
            backend=document["backend"],
            jobs=document["jobs"],
            job_ids=tuple(document["job_ids"]),
            source=document["source"],
            circuits=document["circuits"],
            counts=document["counts"],
            postprocessing=document["postprocessing"],
            calibration=document["calibration"],
            result=document["result"],
            result_artifact=document["result_artifact"],
            failure=document["failure"],
        )

    @classmethod
    def load(cls, artifact_dir: Path | str) -> "RunManifest":
        try:
            run = Path(artifact_dir).expanduser().resolve(strict=True)
        except (OSError, RuntimeError, TypeError):
            raise ExperimentPersistenceError(
                "artifact_dir must identify an existing run directory"
            ) from None
        if not run.is_dir() or len(run.parents) < 2:
            raise ExperimentPersistenceError(
                "artifact_dir must identify a run directory"
            ) from None
        store = ExperimentStore(run.parents[1])
        document = store.read_experiment(run)
        try:
            manifest = cls.from_safe_dict(document)
        except ExperimentPersistenceError:
            raise
        except ExperimentValidationError:
            raise ExperimentPersistenceError(
                "experiment manifest is invalid"
            ) from None
        if manifest.experiment_id != run.name:
            raise ExperimentPersistenceError(
                "experiment ID does not match run directory"
            ) from None
        return manifest


__all__ = ["MANIFEST_SCHEMA_VERSION", "RunManifest"]
