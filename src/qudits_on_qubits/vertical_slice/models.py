"""Immutable contracts for the generic qudit experiment vertical slice."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from types import MappingProxyType
from typing import Any, Mapping, Protocol, Sequence, TypeAlias, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from ..experiments.models import AerIdeal, TranspilationConfig
from ..experiments.safety import validate_persisted_strings


JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]

_ENCODING_SCHEMA = "isometric-encoding-v1"
_EXPERIMENT_SCHEMA = "qudit-experiment-spec-v1"
_MANIFEST_SCHEMA = "run-manifest-v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class SpecValidationError(ValueError):
    """Experiment specification violates a public contract."""


class EncodingValidationError(SpecValidationError):
    """Qudit encoding is invalid or incompatible."""


class ManifestValidationError(ValueError):
    """Persisted manifest is invalid or unsupported."""


class ArtifactIntegrityError(ValueError):
    """Persisted artifact does not match its declared identity."""


class ExecutionMode(str, Enum):
    IDEAL_SIMULATOR = "ideal_simulator"
    NOISY_SIMULATOR = "noisy_simulator"
    HARDWARE = "hardware"


def _safe_text(value: object, field_name: str, error_type: type[ValueError]) -> str:
    if not isinstance(value, str) or not value:
        raise error_type(f"{field_name} must be a non-empty string")
    validate_persisted_strings(
        value, description=field_name, error_type=error_type
    )
    return value


def _validate_json(
    value: object,
    field_name: str,
    error_type: type[ValueError],
    *,
    active: set[int] | None = None,
) -> None:
    """Accept strict JSON values only, including finite floating-point values."""

    validate_persisted_strings(
        value, description=field_name, error_type=error_type
    )
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise error_type(f"{field_name} floating-point values must be finite")
        return
    if not isinstance(value, (Mapping, list, tuple)):
        raise error_type(f"{field_name} must contain JSON-compatible values")

    if active is None:
        active = set()
    identity = id(value)
    if identity in active:
        raise error_type(f"{field_name} must not contain cyclic values")
    active.add(identity)
    try:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if not isinstance(key, str):
                    raise error_type(f"{field_name} JSON object keys must be strings")
                _validate_json(item, field_name, error_type, active=active)
        else:
            for item in value:
                _validate_json(item, field_name, error_type, active=active)
    finally:
        active.remove(identity)


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> JsonValue:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json(item) for item in value]
    return value


def _safe_mapping(
    value: object, field_name: str, error_type: type[ValueError]
) -> Mapping[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise error_type(f"{field_name} must be a mapping")
    _validate_json(value, field_name, error_type)
    return _freeze_json(value)


def _canonical_hash(value: Mapping[str, JsonValue]) -> str:
    _validate_json(value, "hash input", SpecValidationError)
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ManifestValidationError(f"{field_name} must be a lowercase SHA-256 digest")
    return value


def _experiment_snapshot_hash(value: Mapping[str, JsonValue]) -> str:
    payload = _thaw_json(value)
    if not isinstance(payload, dict):
        raise SpecValidationError('experiment snapshot must be a JSON object')
    if payload.get('schema_version') == _EXPERIMENT_SCHEMA:
        payload.pop('output_root', None)
        payload.pop('tags', None)
    return _canonical_hash(payload)


def _require_exact_keys(
    data: Mapping[str, Any], expected: set[str], description: str, error_type: type[ValueError]
) -> None:
    if set(data) != expected:
        missing = sorted(expected - set(data))
        extra = sorted(set(data) - expected)
        raise error_type(
            f"{description} fields do not match schema (missing={missing}, extra={extra})"
        )


@dataclass(frozen=True)
class LogicalOutcome:
    value: int | None
    leaked: bool

    def __post_init__(self) -> None:
        if type(self.leaked) is not bool:
            raise EncodingValidationError("leaked must be a boolean")
        if (self.value is None) != self.leaked:
            raise EncodingValidationError("leaked must be true exactly when value is None")
        if self.value is not None and (
            isinstance(self.value, bool) or not isinstance(self.value, int) or self.value < 0
        ):
            raise EncodingValidationError("logical outcome value must be a non-negative integer")


@runtime_checkable
class QuditEncoding(Protocol):
    @property
    def kind(self) -> str: ...

    @property
    def encoding_id(self) -> str: ...

    @property
    def logical_dimension(self) -> int: ...

    @property
    def physical_qubits(self) -> int: ...

    def isometry(self) -> NDArray[Any]: ...

    def decode(self, physical_bits: Sequence[int]) -> LogicalOutcome: ...

    def to_manifest_dict(self) -> Mapping[str, JsonValue]: ...

    def stable_hash(self) -> str: ...


@runtime_checkable
class CircuitSpec(Protocol):
    @property
    def kind(self) -> str: ...

    @property
    def circuit_id(self) -> str: ...

    @property
    def logical_dimensions(self) -> tuple[int, ...]: ...

    def prepare(self, encoding: QuditEncoding) -> "PreparedExperiment": ...

    def to_manifest_dict(self) -> Mapping[str, JsonValue]: ...

    def stable_hash(self) -> str: ...


@runtime_checkable
class PostprocessorSpec(Protocol):
    @property
    def kind(self) -> str: ...

    def evaluate(
        self, counts_by_circuit: Sequence[Mapping[str, int]]
    ) -> Mapping[str, JsonValue]: ...

    def to_manifest_dict(self) -> Mapping[str, JsonValue]: ...


@dataclass(frozen=True)
class PreparedExperiment:
    source_circuits: tuple[Any, ...]
    executable_circuits: tuple[Any, ...]
    postprocessor: PostprocessorSpec
    provenance: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_circuits", tuple(self.source_circuits))
        object.__setattr__(self, "executable_circuits", tuple(self.executable_circuits))
        if not self.executable_circuits:
            raise SpecValidationError("at least one executable circuit is required")
        if not isinstance(self.postprocessor, PostprocessorSpec):
            raise SpecValidationError("postprocessor must implement PostprocessorSpec")
        object.__setattr__(
            self,
            "provenance",
            _safe_mapping(self.provenance, "provenance", SpecValidationError),
        )


@dataclass(frozen=True, eq=False)
class IsometricQuditEncoding:
    encoding_id: str
    logical_dimension: int
    physical_qubits: int
    matrix: NDArray[Any] = field(repr=False)
    decode_table: tuple[LogicalOutcome, ...]

    @property
    def kind(self) -> str:
        return "isometric"

    def __post_init__(self) -> None:
        _safe_text(self.encoding_id, "encoding_id", EncodingValidationError)
        if (
            isinstance(self.logical_dimension, bool)
            or not isinstance(self.logical_dimension, int)
            or self.logical_dimension < 2
        ):
            raise EncodingValidationError("logical_dimension must be an integer of at least 2")
        if (
            isinstance(self.physical_qubits, bool)
            or not isinstance(self.physical_qubits, int)
            or self.physical_qubits < 1
        ):
            raise EncodingValidationError("physical_qubits must be a positive integer")
        physical_dimension = 2**self.physical_qubits
        if self.logical_dimension > physical_dimension:
            raise EncodingValidationError(
                "logical_dimension must not exceed encoded physical dimension"
            )
        try:
            matrix = np.array(self.matrix, dtype=np.complex128, copy=True)
        except (TypeError, ValueError, OverflowError) as error:
            raise EncodingValidationError("isometry matrix must be numeric") from error
        if matrix.shape != (physical_dimension, self.logical_dimension):
            raise EncodingValidationError(
                "isometry matrix shape must be "
                f"({physical_dimension}, {self.logical_dimension})"
            )
        if not bool(np.all(np.isfinite(matrix))):
            raise EncodingValidationError("isometry matrix values must be finite")
        gram = matrix.conj().T @ matrix
        if not bool(np.allclose(gram, np.eye(self.logical_dimension), rtol=1e-10, atol=1e-12)):
            raise EncodingValidationError("isometry matrix columns must be orthonormal")

        table = tuple(self.decode_table)
        if len(table) != physical_dimension:
            raise EncodingValidationError(
                "decode_table must contain one outcome for every physical word"
            )
        for outcome in table:
            if not isinstance(outcome, LogicalOutcome):
                raise EncodingValidationError(
                    "decode_table entries must be LogicalOutcome values"
                )
            if outcome.value is not None and outcome.value >= self.logical_dimension:
                raise EncodingValidationError(
                    "decode_table logical value exceeds logical_dimension"
                )
        matrix.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "decode_table", table)

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, IsometricQuditEncoding)
            and self.encoding_id == other.encoding_id
            and self.logical_dimension == other.logical_dimension
            and self.physical_qubits == other.physical_qubits
            and self.decode_table == other.decode_table
            and bool(np.array_equal(self.matrix, other.matrix))
        )

    def isometry(self) -> NDArray[Any]:
        return self.matrix.copy()

    def decode(self, physical_bits: Sequence[int]) -> LogicalOutcome:
        bits = tuple(physical_bits)
        if len(bits) != self.physical_qubits:
            raise EncodingValidationError(
                f"physical_bits must contain exactly {self.physical_qubits} bits"
            )
        if any(type(bit) is not int or bit not in (0, 1) for bit in bits):
            raise EncodingValidationError("physical_bits entries must be bits (0 or 1)")
        index = 0
        for bit in bits:
            index = (index << 1) | bit
        return self.decode_table[index]

    def to_manifest_dict(self) -> Mapping[str, JsonValue]:
        matrix = [
            [[float(value.real), float(value.imag)] for value in row]
            for row in self.matrix
        ]
        return {
            "kind": self.kind,
            "schema_version": _ENCODING_SCHEMA,
            "encoding_id": self.encoding_id,
            "logical_dimension": self.logical_dimension,
            "physical_qubits": self.physical_qubits,
            "matrix": matrix,
            "decode_table": [
                {"value": outcome.value, "leaked": outcome.leaked}
                for outcome in self.decode_table
            ],
        }

    def stable_hash(self) -> str:
        return _canonical_hash(dict(self.to_manifest_dict()))

    @classmethod
    def from_manifest_dict(
        cls, data: Mapping[str, Any]
    ) -> "IsometricQuditEncoding":
        if not isinstance(data, Mapping):
            raise EncodingValidationError("encoding manifest must be a mapping")
        expected = {
            "kind",
            "schema_version",
            "encoding_id",
            "logical_dimension",
            "physical_qubits",
            "matrix",
            "decode_table",
        }
        _require_exact_keys(data, expected, "encoding manifest", EncodingValidationError)
        if data["kind"] != "isometric" or data["schema_version"] != _ENCODING_SCHEMA:
            raise EncodingValidationError("unsupported encoding kind or schema_version")
        try:
            matrix = np.asarray(data["matrix"], dtype=float)
            if matrix.ndim != 3 or matrix.shape[-1] != 2:
                raise ValueError
            complex_matrix = matrix[..., 0] + 1j * matrix[..., 1]
            decode_table = tuple(
                LogicalOutcome(value=item["value"], leaked=item["leaked"])
                for item in data["decode_table"]
            )
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise EncodingValidationError("encoding manifest payload is invalid") from error
        return cls(
            encoding_id=data["encoding_id"],
            logical_dimension=data["logical_dimension"],
            physical_qubits=data["physical_qubits"],
            matrix=complex_matrix,
            decode_table=decode_table,
        )


@dataclass(frozen=True)
class ExecutionSpec:
    shots: int
    transpilation: TranspilationConfig = field(default_factory=TranspilationConfig)
    mitigation: Mapping[str, JsonValue] = field(default_factory=dict)
    uncertainty: Mapping[str, JsonValue] = field(default_factory=dict)
    retry: Mapping[str, JsonValue] = field(default_factory=dict)
    seed: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.shots, bool) or not isinstance(self.shots, int) or self.shots <= 0:
            raise SpecValidationError("shots must be a positive integer")
        if not isinstance(self.transpilation, TranspilationConfig):
            raise SpecValidationError("transpilation must be TranspilationConfig")
        _validate_json(
            self.transpilation.to_safe_dict(),
            'transpilation',
            SpecValidationError,
        )
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise SpecValidationError("seed must be an integer or None")
        for field_name in ("mitigation", "uncertainty", "retry"):
            object.__setattr__(
                self,
                field_name,
                _safe_mapping(getattr(self, field_name), field_name, SpecValidationError),
            )

    def to_manifest_dict(self) -> Mapping[str, JsonValue]:
        return {
            "shots": self.shots,
            "transpilation": self.transpilation.to_safe_dict(),
            "mitigation": _thaw_json(self.mitigation),
            "uncertainty": _thaw_json(self.uncertainty),
            "retry": _thaw_json(self.retry),
            "seed": self.seed,
        }


@dataclass(frozen=True)
class QuditExperimentSpec:
    circuit: CircuitSpec
    encoding: QuditEncoding
    backend: AerIdeal
    execution: ExecutionSpec
    output_root: Path = Path("artifacts/vertical_slice_runs")
    tags: Mapping[str, str] = field(default_factory=dict)

    @property
    def schema_version(self) -> str:
        return _EXPERIMENT_SCHEMA

    def __post_init__(self) -> None:
        if not isinstance(self.circuit, CircuitSpec):
            raise SpecValidationError("circuit must implement CircuitSpec")
        if not isinstance(self.encoding, QuditEncoding):
            raise SpecValidationError("encoding must implement QuditEncoding")
        if not isinstance(self.backend, AerIdeal):
            raise SpecValidationError("backend must be AerIdeal")
        if not isinstance(self.execution, ExecutionSpec):
            raise SpecValidationError("execution must be ExecutionSpec")

        dimensions = self.circuit.logical_dimensions
        if (
            not isinstance(dimensions, tuple)
            or not dimensions
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 2
                for value in dimensions
            )
        ):
            raise SpecValidationError(
                "circuit logical dimensions must be a non-empty tuple of integers at least 2"
            )
        if any(value != self.encoding.logical_dimension for value in dimensions):
            raise SpecValidationError(
                "circuit logical dimensions must match encoding logical_dimension"
            )
        if (
            self.execution.seed is not None
            and self.execution.seed != self.backend.seed_simulator
        ):
            raise SpecValidationError(
                "execution seed must match AerIdeal seed_simulator when provided"
            )

        output_root = Path(self.output_root)
        _safe_text(str(output_root), "output_root", SpecValidationError)
        object.__setattr__(self, "output_root", output_root)
        if not isinstance(self.tags, Mapping) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in self.tags.items()
        ):
            raise SpecValidationError("tags must be a mapping of strings")
        validate_persisted_strings(
            self.tags, description="tags", error_type=SpecValidationError
        )
        object.__setattr__(self, "tags", MappingProxyType(dict(self.tags)))

    def to_manifest_dict(self) -> Mapping[str, JsonValue]:
        circuit = dict(self.circuit.to_manifest_dict())
        encoding = dict(self.encoding.to_manifest_dict())
        _validate_json(circuit, "circuit manifest", SpecValidationError)
        _validate_json(encoding, "encoding manifest", SpecValidationError)
        return {
            "schema_version": self.schema_version,
            "circuit": circuit,
            "encoding": encoding,
            "backend": self.backend.to_safe_dict(),
            "execution": dict(self.execution.to_manifest_dict()),
            "output_root": str(self.output_root),
            "tags": dict(self.tags),
        }

    def stable_hash(self) -> str:
        payload = dict(self.to_manifest_dict())
        payload.pop("output_root")
        payload.pop("tags")
        return _canonical_hash(payload)


@dataclass(frozen=True)
class ArtifactRef:
    role: str
    path: str
    sha256: str
    media_type: str

    def __post_init__(self) -> None:
        _safe_text(self.role, "artifact role", ManifestValidationError)
        path = _safe_text(self.path, "artifact path", ManifestValidationError)
        portable_parts = path.replace("\\", "/").split("/")
        windows_path = PureWindowsPath(path)
        posix_path = PurePosixPath(path)
        if (
            windows_path.drive
            or windows_path.root
            or windows_path.is_absolute()
            or posix_path.drive
            or posix_path.root
            or posix_path.is_absolute()
            or any(part in {"", ".", ".."} for part in portable_parts)
        ):
            raise ManifestValidationError(
                "artifact path must be a non-empty contained relative path"
            )
        path = "/".join(portable_parts)
        object.__setattr__(self, "path", path)
        _require_sha256(self.sha256, "artifact sha256")
        _safe_text(self.media_type, "artifact media_type", ManifestValidationError)

    def to_safe_dict(self) -> dict[str, JsonValue]:
        return {
            "role": self.role,
            "path": self.path,
            "sha256": self.sha256,
            "media_type": self.media_type,
        }

    @classmethod
    def from_safe_dict(cls, data: Mapping[str, Any]) -> "ArtifactRef":
        _require_exact_keys(
            data, {"role", "path", "sha256", "media_type"}, "artifact", ManifestValidationError
        )
        return cls(**dict(data))


@dataclass(frozen=True)
class SoftwareProvenance:
    git_commit: str | None
    package_version: str
    python_version: str
    dependencies: Mapping[str, str]
    dirty_worktree: bool | None

    def __post_init__(self) -> None:
        if self.git_commit is not None:
            _safe_text(self.git_commit, "git_commit", SpecValidationError)
        _safe_text(self.package_version, "package_version", SpecValidationError)
        _safe_text(self.python_version, "python_version", SpecValidationError)
        if self.dirty_worktree is not None and type(self.dirty_worktree) is not bool:
            raise SpecValidationError("dirty_worktree must be a boolean or None")
        if not isinstance(self.dependencies, Mapping) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in self.dependencies.items()
        ):
            raise SpecValidationError("dependencies must be a mapping of strings")
        validate_persisted_strings(
            self.dependencies, description="dependencies", error_type=SpecValidationError
        )
        object.__setattr__(
            self, "dependencies", MappingProxyType(dict(self.dependencies))
        )

    def to_safe_dict(self) -> dict[str, JsonValue]:
        return {
            "git_commit": self.git_commit,
            "package_version": self.package_version,
            "python_version": self.python_version,
            "dependencies": dict(self.dependencies),
            "dirty_worktree": self.dirty_worktree,
        }

    @classmethod
    def from_safe_dict(cls, data: Mapping[str, Any]) -> "SoftwareProvenance":
        _require_exact_keys(
            data,
            {
                "git_commit",
                "package_version",
                "python_version",
                "dependencies",
                "dirty_worktree",
            },
            "software provenance",
            ManifestValidationError,
        )
        try:
            return cls(**dict(data))
        except SpecValidationError as error:
            raise ManifestValidationError(str(error)) from error


@dataclass(frozen=True)
class BackendSnapshot:
    provider: str
    backend_name: str
    execution_mode: ExecutionMode
    identity: Mapping[str, JsonValue]
    capabilities: Mapping[str, JsonValue]
    calibration_reference: Mapping[str, JsonValue] | None = None

    def __post_init__(self) -> None:
        _safe_text(self.provider, "backend provider", ManifestValidationError)
        _safe_text(self.backend_name, "backend name", ManifestValidationError)
        if not isinstance(self.execution_mode, ExecutionMode):
            raise ManifestValidationError("execution_mode must be ExecutionMode")
        object.__setattr__(
            self,
            "identity",
            _safe_mapping(self.identity, "backend identity", ManifestValidationError),
        )
        object.__setattr__(
            self,
            "capabilities",
            _safe_mapping(
                self.capabilities, "backend capabilities", ManifestValidationError
            ),
        )
        if self.calibration_reference is not None:
            object.__setattr__(
                self,
                "calibration_reference",
                _safe_mapping(
                    self.calibration_reference,
                    "calibration reference",
                    ManifestValidationError,
                ),
            )

    def to_safe_dict(self) -> dict[str, JsonValue]:
        return {
            "provider": self.provider,
            "backend_name": self.backend_name,
            "execution_mode": self.execution_mode.value,
            "identity": _thaw_json(self.identity),
            "capabilities": _thaw_json(self.capabilities),
            "calibration_reference": (
                None
                if self.calibration_reference is None
                else _thaw_json(self.calibration_reference)
            ),
        }

    @classmethod
    def from_safe_dict(cls, data: Mapping[str, Any]) -> "BackendSnapshot":
        _require_exact_keys(
            data,
            {
                "provider",
                "backend_name",
                "execution_mode",
                "identity",
                "capabilities",
                "calibration_reference",
            },
            "backend snapshot",
            ManifestValidationError,
        )
        try:
            mode = ExecutionMode(data["execution_mode"])
        except (TypeError, ValueError) as error:
            raise ManifestValidationError("unsupported execution_mode") from error
        return cls(
            provider=data["provider"],
            backend_name=data["backend_name"],
            execution_mode=mode,
            identity=data["identity"],
            capabilities=data["capabilities"],
            calibration_reference=data["calibration_reference"],
        )


_TRANSITIONS: Mapping[str, frozenset[str]] = {
    "created": frozenset({"validated", "failed"}),
    "validated": frozenset({"compiled", "failed"}),
    "compiled": frozenset({"running", "failed"}),
    "running": frozenset({"postprocessing", "failed"}),
    "postprocessing": frozenset({"completed", "failed"}),
    "completed": frozenset(),
    "failed": frozenset(),
}


def _safe_warnings(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value, (list, tuple)
    ):
        raise ManifestValidationError("warnings must be a list or tuple of strings")
    warnings = tuple(value)
    for warning in warnings:
        _safe_text(warning, "warning", ManifestValidationError)
    return warnings


@dataclass(frozen=True)
class RunManifest:
    schema_version: str
    run_id: str
    experiment_spec: Mapping[str, JsonValue]
    experiment_hash: str
    encoding: Mapping[str, JsonValue]
    encoding_hash: str
    backend: BackendSnapshot | None
    software: SoftwareProvenance
    status: str
    timestamps: Mapping[str, str]
    status_history: tuple[Mapping[str, JsonValue], ...]
    jobs: Mapping[str, Mapping[str, JsonValue]]
    artifacts: tuple[ArtifactRef, ...]
    result: Mapping[str, JsonValue] | None
    warnings: tuple[str, ...]
    failure: Mapping[str, JsonValue] | None

    def __post_init__(self) -> None:
        if self.schema_version != _MANIFEST_SCHEMA:
            raise ManifestValidationError("unsupported run manifest schema_version")
        _safe_text(self.run_id, "run_id", ManifestValidationError)
        object.__setattr__(
            self,
            "experiment_spec",
            _safe_mapping(
                self.experiment_spec, "experiment_spec", ManifestValidationError
            ),
        )
        object.__setattr__(
            self,
            "encoding",
            _safe_mapping(self.encoding, "encoding", ManifestValidationError),
        )
        _require_sha256(self.experiment_hash, "experiment_hash")
        _require_sha256(self.encoding_hash, "encoding_hash")
        if _experiment_snapshot_hash(self.experiment_spec) != self.experiment_hash:
            raise ManifestValidationError(
                'experiment_hash does not match experiment_spec'
            )
        if _canonical_hash(_thaw_json(self.encoding)) != self.encoding_hash:
            raise ManifestValidationError('encoding_hash does not match encoding')
        if self.backend is not None and not isinstance(self.backend, BackendSnapshot):
            raise ManifestValidationError("backend must be BackendSnapshot or None")
        if not isinstance(self.software, SoftwareProvenance):
            raise ManifestValidationError("software must be SoftwareProvenance")
        if self.status not in _TRANSITIONS:
            raise ManifestValidationError("unsupported run status")

        if not isinstance(self.timestamps, Mapping):
            raise ManifestValidationError("timestamps must be a mapping")
        timestamps: dict[str, str] = {}
        for key, timestamp in self.timestamps.items():
            key = _safe_text(key, "timestamp name", ManifestValidationError)
            timestamps[key] = _safe_text(timestamp, "timestamp", ManifestValidationError)
        if not {"created", "updated", self.status}.issubset(timestamps):
            raise ManifestValidationError(
                "timestamps must contain created, updated, and current status"
            )
        object.__setattr__(self, "timestamps", MappingProxyType(timestamps))

        history = tuple(self.status_history)
        if not history:
            raise ManifestValidationError("status_history must not be empty")
        normalized_history: list[Mapping[str, JsonValue]] = []
        prior: str | None = None
        for entry in history:
            if not isinstance(entry, Mapping) or set(entry) != {"status", "timestamp"}:
                raise ManifestValidationError("status_history entry is invalid")
            entry_status = entry["status"]
            timestamp = entry["timestamp"]
            if not isinstance(entry_status, str) or entry_status not in _TRANSITIONS:
                raise ManifestValidationError("status_history status is invalid")
            _safe_text(timestamp, "status_history timestamp", ManifestValidationError)
            if prior is None and entry_status != "created":
                raise ManifestValidationError("status_history must start with created")
            if prior is not None and entry_status not in _TRANSITIONS[prior]:
                raise ManifestValidationError("status_history contains illegal transition")
            normalized_history.append(
                MappingProxyType({"status": entry_status, "timestamp": timestamp})
            )
            prior = entry_status
        first_timestamp = normalized_history[0]['timestamp']
        final_timestamp = normalized_history[-1]['timestamp']
        if timestamps['created'] != first_timestamp:
            raise ManifestValidationError(
                'created timestamp does not match status_history'
            )
        if (
            timestamps['updated'] != final_timestamp
            or timestamps[self.status] != final_timestamp
        ):
            raise ManifestValidationError(
                'current status timestamp does not match status_history'
            )
        latest_timestamps = {
            entry['status']: entry['timestamp'] for entry in normalized_history
        }
        if any(
            timestamps.get(history_status) != history_timestamp
            for history_status, history_timestamp in latest_timestamps.items()
        ):
            raise ManifestValidationError(
                'status timestamp does not match status_history'
            )
        if prior != self.status:
            raise ManifestValidationError("status_history must end with current status")
        object.__setattr__(self, "status_history", tuple(normalized_history))

        if not isinstance(self.jobs, Mapping):
            raise ManifestValidationError("jobs must be a mapping")
        normalized_jobs: dict[str, Mapping[str, JsonValue]] = {}
        for key, value in self.jobs.items():
            key = _safe_text(key, "job key", ManifestValidationError)
            normalized_jobs[key] = _safe_mapping(
                value, "job record", ManifestValidationError
            )
        object.__setattr__(self, "jobs", MappingProxyType(normalized_jobs))

        artifacts = tuple(self.artifacts)
        if any(not isinstance(item, ArtifactRef) for item in artifacts):
            raise ManifestValidationError("artifacts must contain ArtifactRef values")
        object.__setattr__(self, "artifacts", artifacts)
        if self.result is not None:
            object.__setattr__(
                self,
                "result",
                _safe_mapping(self.result, "result", ManifestValidationError),
            )
        warnings = _safe_warnings(self.warnings)
        object.__setattr__(self, "warnings", warnings)
        if self.failure is not None:
            object.__setattr__(
                self,
                "failure",
                _safe_mapping(self.failure, "failure", ManifestValidationError),
            )

        if self.status == "completed":
            if self.result is None or self.failure is not None:
                raise ManifestValidationError(
                    "completed manifest requires result and no failure"
                )
        elif self.status == "failed":
            if self.failure is None or self.result is not None:
                raise ManifestValidationError(
                    "failed manifest requires failure and no result"
                )
        elif self.result is not None or self.failure is not None:
            raise ManifestValidationError(
                "nonterminal manifest must not contain result or failure"
            )

    @classmethod
    def initial(
        cls,
        run_id: str,
        experiment_spec: Mapping[str, JsonValue],
        experiment_hash: str,
        encoding: Mapping[str, JsonValue],
        encoding_hash: str,
        software: SoftwareProvenance,
        timestamp: str,
    ) -> "RunManifest":
        _safe_text(timestamp, "timestamp", ManifestValidationError)
        return cls(
            schema_version=_MANIFEST_SCHEMA,
            run_id=run_id,
            experiment_spec=experiment_spec,
            experiment_hash=experiment_hash,
            encoding=encoding,
            encoding_hash=encoding_hash,
            backend=None,
            software=software,
            status="created",
            timestamps={"created": timestamp, "updated": timestamp},
            status_history=({"status": "created", "timestamp": timestamp},),
            jobs={},
            artifacts=(),
            result=None,
            warnings=(),
            failure=None,
        )

    def transition(
        self,
        status: str,
        *,
        timestamp: str,
        backend: BackendSnapshot | None = None,
        jobs: Mapping[str, Mapping[str, JsonValue]] | None = None,
        artifacts: tuple[ArtifactRef, ...] | None = None,
        result: Mapping[str, JsonValue] | None = None,
        warnings: tuple[str, ...] | None = None,
        failure: Mapping[str, JsonValue] | None = None,
    ) -> "RunManifest":
        if status not in _TRANSITIONS.get(self.status, frozenset()):
            raise ManifestValidationError(
                f"illegal status transition from {self.status!r} to {status!r}"
            )
        _safe_text(timestamp, "timestamp", ManifestValidationError)
        new_timestamps = dict(self.timestamps)
        new_timestamps.update({"updated": timestamp, status: timestamp})
        new_history = self.status_history + (
            {"status": status, "timestamp": timestamp},
        )
        return replace(
            self,
            status=status,
            timestamps=new_timestamps,
            status_history=new_history,
            backend=self.backend if backend is None else backend,
            jobs=self.jobs if jobs is None else jobs,
            artifacts=self.artifacts if artifacts is None else tuple(artifacts),
            result=self.result if result is None else result,
            warnings=self.warnings if warnings is None else _safe_warnings(warnings),
            failure=self.failure if failure is None else failure,
        )

    def to_safe_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "experiment_spec": _thaw_json(self.experiment_spec),
            "experiment_hash": self.experiment_hash,
            "encoding": _thaw_json(self.encoding),
            "encoding_hash": self.encoding_hash,
            "backend": None if self.backend is None else self.backend.to_safe_dict(),
            "software": self.software.to_safe_dict(),
            "status": self.status,
            "timestamps": dict(self.timestamps),
            "status_history": [_thaw_json(item) for item in self.status_history],
            "jobs": _thaw_json(self.jobs),
            "artifacts": [item.to_safe_dict() for item in self.artifacts],
            "result": None if self.result is None else _thaw_json(self.result),
            "warnings": list(self.warnings),
            "failure": None if self.failure is None else _thaw_json(self.failure),
        }

    @classmethod
    def from_safe_dict(cls, data: Mapping[str, Any]) -> "RunManifest":
        if not isinstance(data, Mapping):
            raise ManifestValidationError("run manifest must be a mapping")
        expected = {
            "schema_version",
            "run_id",
            "experiment_spec",
            "experiment_hash",
            "encoding",
            "encoding_hash",
            "backend",
            "software",
            "status",
            "timestamps",
            "status_history",
            "jobs",
            "artifacts",
            "result",
            "warnings",
            "failure",
        }
        _require_exact_keys(data, expected, "run manifest", ManifestValidationError)
        if data["schema_version"] != _MANIFEST_SCHEMA:
            raise ManifestValidationError("unsupported run manifest schema_version")
        try:
            backend = (
                None
                if data["backend"] is None
                else BackendSnapshot.from_safe_dict(data["backend"])
            )
            software = SoftwareProvenance.from_safe_dict(data["software"])
            artifacts = tuple(
                ArtifactRef.from_safe_dict(item) for item in data["artifacts"]
            )
            return cls(
                schema_version=data["schema_version"],
                run_id=data["run_id"],
                experiment_spec=data["experiment_spec"],
                experiment_hash=data["experiment_hash"],
                encoding=data["encoding"],
                encoding_hash=data["encoding_hash"],
                backend=backend,
                software=software,
                status=data["status"],
                timestamps=data["timestamps"],
                status_history=tuple(data["status_history"]),
                jobs=data["jobs"],
                artifacts=artifacts,
                result=data["result"],
                warnings=_safe_warnings(data["warnings"]),
                failure=data["failure"],
            )
        except ManifestValidationError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise ManifestValidationError("run manifest payload is invalid") from error


@dataclass(frozen=True)
class QuditExperimentResult:
    artifact_dir: Path
    manifest: RunManifest
    result: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        artifact_dir = Path(self.artifact_dir)
        _safe_text(str(artifact_dir), "artifact_dir", SpecValidationError)
        object.__setattr__(self, "artifact_dir", artifact_dir)
        if not isinstance(self.manifest, RunManifest):
            raise SpecValidationError("manifest must be RunManifest")
        object.__setattr__(
            self,
            "result",
            _safe_mapping(self.result, "result", SpecValidationError),
        )


__all__ = [
    "ArtifactIntegrityError",
    "ArtifactRef",
    "BackendSnapshot",
    "CircuitSpec",
    "EncodingValidationError",
    "ExecutionMode",
    "ExecutionSpec",
    "IsometricQuditEncoding",
    "JsonValue",
    "LogicalOutcome",
    "ManifestValidationError",
    "PostprocessorSpec",
    "PreparedExperiment",
    "QuditEncoding",
    "QuditExperimentResult",
    "QuditExperimentSpec",
    "RunManifest",
    "SoftwareProvenance",
    "SpecValidationError",
]
