"""Immutable, validated models for reproducible experiment runs."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .errors import ExperimentValidationError
from .safety import unsafe_persisted_text, validate_persisted_strings


_STATES = {"two_qutrit", "ghz3", "ame43"}


class _Unset:
    __slots__ = ()

    def __repr__(self) -> str:
        return "<factory>"


_UNSET = _Unset()


def _safe_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ExperimentValidationError(f"{field_name} must be a non-empty string")
    if unsafe_persisted_text(value):
        raise ExperimentValidationError(
            f"{field_name} must not contain credential material"
        ) from None
    return value


def _safe_optional_path(value: Path | str | None, field_name: str) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    _safe_text(str(path), field_name)
    return path


def _safe_tags(value: Mapping[str, str]) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise ExperimentValidationError("tags must be a mapping of strings")
    validate_persisted_strings(
        value,
        description="tags",
        error_type=ExperimentValidationError,
    )
    tags: dict[str, str] = {}
    for key, item in value.items():
        tags[_safe_text(key, "tag key")] = _safe_text(item, "tag value")
    return MappingProxyType(tags)


def _require_bool(value: object, field_name: str) -> None:
    if type(value) is not bool:
        raise ExperimentValidationError(f"{field_name} must be a boolean")


def _require_finite_real(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ExperimentValidationError(f"{field_name} must be a finite real value")


@dataclass(frozen=True)
class PathBasis:
    directory: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "directory", _safe_optional_path(self.directory, "basis directory"))
        if self.directory is None:
            raise ExperimentValidationError("basis directory is required")

    def to_safe_dict(self) -> dict[str, str]:
        return {"kind": "path", "directory": str(self.directory)}

    @classmethod
    def from_safe_dict(cls, data: Mapping[str, Any]) -> "PathBasis":
        return cls(directory=Path(data["directory"]))


@dataclass(frozen=True)
class BenchmarkBasis:
    run_kind: str
    run_id: str
    selection: str
    rank: int | None = None
    candidate: str | None = None

    def __post_init__(self) -> None:
        if self.run_kind not in {"direct_basis_runs", "iqm_runs"}:
            raise ExperimentValidationError("run_kind must be direct_basis_runs or iqm_runs")
        _safe_text(self.run_id, "run_id")
        _safe_text(self.selection, "selection")
        if (self.rank is None) == (self.candidate is None):
            raise ExperimentValidationError("exactly one of rank or candidate is required")
        if self.rank is not None and (isinstance(self.rank, bool) or not isinstance(self.rank, int) or self.rank < 0):
            raise ExperimentValidationError("rank must be a non-negative integer")
        if self.candidate is not None:
            _safe_text(self.candidate, "candidate")

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "kind": "benchmark",
            "run_kind": self.run_kind,
            "run_id": self.run_id,
            "selection": self.selection,
            "rank": self.rank,
            "candidate": self.candidate,
        }

    @classmethod
    def from_safe_dict(cls, data: Mapping[str, Any]) -> "BenchmarkBasis":
        return cls(
            run_kind=data["run_kind"], run_id=data["run_id"], selection=data["selection"],
            rank=data.get("rank"), candidate=data.get("candidate"),
        )


@dataclass(frozen=True)
class AerIdeal:
    seed_simulator: int = 123

    def __post_init__(self) -> None:
        if isinstance(self.seed_simulator, bool) or not isinstance(self.seed_simulator, int):
            raise ExperimentValidationError("seed_simulator must be an integer")

    def to_safe_dict(self) -> dict[str, Any]:
        return {"kind": "aer_ideal", "seed_simulator": self.seed_simulator}

    @classmethod
    def from_safe_dict(cls, data: Mapping[str, Any]) -> "AerIdeal":
        return cls(seed_simulator=data.get("seed_simulator", 123))


@dataclass(frozen=True)
class IQMHardware:
    device: str
    use_metrics: bool = False
    env_path: Path | None = None

    def __post_init__(self) -> None:
        _require_bool(self.use_metrics, "use_metrics")
        _safe_text(self.device, "device")
        object.__setattr__(self, "env_path", _safe_optional_path(self.env_path, "env_path"))

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "kind": "iqm_hardware",
            "device": self.device,
            "use_metrics": self.use_metrics,
        }

    @classmethod
    def from_safe_dict(cls, data: Mapping[str, Any]) -> "IQMHardware":
        return cls(data["device"], data.get("use_metrics", False))


@dataclass(frozen=True)
class PiastQHardware:
    mode: str = "auto"
    owner: str | None = None
    env_path: Path | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"auto", "managed", "direct"}:
            raise ExperimentValidationError("mode must be auto, managed, or direct")
        if self.owner is not None:
            _safe_text(self.owner, "owner")
        object.__setattr__(self, "env_path", _safe_optional_path(self.env_path, "env_path"))

    def to_safe_dict(self) -> dict[str, Any]:
        return {"kind": "piastq_hardware", "mode": self.mode, "owner": self.owner}

    @classmethod
    def from_safe_dict(cls, data: Mapping[str, Any]) -> "PiastQHardware":
        return cls(data.get("mode", "auto"), data.get("owner"))


@dataclass(frozen=True)
class CustomBackend:
    instance: Any = field(repr=False, compare=False)
    identity: str = "custom"
    supports_resume: bool = False

    def __post_init__(self) -> None:
        _safe_text(self.identity, "identity")
        _require_bool(self.supports_resume, "supports_resume")

    def to_safe_dict(self) -> dict[str, Any]:
        return {"kind": "custom", "identity": self.identity, "supports_resume": self.supports_resume}

    @classmethod
    def from_safe_dict(cls, data: Mapping[str, Any], *, instance: Any = None) -> "CustomBackend":
        if instance is None:
            raise ExperimentValidationError("custom backend reconstruction requires instance injection")
        return cls(instance=instance, identity=data["identity"], supports_resume=data.get("supports_resume", False))


@dataclass(frozen=True)
class NoisySimulator:
    source: Any = field(default=None, repr=False, compare=False)
    noise_model: Any = field(default=None, repr=False, compare=False)
    target_backend: Any = field(default=None, repr=False, compare=False)
    identity: str | None = None

    def __post_init__(self) -> None:
        source_mode = self.source is not None and self.noise_model is None and self.target_backend is None
        model_mode = self.source is None and self.noise_model is not None and self.target_backend is not None
        if not (source_mode or model_mode):
            raise ExperimentValidationError("provide exactly either source or noise_model with target_backend")
        if self.identity is not None:
            _safe_text(self.identity, "identity")

    def to_safe_dict(self) -> dict[str, Any]:
        return {"kind": "noisy_simulator", "identity": self.identity, "source_mode": self.source is not None}

    @classmethod
    def from_safe_dict(cls, data: Mapping[str, Any], **injected: Any) -> "NoisySimulator":
        if not injected:
            raise ExperimentValidationError("noisy simulator reconstruction requires object injection")
        source_mode = data.get("source_mode")
        if type(source_mode) is not bool:
            raise ExperimentValidationError("noisy simulator source_mode must be a boolean")
        reconstructed = cls(identity=data.get("identity"), **injected)
        if (reconstructed.source is not None) != source_mode:
            raise ExperimentValidationError("noisy simulator source_mode does not match injected objects")
        return reconstructed


@dataclass(frozen=True)
class MitigationConfig:
    readout: bool = False
    zne: bool = False
    zne_factors: tuple[int, ...] = (1, 3, 5)
    zne_model: str = "linear"
    readout_max_age_hours: float = 24.0
    force_recalibration: bool = False

    def __post_init__(self) -> None:
        factors = tuple(self.zne_factors)
        _require_bool(self.readout, "readout")
        _require_bool(self.zne, "zne")
        _require_bool(self.force_recalibration, "force_recalibration")
        object.__setattr__(self, "zne_factors", factors)
        if self.zne and (not factors or 1 not in factors or len(set(factors)) != len(factors) or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 or item % 2 == 0 for item in factors)):
            raise ExperimentValidationError("zne_factors must be unique positive odd factors including 1")
        if self.zne_model != "linear":
            raise ExperimentValidationError("zne_model must be linear")
        if not isinstance(self.readout_max_age_hours, (int, float)) or isinstance(self.readout_max_age_hours, bool) or not math.isfinite(self.readout_max_age_hours) or self.readout_max_age_hours <= 0:
            raise ExperimentValidationError("readout_max_age_hours must be positive")

    def to_safe_dict(self) -> dict[str, Any]:
        return {"readout": self.readout, "zne": self.zne, "zne_factors": list(self.zne_factors), "zne_model": self.zne_model, "readout_max_age_hours": self.readout_max_age_hours, "force_recalibration": self.force_recalibration}

    @classmethod
    def from_safe_dict(cls, data: Mapping[str, Any]) -> "MitigationConfig":
        return cls(**{**cls().to_safe_dict(), **dict(data), "zne_factors": tuple(data.get("zne_factors", (1, 3, 5)))})


@dataclass(frozen=True)
class BootstrapConfig:
    samples: int = 2000
    confidence_level: float = 0.95
    seed: int = 12345
    include_readout_calibration: bool = True

    def __post_init__(self) -> None:
        _require_bool(self.include_readout_calibration, "include_readout_calibration")
        if isinstance(self.samples, bool) or not isinstance(self.samples, int) or self.samples < 2:
            raise ExperimentValidationError("samples must be an integer of at least 2")
        if not isinstance(self.confidence_level, (int, float)) or isinstance(self.confidence_level, bool) or not math.isfinite(self.confidence_level) or not 0 < self.confidence_level < 1:
            raise ExperimentValidationError("confidence_level must be between 0 and 1")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ExperimentValidationError("seed must be a non-negative integer")

    def to_safe_dict(self) -> dict[str, Any]:
        return {"samples": self.samples, "confidence_level": self.confidence_level, "seed": self.seed, "include_readout_calibration": self.include_readout_calibration}

    @classmethod
    def from_safe_dict(cls, data: Mapping[str, Any]) -> "BootstrapConfig":
        return cls(**dict(data))


@dataclass(frozen=True)
class TranspilationConfig:
    optimization_level: int = 3
    seed_transpiler: int | None = None
    layout_method: str | None = None
    routing_method: str | None = None
    scheduling_method: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.optimization_level, bool) or not isinstance(self.optimization_level, int) or self.optimization_level not in range(4):
            raise ExperimentValidationError("optimization_level must be an integer from 0 to 3")

    def to_safe_dict(self) -> dict[str, Any]:
        return {"optimization_level": self.optimization_level, "seed_transpiler": self.seed_transpiler, "layout_method": self.layout_method, "routing_method": self.routing_method, "scheduling_method": self.scheduling_method}

    @classmethod
    def from_safe_dict(cls, data: Mapping[str, Any]) -> "TranspilationConfig":
        return cls(**dict(data))


@dataclass(frozen=True)
class RetryConfig:
    max_attempts: int = 3
    initial_delay: float = 1.0
    multiplier: float = 2.0
    max_delay: float = 30.0

    def __post_init__(self) -> None:
        if isinstance(self.max_attempts, bool) or not isinstance(self.max_attempts, int) or self.max_attempts <= 0:
            raise ExperimentValidationError("max_attempts must be a positive integer")
        for field_name in ("initial_delay", "multiplier", "max_delay"):
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ExperimentValidationError(f"{field_name} must be positive")
        if self.max_delay < self.initial_delay:
            raise ExperimentValidationError("max_delay must be at least initial_delay")

    def to_safe_dict(self) -> dict[str, Any]:
        return {"max_attempts": self.max_attempts, "initial_delay": self.initial_delay, "multiplier": self.multiplier, "max_delay": self.max_delay}

    @classmethod
    def from_safe_dict(cls, data: Mapping[str, Any]) -> "RetryConfig":
        return cls(**dict(data))


Basis = PathBasis | BenchmarkBasis
Backend = AerIdeal | IQMHardware | PiastQHardware | CustomBackend | NoisySimulator


def _normalize_experiment_spec_dict(data: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(data)
    canonical = normalized.get("uncertainty", _UNSET)
    legacy = normalized.pop("bootstrap", _UNSET)
    if canonical is not _UNSET and legacy is not _UNSET and canonical != legacy:
        raise ExperimentValidationError(
            "bootstrap and uncertainty contain conflicting values"
        )
    if canonical is _UNSET and legacy is not _UNSET:
        normalized["uncertainty"] = legacy

    backend = normalized.get("backend")
    if isinstance(backend, Mapping) and backend.get("kind") in {
        "iqm_hardware",
        "piastq_hardware",
    }:
        normalized_backend = dict(backend)
        normalized_backend.pop("env_path", None)
        normalized["backend"] = normalized_backend
    return normalized


@dataclass(frozen=True, init=False)
class ExperimentSpec:
    state: str
    basis: Basis
    backend: Backend
    shots: int = 20480
    mitigation: MitigationConfig = field(default_factory=MitigationConfig)
    uncertainty: BootstrapConfig = field(default_factory=BootstrapConfig)
    transpilation: TranspilationConfig = field(default_factory=TranspilationConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    output_root: Path = Path("artifacts/experiment_runs")
    tags: Mapping[str, str] = field(default_factory=dict)

    def __init__(
        self,
        state: str,
        basis: Basis,
        backend: Backend,
        shots: int = 20480,
        mitigation: MitigationConfig | object = _UNSET,
        uncertainty: BootstrapConfig | object = _UNSET,
        transpilation: TranspilationConfig | object = _UNSET,
        retry: RetryConfig | object = _UNSET,
        output_root: Path | str = Path("artifacts/experiment_runs"),
        tags: Mapping[str, str] | object = _UNSET,
        *,
        bootstrap: BootstrapConfig | object = _UNSET,
    ) -> None:
        if bootstrap is not _UNSET and not isinstance(bootstrap, BootstrapConfig):
            raise ExperimentValidationError("bootstrap must be BootstrapConfig")
        if uncertainty is not _UNSET and not isinstance(uncertainty, BootstrapConfig):
            raise ExperimentValidationError("uncertainty must be BootstrapConfig")
        if (
            bootstrap is not _UNSET
            and uncertainty is not _UNSET
            and bootstrap != uncertainty
        ):
            raise ExperimentValidationError(
                "bootstrap and uncertainty contain conflicting values"
            )
        if uncertainty is not _UNSET:
            uncertainty_config = uncertainty
        elif bootstrap is not _UNSET:
            uncertainty_config = bootstrap
        else:
            uncertainty_config = BootstrapConfig()
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "shots", shots)
        object.__setattr__(
            self,
            "mitigation",
            MitigationConfig() if mitigation is _UNSET else mitigation,
        )
        object.__setattr__(self, "uncertainty", uncertainty_config)
        object.__setattr__(
            self,
            "transpilation",
            TranspilationConfig() if transpilation is _UNSET else transpilation,
        )
        object.__setattr__(self, "retry", RetryConfig() if retry is _UNSET else retry)
        object.__setattr__(self, "output_root", output_root)
        object.__setattr__(self, "tags", {} if tags is _UNSET else tags)
        self.__post_init__()

    @property
    def bootstrap(self) -> BootstrapConfig:
        """Backward-compatible alias for :attr:`uncertainty`."""

        return self.uncertainty

    def __post_init__(self) -> None:
        state = "ghz3" if self.state == "ghz" else self.state
        if state not in _STATES:
            raise ExperimentValidationError("state must be two_qutrit, ghz3, or ame43")
        object.__setattr__(self, "state", state)
        if isinstance(self.shots, bool) or not isinstance(self.shots, int) or self.shots <= 0:
            raise ExperimentValidationError("shots must be a positive integer")
        if not isinstance(self.basis, (PathBasis, BenchmarkBasis)):
            raise ExperimentValidationError("basis must be a supported basis specification")
        if not isinstance(self.backend, (AerIdeal, IQMHardware, PiastQHardware, CustomBackend, NoisySimulator)):
            raise ExperimentValidationError("backend must be a supported backend specification")
        if not isinstance(self.uncertainty, BootstrapConfig):
            raise ExperimentValidationError("uncertainty must be BootstrapConfig")
        object.__setattr__(self, "output_root", _safe_optional_path(self.output_root, "output_root"))
        if self.output_root is None:
            raise ExperimentValidationError("output_root is required")
        object.__setattr__(self, "tags", _safe_tags(self.tags))

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "state": self.state, "basis": self.basis.to_safe_dict(), "backend": self.backend.to_safe_dict(),
            "shots": self.shots, "mitigation": self.mitigation.to_safe_dict(), "uncertainty": self.uncertainty.to_safe_dict(),
            "transpilation": self.transpilation.to_safe_dict(), "retry": self.retry.to_safe_dict(),
            "output_root": str(self.output_root), "tags": dict(self.tags),
        }

    @classmethod
    def from_safe_dict(cls, data: Mapping[str, Any]) -> "ExperimentSpec":
        data = _normalize_experiment_spec_dict(data)
        return cls(
            state=data["state"], basis=_basis_from_safe_dict(data["basis"]), backend=_backend_from_safe_dict(data["backend"]),
            shots=data.get("shots", 20480), mitigation=MitigationConfig.from_safe_dict(data.get("mitigation", {})),
            uncertainty=BootstrapConfig.from_safe_dict(data.get("uncertainty", {})),
            transpilation=TranspilationConfig.from_safe_dict(data.get("transpilation", {})), retry=RetryConfig.from_safe_dict(data.get("retry", {})),
            output_root=Path(data.get("output_root", "artifacts/experiment_runs")), tags=data.get("tags", {}),
        )


def _basis_from_safe_dict(data: Mapping[str, Any]) -> Basis:
    kind = data.get("kind")
    if kind == "path":
        return PathBasis.from_safe_dict(data)
    if kind == "benchmark":
        return BenchmarkBasis.from_safe_dict(data)
    raise ExperimentValidationError(f"unsupported basis kind {kind!r}")


def _backend_from_safe_dict(data: Mapping[str, Any]) -> Backend:
    kind = data.get("kind")
    classes: dict[str, Any] = {"aer_ideal": AerIdeal, "iqm_hardware": IQMHardware, "piastq_hardware": PiastQHardware}
    if kind in classes:
        return classes[kind].from_safe_dict(data)
    if kind in {"custom", "noisy_simulator"}:
        raise ExperimentValidationError(f"{kind} reconstruction requires object injection")
    raise ExperimentValidationError(f"unsupported backend kind {kind!r}")


class ExperimentStatus(str, Enum):
    CREATED = "created"
    VALIDATED = "validated"
    COMPILED = "compiled"
    SUBMITTED = "submitted"
    RUNNING = "running"
    POSTPROCESSING = "postprocessing"
    COMPLETED = "completed"
    FAILED = "failed"
    SUBMISSION_UNKNOWN = "submission_unknown"


BackendStatus = ExperimentStatus


@dataclass(frozen=True)
class ComplexComponents:
    real: float
    imag: float

    def __post_init__(self) -> None:
        _require_finite_real(self.real, "real")
        _require_finite_real(self.imag, "imag")

    def to_safe_dict(self) -> dict[str, float]:
        return {"real": self.real, "imag": self.imag}


@dataclass(frozen=True)
class ConfidenceInterval:
    low: float
    high: float

    def __post_init__(self) -> None:
        _require_finite_real(self.low, "low")
        _require_finite_real(self.high, "high")
        if self.low > self.high:
            raise ExperimentValidationError("confidence interval low must not exceed high")

    def to_safe_dict(self) -> dict[str, float]:
        return {"low": self.low, "high": self.high}


@dataclass(frozen=True)
class ComplexConfidenceInterval:
    real: ConfidenceInterval
    imag: ConfidenceInterval

    def __post_init__(self) -> None:
        if not isinstance(self.real, ConfidenceInterval):
            raise ExperimentValidationError("real must be ConfidenceInterval")
        if not isinstance(self.imag, ConfidenceInterval):
            raise ExperimentValidationError("imag must be ConfidenceInterval")

    def to_safe_dict(self) -> dict[str, dict[str, float]]:
        return {"real": self.real.to_safe_dict(), "imag": self.imag.to_safe_dict()}


@dataclass(frozen=True)
class BellEstimate:
    estimate: ComplexComponents
    standard_error: ComplexComponents
    confidence_interval: ComplexConfidenceInterval

    def __post_init__(self) -> None:
        if not isinstance(self.estimate, ComplexComponents):
            raise ExperimentValidationError("estimate must be ComplexComponents")
        if not isinstance(self.standard_error, ComplexComponents):
            raise ExperimentValidationError("standard_error must be ComplexComponents")
        if not isinstance(self.confidence_interval, ComplexConfidenceInterval):
            raise ExperimentValidationError("confidence_interval must be ComplexConfidenceInterval")
        if self.standard_error.real < 0 or self.standard_error.imag < 0:
            raise ExperimentValidationError("standard_error must be non-negative")

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "estimate": self.estimate.to_safe_dict(),
            "standard_error": self.standard_error.to_safe_dict(),
            "confidence_interval": self.confidence_interval.to_safe_dict(),
        }


@dataclass(frozen=True)
class ExperimentResult:
    experiment_id: str
    status: ExperimentStatus
    artifact_dir: Path
    values: Mapping[str, Any]
    backend: Mapping[str, Any]
    job_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _safe_text(self.experiment_id, "experiment_id")
        object.__setattr__(self, "artifact_dir", Path(self.artifact_dir))
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))
        object.__setattr__(self, "backend", MappingProxyType(dict(self.backend)))
        object.__setattr__(self, "job_ids", tuple(self.job_ids))

    def to_safe_dict(self) -> dict[str, Any]:
        return {"experiment_id": self.experiment_id, "status": self.status.value, "artifact_dir": str(self.artifact_dir), "values": dict(self.values), "backend": dict(self.backend), "job_ids": list(self.job_ids)}
