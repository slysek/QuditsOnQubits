"""Immutable, validated models for reproducible experiment runs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
import math
from enum import Enum
from numbers import Real
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Mapping

from .errors import ExperimentValidationError
from .execution import ExecutionMode, validate_backend_execution_mode
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
    execution_mode: ClassVar[ExecutionMode] = ExecutionMode.IDEAL_SIMULATOR

    def __post_init__(self) -> None:
        if isinstance(self.seed_simulator, bool) or not isinstance(self.seed_simulator, int):
            raise ExperimentValidationError("seed_simulator must be an integer")

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "kind": "aer_ideal",
            "seed_simulator": self.seed_simulator,
            "execution_mode": self.execution_mode.value,
        }

    @classmethod
    def from_safe_dict(cls, data: Mapping[str, Any]) -> "AerIdeal":
        validate_backend_execution_mode(
            "aer_ideal",
            data.get("execution_mode", ExecutionMode.IDEAL_SIMULATOR.value),
        )
        return cls(seed_simulator=data.get("seed_simulator", 123))


@dataclass(frozen=True)
class IQMHardware:
    device: str
    use_metrics: bool = False
    env_path: Path | None = None
    execution_mode: ClassVar[ExecutionMode] = ExecutionMode.HARDWARE

    def __post_init__(self) -> None:
        _require_bool(self.use_metrics, "use_metrics")
        _safe_text(self.device, "device")
        object.__setattr__(self, "env_path", _safe_optional_path(self.env_path, "env_path"))

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "kind": "iqm_hardware",
            "device": self.device,
            "use_metrics": self.use_metrics,
            "execution_mode": self.execution_mode.value,
        }

    @classmethod
    def from_safe_dict(cls, data: Mapping[str, Any]) -> "IQMHardware":
        validate_backend_execution_mode(
            "iqm_hardware",
            data.get("execution_mode", ExecutionMode.HARDWARE.value),
        )
        return cls(data["device"], data.get("use_metrics", False))


@dataclass(frozen=True)
class PiastQHardware:
    mode: str = "managed"
    owner: str | None = None
    env_path: Path | None = None
    execution_mode: ClassVar[ExecutionMode] = ExecutionMode.HARDWARE

    def __post_init__(self) -> None:
        if self.mode != "managed":
            raise ExperimentValidationError(
                "PiastQHardware supports only managed mode; direct access requires a separate environment"
            )
        if self.owner is not None:
            _safe_text(self.owner, "owner")
        object.__setattr__(self, "env_path", _safe_optional_path(self.env_path, "env_path"))

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "kind": "piastq_hardware",
            "mode": self.mode,
            "owner": self.owner,
            "execution_mode": self.execution_mode.value,
        }

    @classmethod
    def from_safe_dict(cls, data: Mapping[str, Any]) -> "PiastQHardware":
        validate_backend_execution_mode(
            "piastq_hardware",
            data.get("execution_mode", ExecutionMode.HARDWARE.value),
        )
        return cls(data.get("mode", "managed"), data.get("owner"))


@dataclass(frozen=True)
class CustomBackend:
    instance: Any = field(repr=False, compare=False)
    identity: str = "custom"
    supports_resume: bool = False
    execution_mode: ExecutionMode = field(kw_only=True)

    def __post_init__(self) -> None:
        _safe_text(self.identity, "identity")
        _require_bool(self.supports_resume, "supports_resume")
        if not isinstance(self.execution_mode, ExecutionMode):
            raise ExperimentValidationError(
                "execution_mode must be ExecutionMode"
            ) from None

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "kind": "custom",
            "identity": self.identity,
            "supports_resume": self.supports_resume,
            "execution_mode": self.execution_mode.value,
        }

    @classmethod
    def from_safe_dict(cls, data: Mapping[str, Any], *, instance: Any = None) -> "CustomBackend":
        if instance is None:
            raise ExperimentValidationError("custom backend reconstruction requires instance injection")
        execution_mode = validate_backend_execution_mode(
            "custom",
            data.get("execution_mode"),
        )
        return cls(
            instance=instance,
            identity=data["identity"],
            supports_resume=data.get("supports_resume", False),
            execution_mode=execution_mode,
        )


@dataclass(frozen=True)
class NoisySimulator:
    source: Any = field(default=None, repr=False, compare=False)
    noise_model: Any = field(default=None, repr=False, compare=False)
    target_backend: Any = field(default=None, repr=False, compare=False)
    identity: str | None = None
    execution_mode: ClassVar[ExecutionMode] = ExecutionMode.NOISY_SIMULATOR

    def __post_init__(self) -> None:
        source_mode = self.source is not None and self.noise_model is None and self.target_backend is None
        model_mode = self.source is None and self.noise_model is not None and self.target_backend is not None
        if not (source_mode or model_mode):
            raise ExperimentValidationError("provide exactly either source or noise_model with target_backend")
        if self.identity is not None:
            _safe_text(self.identity, "identity")

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "kind": "noisy_simulator",
            "identity": self.identity,
            "source_mode": self.source is not None,
            "execution_mode": self.execution_mode.value,
        }

    @classmethod
    def from_safe_dict(cls, data: Mapping[str, Any], **injected: Any) -> "NoisySimulator":
        if not injected:
            raise ExperimentValidationError("noisy simulator reconstruction requires object injection")
        validate_backend_execution_mode(
            "noisy_simulator",
            data.get("execution_mode", ExecutionMode.NOISY_SIMULATOR.value),
        )
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
    circuit_twirling: bool = False
    twirling_instances: int = 20
    twirling_seed: int | None = None

    def __post_init__(self) -> None:
        factors = tuple(self.zne_factors)
        _require_bool(self.readout, "readout")
        _require_bool(self.zne, "zne")
        _require_bool(self.force_recalibration, "force_recalibration")
        _require_bool(self.circuit_twirling, "circuit_twirling")
        object.__setattr__(self, "zne_factors", factors)
        if self.zne and (not factors or 1 not in factors or len(set(factors)) != len(factors) or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 or item % 2 == 0 for item in factors)):
            raise ExperimentValidationError("zne_factors must be unique positive odd factors including 1")
        if self.zne_model != "linear":
            raise ExperimentValidationError("zne_model must be linear")
        if not isinstance(self.readout_max_age_hours, (int, float)) or isinstance(self.readout_max_age_hours, bool) or not math.isfinite(self.readout_max_age_hours) or self.readout_max_age_hours <= 0:
            raise ExperimentValidationError("readout_max_age_hours must be positive")
        if type(self.twirling_instances) is not int or self.twirling_instances <= 0:
            raise ExperimentValidationError(
                "twirling_instances must be a positive integer"
            )
        if self.twirling_seed is not None and (
            type(self.twirling_seed) is not int or self.twirling_seed < 0
        ):
            raise ExperimentValidationError(
                "twirling_seed must be a non-negative integer or None"
            )

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "readout": self.readout,
            "zne": self.zne,
            "zne_factors": list(self.zne_factors),
            "zne_model": self.zne_model,
            "readout_max_age_hours": self.readout_max_age_hours,
            "force_recalibration": self.force_recalibration,
            "circuit_twirling": self.circuit_twirling,
            "twirling_instances": self.twirling_instances,
            "twirling_seed": self.twirling_seed,
        }

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
class IQMQubitSelectorConfig:
    top_k: int = 10
    num_trials: int = 2000
    cost_function: str = "cz"
    readout_mode: str = "none"
    remove_qubits: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if type(self.top_k) is not int or self.top_k <= 0:
            raise ExperimentValidationError("top_k must be a positive integer")
        if type(self.num_trials) is not int or self.num_trials <= 0:
            raise ExperimentValidationError("num_trials must be a positive integer")
        if not isinstance(self.cost_function, str) or self.cost_function not in {
            "cz",
            "clifford",
        }:
            raise ExperimentValidationError(
                "cost_function must be cz or clifford"
            )
        if not isinstance(self.readout_mode, str) or self.readout_mode not in {
            "none",
            "fidelity",
            "qndness",
        }:
            raise ExperimentValidationError(
                "readout_mode must be none, fidelity, or qndness"
            )
        if not isinstance(self.remove_qubits, Sequence) or isinstance(
            self.remove_qubits, (str, bytes)
        ):
            raise ExperimentValidationError(
                "remove_qubits must be a sequence of distinct non-negative integers"
            )
        remove_qubits = tuple(self.remove_qubits)
        if (
            any(type(qubit) is not int or qubit < 0 for qubit in remove_qubits)
            or len(set(remove_qubits)) != len(remove_qubits)
        ):
            raise ExperimentValidationError(
                "remove_qubits must be distinct non-negative integers"
            )
        object.__setattr__(self, "remove_qubits", remove_qubits)

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "top_k": self.top_k,
            "num_trials": self.num_trials,
            "cost_function": self.cost_function,
            "readout_mode": self.readout_mode,
            "remove_qubits": list(self.remove_qubits),
        }

    @classmethod
    def from_safe_dict(cls, data: Mapping[str, Any]) -> "IQMQubitSelectorConfig":
        return cls(**dict(data))


@dataclass(frozen=True)
class WorkloadOptimizationConfig:
    initial_layouts: tuple[tuple[int, ...], ...]
    seed_transpilers: tuple[int, ...] = (0,)
    require_exact_physical_qubit_set: bool = True
    prefer_calibration_metrics: bool = True
    iqm_qubit_selector: IQMQubitSelectorConfig | None = None

    def __post_init__(self) -> None:
        if (
            self.iqm_qubit_selector is not None
            and not isinstance(self.iqm_qubit_selector, IQMQubitSelectorConfig)
        ):
            raise ExperimentValidationError(
                "iqm_qubit_selector must be IQMQubitSelectorConfig or None"
            )
        if not isinstance(self.initial_layouts, Sequence) or isinstance(
            self.initial_layouts, (str, bytes)
        ):
            raise ExperimentValidationError(
                "initial_layouts must be a sequence of equal-width layouts"
            )

        layouts: list[tuple[int, ...]] = []
        for candidate in self.initial_layouts:
            if not isinstance(candidate, Sequence) or isinstance(
                candidate, (str, bytes)
            ):
                raise ExperimentValidationError(
                    "initial_layouts must contain sequences of distinct non-negative integers"
                )
            layout = tuple(candidate)
            if (
                not layout
                or any(type(index) is not int or index < 0 for index in layout)
                or len(set(layout)) != len(layout)
            ):
                raise ExperimentValidationError(
                    "initial_layouts must contain non-empty layouts of distinct non-negative integers"
                )
            layouts.append(layout)

        normalized_layouts = tuple(layouts)
        if not normalized_layouts and self.iqm_qubit_selector is None:
            raise ExperimentValidationError(
                "initial_layouts require at least one layout source"
            )
        if normalized_layouts and (
            any(
                len(layout) != len(normalized_layouts[0])
                for layout in normalized_layouts[1:]
            )
            or len(set(normalized_layouts)) != len(normalized_layouts)
        ):
            raise ExperimentValidationError(
                "initial_layouts must be non-empty, equal-width, and unique"
            )

        if not isinstance(self.seed_transpilers, Sequence) or isinstance(
            self.seed_transpilers, (str, bytes)
        ):
            raise ExperimentValidationError(
                "seed_transpilers must be a non-empty sequence of distinct non-negative integers"
            )
        seeds = tuple(self.seed_transpilers)
        if (
            not seeds
            or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)
        ):
            raise ExperimentValidationError(
                "seed_transpilers must be a non-empty sequence of distinct non-negative integers"
            )

        _require_bool(
            self.require_exact_physical_qubit_set,
            "require_exact_physical_qubit_set",
        )
        _require_bool(self.prefer_calibration_metrics, "prefer_calibration_metrics")
        object.__setattr__(self, "initial_layouts", normalized_layouts)
        object.__setattr__(self, "seed_transpilers", seeds)

    def to_safe_dict(self) -> dict[str, Any]:
        payload = {
            "initial_layouts": [list(layout) for layout in self.initial_layouts],
            "seed_transpilers": list(self.seed_transpilers),
            "require_exact_physical_qubit_set": self.require_exact_physical_qubit_set,
            "prefer_calibration_metrics": self.prefer_calibration_metrics,
        }
        if self.iqm_qubit_selector is not None:
            payload["iqm_qubit_selector"] = self.iqm_qubit_selector.to_safe_dict()
        return payload

    @classmethod
    def from_safe_dict(cls, data: Mapping[str, Any]) -> "WorkloadOptimizationConfig":
        selector = None
        selector_payload = data.get("iqm_qubit_selector")
        if selector_payload is not None:
            if not isinstance(selector_payload, Mapping):
                raise ExperimentValidationError(
                    "iqm_qubit_selector must be a mapping"
                )
            selector = IQMQubitSelectorConfig.from_safe_dict(selector_payload)
        return cls(
            initial_layouts=data["initial_layouts"],
            seed_transpilers=data.get("seed_transpilers", (0,)),
            require_exact_physical_qubit_set=data.get(
                "require_exact_physical_qubit_set", True
            ),
            prefer_calibration_metrics=data.get("prefer_calibration_metrics", True),
            iqm_qubit_selector=selector,
        )


@dataclass(frozen=True)
class TranspilationConfig:
    optimization_level: int = 3
    seed_transpiler: int | None = None
    layout_method: str | None = None
    routing_method: str | None = None
    scheduling_method: str | None = None
    initial_layout: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.optimization_level, bool) or not isinstance(self.optimization_level, int) or self.optimization_level not in range(4):
            raise ExperimentValidationError("optimization_level must be an integer from 0 to 3")
        if self.initial_layout is None:
            return
        if not isinstance(self.initial_layout, Sequence) or isinstance(
            self.initial_layout, (str, bytes)
        ):
            raise ExperimentValidationError(
                "initial_layout must be a non-empty sequence of distinct non-negative integers"
            )
        layout = tuple(self.initial_layout)
        if (
            not layout
            or any(type(index) is not int or index < 0 for index in layout)
            or len(set(layout)) != len(layout)
        ):
            raise ExperimentValidationError(
                "initial_layout must be a non-empty sequence of distinct non-negative integers"
            )
        object.__setattr__(self, "initial_layout", layout)

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "optimization_level": self.optimization_level,
            "seed_transpiler": self.seed_transpiler,
            "layout_method": self.layout_method,
            "routing_method": self.routing_method,
            "scheduling_method": self.scheduling_method,
            "initial_layout": list(self.initial_layout) if self.initial_layout else None,
        }

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
    workload_optimization: WorkloadOptimizationConfig | None = field(
        default=None,
        kw_only=True,
    )
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
        workload_optimization: WorkloadOptimizationConfig | None = None,
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
        object.__setattr__(self, "workload_optimization", workload_optimization)
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
        if self.workload_optimization is not None and not isinstance(
            self.workload_optimization, WorkloadOptimizationConfig
        ):
            raise ExperimentValidationError(
                "workload_optimization must be WorkloadOptimizationConfig or None"
            )
        object.__setattr__(self, "output_root", _safe_optional_path(self.output_root, "output_root"))
        if self.output_root is None:
            raise ExperimentValidationError("output_root is required")
        object.__setattr__(self, "tags", _safe_tags(self.tags))

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "state": self.state, "basis": self.basis.to_safe_dict(), "backend": self.backend.to_safe_dict(),
            "shots": self.shots, "mitigation": self.mitigation.to_safe_dict(), "uncertainty": self.uncertainty.to_safe_dict(),
            "transpilation": self.transpilation.to_safe_dict(),
            "workload_optimization": (
                self.workload_optimization.to_safe_dict()
                if self.workload_optimization is not None
                else None
            ),
            "retry": self.retry.to_safe_dict(),
            "output_root": str(self.output_root), "tags": dict(self.tags),
        }

    @classmethod
    def from_safe_dict(cls, data: Mapping[str, Any]) -> "ExperimentSpec":
        data = _normalize_experiment_spec_dict(data)
        return cls(
            state=data["state"], basis=_basis_from_safe_dict(data["basis"]), backend=_backend_from_safe_dict(data["backend"]),
            shots=data.get("shots", 20480), mitigation=MitigationConfig.from_safe_dict(data.get("mitigation", {})),
            uncertainty=BootstrapConfig.from_safe_dict(data.get("uncertainty", {})),
            transpilation=TranspilationConfig.from_safe_dict(data.get("transpilation", {})),
            workload_optimization=(
                WorkloadOptimizationConfig.from_safe_dict(data["workload_optimization"])
                if data.get("workload_optimization") is not None
                else None
            ),
            retry=RetryConfig.from_safe_dict(data.get("retry", {})),
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
class ScalarEstimate:
    estimate: float
    standard_error: float
    confidence_interval: ConfidenceInterval

    def __post_init__(self) -> None:
        normalized_values: dict[str, float] = {}
        for field_name in ("estimate", "standard_error"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ExperimentValidationError(
                    f"{field_name} must be a finite real value"
                )
            try:
                normalized_value = float(value)
            except (OverflowError, TypeError, ValueError):
                raise ExperimentValidationError(
                    f"{field_name} must be a finite real value"
                ) from None
            if not math.isfinite(normalized_value):
                raise ExperimentValidationError(
                    f"{field_name} must be a finite real value"
                )
            normalized_values[field_name] = normalized_value
        if normalized_values["standard_error"] < 0:
            raise ExperimentValidationError("standard_error must be non-negative")
        if not isinstance(self.confidence_interval, ConfidenceInterval):
            raise ExperimentValidationError(
                "confidence_interval must be ConfidenceInterval"
            )
        object.__setattr__(self, "estimate", normalized_values["estimate"])
        object.__setattr__(
            self, "standard_error", normalized_values["standard_error"]
        )

    def to_safe_dict(self) -> dict[str, Any]:
        return {
            "estimate": self.estimate,
            "standard_error": self.standard_error,
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
