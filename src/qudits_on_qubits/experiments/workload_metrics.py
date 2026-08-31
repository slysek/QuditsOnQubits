"""Backend-neutral metrics for complete compiled experiment workloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from numbers import Real
from types import MappingProxyType
from typing import Any

from .errors import ExperimentValidationError


@dataclass(frozen=True, slots=True)
class WorkloadMetrics:
    """Immutable per-circuit and aggregate compiled-workload metrics."""

    circuits: tuple[Mapping[str, object], ...]
    aggregate: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.circuits, tuple) or not all(
            isinstance(circuit, Mapping) for circuit in self.circuits
        ):
            raise ExperimentValidationError(
                "circuits must be a tuple of JSON-safe mappings"
            )
        if not isinstance(self.aggregate, Mapping):
            raise ExperimentValidationError("aggregate must be a JSON-safe mapping")
        object.__setattr__(
            self,
            "circuits",
            tuple(_freeze_value(circuit) for circuit in self.circuits),
        )
        object.__setattr__(self, "aggregate", _freeze_value(self.aggregate))

    def to_safe_dict(self) -> dict[str, object]:
        """Return a fresh JSON-safe representation."""
        return {
            "circuits": [_thaw_value(circuit) for circuit in self.circuits],
            "aggregate": _thaw_value(self.aggregate),
        }


def summarize_compiled_workload(
    circuits: Sequence[Any],
    *,
    settings: Sequence[Sequence[str]],
    physical_mappings: Sequence[Sequence[int]],
    requested_physical_qubits: Sequence[int],
    target: Any | None = None,
) -> WorkloadMetrics:
    """Summarize all compiled Bell-measurement circuits in a workload."""
    normalized_circuits = _as_sequence(circuits, "circuits")
    raw_settings = _as_sequence(settings, "settings")
    raw_mappings = _as_sequence(physical_mappings, "physical_mappings")
    requested = _validate_qubit_sequence(
        requested_physical_qubits,
        "requested_physical_qubits",
    )
    normalized_settings = tuple(
        _validate_setting(setting) for setting in raw_settings
    )
    normalized_mappings = tuple(
        _validate_qubit_sequence(mapping, "physical_mappings")
        for mapping in raw_mappings
    )
    if (
        not normalized_circuits
        or len(normalized_circuits) != len(normalized_settings)
        or len(normalized_circuits) != len(normalized_mappings)
    ):
        raise ExperimentValidationError(
            "circuits, settings, and physical_mappings must have the same non-zero length"
        )

    per_circuit: list[dict[str, object]] = []
    for index, (circuit, setting, mapping) in enumerate(
        zip(
            normalized_circuits,
            normalized_settings,
            normalized_mappings,
            strict=True,
        )
    ):
        instructions = tuple(circuit.data)
        executed_instructions = tuple(
            instruction
            for instruction in instructions
            if not _instruction_is_directive(instruction)
        )
        operation_counts = {
            name: count for name, count in sorted(circuit.count_ops().items())
        }
        instruction_error_cost, instruction_duration = _calibration_totals(
            circuit,
            executed_instructions,
            target,
        )
        per_circuit.append(
            {
                "circuit_index": index,
                "setting": tuple(setting),
                "depth": circuit.depth(),
                "size": circuit.size(),
                "operation_counts": operation_counts,
                "two_qubit_gate_count": sum(
                    len(instruction.qubits) == 2
                    for instruction in executed_instructions
                ),
                "native_cz_count": sum(
                    instruction.operation.name == "cz"
                    for instruction in executed_instructions
                ),
                "physical_qubit_mapping": tuple(mapping),
                "instruction_error_cost": instruction_error_cost,
                "instruction_duration": instruction_duration,
            }
        )

    depths = [item["depth"] for item in per_circuit]
    sizes = [item["size"] for item in per_circuit]
    two_qubit_counts = [item["two_qubit_gate_count"] for item in per_circuit]
    native_cz_counts = [item["native_cz_count"] for item in per_circuit]
    requested_set = set(requested)
    aggregate = {
        "circuit_count": len(per_circuit),
        "maximum_depth": max(depths),
        "total_depth": sum(depths),
        "maximum_two_qubit_gate_count": max(two_qubit_counts),
        "total_two_qubit_gate_count": sum(two_qubit_counts),
        "maximum_native_cz_count": max(native_cz_counts),
        "total_native_cz_count": sum(native_cz_counts),
        "maximum_size": max(sizes),
        "total_size": sum(sizes),
        "physical_qubit_union": tuple(
            sorted({qubit for mapping in normalized_mappings for qubit in mapping})
        ),
        "uses_exact_physical_qubit_set": all(
            set(mapping) == requested_set for mapping in normalized_mappings
        ),
        "total_instruction_error_cost": _complete_total(
            per_circuit,
            "instruction_error_cost",
        ),
        "total_instruction_duration": _complete_total(
            per_circuit,
            "instruction_duration",
        ),
    }
    return WorkloadMetrics(circuits=tuple(per_circuit), aggregate=aggregate)


def choose_workload_ranking_basis(
    candidates: Sequence[WorkloadMetrics],
    *,
    prefer_calibration: bool,
) -> tuple[bool, bool]:
    """Choose calibrated ranking only when every candidate supports both totals."""
    if type(prefer_calibration) is not bool:
        raise ExperimentValidationError("prefer_calibration must be a boolean")
    normalized_candidates = _as_sequence(candidates, "candidates")
    if not normalized_candidates or not all(
        isinstance(candidate, WorkloadMetrics)
        for candidate in normalized_candidates
    ):
        raise ExperimentValidationError(
            "candidates must be a non-empty sequence of WorkloadMetrics"
        )
    if not prefer_calibration:
        return False, False
    calibration_available = all(
        _aggregate_nonnegative_finite_real(
            candidate,
            "total_instruction_error_cost",
        )
        is not None
        and _aggregate_nonnegative_finite_real(
            candidate,
            "total_instruction_duration",
        )
        is not None
        for candidate in normalized_candidates
    )
    return (True, True) if calibration_available else (False, False)


def workload_rank_key(
    metrics: WorkloadMetrics,
    *,
    use_error: bool,
    use_duration: bool,
    seed: int,
    layout: Sequence[int],
) -> tuple[object, ...]:
    """Build the deterministic rank key for one compiled workload candidate."""
    if not isinstance(metrics, WorkloadMetrics):
        raise ExperimentValidationError("metrics must be WorkloadMetrics")
    if (
        type(use_error) is not bool
        or type(use_duration) is not bool
        or use_error != use_duration
    ):
        raise ExperimentValidationError(
            "use_error and use_duration must be equal boolean values"
        )
    if type(seed) is not int or seed < 0:
        raise ExperimentValidationError("seed must be a non-negative integer")
    normalized_layout = _validate_qubit_sequence(layout, "layout")

    structural = (
        _aggregate_nonnegative_int(metrics, "maximum_two_qubit_gate_count"),
        _aggregate_nonnegative_int(metrics, "total_two_qubit_gate_count"),
        _aggregate_nonnegative_int(metrics, "maximum_depth"),
        _aggregate_nonnegative_int(metrics, "total_depth"),
    )
    suffix = (*structural, seed, normalized_layout)
    if not use_error:
        return suffix

    error = _aggregate_nonnegative_finite_real(
        metrics,
        "total_instruction_error_cost",
    )
    duration = _aggregate_nonnegative_finite_real(
        metrics,
        "total_instruction_duration",
    )
    if error is None or duration is None:
        raise ExperimentValidationError(
            "requested calibration ranking metrics are unavailable"
        )
    return error, duration, *suffix


def _as_sequence(value: Any, field_name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ExperimentValidationError(f"{field_name} must be a non-string sequence")
    return tuple(value)


def _validate_setting(value: Any) -> tuple[str, ...]:
    setting = _as_sequence(value, "settings entries")
    if not setting or any(
        not isinstance(label, str) or not label for label in setting
    ):
        raise ExperimentValidationError(
            "settings entries must be non-empty and contain only non-empty string labels"
        )
    return tuple(str(label) for label in setting)


def _validate_qubit_sequence(value: Any, field_name: str) -> tuple[int, ...]:
    qubits = _as_sequence(value, field_name)
    if not qubits:
        raise ExperimentValidationError(f"{field_name} must be non-empty")
    if any(type(qubit) is not int or qubit < 0 for qubit in qubits):
        raise ExperimentValidationError(
            f"{field_name} must contain only non-negative integers"
        )
    if len(set(qubits)) != len(qubits):
        raise ExperimentValidationError(f"{field_name} must not contain duplicates")
    return qubits


def _instruction_is_directive(instruction: Any) -> bool:
    try:
        return bool(getattr(instruction.operation, "_directive", False))
    except (KeyboardInterrupt, SystemExit, MemoryError):
        raise
    except Exception:
        raise ExperimentValidationError(
            "could not inspect circuit instruction directive flag"
        ) from None


def _calibration_totals(
    circuit: Any,
    instructions: Sequence[Any],
    target: Any | None,
) -> tuple[float | None, float | None]:
    if target is None:
        return None, None

    error_costs: list[float] = []
    durations: list[float] = []
    error_available = True
    duration_available = True
    for instruction in instructions:
        try:
            qargs = tuple(
                circuit.find_bit(qubit).index for qubit in instruction.qubits
            )
            properties = target[instruction.operation.name][qargs]
        except MemoryError:
            raise
        except Exception:
            error_available = False
            duration_available = False
            continue

        if error_available:
            try:
                error = properties.error
            except MemoryError:
                raise
            except Exception:
                error_available = False
            else:
                normalized_error = _finite_real(error)
                if (
                    normalized_error is None
                    or normalized_error < 0.0
                    or normalized_error >= 1.0
                ):
                    error_available = False
                else:
                    error_costs.append(-math.log1p(-normalized_error))

        if duration_available:
            try:
                duration = properties.duration
            except MemoryError:
                raise
            except Exception:
                duration_available = False
            else:
                normalized_duration = _finite_real(duration)
                if normalized_duration is None or normalized_duration < 0.0:
                    duration_available = False
                else:
                    durations.append(normalized_duration)

    return (
        _finite_sum(error_costs) if error_available else None,
        _finite_sum(durations) if duration_available else None,
    )


def _finite_real(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        normalized = float(value)
    except (KeyboardInterrupt, SystemExit, MemoryError):
        raise
    except Exception:
        return None
    return normalized if math.isfinite(normalized) else None


def _finite_sum(values: Sequence[float]) -> float | None:
    try:
        total = math.fsum(values)
    except (OverflowError, ValueError):
        return None
    return total if math.isfinite(total) else None


def _complete_total(
    per_circuit: Sequence[Mapping[str, object]],
    field_name: str,
) -> float | None:
    values = [circuit[field_name] for circuit in per_circuit]
    if any(value is None for value in values):
        return None
    return _finite_sum([float(value) for value in values])


def _aggregate_finite_real(
    metrics: WorkloadMetrics,
    field_name: str,
) -> float | None:
    return _finite_real(metrics.aggregate.get(field_name))


def _aggregate_nonnegative_finite_real(
    metrics: WorkloadMetrics,
    field_name: str,
) -> float | None:
    value = _aggregate_finite_real(metrics, field_name)
    if value is None or value < 0.0:
        return None
    return value


def _aggregate_nonnegative_int(
    metrics: WorkloadMetrics,
    field_name: str,
) -> int:
    value = metrics.aggregate.get(field_name)
    if type(value) is not int or value < 0:
        raise ExperimentValidationError(
            f"metrics aggregate {field_name} must be a non-negative integer"
        )
    return value


def _freeze_value(value: Any, active: set[int] | None = None) -> Any:
    if value is None or type(value) is bool:
        return value
    if isinstance(value, str):
        return str(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        normalized = float(value)
        if math.isfinite(normalized):
            return normalized
        raise ExperimentValidationError("metrics must contain only JSON-safe values")

    if active is None:
        active = set()
    identity = id(value)
    if identity in active:
        raise ExperimentValidationError("metrics must not contain recursive values")
    if isinstance(value, Mapping):
        if not all(type(key) is str for key in value):
            raise ExperimentValidationError(
                "metrics must contain only JSON-safe string mapping keys"
            )
        active.add(identity)
        try:
            return MappingProxyType(
                {
                    key: _freeze_value(item, active)
                    for key, item in value.items()
                }
            )
        finally:
            active.remove(identity)
    if isinstance(value, (list, tuple)):
        active.add(identity)
        try:
            return tuple(_freeze_value(item, active) for item in value)
        finally:
            active.remove(identity)
    raise ExperimentValidationError("metrics must contain only JSON-safe values")


def _thaw_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_value(item) for item in value]
    return value


__all__ = [
    "WorkloadMetrics",
    "choose_workload_ranking_basis",
    "summarize_compiled_workload",
    "workload_rank_key",
]
