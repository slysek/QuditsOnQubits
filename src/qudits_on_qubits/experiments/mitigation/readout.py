"""Readout calibration evidence and M3 correction helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
import math
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any

import numpy as np
from qiskit import QuantumCircuit

from ..errors import ExperimentValidationError, JobResultError, OptionalDependencyError
from .base import ReadoutMitigationStrategy


_CREDENTIAL_MARKERS = ("token=", "api_key=", "password=", "secret=")
_QUASI_TOTAL_TOLERANCE = 1e-6


def _safe_identity(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ExperimentValidationError(f"{field_name} must be a non-empty string")
    if any(marker in value.lower() for marker in _CREDENTIAL_MARKERS):
        raise ExperimentValidationError(f"{field_name} must not contain credential material")
    return value


def _validate_qubit_mapping(qubits: Sequence[int]) -> tuple[int, ...]:
    try:
        normalized = tuple(qubits)
    except TypeError as error:
        raise ExperimentValidationError("qubit mapping must be a sequence") from error
    if not normalized:
        raise ExperimentValidationError("qubit mapping must not be empty")
    if any(type(qubit) is not int or qubit < 0 for qubit in normalized):
        raise ExperimentValidationError("qubit mapping must contain non-negative integers")
    if len(set(normalized)) != len(normalized):
        raise ExperimentValidationError("qubit mapping must contain unique qubits")
    return normalized


def _validate_shots(shots: object) -> int:
    if type(shots) is not int or shots <= 0:
        raise ExperimentValidationError("shots must be a positive integer")
    return shots


def _validated_calibration_counts(
    counts: object, *, shots: int | None = None
) -> MappingProxyType:
    if not isinstance(counts, Mapping):
        raise ExperimentValidationError("calibration counts must be mappings")
    normalized = {"0": 0, "1": 0}
    for outcome, count in counts.items():
        if outcome not in {"0", "1"}:
            raise ExperimentValidationError("calibration counts outcomes must be '0' or '1'")
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
            raise ExperimentValidationError("calibration counts must be non-negative integers")
        normalized[outcome] = int(count)
    total = sum(normalized.values())
    if total <= 0:
        raise ExperimentValidationError("calibration counts total must be positive")
    if shots is not None and total != shots:
        raise ExperimentValidationError("calibration counts total must equal shots")
    return MappingProxyType(normalized)


def build_readout_calibration_circuits(
    qubit_mapping: Sequence[int],
) -> tuple[QuantumCircuit, ...]:
    """Build prepare-0 then prepare-1 evidence circuits in mapping order."""

    qubits = _validate_qubit_mapping(qubit_mapping)
    width = max(qubits) + 1
    circuits: list[QuantumCircuit] = []
    for qubit in qubits:
        for prepared_state in (0, 1):
            circuit = QuantumCircuit(width, 1, name=f"readout_q{qubit}_prep{prepared_state}")
            circuit.metadata = {
                "calibration_kind": "readout_assignment",
                "physical_qubit": qubit,
                "prepared_state": prepared_state,
            }
            if prepared_state:
                circuit.x(qubit)
            circuit.measure(qubit, 0)
            circuits.append(circuit)
    return tuple(circuits)


def assignment_matrices_from_counts(
    qubit_mapping: Sequence[int],
    raw_counts: Sequence[Mapping[str, int]],
    *,
    shots: int | None = None,
) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    """Derive matrices with columns prepared state and rows measured state."""

    qubits = _validate_qubit_mapping(qubit_mapping)
    validated_shots = _validate_shots(shots) if shots is not None else None
    try:
        evidence = tuple(raw_counts)
    except TypeError as error:
        raise ExperimentValidationError("raw calibration counts must be a sequence") from error
    if len(evidence) != 2 * len(qubits):
        raise ExperimentValidationError("raw calibration counts must contain two settings per qubit")

    matrices: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for index in range(len(qubits)):
        prepared_zero = _validated_calibration_counts(evidence[2 * index], shots=validated_shots)
        prepared_one = _validated_calibration_counts(evidence[2 * index + 1], shots=validated_shots)
        total_zero = prepared_zero["0"] + prepared_zero["1"]
        total_one = prepared_one["0"] + prepared_one["1"]
        matrix = (
            (prepared_zero["0"] / total_zero, prepared_one["0"] / total_one),
            (prepared_zero["1"] / total_zero, prepared_one["1"] / total_one),
        )
        if not all(math.isfinite(value) for row in matrix for value in row):
            raise ExperimentValidationError("assignment matrices must contain finite probabilities")
        matrices.append(matrix)
    return tuple(matrices)


def _validated_matrix(
    matrix: object,
) -> tuple[tuple[float, float], tuple[float, float]]:
    try:
        rows = tuple(tuple(row) for row in matrix)  # type: ignore[union-attr]
    except TypeError as error:
        raise ExperimentValidationError("assignment matrix must be 2x2") from error
    if len(rows) != 2 or any(len(row) != 2 for row in rows):
        raise ExperimentValidationError("assignment matrix must be 2x2")
    normalized_rows: list[tuple[float, float]] = []
    for row in rows:
        values: list[float] = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
                raise ExperimentValidationError("assignment matrices must contain finite probabilities")
            number = float(value)
            if not 0.0 <= number <= 1.0:
                raise ExperimentValidationError("assignment matrix probabilities must be between 0 and 1")
            values.append(number)
        normalized_rows.append((values[0], values[1]))
    normalized = (normalized_rows[0], normalized_rows[1])
    for column in range(2):
        if not math.isclose(normalized[0][column] + normalized[1][column], 1.0, abs_tol=1e-12):
            raise ExperimentValidationError("assignment matrix probability columns must sum to 1")
    return normalized


@dataclass(frozen=True)
class ReadoutCalibration:
    """Immutable raw calibration evidence plus derived assignment matrices."""

    backend_identity: str
    calibration_id: str
    qubit_mapping: tuple[int, ...]
    timestamp: datetime
    shots: int
    raw_counts: tuple[Mapping[str, int], ...]
    assignment_matrices: tuple[tuple[tuple[float, float], tuple[float, float]], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "backend_identity", _safe_identity(self.backend_identity, "backend_identity"))
        object.__setattr__(self, "calibration_id", _safe_identity(self.calibration_id, "calibration_id"))
        qubits = _validate_qubit_mapping(self.qubit_mapping)
        object.__setattr__(self, "qubit_mapping", qubits)
        shots = _validate_shots(self.shots)
        object.__setattr__(self, "shots", shots)
        if not isinstance(self.timestamp, datetime) or self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ExperimentValidationError("timestamp must be a timezone-aware datetime")
        object.__setattr__(self, "timestamp", self.timestamp.astimezone(timezone.utc))

        try:
            raw_source = tuple(self.raw_counts)
        except TypeError as error:
            raise ExperimentValidationError("raw calibration counts must be a sequence") from error
        raw = tuple(_validated_calibration_counts(item, shots=shots) for item in raw_source)
        if len(raw) != 2 * len(qubits):
            raise ExperimentValidationError("raw calibration counts must contain two settings per qubit")
        object.__setattr__(self, "raw_counts", raw)

        try:
            matrices = tuple(_validated_matrix(matrix) for matrix in self.assignment_matrices)
        except TypeError as error:
            raise ExperimentValidationError("assignment matrices must be a sequence") from error
        if len(matrices) != len(qubits):
            raise ExperimentValidationError("assignment matrices must contain one matrix per qubit")
        derived = assignment_matrices_from_counts(qubits, raw, shots=shots)
        if any(
            not math.isclose(matrices[index][row][column], derived[index][row][column], abs_tol=1e-12)
            for index in range(len(qubits))
            for row in range(2)
            for column in range(2)
        ):
            raise ExperimentValidationError("assignment matrices do not match raw calibration counts")
        object.__setattr__(self, "assignment_matrices", matrices)

    def to_safe_dict(self) -> dict[str, Any]:
        """Return JSON-compatible calibration evidence for persistence."""

        return {
            "backend_identity": self.backend_identity,
            "calibration_id": self.calibration_id,
            "qubit_mapping": list(self.qubit_mapping),
            "timestamp": self.timestamp.isoformat(),
            "shots": self.shots,
            "raw_counts": [dict(counts) for counts in self.raw_counts],
            "assignment_matrices": [[list(row) for row in matrix] for matrix in self.assignment_matrices],
        }


def build_m3_mitigation(
    calibration: ReadoutCalibration,
    *,
    mitigation: ReadoutMitigationStrategy | None = None,
) -> ReadoutMitigationStrategy:
    """Create or inject M3, then load physical-qubit assignment matrices."""

    if not isinstance(calibration, ReadoutCalibration):
        raise ExperimentValidationError("calibration must be a ReadoutCalibration")
    if mitigation is None:
        try:
            mthree = importlib.import_module("mthree")
        except ImportError as error:
            raise OptionalDependencyError(
                "readout mitigation requires mthree; install with pip install -e .[mitigation]"
            ) from error
        mitigation = mthree.M3Mitigation()
    loader = getattr(mitigation, "cals_from_matrices", None)
    if not callable(loader):
        raise ExperimentValidationError("mitigation object must provide cals_from_matrices")

    matrices: list[np.ndarray | None] = [None] * (max(calibration.qubit_mapping) + 1)
    for qubit, matrix in zip(calibration.qubit_mapping, calibration.assignment_matrices, strict=True):
        matrices[qubit] = np.array(matrix, dtype=float, copy=True)
    loader(matrices)
    return mitigation


def _validate_bitstring(outcome: object, expected_bits: int, field_name: str) -> str:
    if (
        not isinstance(outcome, str)
        or len(outcome) != expected_bits
        or any(bit not in "01" for bit in outcome)
    ):
        raise ExperimentValidationError(
            f"{field_name} must be binary bitstrings of length {expected_bits}"
        )
    return outcome


def _validated_setting_counts(counts: object, expected_bits: int) -> dict[str, int]:
    if not isinstance(counts, Mapping):
        raise ExperimentValidationError("setting counts must be mappings")
    normalized: dict[str, int] = {}
    for outcome, count in counts.items():
        bitstring = _validate_bitstring(outcome, expected_bits, "setting count outcomes")
        if isinstance(count, bool) or not isinstance(count, Integral) or count < 0:
            raise ExperimentValidationError("setting counts must be non-negative integers")
        normalized[bitstring] = int(count)
    if sum(normalized.values()) <= 0:
        raise ExperimentValidationError("setting counts total must be positive")
    return normalized


def _plain_finite_quasi(output: object, expected_bits: int) -> dict[str, float]:
    if not isinstance(output, Mapping) or not output:
        raise ExperimentValidationError("readout correction must return a non-empty mapping")
    normalized: dict[str, float] = {}
    for outcome, weight in output.items():
        bitstring = _validate_bitstring(outcome, expected_bits, "corrected outcomes")
        if isinstance(weight, bool) or not isinstance(weight, Real) or not math.isfinite(weight):
            raise ExperimentValidationError("corrected weights must be finite real values")
        normalized[bitstring] = float(weight)
    total = sum(normalized.values())
    if not math.isfinite(total) or not math.isclose(
        total,
        1.0,
        rel_tol=_QUASI_TOTAL_TOLERANCE,
        abs_tol=_QUASI_TOTAL_TOLERANCE,
    ):
        raise ExperimentValidationError(
            "corrected weights must sum to 1 within absolute and relative tolerance 1e-6"
        )
    return normalized


def _physical_qubits(
    mapping: Sequence[int] | Mapping[int, int],
) -> tuple[int, ...]:
    if not isinstance(mapping, Mapping):
        return _validate_qubit_mapping(mapping)
    logical_qubits = tuple(mapping)
    if any(type(qubit) is not int or qubit < 0 for qubit in logical_qubits):
        raise ExperimentValidationError("logical qubits must be non-negative integers")
    if set(logical_qubits) != set(range(len(logical_qubits))):
        raise ExperimentValidationError("logical qubits must be contiguous from zero")
    return _validate_qubit_mapping(tuple(mapping[index] for index in range(len(mapping))))


def apply_readout_mitigation(
    counts_by_setting: Mapping[str, Mapping[str, int]],
    *,
    mapping: Sequence[int] | Mapping[int, int] | None = None,
    mapping_by_setting: Mapping[
        str, Sequence[int] | Mapping[int, int]
    ] | None = None,
    mitigation: ReadoutMitigationStrategy,
) -> dict[str, dict[str, float]]:
    """Apply M3 using one common mapping or one physical mapping per setting."""

    if not isinstance(counts_by_setting, Mapping):
        raise ExperimentValidationError("counts_by_setting must be a mapping")
    if (mapping is None) == (mapping_by_setting is None):
        raise ExperimentValidationError(
            "provide exactly one of mapping or mapping_by_setting"
        )

    setting_names = tuple(counts_by_setting)
    if mapping_by_setting is not None:
        if not isinstance(mapping_by_setting, Mapping):
            raise ExperimentValidationError("mapping_by_setting must be a mapping")
        if tuple(mapping_by_setting) != setting_names:
            raise ExperimentValidationError(
                "mapping_by_setting must have the same keys and order as counts_by_setting"
            )
        physical_qubits_by_setting = {
            setting: _physical_qubits(mapping_by_setting[setting])
            for setting in setting_names
        }
    else:
        common_physical_qubits = _physical_qubits(mapping)
        physical_qubits_by_setting = {
            setting: common_physical_qubits for setting in setting_names
        }

    correct = getattr(mitigation, "apply_correction", None)
    if not callable(correct):
        raise ExperimentValidationError("mitigation object must provide apply_correction")

    corrected: dict[str, dict[str, float]] = {}
    for setting, counts in counts_by_setting.items():
        if not isinstance(setting, str) or not setting:
            raise ExperimentValidationError("setting names must be non-empty strings")
        physical_qubits = physical_qubits_by_setting[setting]
        normalized_counts = _validated_setting_counts(counts, len(physical_qubits))
        try:
            output = correct(normalized_counts, physical_qubits)
        except Exception:
            raise JobResultError("readout mitigation correction failed") from None
        corrected[setting] = _plain_finite_quasi(output, len(physical_qubits))
    return corrected


def calibration_cache_is_valid(
    calibration: ReadoutCalibration,
    *,
    backend_identity: str,
    calibration_id: str,
    qubit_mapping: Sequence[int],
    now: datetime,
    max_age_hours: float,
) -> bool:
    """Return whether cached evidence exactly matches identity, mapping, and age."""

    if not isinstance(calibration, ReadoutCalibration):
        raise ExperimentValidationError("calibration must be a ReadoutCalibration")
    expected_backend = _safe_identity(backend_identity, "backend_identity")
    expected_calibration = _safe_identity(calibration_id, "calibration_id")
    expected_qubits = _validate_qubit_mapping(qubit_mapping)
    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise ExperimentValidationError("now must be a timezone-aware datetime")
    if isinstance(max_age_hours, bool) or not isinstance(max_age_hours, Real) or not math.isfinite(max_age_hours) or max_age_hours <= 0:
        raise ExperimentValidationError("max_age_hours must be a positive finite value")
    age_seconds = (now.astimezone(timezone.utc) - calibration.timestamp).total_seconds()
    return (
        calibration.backend_identity == expected_backend
        and calibration.calibration_id == expected_calibration
        and calibration.qubit_mapping == expected_qubits
        and 0.0 <= age_seconds <= float(max_age_hours) * 3600.0
    )
