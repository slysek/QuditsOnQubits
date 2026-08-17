"""Controlled-Z gate folding and linear zero-noise extrapolation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Sequence
import warnings

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ControlFlowOp
from qiskit.circuit.library import CZGate

from ..errors import ExperimentValidationError


def validate_zne_factors(factors: Iterable[int]) -> tuple[int, ...]:
    """Validate factors and return them sorted for deterministic persistence."""

    try:
        normalized = tuple(factors)
    except TypeError as error:
        raise ExperimentValidationError("ZNE factors must be an iterable") from error
    if not normalized:
        raise ExperimentValidationError("ZNE factors must not be empty")
    if any(type(factor) is not int or factor <= 0 or factor % 2 == 0 for factor in normalized):
        raise ExperimentValidationError("ZNE factors must be positive odd integers")
    if len(set(normalized)) != len(normalized):
        raise ExperimentValidationError("ZNE factors must be unique")
    if 1 not in normalized:
        raise ExperimentValidationError("ZNE factors must include 1")
    return tuple(sorted(normalized))


def _validate_fold_factor(factor: object) -> int:
    if type(factor) is not int or factor <= 0 or factor % 2 == 0:
        raise ExperimentValidationError("CZ fold factor must be a positive odd integer")
    return factor


def _operation_has_condition(operation: Any) -> bool:
    return getattr(operation, "condition", None) is not None or getattr(
        operation, "_condition", None
    ) is not None


def _contains_cz(circuit: QuantumCircuit) -> bool:
    for instruction in circuit.data:
        operation = instruction.operation
        if operation.name == "cz":
            return True
        if isinstance(operation, ControlFlowOp) and any(_contains_cz(block) for block in operation.blocks):
            return True
    return False


def _has_cz_calibration(circuit: QuantumCircuit) -> bool:
    calibrations = getattr(circuit, "calibrations", None)
    if calibrations is None:
        calibrations = getattr(circuit, "_calibrations", None)
    return bool(calibrations and calibrations.get("cz"))


def _validate_fold_safety(circuit: QuantumCircuit) -> None:
    if _has_cz_calibration(circuit):
        raise ExperimentValidationError("calibrated CZ gates cannot be folded safely")
    for instruction in circuit.data:
        operation = instruction.operation
        if isinstance(operation, ControlFlowOp):
            if any(_contains_cz(block) for block in operation.blocks):
                raise ExperimentValidationError("CZ inside control-flow cannot be folded safely")
            continue
        if operation.name != "cz":
            continue
        if not isinstance(operation, CZGate):
            raise ExperimentValidationError("custom CZ instructions cannot be folded safely")
        if _operation_has_condition(operation):
            raise ExperimentValidationError("conditioned CZ gates cannot be folded safely")


def _empty_structural_copy(circuit: QuantumCircuit) -> QuantumCircuit:
    try:
        return circuit.copy_empty_like(vars_mode="alike")
    except TypeError:  # Qiskit versions predating vars_mode keyword.
        return circuit.copy_empty_like()


def fold_cz_batch(
    circuits: Sequence[QuantumCircuit], factor: int
) -> tuple[QuantumCircuit, ...]:
    """Repeat each standard self-inverse CZ ``factor`` times without mutation."""

    fold_factor = _validate_fold_factor(factor)
    try:
        source_circuits = tuple(circuits)
    except TypeError as error:
        raise ExperimentValidationError("circuits must be a sequence") from error
    if any(not isinstance(circuit, QuantumCircuit) for circuit in source_circuits):
        raise ExperimentValidationError("all circuits must be QuantumCircuit instances")

    folded_circuits: list[QuantumCircuit] = []
    for circuit in source_circuits:
        _validate_fold_safety(circuit)
        if fold_factor == 1:
            folded_circuits.append(circuit.copy())
            continue

        folded = _empty_structural_copy(circuit)
        for instruction in circuit.data:
            operation = instruction.operation
            qargs = tuple(folded.qubits[circuit.find_bit(bit).index] for bit in instruction.qubits)
            cargs = tuple(folded.clbits[circuit.find_bit(bit).index] for bit in instruction.clbits)
            repetitions = fold_factor if operation.name == "cz" else 1
            for _ in range(repetitions):
                folded.append(operation.copy(), qargs, cargs, copy=False)
        folded_circuits.append(folded)
    return tuple(folded_circuits)


@dataclass(frozen=True)
class LinearZNEFit:
    """Immutable evidence from separate real and imaginary linear fits."""

    factors: tuple[int, ...]
    values: tuple[complex, ...]
    real_coefficients: tuple[float, float]
    imag_coefficients: tuple[float, float]
    residuals: tuple[complex, ...]

    def to_safe_dict(self) -> dict[str, Any]:
        """Return JSON-compatible fit evidence for artifact persistence."""

        return {
            "factors": list(self.factors),
            "values": [{"real": value.real, "imag": value.imag} for value in self.values],
            "real_coefficients": list(self.real_coefficients),
            "imag_coefficients": list(self.imag_coefficients),
            "residuals": [
                {"real": residual.real, "imag": residual.imag}
                for residual in self.residuals
            ],
        }


def _finite_complex_values(values: Sequence[complex]) -> tuple[complex, ...]:
    try:
        source = tuple(values)
    except TypeError as error:
        raise ExperimentValidationError("ZNE values must be a sequence") from error
    normalized: list[complex] = []
    for value in source:
        if isinstance(value, (bool, np.bool_)):
            raise ExperimentValidationError("ZNE values must be finite complex numbers")
        try:
            item = complex(value)
        except (TypeError, ValueError, OverflowError) as error:
            raise ExperimentValidationError("ZNE values must be finite complex numbers") from error
        if not math.isfinite(item.real) or not math.isfinite(item.imag):
            raise ExperimentValidationError("ZNE values must be finite complex numbers")
        normalized.append(item)
    return tuple(normalized)


def linear_zne_extrapolate(
    factors: Sequence[int], values: Sequence[complex]
) -> tuple[complex, LinearZNEFit]:
    """Fit complex line and return its zero-noise intercept plus fit evidence."""

    original_factors = tuple(factors)
    normalized_factors = validate_zne_factors(original_factors)
    normalized_values = _finite_complex_values(values)
    if len(normalized_factors) != len(normalized_values):
        raise ExperimentValidationError("ZNE factors and values must have matching lengths")
    if len(normalized_factors) < 2:
        raise ExperimentValidationError("linear ZNE requires at least two unique factors")

    value_by_factor = dict(zip(original_factors, normalized_values, strict=True))
    sorted_values = tuple(value_by_factor[factor] for factor in normalized_factors)
    try:
        x = np.asarray(normalized_factors, dtype=float)
    except (OverflowError, ValueError) as error:
        raise ExperimentValidationError("ZNE factors cannot be represented as finite values") from error
    if not np.all(np.isfinite(x)):
        raise ExperimentValidationError("ZNE factors cannot be represented as finite values")
    design = np.column_stack((x, np.ones_like(x)))
    condition = float(np.linalg.cond(design))
    if not math.isfinite(condition) or condition > 1.0 / np.finfo(float).eps:
        raise ExperimentValidationError("linear ZNE fit is ill-conditioned")

    y = np.asarray(sorted_values, dtype=complex)
    rank_warning = getattr(np.exceptions, "RankWarning", Warning)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", rank_warning)
            real_coefficients = np.polyfit(x, y.real, 1)
            imag_coefficients = np.polyfit(x, y.imag, 1)
    except (FloatingPointError, ValueError, np.linalg.LinAlgError, rank_warning) as error:
        raise ExperimentValidationError("linear ZNE fit is ill-conditioned") from error

    fitted = np.polyval(real_coefficients, x) + 1j * np.polyval(imag_coefficients, x)
    residuals = tuple(complex(value) for value in y - fitted)
    real_pair = (float(real_coefficients[0]), float(real_coefficients[1]))
    imag_pair = (float(imag_coefficients[0]), float(imag_coefficients[1]))
    all_numbers = (*real_pair, *imag_pair, *(part for item in residuals for part in (item.real, item.imag)))
    if not all(math.isfinite(value) for value in all_numbers):
        raise ExperimentValidationError("linear ZNE fit produced non-finite values")
    fit = LinearZNEFit(
        factors=normalized_factors,
        values=sorted_values,
        real_coefficients=real_pair,
        imag_coefficients=imag_pair,
        residuals=residuals,
    )
    return complex(real_pair[1], imag_pair[1]), fit
