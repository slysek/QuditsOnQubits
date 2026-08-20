"""Execution provenance shared by experiment specifications and manifests."""

from __future__ import annotations

from enum import Enum
from typing import Any

from .errors import ExperimentValidationError


class ExecutionMode(str, Enum):
    IDEAL_SIMULATOR = "ideal_simulator"
    NOISY_SIMULATOR = "noisy_simulator"
    HARDWARE = "hardware"


_FIXED_MODES = {
    "aer_ideal": ExecutionMode.IDEAL_SIMULATOR,
    "noisy_simulator": ExecutionMode.NOISY_SIMULATOR,
    "iqm_hardware": ExecutionMode.HARDWARE,
    "piastq_hardware": ExecutionMode.HARDWARE,
}
_IDENTITY_KINDS = {
    "aer_ideal": "aer_ideal",
    "noisy_simulator": "noisy",
    "iqm_hardware": "iqm",
    "piastq_hardware": "piastq",
    "custom": "custom",
}


def fixed_execution_mode(backend_kind: str) -> ExecutionMode | None:
    if backend_kind == "custom":
        return None
    try:
        return _FIXED_MODES[backend_kind]
    except KeyError:
        raise ExperimentValidationError("backend kind is unsupported") from None


def validate_backend_execution_mode(
    backend_kind: str,
    value: Any,
) -> ExecutionMode:
    try:
        mode = value if isinstance(value, ExecutionMode) else ExecutionMode(value)
    except (TypeError, ValueError):
        raise ExperimentValidationError("execution_mode is invalid") from None
    fixed = fixed_execution_mode(backend_kind)
    if fixed is not None and mode is not fixed:
        raise ExperimentValidationError(
            "execution_mode does not match backend kind"
        ) from None
    return mode


def expected_backend_identity_kind(backend_kind: str) -> str:
    try:
        return _IDENTITY_KINDS[backend_kind]
    except KeyError:
        raise ExperimentValidationError("backend kind is unsupported") from None


__all__ = ["ExecutionMode"]
