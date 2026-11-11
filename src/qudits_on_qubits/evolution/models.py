"""Inputs and serializable results for dense logical-gate evaluation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from numbers import Integral, Real
from pathlib import Path

import numpy as np

from qudits_on_qubits.benchmarks.direct_basis.math_utils import validate_unitary


def _integer_at_least(value: int, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _finite_matrix(value: np.ndarray, name: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.complex128)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a numeric matrix") from exc
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return matrix


@dataclass(frozen=True, eq=False)
class EvaluationTask:
    """Set the target and the limits for eps and L.

    Qudit 0 is the least significant qudit. The limits use normalized metrics.
    """

    d: int
    n: int
    target: np.ndarray = field(repr=False)
    logical_error_tolerance: float = 1e-10
    leakage_tolerance: float = 1e-10
    allowed_gates: tuple[str, ...] = ("u", "u3", "cz", "cx")

    def __post_init__(self) -> None:
        d = _integer_at_least(self.d, "d", 2)
        n = _integer_at_least(self.n, "n", 1)
        for name in ("logical_error_tolerance", "leakage_tolerance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} must be a finite nonnegative number")
            try:
                value = float(value)
            except OverflowError as exc:
                raise ValueError(f"{name} must be a finite nonnegative number") from exc
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative number")
            object.__setattr__(self, name, value)
        if not isinstance(self.allowed_gates, (tuple, list)) or not self.allowed_gates:
            raise ValueError("allowed_gates must be a nonempty list of gate names")
        if any(not isinstance(gate, str) or gate not in {"u", "u3", "cz", "cx"} for gate in self.allowed_gates):
            raise ValueError("allowed_gates must contain only u, u3, cz or cx")
        object.__setattr__(self, "allowed_gates", tuple(self.allowed_gates))
        target = validate_unitary(_finite_matrix(self.target, "target"), d**n, "target")
        # Keep the validated task reusable even if the caller edits the input array.
        target = np.frombuffer(target.tobytes(), dtype=target.dtype).reshape(target.shape)
        object.__setattr__(self, "d", d)
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "target", target)

    @property
    def num_qubits(self) -> int:
        return self.n * (self.d - 1).bit_length()


@dataclass(frozen=True)
class EvaluationResult:
    """Code-space fidelity, final leakage probability, aligned RMS error and gate costs."""

    d: int
    n: int
    num_qubits: int
    F: float
    L: float
    eps: float
    error_norm: float
    is_valid: bool
    evaluation_seconds: float
    cz_count: int
    cx_count: int
    two_qubit_gate_count: int
    depth: int
    two_qubit_depth: int
    one_qubit_gate_count: int

    def to_json(self) -> str:
        """Serialize all fields with stable key order and full float precision."""
        return json.dumps(asdict(self), sort_keys=True, indent=2, allow_nan=False) + "\n"

    def write_json(self, path: str | Path) -> None:
        """Write the UTF-8 report with the same line endings on every platform."""
        Path(path).write_bytes(self.to_json().encode("utf-8"))
