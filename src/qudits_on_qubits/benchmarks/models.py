"""Immutable encoding inputs for the unified benchmark pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from numbers import Integral, Real
import re
from types import MappingProxyType
from typing import Any

import numpy as np


def validate_encoding(encoding: np.ndarray) -> np.ndarray:
    """Return a finite 4x3 isometry, with absolute orthogonality tolerance 1e-10."""
    try:
        matrix = np.asarray(encoding, dtype=np.complex128)
    except (TypeError, ValueError) as error:
        raise ValueError("encoding must be a finite 4x3 isometry") from error
    if matrix.shape != (4, 3) or not np.isfinite(matrix).all():
        raise ValueError("encoding must be a finite 4x3 isometry")
    if not np.allclose(matrix.conj().T @ matrix, np.eye(3), atol=1e-10, rtol=0.0):
        raise ValueError("encoding must be an isometry: encoding.conj().T @ encoding = I")
    return matrix


def _stable_identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"{name} must be a nonempty stable identifier using letters, digits, '.', '_' or '-'")
    return value


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("parameters must use string mapping keys")
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real) and np.isfinite(value):
        return float(value)
    raise ValueError("parameters must contain only finite JSON-compatible values")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True)
class EncodingCandidate:
    """An identified qutrit isometry and its immutable JSON-compatible provenance.

    Matrix rows are ordered |00>, |01>, |10>, |11>. Columns are logical
    |0>, |1>, |2>. The matrix is copied into an immutable buffer, so callers
    cannot mutate a candidate through their original input array.
    """

    candidate_id: str
    family: str
    encoding: np.ndarray = field(repr=False, compare=False)
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _stable_identifier(self.candidate_id, "candidate_id")
        _stable_identifier(self.family, "family")
        matrix = validate_encoding(self.encoding)
        immutable = np.frombuffer(matrix.tobytes(order="C"), dtype=np.complex128).reshape(4, 3)
        if not isinstance(self.parameters, Mapping):
            raise TypeError("parameters must be a mapping")
        object.__setattr__(self, "encoding", immutable)
        object.__setattr__(self, "parameters", _freeze_json(self.parameters))

    def to_dict(self) -> dict[str, Any]:
        """Return independent JSON metadata; the matrix is persisted separately."""
        return {
            "candidate_id": self.candidate_id,
            "family": self.family,
            "parameters": _thaw_json(self.parameters),
        }
