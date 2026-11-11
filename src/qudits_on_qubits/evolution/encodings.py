"""Isometries from logical qudits to contiguous blocks of physical qubits."""

from collections.abc import Sequence

import numpy as np

from qudits_on_qubits.benchmarks.direct_basis.math_utils import validate_isometry

from .models import _finite_matrix, _integer_at_least


def canonical_encoding(d: int) -> np.ndarray:
    """Map logical level j to computational basis word j on ceil(log2(d)) qubits."""
    d = _integer_at_least(d, "d", 2)
    return np.eye(1 << (d - 1).bit_length(), d, dtype=complex)


def tensor_encoding(encodings: Sequence[np.ndarray], *, d: int | None = None) -> np.ndarray:
    """Return E[n-1] ⊗ ... ⊗ E[0]. E[0] acts on the lowest numbered qubits."""
    if len(encodings) == 0:
        raise ValueError("at least one encoding is required")
    if d is None:
        first = np.asarray(encodings[0])
        if first.ndim != 2:
            raise ValueError("encoding[0] must be a matrix")
        d = first.shape[1]
    d = _integer_at_least(d, "d", 2)
    rows = 1 << (d - 1).bit_length()
    product = np.ones((1, 1), dtype=complex)
    for i, encoding in enumerate(encodings):
        name = f"encoding[{i}]"
        local = validate_isometry(_finite_matrix(encoding, name), rows, d, name)
        product = np.kron(local, product)
    return product
