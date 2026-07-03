from __future__ import annotations

import math
from functools import reduce
from typing import Iterable

import numpy as np
from qiskit.quantum_info import Statevector


def default_qutrit_encoding() -> np.ndarray:
    """Return E_Z: |0>, |1>, |2> -> |00>, |01>, |10>."""
    return np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ],
        dtype=complex,
    )


def validate_isometry(E: np.ndarray, atol: float = 1e-10) -> bool:
    matrix = np.asarray(E, dtype=complex)
    if matrix.shape != (4, 3):
        return False
    return np.allclose(matrix.conj().T @ matrix, np.eye(3), atol=atol)


def _require_isometry(E: np.ndarray) -> np.ndarray:
    matrix = np.asarray(E, dtype=complex)
    if not validate_isometry(matrix):
        raise ValueError("E must have shape (4, 3) and satisfy E^dagger E = I_3")
    return matrix


def projector_E(E: np.ndarray) -> np.ndarray:
    matrix = _require_isometry(E)
    return matrix @ matrix.conj().T


def embed_operator_E(E: np.ndarray, O: np.ndarray, full: bool = True) -> np.ndarray:
    """Embed a qutrit operator into one two-qubit block.

    The full form acts as O on the code space and as identity on the orthogonal
    complement, so unitary qutrit observables remain unitary 4x4 observables.
    """
    matrix = _require_isometry(E)
    operator = np.asarray(O, dtype=complex)
    if operator.shape != (3, 3):
        raise ValueError("O must be a 3x3 qutrit operator")
    physical = matrix @ operator @ matrix.conj().T
    if not full:
        return physical
    return physical + np.eye(4, dtype=complex) - matrix @ matrix.conj().T


def embed_projector_E(E: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Embed a qutrit outcome projector without adding I_perp."""
    matrix = _require_isometry(E)
    projector = np.asarray(P, dtype=complex)
    if projector.shape != (3, 3):
        raise ValueError("P must be a 3x3 qutrit projector")
    return matrix @ projector @ matrix.conj().T


def kron_all(matrices: Iterable[np.ndarray]) -> np.ndarray:
    items = list(matrices)
    if not items:
        raise ValueError("kron_all needs at least one matrix")
    return reduce(np.kron, items)


def kron_power(matrix: np.ndarray, count: int) -> np.ndarray:
    if count < 1:
        raise ValueError("count must be positive")
    return kron_all([matrix] * count)


def encode_qutrit_state(state: np.ndarray, E: np.ndarray, num_qutrits: int) -> np.ndarray:
    """Apply E^tensor N to a qutrit state vector."""
    matrix = _require_isometry(E)
    vector = np.asarray(state, dtype=complex).reshape(-1)
    expected = 3**num_qutrits
    if vector.size != expected:
        raise ValueError(f"expected a qutrit state of length {expected}")
    encoded = kron_power(matrix, num_qutrits) @ vector
    norm = np.linalg.norm(encoded)
    if norm == 0:
        raise ValueError("encoded state has zero norm")
    return encoded / norm


def statevector_data(state: Statevector | np.ndarray) -> np.ndarray:
    if isinstance(state, Statevector):
        return np.asarray(state.data, dtype=complex)
    return np.asarray(state, dtype=complex).reshape(-1)


def infer_num_qutrits_from_state(state: Statevector | np.ndarray) -> int:
    vector = statevector_data(state)
    if vector.size < 1:
        raise ValueError("state vector is empty")
    n_float = math.log(vector.size, 4)
    n = round(n_float)
    if 4**n != vector.size:
        raise ValueError("encoded qutrit states must have dimension 4^N")
    return n


def leakage_probability(
    state: Statevector | np.ndarray,
    E: np.ndarray,
    num_qutrits: int | None = None,
) -> float:
    """Return 1 - <psi|P_E^tensor N|psi>."""
    vector = statevector_data(state)
    n = infer_num_qutrits_from_state(vector) if num_qutrits is None else num_qutrits
    expected = 4**n
    if vector.size != expected:
        raise ValueError(f"expected encoded state length {expected}")
    code_projector = kron_power(projector_E(E), n)
    in_code = np.vdot(vector, code_projector @ vector)
    leakage = 1.0 - float(np.real_if_close(in_code))
    if abs(leakage) < 1e-12:
        return 0.0
    return max(0.0, min(1.0, leakage))
