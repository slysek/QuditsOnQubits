from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def omega(d: int) -> complex:
    """Return the primitive d-th root exp(2 pi i / d)."""
    if d < 1:
        raise ValueError("d must be positive")
    return complex(np.exp(2j * np.pi / d))


def physical_index_from_bits(bit0: int, bit1: int) -> int:
    """Map ``(bit0, bit1)`` to the physical index in ``|00>,|01>,|10>,|11>`` order.

    This convention matches encoded-state bookkeeping where ``bit0`` is the
    more significant bit of the two-qubit block.
    """
    if bit0 not in (0, 1) or bit1 not in (0, 1):
        raise ValueError("bits must be 0 or 1")
    return 2 * bit0 + bit1


def measurement_physical_index_from_bits(bit0: int, bit1: int) -> int:
    """Map measured classical bits to indices after ``W_dag`` gates in Qiskit order."""
    if bit0 not in (0, 1) or bit1 not in (0, 1):
        raise ValueError("bits must be 0 or 1")
    return bit0 + 2 * bit1


def bits_from_physical_index(index: int) -> tuple[int, int]:
    """Inverse of :func:`physical_index_from_bits`."""
    if index not in (0, 1, 2, 3):
        raise ValueError("physical index must be in {0, 1, 2, 3}")
    return index // 2, index % 2


def encoding_leakage_subspace(E: np.ndarray, d: int = 3, tol: float = 1e-10) -> np.ndarray:
    """Return an orthonormal basis of the leakage subspace as columns."""
    matrix = np.asarray(E, dtype=complex)
    if matrix.shape[0] <= d:
        raise ValueError("encoding must have a nontrivial leakage subspace")
    completed = complete_isometry_to_unitary(matrix, tol=tol)
    return completed[:, d:]


def physical_to_logical_outcome_map(
    E: np.ndarray,
    d: int = 3,
    tol: float = 1e-10,
) -> dict[int, int | None]:
    """Map each physical index in ``|00>,|01>,|10>,|11>`` order to a qutrit outcome.

    Physical indices are ordered as ``|00> -> 0``, ``|01> -> 1``, ``|10> -> 2``,
    ``|11> -> 3``. For each logical outcome ``j``, the dominant physical index of
    column ``j`` of ``E`` is assigned to ``j``. Any remaining physical index is
    treated as leakage and mapped to ``None``.
    """
    matrix = np.asarray(E, dtype=complex)
    if matrix.shape != (4, d):
        raise ValueError(f"E must have shape (4, {d})")
    if not np.allclose(matrix.conj().T @ matrix, np.eye(d), atol=tol):
        raise ValueError("E must satisfy E^dagger E = I")

    logical_to_physical: dict[int, int] = {}
    used_physical: set[int] = set()
    for logical in range(d):
        column = matrix[:, logical]
        physical = int(np.argmax(np.abs(column)))
        peak = abs(column[physical])
        if peak < 1.0 - tol:
            raise ValueError(
                f"logical outcome {logical} is not supported on a single "
                "computational basis state"
            )
        if physical in used_physical:
            raise ValueError("two logical outcomes map to the same physical index")
        used_physical.add(physical)
        logical_to_physical[logical] = physical

    outcome_map: dict[int, int | None] = {physical: None for physical in range(4)}
    for logical, physical in logical_to_physical.items():
        outcome_map[physical] = logical
    return outcome_map


def measurement_basis_outcome_map(d: int = 3) -> dict[int, int | None]:
    """Map post-measurement computational indices to qutrit outcomes.

    After ``local_measurement_basis_unitary`` gates are applied, classical
    outcomes index the measurement eigenstates directly:
    ``|00> -> 0``, ``|01> -> 1``, ``|10> -> 2``, ``|11> -> leakage``.
    """
    if d != 3:
        raise ValueError("measurement basis outcome map is implemented only for d=3")
    return {0: 0, 1: 1, 2: 2, 3: None}


def canonical_Ez(d: int = 3) -> np.ndarray:
    """Return the canonical computational isometry E_z: C^d -> C^m.

    For the qutrit case this is the two-qubit encoding
    |0> -> |00>, |1> -> |01>, |2> -> |10>, with |11> as leakage.
    """
    if d < 1:
        raise ValueError("d must be positive")
    physical_dim = 1 << math.ceil(math.log2(d))
    E = np.zeros((physical_dim, d), dtype=complex)
    E[:d, :] = np.eye(d, dtype=complex)
    return E


def complete_isometry_to_unitary(E: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Complete an isometry to a square unitary whose first columns are E."""
    matrix = np.asarray(E, dtype=complex)
    if matrix.ndim != 2:
        raise ValueError("E must be a matrix")
    physical_dim, logical_dim = matrix.shape
    if physical_dim < logical_dim:
        raise ValueError("E must have at least as many rows as columns")
    if not np.allclose(matrix.conj().T @ matrix, np.eye(logical_dim), atol=tol):
        raise ValueError("E must satisfy E^dagger E = I")

    if physical_dim == logical_dim:
        return matrix.copy()

    _, singular_values, vh = np.linalg.svd(matrix.conj().T, full_matrices=True)
    rank = int(np.sum(singular_values > tol))
    if rank != logical_dim:
        raise ValueError("E columns must be linearly independent")

    complement = vh.conj().T[:, logical_dim:]
    U = np.column_stack((matrix, complement))
    if not np.allclose(U.conj().T @ U, np.eye(physical_dim), atol=tol):
        raise ValueError("failed to complete E to a unitary")
    return U


def logical_part_from_matrix(
    O: np.ndarray,
    E: np.ndarray | None = None,
    d: int = 3,
) -> np.ndarray:
    """Return a d x d logical block for a physical operator and encoding ``E``."""
    matrix = np.asarray(O, dtype=complex)
    if matrix.shape == (d, d):
        return matrix.copy()

    encoding = np.asarray(canonical_Ez(d) if E is None else E, dtype=complex)
    physical_dim = encoding.shape[0]
    if matrix.shape != (physical_dim, physical_dim):
        raise ValueError(
            f"O must have shape ({d}, {d}) or ({physical_dim}, {physical_dim})"
        )
    if encoding.shape != (physical_dim, d):
        raise ValueError(f"E must have shape ({physical_dim}, {d})")
    return encoding.conj().T @ matrix @ encoding


def ordered_qutrit_eigenbasis(
    O: np.ndarray,
    d: int = 3,
    tol: float = 1e-7,
    allow_global_phase: bool = True,
) -> tuple[np.ndarray, complex]:
    """Diagonalize O and order eigenvectors by outcomes 0, 1, ..., d - 1.

    The accepted spectrum is ``gamma * {omega(d)^a}``. The returned matrix V
    has column ``a`` for eigenvalue ``gamma * omega(d)^a``. Since gamma is
    defined only up to multiplication by roots of unity, this function chooses
    the matching gamma whose principal phase is closest to zero.
    """
    matrix = np.asarray(O, dtype=complex)
    if matrix.shape != (d, d):
        raise ValueError(f"O must have shape ({d}, {d})")

    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    root_powers = [omega(d) ** outcome for outcome in range(d)]

    if allow_global_phase:
        candidates = [
            eigenvalue / root
            for eigenvalue in eigenvalues
            for root in root_powers
        ]
    else:
        candidates = [1.0 + 0.0j]

    best_match: tuple[float, float, complex, list[int]] | None = None
    for gamma_candidate in candidates:
        order = _match_eigenvalues(eigenvalues, gamma_candidate, root_powers, tol)
        if order is None:
            continue
        max_error = max(
            abs(eigenvalues[idx] - gamma_candidate * root_powers[outcome])
            for outcome, idx in enumerate(order)
        )
        phase_score = abs(np.angle(gamma_candidate))
        score = (phase_score, max_error, complex(gamma_candidate), order)
        if best_match is None or score[:2] < best_match[:2]:
            best_match = score

    if best_match is None:
        raise ValueError("O spectrum is not gamma * {omega^a} within tolerance")

    _, _, gamma, order = best_match
    V = np.column_stack(
        [_canonicalize_vector_phase(eigenvectors[:, idx]) for idx in order]
    )

    if not np.allclose(V.conj().T @ V, np.eye(d), atol=max(10 * tol, 1e-10)):
        raise ValueError("O eigenvectors are not orthonormal; expected a normal observable")

    return V, gamma


def local_measurement_basis_unitary(
    E: np.ndarray,
    observable: np.ndarray,
    d: int = 3,
    tol: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, complex]:
    """Build the local physical basis-change unitary for one encoded qutrit.

    Returns ``W, W_dag, V, gamma`` where the columns of W are
    ``E @ v_0, E @ v_1, E @ v_2, leakage`` and ``W_dag`` is the gate to apply
    before computational-basis measurement on the two physical qubits.
    """
    matrix = np.asarray(E, dtype=complex)
    if matrix.shape != (4, d):
        raise ValueError(f"E must have shape (4, {d})")
    if not np.allclose(matrix.conj().T @ matrix, np.eye(d), atol=tol):
        raise ValueError("E must satisfy E^dagger E = I")

    obs = np.asarray(observable, dtype=complex)
    if obs.shape == (4, 4):
        logical_observable = logical_part_from_matrix(obs, E=matrix, d=d)
    elif obs.shape == (d, d):
        logical_observable = obs
    else:
        raise ValueError(f"observable must have shape ({d}, {d}) or (4, 4)")

    V, gamma = ordered_qutrit_eigenbasis(logical_observable, d=d, tol=tol)
    completed = complete_isometry_to_unitary(matrix, tol=tol)
    leakage_columns = completed[:, d:]
    if leakage_columns.shape != (4, 1):
        raise ValueError("two-qubit qutrit encoding must have exactly one leakage vector")

    W = np.column_stack((matrix @ V, leakage_columns[:, 0]))
    W_dag = W.conj().T
    if not np.allclose(W_dag @ W, np.eye(4), atol=tol):
        raise ValueError("constructed measurement basis is not unitary")
    return W, W_dag, V, gamma


def _match_eigenvalues(
    eigenvalues: Sequence[complex],
    gamma: complex,
    root_powers: Sequence[complex],
    tol: float,
) -> list[int] | None:
    unused = set(range(len(eigenvalues)))
    order: list[int] = []
    for root in root_powers:
        target = gamma * root
        idx = min(unused, key=lambda item: abs(eigenvalues[item] - target))
        error = abs(eigenvalues[idx] - target)
        if error > tol * max(1.0, abs(target)):
            return None
        order.append(idx)
        unused.remove(idx)
    return order


def _canonicalize_vector_phase(vector: np.ndarray) -> np.ndarray:
    normalized = np.asarray(vector, dtype=complex)
    norm = np.linalg.norm(normalized)
    if norm == 0:
        raise ValueError("eigenvector has zero norm")
    normalized = normalized / norm
    pivot = int(np.argmax(np.abs(normalized)))
    pivot_value = normalized[pivot]
    if abs(pivot_value) > 0:
        normalized = normalized / (pivot_value / abs(pivot_value))
    return normalized
