from __future__ import annotations

from itertools import product
from typing import Iterable

import numpy as np


OMEGA = np.exp(2j * np.pi / 3)
_CODE_LEVELS = (0, 1, 2)


def canonical_qutrit_embedding() -> np.ndarray:
    """Return E_Z for |0>, |1>, |2> mapped to |00>, |01>, |10>."""
    return np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
        ],
        dtype=complex,
    )


def qutrit_plus_state() -> np.ndarray:
    """Return |+> = (|0> + |1> + |2>) / sqrt(3)."""
    return np.ones(3, dtype=complex) / np.sqrt(3)


def qutrit_fourier() -> np.ndarray:
    """Return the qutrit Fourier matrix F3."""
    return np.array(
        [
            [1, 1, 1],
            [1, OMEGA, OMEGA**2],
            [1, OMEGA**2, OMEGA**4],
        ],
        dtype=complex,
    ) / np.sqrt(3)


def is_unitary(matrix: np.ndarray, tol: float = 1e-10) -> bool:
    matrix = np.asarray(matrix, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False
    return bool(np.allclose(matrix.conj().T @ matrix, np.eye(matrix.shape[0]), atol=tol))


def validate_unitary(matrix: np.ndarray, dimension: int, name: str = "matrix") -> np.ndarray:
    matrix = np.asarray(matrix, dtype=complex)
    if matrix.shape != (dimension, dimension):
        raise ValueError(f"{name} must have shape {(dimension, dimension)}, got {matrix.shape}.")
    if not is_unitary(matrix):
        raise ValueError(f"{name} must be unitary.")
    return matrix


def direct_basis_embedding(w: np.ndarray) -> np.ndarray:
    """Return E_W = E_Z @ W for the direct W-defined qutrit basis."""
    w = validate_unitary(w, 3, name="W")
    return canonical_qutrit_embedding() @ w


def local_plus_state_in_direct_basis(w: np.ndarray) -> np.ndarray:
    """Return the physical two-qubit state E_Z @ W @ |+>."""
    w = validate_unitary(w, 3, name="W")
    state = canonical_qutrit_embedding() @ w @ qutrit_plus_state()
    return state / np.linalg.norm(state)


def qutrit_cz() -> np.ndarray:
    """Return qutrit CZ with phases omega**(j*k) in lexicographic |j,k> order."""
    diagonal = [OMEGA ** (j * k) for j in range(3) for k in range(3)]
    return np.diag(np.array(diagonal, dtype=complex))


def conjugated_qutrit_fourier(w: np.ndarray) -> np.ndarray:
    """Return F3^(W) = W F3 W^dag for the W-defined qutrit basis."""
    w = validate_unitary(w, 3, name="W")
    f3_w = w @ qutrit_fourier() @ w.conj().T
    if not is_unitary(f3_w):
        raise ValueError("Conjugated qutrit Fourier gate is not unitary.")
    return f3_w


def conjugated_qutrit_cz(w: np.ndarray) -> np.ndarray:
    """Return CZ^(W) = (W kron W) CZ (W^dag kron W^dag)."""
    w = validate_unitary(w, 3, name="W")
    ww = np.kron(w, w)
    ww_dag = np.kron(w.conj().T, w.conj().T)
    cz_w = ww @ qutrit_cz() @ ww_dag
    if not is_unitary(cz_w):
        raise ValueError("Conjugated qutrit CZ is not unitary.")
    return cz_w


def code_subspace_indices() -> tuple[int, ...]:
    """Return 16-dimensional physical indices for the 3x3 code subspace."""
    return tuple(left * 4 + right for left, right in product(_CODE_LEVELS, repeat=2))


def embed_single_qutrit_gate_identity_leakage(qutrit_gate: np.ndarray) -> np.ndarray:
    """Embed a 3x3 qutrit gate into two qubits with |11> as identity leakage."""
    gate = validate_unitary(qutrit_gate, 3, name="qutrit_gate")
    embedded = np.eye(4, dtype=complex)
    embedded[np.ix_(_CODE_LEVELS, _CODE_LEVELS)] = gate
    if not is_unitary(embedded):
        raise ValueError("Embedded two-qubit gate is not unitary.")
    return embedded


def embed_two_qutrit_gate_identity_leakage(two_qutrit_gate: np.ndarray) -> np.ndarray:
    """Embed a 9x9 two-qutrit gate into four qubits with identity leakage.

    The code subspace uses physical levels |00>, |01>, |10> for each qutrit.
    Any basis vector where either encoded qutrit is |11> is leakage.  This
    direct-basis benchmark keeps that leakage subspace fixed as identity and
    never couples it to the code subspace.
    """
    gate = validate_unitary(two_qutrit_gate, 9, name="two_qutrit_gate")
    embedded = np.eye(16, dtype=complex)
    code = code_subspace_indices()
    embedded[np.ix_(code, code)] = gate
    if not is_unitary(embedded):
        raise ValueError("Embedded four-qubit gate is not unitary.")
    return embedded


def extract_qutrit_unitary_from_embedding(e_new: np.ndarray | None) -> np.ndarray:
    """Recover W from E_new = E_Z @ W when the embedding stays in code space."""
    if e_new is None:
        return np.eye(3, dtype=complex)

    e_new = np.asarray(e_new, dtype=complex)
    if e_new.shape != (4, 3):
        raise ValueError(f"E_new must have shape (4, 3), got {e_new.shape}.")

    e_z = canonical_qutrit_embedding()
    w = e_z.conj().T @ e_new
    if not np.allclose(e_z @ w, e_new, atol=1e-10):
        raise ValueError(
            "Candidate embedding is not of the form E_Z @ W; direct basis encoding "
            "only supports qutrit U(3) basis changes in the canonical code space."
        )
    validate_unitary(w, 3, name="extracted W")
    return w


def has_code_leakage_identity(embedded: np.ndarray, tol: float = 1e-10) -> bool:
    """Check that leakage basis vectors are identity and uncoupled."""
    embedded = np.asarray(embedded, dtype=complex)
    if embedded.shape != (16, 16):
        return False
    code = set(code_subspace_indices())
    leakage = [idx for idx in range(16) if idx not in code]
    for idx in leakage:
        expected = np.zeros(16, dtype=complex)
        expected[idx] = 1.0
        if not np.allclose(embedded[:, idx], expected, atol=tol):
            return False
        if not np.allclose(embedded[idx, :], expected, atol=tol):
            return False
    return True


def rounded_complex_tuple(values: Iterable[complex], digits: int = 12) -> tuple[complex, ...]:
    return tuple(np.round(np.asarray(tuple(values), dtype=complex), digits))
