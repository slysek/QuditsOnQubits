from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def make_XZ_qutrit() -> tuple[np.ndarray, np.ndarray, complex]:
    """Return qutrit shift X, clock Z, and omega = exp(2*pi*i/3)."""
    omega = np.exp(2j * np.pi / 3)
    x = np.zeros((3, 3), dtype=complex)
    for j in range(3):
        x[(j + 1) % 3, j] = 1.0
    z = np.diag([omega**j for j in range(3)]).astype(complex)
    return x, z, omega


def qutrit_lambda(n: int) -> complex:
    """Return lambda_n for d=3 and n in {1, 2}."""
    if n == 1:
        return np.exp(1j * np.pi / 18)
    if n == 2:
        return np.exp(-1j * np.pi / 18)
    raise ValueError("qutrit lambda is defined here only for n=1 or n=2")


def _validate_power(n: int) -> int:
    if n not in (1, 2):
        raise ValueError("qutrit Bell powers are n=1 or n=2")
    return n


def _replacement_phase_exponent(k: int) -> int:
    """Phase convention used by the audited qutrit graph-state notebooks.

    With this convention the inverse measurement observables below satisfy
    make_A_tilde_qutrit_d3(make_measurement_observables_qutrit_d3(n), k, n)
    = (X Z^k)^n.
    """
    return k * (k + 1)


def make_measurement_observables_qutrit_d3(n: int = 1) -> list[np.ndarray]:
    """Build the three realizational qutrit observables A_t^n.

    These are the inverse-DFT observables used in the static notebook audit. The
    A-tilde replacement constructed from them recovers the stabilizer factor
    (X Z^k)^n exactly up to floating-point tolerance.
    """
    n = _validate_power(n)
    x, z, omega = make_XZ_qutrit()
    lam = qutrit_lambda(n)
    observables: list[np.ndarray] = []
    for t in range(3):
        matrix = np.zeros((3, 3), dtype=complex)
        for k in range(3):
            phase = omega ** (n * t * k)
            replacement_phase = omega ** (n * _replacement_phase_exponent(k))
            xzk = x @ np.linalg.matrix_power(z, k)
            matrix += phase * replacement_phase * np.linalg.matrix_power(xzk, n)
        observables.append(lam * matrix / math.sqrt(3))
    return observables


def make_A_tilde_qutrit_d3(
    observables: Sequence[np.ndarray],
    k: int,
    n: int = 1,
) -> np.ndarray:
    """Apply the d=3 replacement A_t -> A_tilde_k^(n)."""
    n = _validate_power(n)
    if len(observables) != 3:
        raise ValueError("expected exactly three qutrit observables")
    if k not in (0, 1, 2):
        raise ValueError("k must be 0, 1, or 2")

    _, _, omega = make_XZ_qutrit()
    lam = qutrit_lambda(n)
    prefactor = omega ** (-n * _replacement_phase_exponent(k)) / (
        math.sqrt(3) * lam
    )
    total = np.zeros((3, 3), dtype=complex)
    for t, observable in enumerate(observables):
        total += (omega ** (-n * t * k)) * np.asarray(observable, dtype=complex)
    return prefactor * total


def split_nonhermitian(O: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return Hermitian real and imaginary parts of a square operator."""
    matrix = np.asarray(O, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("operator must be a square matrix")
    real = 0.5 * (matrix + matrix.conj().T)
    imag = (matrix - matrix.conj().T) / (2j)
    return real, imag


def is_hermitian(O: np.ndarray, atol: float = 1e-10) -> bool:
    matrix = np.asarray(O, dtype=complex)
    return np.allclose(matrix.conj().T, matrix, atol=atol)


def is_unitary(O: np.ndarray, atol: float = 1e-10) -> bool:
    matrix = np.asarray(O, dtype=complex)
    return np.allclose(matrix.conj().T @ matrix, np.eye(matrix.shape[0]), atol=atol)
