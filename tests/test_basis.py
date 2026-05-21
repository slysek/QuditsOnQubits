from __future__ import annotations

import numpy as np

from qutrit_bell_measurements import (
    canonical_Ez,
    complete_isometry_to_unitary,
    local_measurement_basis_unitary,
    ordered_qutrit_eigenbasis,
)
from qutrit_bell_measurements.basis import omega
from qutrit_bell_measurements.postprocessing import compute_complex_expectation


def _qutrit_xz() -> tuple[np.ndarray, np.ndarray, complex]:
    w = omega(3)
    x = np.zeros((3, 3), dtype=complex)
    for j in range(3):
        x[(j + 1) % 3, j] = 1.0
    z = np.diag([w**j for j in range(3)]).astype(complex)
    return x, z, w


def _phase_aligned_columns(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    aligned = np.array(actual, dtype=complex, copy=True)
    for col in range(aligned.shape[1]):
        overlap = np.vdot(expected[:, col], aligned[:, col])
        if abs(overlap) > 1e-12:
            aligned[:, col] *= np.conj(overlap / abs(overlap))
    return aligned


def test_canonical_Ez_is_four_by_three_isometry() -> None:
    E = canonical_Ez()

    assert E.shape == (4, 3)
    np.testing.assert_allclose(E.conj().T @ E, np.eye(3), atol=1e-12)


def test_complete_isometry_to_unitary_returns_four_by_four_unitary() -> None:
    E = canonical_Ez()

    U = complete_isometry_to_unitary(E)

    assert U.shape == (4, 4)
    np.testing.assert_allclose(U[:, :3], E, atol=1e-12)
    np.testing.assert_allclose(U.conj().T @ U, np.eye(4), atol=1e-12)


def test_local_measurement_basis_unitary_handles_qutrit_z_and_x() -> None:
    E = canonical_Ez()
    X, Z, _ = _qutrit_xz()

    for observable in (Z, X):
        W, W_dag, V, gamma = local_measurement_basis_unitary(E, observable)

        np.testing.assert_allclose(W.conj().T @ W, np.eye(4), atol=1e-10)
        np.testing.assert_allclose(W_dag, W.conj().T, atol=1e-12)
        np.testing.assert_allclose(V.conj().T @ V, np.eye(3), atol=1e-10)
        assert abs(gamma - 1.0) < 1e-10


def test_ordered_qutrit_eigenbasis_accepts_global_phase() -> None:
    _, Z, w = _qutrit_xz()
    expected_gamma = np.exp(1j * np.pi / 9)

    V, gamma = ordered_qutrit_eigenbasis(expected_gamma * Z)

    assert abs(gamma - expected_gamma) < 1e-10
    diagonalized = V.conj().T @ (expected_gamma * Z) @ V
    np.testing.assert_allclose(
        diagonalized,
        np.diag([gamma * (w**outcome) for outcome in range(3)]),
        atol=1e-10,
    )


def test_z_measurement_for_canonical_encoding_is_computational_basis() -> None:
    E = canonical_Ez()
    _, Z, _ = _qutrit_xz()

    W, W_dag, _, _ = local_measurement_basis_unitary(E, Z)

    np.testing.assert_allclose(
        _phase_aligned_columns(W, np.eye(4)),
        np.eye(4),
        atol=1e-10,
    )
    np.testing.assert_allclose(W_dag @ W, np.eye(4), atol=1e-12)


def test_compute_complex_expectation_uses_qiskit_bit_order_and_discards_leakage() -> None:
    _, _, w = _qutrit_xz()
    counts = {
        "0000": 2,  # c0,c1 = 00 -> outcome 0
        "0001": 3,  # c0,c1 = 10 -> outcome 2
        "0010": 5,  # c0,c1 = 01 -> outcome 1
        "0011": 7,  # c0,c1 = 11 -> leakage
    }

    value = compute_complex_expectation(
        counts,
        powers=(1,),
        qutrit_bit_indices=[(0, 1)],
        bit_order="qiskit",
    )
    expected = (2 + 3 * (w**2) + 5 * w) / 10

    np.testing.assert_allclose(value, expected, atol=1e-12)

    unnormalized = compute_complex_expectation(
        counts,
        powers=(1,),
        qutrit_bit_indices=[(0, 1)],
        bit_order="qiskit",
        renormalize_after_discard=False,
    )

    np.testing.assert_allclose(unnormalized, (2 + 3 * (w**2) + 5 * w) / 17, atol=1e-12)
