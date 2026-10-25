import numpy as np
import pytest

from qudits_on_qubits.core.encoding_change_unitary import (
    build_encoding_change_unitary,
    validate_encoding_map,
)


@pytest.mark.parametrize("encoding", [
    np.eye(4, 3),
    np.eye(4)[:, [3, 1, 0]] * [1j, -1, np.exp(0.3j)],
    np.array([[1, 1, 1], [1, -1, 1], [1, 1, -1], [1, -1, -1]]) / 2,
    np.exp(2j * np.pi * np.outer(np.arange(4), np.arange(3)) / 4) / 2,
])
def test_unitary_preserves_encoding_and_input(encoding):
    original = encoding.copy()
    report = validate_encoding_map(encoding)
    assert report["is_valid"]
    assert report["rank"] == 3
    unitary = build_encoding_change_unitary(encoding)
    np.testing.assert_allclose(unitary[:, :3], encoding, rtol=0, atol=1e-12)
    np.testing.assert_allclose(unitary.conj().T @ unitary, np.eye(4), rtol=0, atol=1e-12)
    np.testing.assert_array_equal(encoding, original)


@pytest.mark.parametrize("dtype", [bool, np.int64, np.float16, np.float32, np.complex64])
def test_canonical_encoding_preserves_identity_completion(dtype):
    encoding = np.eye(4, 3, dtype=dtype)
    assert validate_encoding_map(encoding)["is_valid"]
    np.testing.assert_allclose(
        build_encoding_change_unitary(encoding), np.eye(4), rtol=0, atol=0,
    )


def test_zero_tolerance_accepts_exact_canonical_encoding():
    encoding = np.eye(4, 3)
    assert validate_encoding_map(encoding, tol=0)["is_valid"]
    np.testing.assert_array_equal(build_encoding_change_unitary(encoding, tol=0), np.eye(4))


@pytest.mark.parametrize("tolerance, valid", [(1e-10, False), (1e-6, True)])
def test_tolerance_is_absolute(tolerance, valid):
    encoding = (1 + 1e-7) * np.eye(4, 3)
    report = validate_encoding_map(encoding, tol=tolerance)
    assert report["is_valid"] is valid
    if valid:
        unitary = build_encoding_change_unitary(encoding, tol=tolerance)
        assert np.max(np.abs(unitary.conj().T @ unitary - np.eye(4))) <= tolerance
    else:
        assert not report["columns_normalized"]
        with pytest.raises(ValueError, match="not a valid encoding map"):
            build_encoding_change_unitary(encoding, tol=tolerance)


@pytest.mark.parametrize("entry", [np.nan, np.inf, -np.inf, complex(0, np.inf)])
def test_nonfinite_encoding_returns_invalid_report(entry):
    encoding = np.eye(4, 3, dtype=complex)
    encoding[0, 0] = entry
    report = validate_encoding_map(encoding)
    assert not report["is_valid"]
    assert report["message"]
    with pytest.raises(ValueError, match="not a valid encoding map"):
        build_encoding_change_unitary(encoding)


@pytest.mark.parametrize("encoding", [
    [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]],
    np.eye(3),
    np.ones((4, 3), dtype=str),
    np.eye(4, 3).astype(object),
    np.zeros((4, 3)),
    np.ones((4, 3)),
    np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool),
])
def test_invalid_encoding_returns_report_and_builder_rejects_it(encoding):
    report = validate_encoding_map(encoding)
    assert not report["is_valid"]
    assert report["message"]
    with pytest.raises(ValueError, match="not a valid encoding map"):
        build_encoding_change_unitary(encoding)


@pytest.mark.parametrize("tolerance", [
    -1, np.nan, np.inf, -np.inf, "1e-10", 1j, True, None, np.array([1e-10]),
])
@pytest.mark.parametrize("function", [validate_encoding_map, build_encoding_change_unitary])
def test_invalid_tolerance_raises_value_error(function, tolerance):
    with pytest.raises(ValueError, match="tol"):
        function(np.eye(4, 3), tol=tolerance)
