import numpy as np
import pytest

from qudits_on_qubits.benchmarks.direct_basis.math_utils import (
    encoding_embedding,
    has_code_leakage_identity,
    is_isometry,
    is_unitary,
    validate_unitary,
)


@pytest.mark.parametrize("rows, predicate", [(3, is_unitary), (4, is_isometry)])
@pytest.mark.parametrize("error, accepted", [(1e-12, True), (1e-7, False), (-1e-7, False)])
def test_orthogonality_uses_absolute_tolerance(rows, predicate, error, accepted):
    matrix = np.eye(rows, 3, dtype=complex)
    matrix[0, 0] = np.sqrt(1 + error)

    assert predicate(matrix, tol=1e-10) is accepted


def test_encoding_embedding_rejects_scaled_logical_column():
    encoding = np.eye(4, 3, dtype=complex)
    encoding[0, 0] = np.sqrt(1 + 1e-7)

    with pytest.raises(ValueError, match="isometry"):
        encoding_embedding(encoding)


def test_validate_unitary_rejects_scaled_gate_column():
    gate = np.eye(3, dtype=complex)
    gate[0, 0] = np.sqrt(1 + 1e-7)

    with pytest.raises(ValueError, match="unitary"):
        validate_unitary(gate, 3)


def test_leakage_identity_uses_absolute_tolerance():
    gate = np.eye(16, dtype=complex)
    gate[3, 3] = np.exp(1e-7j)

    assert not has_code_leakage_identity(gate, tol=1e-10)
    assert has_code_leakage_identity(gate, tol=1e-6)
