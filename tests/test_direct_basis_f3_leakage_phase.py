from __future__ import annotations

from itertools import permutations

import numpy as np
import pytest

from qudits_on_qubits.benchmarks.direct_basis.math_utils import (
    F3LeakagePhaseAnalysis,
    encoding_embedding,
    optimal_f3_leakage_phase,
    physical_single_qutrit_gate_in_encoding,
    qutrit_fourier,
)
from qudits_on_qubits.core.benchmark_encoding_bases import (
    generate_monomial_full_bases,
)


def _permutation_matrix(permutation: tuple[int, int, int]) -> np.ndarray:
    return np.eye(3, dtype=complex)[list(permutation)]


@pytest.mark.parametrize("permutation", list(permutations(range(3))))
def test_optimal_phase_for_effective_f3_permutations(permutation):
    encoding = np.eye(4, 3, dtype=complex) @ _permutation_matrix(permutation)

    analysis = optimal_f3_leakage_phase(encoding)

    expected = (
        11 * np.pi / 6
        if permutation in {(1, 2, 0), (2, 1, 0)}
        else np.pi / 2
    )
    assert isinstance(analysis, F3LeakagePhaseAnalysis)
    assert analysis.phase == pytest.approx(expected)
    assert abs(abs(analysis.phase_factor) - 1.0) < 1e-12
    assert analysis.support == (0, 1, 2)


def test_all_generated_monomial_full_phases_are_analytic_two_value_family():
    phases = {
        round(optimal_f3_leakage_phase(encoding).phase / np.pi, 12)
        for _, _, encoding in generate_monomial_full_bases(max_candidates=None)
    }

    assert phases == {0.5, round(11 / 6, 12)}


def test_diagonal_monomial_phases_do_not_change_optimal_phase():
    permutation = _permutation_matrix((1, 2, 0))
    first = np.eye(4, 3, dtype=complex) @ permutation
    diagonal = np.diag(np.exp(1j * np.array([0.2, -0.7, 1.3])))
    second = np.eye(4, 3, dtype=complex) @ diagonal @ permutation

    assert optimal_f3_leakage_phase(first).phase == pytest.approx(
        optimal_f3_leakage_phase(second).phase
    )


def test_support_embedding_is_removed_in_ascending_physical_order():
    support_embedding = np.eye(4, dtype=complex)[:, [0, 2, 3]]
    permutation = _permutation_matrix((2, 0, 1))
    encoding = support_embedding @ permutation

    analysis = optimal_f3_leakage_phase(encoding)

    assert analysis.support == (0, 2, 3)
    np.testing.assert_allclose(
        analysis.effective_fourier,
        permutation @ qutrit_fourier() @ permutation.conj().T,
    )


def test_dense_encoding_is_not_accepted_as_monomial():
    with pytest.raises(ValueError, match="monomial"):
        optimal_f3_leakage_phase(qutrit_fourier())


def test_phase_embedding_changes_only_the_leakage_complement():
    encoding = np.eye(4, 3, dtype=complex)
    analysis = optimal_f3_leakage_phase(encoding)
    baseline = physical_single_qutrit_gate_in_encoding(qutrit_fourier(), encoding)
    optimal = physical_single_qutrit_gate_in_encoding(
        qutrit_fourier(),
        encoding,
        leakage_phase=analysis.phase,
    )
    physical_encoding = encoding_embedding(encoding)
    projector = physical_encoding @ physical_encoding.conj().T

    np.testing.assert_allclose(
        projector @ baseline @ projector,
        projector @ optimal @ projector,
    )
    np.testing.assert_allclose(optimal[3, 3], analysis.phase_factor)


@pytest.mark.parametrize("phase", [True, np.inf, np.nan, "11pi/6"])
def test_phase_embedding_rejects_invalid_phase(phase):
    with pytest.raises(ValueError, match="leakage_phase"):
        physical_single_qutrit_gate_in_encoding(
            qutrit_fourier(),
            np.eye(3),
            leakage_phase=phase,
        )
