from __future__ import annotations

from itertools import combinations, permutations

import numpy as np
import pytest
from qiskit import transpile
from qiskit.quantum_info import Operator, Statevector

from qudits_on_qubits.benchmarks.direct_basis.circuits import (
    build_direct_basis_fourier_gate,
    build_direct_basis_fourier_graph_state_circuit,
    build_direct_basis_graph_state_circuit,
    gate_as_circuit,
)
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
        if permutation in {(0, 1, 2), (0, 2, 1)}
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


def test_support_is_ordered_by_local_x_mapping_unused_state_to_11():
    support_embedding = np.eye(4, dtype=complex)[:, [0, 2, 3]]
    permutation = _permutation_matrix((2, 0, 1))
    encoding = support_embedding @ permutation

    analysis = optimal_f3_leakage_phase(encoding)

    assert analysis.support == (0, 2, 3)
    # Unused |01> is moved to |11> by X on the high bit. In the resulting
    # canonical frame, rows |00>, |01>, |10> came from |10>, |11>, |00>.
    effective_basis = encoding[[2, 3, 0], :]
    np.testing.assert_allclose(
        analysis.effective_fourier,
        effective_basis @ qutrit_fourier() @ effective_basis.conj().T,
    )


@pytest.mark.parametrize("support", list(combinations(range(4), 3)))
@pytest.mark.parametrize("permutation", list(permutations(range(3))))
def test_optimal_phase_reduces_exact_local_synthesis_to_two_cz(support, permutation):
    diagonal = np.diag(np.exp(1j * np.array([0.2, -0.7, 1.3])))
    encoding = np.eye(4)[:, support] @ diagonal @ _permutation_matrix(permutation)
    phase = optimal_f3_leakage_phase(encoding).phase
    counts = []
    for leakage_phase in (0.0, phase):
        source = gate_as_circuit(
            build_direct_basis_fourier_gate(encoding, leakage_phase=leakage_phase),
            2,
            "F3",
        )
        compiled = transpile(
            source, basis_gates=["cz", "rz", "sx", "x"],
            optimization_level=3, seed_transpiler=0,
        )
        assert Operator(source).equiv(Operator(compiled))
        counts.append(compiled.count_ops().get("cz", 0))
    assert counts == [3, 2]


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


def test_fourier_gate_carries_requested_leakage_phase():
    encoding = np.eye(4, 3, dtype=complex)
    phase = optimal_f3_leakage_phase(encoding).phase

    gate = build_direct_basis_fourier_gate(encoding, leakage_phase=phase)

    np.testing.assert_allclose(Operator(gate).data[3, 3], np.exp(1j * phase))


@pytest.mark.parametrize(
    "encoding",
    [
        np.eye(4, 3, dtype=complex),
        np.eye(4, dtype=complex)[:, [0, 2, 3]]
        @ _permutation_matrix((2, 0, 1)),
    ],
)
@pytest.mark.parametrize("state_name", ["two_qutrit", "ghz3", "ame43"])
def test_full_graph_circuits_are_equivalent_and_contain_explicit_f3_per_qutrit(
    encoding, state_name,
):
    phase = optimal_f3_leakage_phase(encoding).phase
    baseline = build_direct_basis_fourier_graph_state_circuit(
        state_name,
        encoding,
        leakage_phase=0.0,
    )
    optimal = build_direct_basis_fourier_graph_state_circuit(
        state_name,
        encoding,
        leakage_phase=phase,
    )

    n_qutrits = baseline.num_qubits // 2
    assert baseline.count_ops()["F3_W"] == n_qutrits
    assert optimal.count_ops()["F3_W"] == n_qutrits
    assert Statevector.from_instruction(baseline).equiv(
        Statevector.from_instruction(optimal)
    )
    assert Statevector.from_instruction(baseline).equiv(
        Statevector.from_instruction(
            build_direct_basis_graph_state_circuit(state_name, encoding)
        )
    )
