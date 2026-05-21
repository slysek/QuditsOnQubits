from __future__ import annotations

import numpy as np

from qutrit_bell_measurements import (
    bit_pair_to_qutrit_outcome,
    build_sampler_circuits_for_candidate,
    canonical_Ez,
    compute_bell_value_from_counts,
    compute_complex_expectation,
    decoding_kwargs_from_metadata,
    logical_part_from_matrix,
    physical_to_logical_outcome_map,
)
from qutrit_bell_measurements.basis import omega, physical_index_from_bits
from qutrit_bell_measurements.postprocessing import bitstring_to_qutrit_outcomes


E_PERM = np.array(
    [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 1],
    ],
    dtype=complex,
)


def test_canonical_encoding_outcome_map() -> None:
    E = canonical_Ez()
    outcome_map = physical_to_logical_outcome_map(E)

    assert outcome_map == {0: 0, 1: 1, 2: 2, 3: None}
    assert bit_pair_to_qutrit_outcome(0, 0, E=E) == 0
    assert bit_pair_to_qutrit_outcome(0, 1, E=E) == 1
    assert bit_pair_to_qutrit_outcome(1, 0, E=E) == 2
    assert bit_pair_to_qutrit_outcome(1, 1, E=E) is None


def test_permuted_encoding_outcome_map() -> None:
    outcome_map = physical_to_logical_outcome_map(E_PERM)

    assert outcome_map[2] is None
    assert outcome_map[3] == 2
    assert bit_pair_to_qutrit_outcome(1, 0, E=E_PERM) is None
    assert bit_pair_to_qutrit_outcome(1, 1, E=E_PERM) == 2


def test_logical_part_from_matrix_uses_provided_encoding() -> None:
    E = canonical_Ez()
    logical_z = np.diag([1.0, 1.0, omega(3) ** 2]).astype(complex)
    physical_z = E @ logical_z @ E.conj().T

    np.testing.assert_allclose(
        logical_part_from_matrix(physical_z, E=E),
        logical_z,
        atol=1e-12,
    )


def test_compute_complex_expectation_with_permuted_encoding() -> None:
    w = omega(3)
    counts = {
        "0000": 2,  # 00 -> logical 0
        "0010": 5,  # 01 -> logical 1
        "0001": 3,  # 10 -> leakage for E_PERM
        "0011": 4,  # 11 -> logical 2
    }

    value = compute_complex_expectation(
        counts,
        powers=(1,),
        qutrit_bit_indices=[(0, 1)],
        E=E_PERM,
        bit_order="qiskit",
    )
    expected = (2 + 5 * w + 4 * (w**2)) / 11

    np.testing.assert_allclose(value, expected, atol=1e-12)


def test_direct_basis_encoding_outcome_map_matches_dominant_columns() -> None:
    w = np.array(
        [
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
        ],
        dtype=complex,
    )
    E = canonical_Ez() @ w
    outcome_map = physical_to_logical_outcome_map(E)

    for logical in range(3):
        physical = int(np.argmax(np.abs(E[:, logical])))
        assert outcome_map[physical] == logical


def test_metadata_carries_encoding_for_postprocessing() -> None:
    from qiskit import QuantumCircuit

    _, metadata = build_sampler_circuits_for_candidate(
        candidate="two_qutrit",
        state_circuit=QuantumCircuit(4),
        E=E_PERM,
    )

    assert "E" in metadata
    assert "encoding_outcome_map" in metadata
    assert "physical_to_logical_outcome_map" in metadata
    np.testing.assert_allclose(metadata["E"], E_PERM, atol=1e-12)
    assert metadata["encoding_outcome_map"][2] is None
    assert metadata["physical_to_logical_outcome_map"] == {0: 0, 1: 1, 2: 2, 3: None}

    kwargs = decoding_kwargs_from_metadata(metadata)
    assert kwargs["outcome_map"] == metadata["physical_to_logical_outcome_map"]


def test_bitstring_decode_uses_physical_index_convention() -> None:
    assert physical_index_from_bits(0, 0) == 0
    assert physical_index_from_bits(0, 1) == 1
    assert physical_index_from_bits(1, 0) == 2
    assert physical_index_from_bits(1, 1) == 3

    outcomes = bitstring_to_qutrit_outcomes(
        "0011",
        qutrit_bit_indices=[(0, 1)],
        E=E_PERM,
        bit_order="qiskit",
    )
    assert outcomes == (2,)
