from functools import reduce
from itertools import product

import numpy as np
import pytest
from igraph import Graph
from qiskit import ClassicalRegister, QuantumCircuit
from qiskit.primitives import StatevectorSampler
from qiskit.quantum_info import Statevector

from qudits_on_qubits.bell_functionals.encoding import projector_E, validate_isometry
from qudits_on_qubits.bell_functionals.operators import (
    is_hermitian,
    is_unitary,
    make_A_tilde_qutrit_d3,
    make_measurement_observables_qutrit_d3,
)
from qudits_on_qubits.bell_measurements import (
    append_measurement_for_global_setting,
    bits_from_physical_index,
    build_sampler_circuits_for_candidate,
    build_sampler_circuits_from_graph,
    complete_isometry_to_unitary,
    compute_bell_value_from_counts,
    counts_by_setting_from_sampler_result,
    encoding_leakage_subspace,
    logical_part_from_matrix,
    local_measurement_basis_unitary,
    ordered_qutrit_eigenbasis,
    physical_to_logical_outcome_map,
    physical_index_from_bits,
    run_sampler_circuits_to_counts_by_setting,
)
from qudits_on_qubits.reference_experiments import get_reference_experiment


def test_bell_encoding_uses_absolute_isometry_tolerance():
    encoding = np.eye(4, 3) * (1 + 1e-6)
    assert not validate_isometry(encoding)
    with pytest.raises(ValueError, match="E"):
        projector_E(encoding)


def test_operator_checks_use_absolute_tolerance():
    assert not is_unitary((1 + 1e-6) * np.eye(3))
    assert not is_hermitian(np.array([[1, 1], [1 + 1e-6, 1]]))


@pytest.mark.parametrize("power", [1, 2])
@pytest.mark.parametrize("setting", [0, 1, 2])
def test_inverse_measurement_transform_recovers_weyl_operator(power, setting):
    x = np.roll(np.eye(3), 1, axis=0)
    z = np.diag(np.exp(2j * np.pi * np.arange(3) / 3))
    actual = make_A_tilde_qutrit_d3(
        make_measurement_observables_qutrit_d3(power), setting, power,
    )
    expected = np.linalg.matrix_power(x @ np.linalg.matrix_power(z, setting), power)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


@pytest.mark.parametrize("build", [complete_isometry_to_unitary, physical_to_logical_outcome_map])
def test_measurement_encoding_uses_absolute_isometry_tolerance(build):
    with pytest.raises(ValueError, match="E"):
        build(np.eye(4, 3) * (1 + 1e-6))


def test_measurement_basis_rejects_nonunitary_observable():
    with pytest.raises(ValueError, match="unitary"):
        ordered_qutrit_eigenbasis(2 * np.diag(np.exp(2j * np.pi * np.arange(3) / 3)))


@pytest.mark.parametrize("build", ["local", "append"])
def test_measurement_gate_rejects_scaled_encoding(build):
    encoding = np.eye(4, 3) * (1 + 1e-6)
    observable = np.diag(np.exp(2j * np.pi * np.arange(3) / 3))
    with pytest.raises(ValueError, match="E"):
        if build == "local":
            local_measurement_basis_unitary(encoding, observable)
        else:
            append_measurement_for_global_setting(
                QuantumCircuit(2), ["A0"], [(0, 1)], encoding, lambda _: observable,
            )


@pytest.mark.parametrize("build", ["candidate", "graph"])
def test_multiple_measurement_settings_reject_inplace_before_mutation(build):
    state = QuantumCircuit(4)
    state.h(0)
    original = state.copy()
    with pytest.raises(ValueError, match="inplace"):
        if build == "candidate":
            build_sampler_circuits_for_candidate("two_qutrit", state, np.eye(4, 3), inplace=True)
        else:
            build_sampler_circuits_from_graph(
                state, Graph(2, [(0, 1)]), np.eye(4, 3), inplace=True,
            )
    assert state == original


def test_one_explicit_setting_preserves_inplace_contract():
    state = QuantumCircuit(4)
    circuits, _ = build_sampler_circuits_for_candidate(
        "two_qutrit", state, np.eye(4, 3), inplace=True,
        explicit_settings=[("A0", "B0")],
    )
    assert circuits == [state]
    assert state.num_clbits == 4


@pytest.mark.parametrize("old_register", ["meas", "existing"])
def test_sampler_result_preserves_global_classical_bit_indices(old_register):
    state = QuantumCircuit(5)
    previous = ClassicalRegister(1, old_register)
    state.add_register(previous)
    state.x(4)
    circuits, metadata = build_sampler_circuits_for_candidate(
        "two_qutrit", state, np.eye(4, 3), explicit_settings=[("A0", "B0")],
    )
    circuits[0].measure(4, previous[0])
    result = StatevectorSampler(seed=47).run(circuits, shots=128).result()
    counts = counts_by_setting_from_sampler_result(result, metadata)
    values = counts[("A0", "B0")]
    assert sum(values.values()) == 128
    assert all(len(key) == 5 and key[-1] == "1" for key in values)


def test_sampler_result_reorders_a_permuted_existing_register():
    state = QuantumCircuit(5)
    previous = ClassicalRegister(2, "existing")
    state.add_bits(list(reversed(previous)))
    state.add_register(previous)
    state.x(4)
    circuits, metadata = build_sampler_circuits_for_candidate(
        "two_qutrit", state, np.eye(4, 3), explicit_settings=[("A0", "B0")],
    )
    circuits[0].measure(4, previous[0])
    result = StatevectorSampler(seed=47).run(circuits, shots=32).result()
    counts = counts_by_setting_from_sampler_result(result, metadata)
    assert all(key[-2:] == "10" for key in counts[("A0", "B0")])
    with pytest.raises(ValueError, match="metadata"):
        counts_by_setting_from_sampler_result(
            result, {"setting_by_circuit_index": [("A0", "B0")]},
        )


@pytest.mark.parametrize("index", range(4))
def test_physical_bit_conversion_is_invertible(index):
    assert physical_index_from_bits(*bits_from_physical_index(index)) == index


def test_dense_encoding_completion_and_logical_compression():
    unitary = np.fft.fft(np.eye(4), norm="ortho")
    encoding = unitary[:, :3]
    leakage = encoding_leakage_subspace(encoding)
    assert leakage.shape == (4, 1)
    np.testing.assert_allclose(encoding.conj().T @ leakage, 0, atol=1e-12)
    np.testing.assert_allclose(leakage.conj().T @ leakage, 1, atol=1e-12)
    np.testing.assert_array_equal(complete_isometry_to_unitary(unitary), unitary)
    logical = np.diag(np.exp(2j * np.pi * np.arange(3) / 3))
    physical = encoding @ logical @ encoding.conj().T + 1j * leakage @ leakage.conj().T
    np.testing.assert_allclose(logical_part_from_matrix(physical, encoding), logical, atol=1e-12)
    np.testing.assert_array_equal(logical_part_from_matrix(logical), logical)


@pytest.mark.parametrize("transpile", [False, True])
def test_local_sampler_execution_returns_actual_counts(transpile):
    circuit = QuantumCircuit(1, 1)
    circuit.x(0)
    circuit.measure(0, 0)
    counts, _ = run_sampler_circuits_to_counts_by_setting(
        [circuit], {"setting_by_circuit_index": [("A0",)]}, shots=8,
        transpile_circuits=transpile,
    )
    assert counts == {("A0",): {"1": 8}}


@pytest.mark.parametrize("candidate, expected", [("two_qutrit", 6), ("ghz3", 6), ("ame43", 8)])
@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("leakage", [0, 0.125])
def test_measurement_circuits_match_independent_graph_phases(candidate, expected, dense, leakage):
    reference = get_reference_experiment(candidate)
    n = reference.state.num_parties
    unitary = np.fft.fft(np.eye(4), norm="ortho") if dense else np.eye(4)
    encoding = unitary[:, :3]
    logical = np.array([
        np.exp(2j * np.pi * sum(
            weight * levels[u] * levels[v]
            for u, v, weight in reference.state.weighted_edges
        ) / 3) / np.sqrt(3**n)
        for levels in product(range(3), repeat=n)
    ])
    physical = np.sqrt(1 - leakage) * reduce(np.kron, [encoding] * n) @ logical
    physical += np.sqrt(leakage) * reduce(np.kron, [unitary[:, 3]] * n)
    state = QuantumCircuit(2 * n)
    state.initialize(physical)
    circuits, metadata = build_sampler_circuits_for_candidate(
        candidate, state, encoding, add_measurements=False,
        qutrit_qubits=[(2 * party, 2 * party + 1) for party in reversed(range(n))],
    )
    counts = {
        setting: {
            format(index, f"0{2*n}b"): probability
            for index, probability in enumerate(
                Statevector.from_instruction(circuit).probabilities()
            )
        }
        for circuit, setting in zip(circuits, metadata["setting_by_circuit_index"])
    }
    indices = {setting: metadata["qutrit_qubits"] for setting in counts}
    for conditional in (True, False):
        value = compute_bell_value_from_counts(
            counts, metadata["terms"], indices,
            outcome_map=metadata["physical_to_logical_outcome_map"],
            discard_leakage=conditional, renormalize_after_discard=conditional,
        )
        assert value == pytest.approx(expected * (1 if conditional else 1 - leakage), abs=1e-9)
