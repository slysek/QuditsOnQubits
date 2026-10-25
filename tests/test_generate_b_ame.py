import numpy as np
import pytest
from igraph import Graph
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from qudits_on_qubits.core.generate_b_ame import (
    Gi1,
    MemoryLimitExceeded,
    create_stab_from_str,
    find_ck,
    find_n1_n2,
    generate_b_ame,
    th_values,
)


def test_root_vertices_must_share_an_edge():
    graph = Graph(4, [(0, 2), (0, 3)])
    assert find_n1_n2(graph) == (1, 3)


def test_neighbor_stabilizer_includes_edge_to_second_root():
    graph = Graph(3, [(0, 1), (0, 2), (1, 2)])
    np.testing.assert_array_equal(Gi1(graph, 0, 1), [["Z", "Z", "X"]])


def test_coefficients_do_not_require_consecutive_vertex_numbers():
    graph = Graph(6, [(0, 1), (0, 3), (0, 5)])
    coefficients, groups = find_ck(graph, 3)
    np.testing.assert_allclose(coefficients, [[1 / 3] * 3, [1, 0, 0]], atol=1e-15)
    assert tuple(groups[1]) == (4, 6)


@pytest.mark.parametrize("graph", [Graph(0), Graph(1), Graph(3)])
def test_root_selection_rejects_graphs_without_edges(graph):
    with pytest.raises(ValueError, match="edge"):
        find_n1_n2(graph)


@pytest.mark.parametrize("operators", [[1], [1, 2, 3]])
def test_experimental_value_requires_one_result_per_coefficient(operators):
    indicator = generate_b_ame(thval=False)
    indicator.ck_final = [0.25, 0.75]
    with pytest.raises(ValueError, match="result"):
        indicator.bame_jobresult(operators)
    assert indicator.bame_jobresult([2, 4]) == 3.5


@pytest.mark.parametrize("dimension", [2, 5])
def test_stabilizer_rejects_unsupported_dimension(dimension):
    with pytest.raises(ValueError, match="dim"):
        create_stab_from_str("XZ", dimension)


@pytest.mark.parametrize("operators", ["Y", "XZq"])
def test_stabilizer_rejects_unknown_operator(operators):
    with pytest.raises(ValueError, match="operator"):
        create_stab_from_str(operators, 3)


def test_weighted_graph_generates_stabilizers_for_both_neighbor_groups():
    graph = Graph(3, [(0, 1), (0, 2), (0, 2)])
    coefficients, _ = find_ck(graph, 3)
    np.testing.assert_allclose(coefficients, [[1, 0], [0.5, 0.5]])
    indicator = generate_b_ame(dim=3, graph=graph)
    assert len(indicator.get_stabilizers()) == len(indicator.ck_final)
    assert indicator.get_expectation_values() == pytest.approx([1] * len(indicator.ck_final))


@pytest.mark.parametrize("dimension", [3, 4])
def test_stabilizers_fix_the_path_graph_state(dimension):
    indicator = generate_b_ame(dim=dimension, graph=Graph(4, [(0, 1), (1, 2), (2, 3)]))
    assert indicator.get_expectation_values() == pytest.approx([1] * len(indicator.ck_final))
    assert indicator.get_value() == pytest.approx(sum(indicator.ck_final))
    assert indicator.get_n1n2() == (2, 3)


def test_stabilizer_expectations_do_not_allocate_dense_operators(monkeypatch):
    state = QuantumCircuit(12)
    state.h(0)
    shift = QuantumCircuit(12)
    shift.x(0)
    shift.global_phase = np.pi / 4
    clock = QuantumCircuit(12)
    clock.z(0)
    original_init = Operator.__init__

    def reject_dense_operator(self, data, *args, **kwargs):
        if isinstance(data, QuantumCircuit):
            pytest.fail("A dense circuit matrix is not required for statevector evolution")
        original_init(self, data, *args, **kwargs)

    monkeypatch.setattr(Operator, "__init__", reject_dense_operator)
    assert th_values(state, [shift, clock, shift]) == pytest.approx(
        [0.7071 + 0.7071j, 0, 0.7071 + 0.7071j], abs=1e-12,
    )
    assert th_values(None, []) == []


def test_stabilizer_expectations_preserve_qubit_limit():
    with pytest.raises(MemoryLimitExceeded):
        th_values(None, [QuantumCircuit(19)])
