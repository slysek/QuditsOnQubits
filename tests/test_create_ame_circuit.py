from itertools import product

import numpy as np
import pytest
from igraph import Graph
from qiskit.quantum_info import Statevector

from qudits_on_qubits import create_ame_circuit
from qudits_on_qubits.core.create_ame_circuit import change_basis


PREPARED = "prepared_w_then_conjugated_entanglers"
DENSE_ENCODING = np.exp(2j * np.pi * np.outer(np.arange(4), np.arange(3)) / 4) / 2


def logical_graph_state(dimension, n, edges):
    state = np.zeros(dimension**n, dtype=complex)
    for levels in product(range(dimension), repeat=n):
        exponent = sum(levels[u] * levels[v] for u, v in edges if u != v)
        index = sum(level * dimension**i for i, level in enumerate(levels))
        state[index] = np.exp(2j * np.pi * exponent / dimension) / dimension**(n / 2)
    return state


@pytest.mark.parametrize("dimension, strategy", [
    (3, "append_w"), (3, PREPARED), (4, "append_w"),
])
@pytest.mark.parametrize("graph_type, edges", [
    ("star", [(0, 1), (0, 2)]),
    ("line", [(0, 1), (1, 2)]),
    (None, [(0, 1), (0, 1), (1, 2), (2, 2)]),
    (None, [(0, 2)]),
])
def test_graph_state_matches_independent_phases(dimension, strategy, graph_type, edges):
    if graph_type is None:
        graph = Graph(3, edges=edges)
        before = graph.get_edgelist()
        circuit, returned = create_ame_circuit(
            n=99, graph_type="ignored", graph=graph,
            dim=dimension, encoding_strategy=strategy,
        )
        assert returned is graph
        assert graph.get_edgelist() == before
    else:
        circuit, returned = create_ame_circuit(
            n=3, dim=dimension, graph_type=graph_type, encoding_strategy=strategy,
        )
        assert returned.get_edgelist() == edges
    encoding = np.eye(4, dimension)
    transform = np.kron(encoding, np.kron(encoding, encoding))
    expected = transform @ logical_graph_state(dimension, 3, edges)
    assert Statevector.from_instruction(circuit).equiv(expected, atol=1e-10, rtol=0)


@pytest.mark.parametrize("dimension, strategy, encoding", [
    (3, "append_w", None),
    (3, PREPARED, None),
    (3, "append_w", DENSE_ENCODING),
    (3, PREPARED, DENSE_ENCODING),
    (4, "append_w", None),
])
@pytest.mark.parametrize("basis", [
    None,
    np.diag(np.exp(1j * 1e-5 * np.array([1, -1, -1, 1]))),
    np.diag(np.exp(1j * 1e-8 * np.array([1, -1, -1, 1]))),
    np.exp(2j * np.pi * np.outer(np.arange(4), np.arange(4)) / 4) / 2,
])
def test_encoding_and_basis_match_independent_state(dimension, strategy, encoding, basis):
    edges = [(0, 1), (0, 2)]
    circuit, _ = create_ame_circuit(
        n=3, dim=dimension, E_new=encoding, basis=basis, encoding_strategy=strategy,
    )
    local = np.eye(4, dimension) if encoding is None else encoding
    if basis is not None:
        local = basis @ local
    transform = np.kron(local, np.kron(local, local))
    expected = transform @ logical_graph_state(dimension, 3, edges)
    assert Statevector.from_instruction(circuit).equiv(expected, atol=1e-10, rtol=0)


@pytest.mark.parametrize("dimension, strategy", [
    (3, "append_w"), (3, PREPARED), (4, "append_w"),
])
@pytest.mark.parametrize("n", [0, 1])
def test_empty_and_single_vertex_graphs_remain_supported(n, dimension, strategy):
    circuit, graph = create_ame_circuit(n, dimension, encoding_strategy=strategy)
    assert graph.vcount() == n
    assert circuit.num_qubits == 2 * n
    expected = np.zeros(4**n, dtype=complex)
    expected[:dimension**n] = 1 / dimension**(n / 2)
    assert Statevector.from_instruction(circuit).equiv(expected, atol=1e-10, rtol=0)


@pytest.mark.parametrize("kwargs, message", [
    ({}, "Provide either"),
    ({"n": 2, "dim": 2}, "Unsupported dimension"),
    ({"n": 2, "graph_type": "cycle"}, "Unknown graph type"),
    ({"n": 2, "encoding_strategy": "unknown"}, "Unknown encoding_strategy"),
    ({"n": 2, "dim": 4, "E_new": np.eye(4, 3)}, "only for dim=3"),
    ({"n": 2, "dim": 4, "encoding_strategy": PREPARED}, "requires dim=3"),
    ({"n": 2, "E_new": np.zeros((4, 3))}, "not a valid encoding map"),
])
def test_invalid_constructor_arguments_remain_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        create_ame_circuit(**kwargs)


@pytest.mark.parametrize("basis", [
    (1 + 1e-7) * np.eye(4),
    np.full((4, 4), np.nan),
    np.ones((4, 4)),
    np.eye(3),
    np.eye(4, 3),
    np.eye(4).tolist(),
    np.eye(4).astype(str),
    0,
])
def test_invalid_basis_is_rejected(basis):
    with pytest.raises(ValueError):
        create_ame_circuit(n=2, basis=basis)


@pytest.mark.parametrize("dimension", [3, 4])
def test_change_basis_returns_independent_physical_matrix(dimension):
    basis = np.exp(2j * np.pi * np.outer(np.arange(4), np.arange(4)) / 4) / 2
    original = basis.copy()
    basis.setflags(write=False)
    result = change_basis(basis, dimension)
    np.testing.assert_allclose(result, original, atol=1e-12, rtol=0)
    result[0, 0] = 0
    np.testing.assert_array_equal(basis, original)
