import pytest

from qudits_on_qubits.core.graph_states import (
    BENCHMARK_SUITES,
    cluster_edges,
    complete_edges,
    cycle_edges,
    get_suite_states,
    is_known_graph_state,
    list_suites,
    path_edges,
    resolve_graph_state,
    resolve_graph_state_or_raise,
    star_edges,
    wheel_edges,
)


@pytest.mark.parametrize(
    ("generator", "args", "expected"),
    [
        (star_edges, (4,), ((0, 1), (0, 2), (0, 3))),
        (path_edges, (4,), ((0, 1), (1, 2), (2, 3))),
        (cycle_edges, (3,), ((0, 1), (1, 2), (2, 0))),
        (wheel_edges, (4,), ((0, 1), (0, 2), (0, 3), (1, 2), (2, 3), (3, 1))),
        (complete_edges, (3,), ((0, 1), (0, 2), (1, 2))),
        (cluster_edges, (2, 3), ((0, 1), (0, 3), (1, 2), (1, 4), (2, 5), (3, 4), (4, 5))),
    ],
)
def test_graph_edges(generator, args, expected):
    assert generator(*args) == expected


@pytest.mark.parametrize(
    ("generator", "args"),
    [(star_edges, (1,)), (path_edges, (1,)), (cycle_edges, (2,)),
     (wheel_edges, (3,)), (complete_edges, (1,)),
     (cluster_edges, (0, 3)), (cluster_edges, (1, 1))],
)
def test_graph_generators_reject_too_few_vertices(generator, args):
    with pytest.raises(ValueError):
        generator(*args)


@pytest.mark.parametrize(
    ("name", "count", "state_id", "edges"),
    [
        ("ghz_star", 4, "ghz_star_4", star_edges(4)),
        ("ghz_n", 4, "ghz_star_4", star_edges(4)),
        ("ghz_n_4", None, "ghz_star_4", star_edges(4)),
        ("ghz_star_4", None, "ghz_star_4", star_edges(4)),
        ("ghz4", None, "ghz4", star_edges(4)),
        ("path4", None, "path4", path_edges(4)),
        ("cycle4", None, "cycle4", cycle_edges(4)),
        ("wheel4", None, "wheel4", wheel_edges(4)),
        ("complete4", None, "complete4", complete_edges(4)),
        ("cluster2x3", None, "cluster2x3", cluster_edges(2, 3)),
    ],
)
def test_graph_state_names(name, count, state_id, edges):
    spec = resolve_graph_state_or_raise(name, count)
    assert spec.state_id == state_id
    assert spec.edges == edges


@pytest.mark.parametrize("name", [
    "two_qutrit", "ghz3", "ame43", "ghz_star_4", "ghz4", "path4", "cluster2x3",
])
def test_graph_state_rejects_conflicting_count(name):
    with pytest.raises(ValueError):
        resolve_graph_state(name, 99)


@pytest.mark.parametrize("name", [
    "", "unknown", "path1", "cycle2", "wheel3", "cluster0x2", "cluster1x1",
])
def test_invalid_graph_state_names_are_not_known(name):
    assert not is_known_graph_state(name)


def test_parameterized_names_and_suite_registry():
    assert is_known_graph_state("ghz_star")
    assert is_known_graph_state("ghz_n")
    assert resolve_graph_state("unknown") is None
    with pytest.raises(ValueError, match="Unknown benchmark state"):
        resolve_graph_state_or_raise("unknown")
    assert list_suites() == tuple(sorted(BENCHMARK_SUITES))
    for suite in list_suites():
        assert all(resolve_graph_state(state) for state in get_suite_states(suite))
    with pytest.raises(ValueError, match="Unknown benchmark suite"):
        get_suite_states("unknown")
