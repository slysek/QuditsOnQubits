import unittest

from QuditsOnQubits.graph_states import (
    BENCHMARK_SUITES,
    EXTENDED_GRAPH_STATES,
    GraphStateSpec,
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


class TestEdgeGenerators(unittest.TestCase):
    def test_star_edges_are_center_zero_to_each_leaf(self):
        self.assertEqual(star_edges(5), ((0, 1), (0, 2), (0, 3), (0, 4)))

    def test_path_edges_form_linear_chain(self):
        self.assertEqual(path_edges(5), ((0, 1), (1, 2), (2, 3), (3, 4)))

    def test_cycle_edges_close_the_loop(self):
        self.assertEqual(
            cycle_edges(4),
            ((0, 1), (1, 2), (2, 3), (3, 0)),
        )

    def test_wheel_edges_combine_spokes_and_outer_cycle(self):
        edges = wheel_edges(5)
        self.assertIn((0, 1), edges)
        self.assertIn((0, 2), edges)
        self.assertIn((0, 3), edges)
        self.assertIn((0, 4), edges)
        # outer cycle 1 -> 2 -> 3 -> 4 -> 1
        self.assertIn((1, 2), edges)
        self.assertIn((2, 3), edges)
        self.assertIn((3, 4), edges)
        self.assertIn((4, 1), edges)

    def test_complete_edges_count_matches_n_choose_2(self):
        n = 5
        self.assertEqual(len(complete_edges(n)), n * (n - 1) // 2)

    def test_cluster_edges_for_2x3_grid_count_matches_grid_topology(self):
        edges = set(cluster_edges(2, 3))
        # rows: 2, cols: 3
        # horizontal: 2 rows * 2 hop = 4
        # vertical: 3 cols * 1 hop = 3
        self.assertEqual(len(edges), 7)
        self.assertIn((0, 1), edges)
        self.assertIn((1, 2), edges)
        self.assertIn((3, 4), edges)
        self.assertIn((4, 5), edges)
        self.assertIn((0, 3), edges)
        self.assertIn((1, 4), edges)
        self.assertIn((2, 5), edges)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            star_edges(1)
        with self.assertRaises(ValueError):
            cycle_edges(2)
        with self.assertRaises(ValueError):
            wheel_edges(3)
        with self.assertRaises(ValueError):
            cluster_edges(0, 5)


class TestStateNameResolver(unittest.TestCase):
    def test_resolve_two_qutrit_returns_star_with_one_edge(self):
        spec = resolve_graph_state("two_qutrit")
        self.assertIsInstance(spec, GraphStateSpec)
        self.assertEqual(spec.num_qutrits, 2)
        self.assertEqual(spec.state_id, "two_qutrit")
        self.assertEqual(spec.edges, ((0, 1),))

    def test_resolve_ghz3_keeps_legacy_state_id(self):
        spec = resolve_graph_state("ghz3")
        self.assertEqual(spec.state_id, "ghz3")
        self.assertEqual(spec.state_family, "ghz_star")
        self.assertEqual(spec.num_qutrits, 3)

    def test_resolve_ghz_short_alias_for_arbitrary_n(self):
        spec = resolve_graph_state("ghz5")
        self.assertEqual(spec.state_id, "ghz5")
        self.assertEqual(spec.state_family, "ghz_star")
        self.assertEqual(spec.num_qutrits, 5)
        self.assertEqual(spec.edges, ((0, 1), (0, 2), (0, 3), (0, 4)))

    def test_resolve_path_cycle_wheel_complete_and_cluster(self):
        self.assertEqual(resolve_graph_state("path4").num_qutrits, 4)
        self.assertEqual(resolve_graph_state("path4").state_family, "path")
        self.assertEqual(resolve_graph_state("cycle6").num_qutrits, 6)
        self.assertEqual(resolve_graph_state("wheel7").num_qutrits, 7)
        self.assertEqual(resolve_graph_state("complete5").num_qutrits, 5)
        cluster = resolve_graph_state("cluster2x3")
        self.assertEqual(cluster.num_qutrits, 6)
        self.assertEqual(cluster.state_family, "cluster")

    def test_resolve_returns_none_for_unknown_name(self):
        self.assertIsNone(resolve_graph_state("nonsense"))

    def test_resolve_or_raise_propagates_for_unknown(self):
        with self.assertRaises(ValueError):
            resolve_graph_state_or_raise("nonsense")

    def test_resolve_ghz_star_requires_n(self):
        with self.assertRaises(ValueError):
            resolve_graph_state("ghz_star")

    def test_is_known_graph_state_does_not_raise_for_n_required_aliases(self):
        self.assertTrue(is_known_graph_state("ghz_star"))
        self.assertTrue(is_known_graph_state("ghz5"))
        self.assertFalse(is_known_graph_state("not_a_state"))


class TestSuiteRegistry(unittest.TestCase):
    def test_extended_suite_does_not_include_legacy_states(self):
        legacy = {"two_qutrit", "ghz3", "ame43"}
        self.assertFalse(legacy.intersection(EXTENDED_GRAPH_STATES))

    def test_extended_suite_includes_all_requested_states(self):
        expected = {
            "ghz4", "ghz5", "ghz6", "ghz7", "ghz8", "ghz9",
            "path4", "path5", "path6", "path7", "path8", "path9",
            "cycle4", "cycle5", "cycle6", "cycle7", "cycle8", "cycle9",
            "wheel5", "wheel6", "wheel7", "wheel8", "wheel9",
            "complete4", "complete5", "complete6",
            "cluster2x2", "cluster2x3",
        }
        self.assertEqual(set(EXTENDED_GRAPH_STATES), expected)

    def test_suite_registry_lookup(self):
        self.assertIn("graph_states_extended", BENCHMARK_SUITES)
        self.assertEqual(
            get_suite_states("graph_states_extended"),
            EXTENDED_GRAPH_STATES,
        )
        self.assertIn("graph_states_extended", list_suites())

    def test_unknown_suite_raises(self):
        with self.assertRaises(ValueError):
            get_suite_states("unknown_suite")


if __name__ == "__main__":
    unittest.main()
