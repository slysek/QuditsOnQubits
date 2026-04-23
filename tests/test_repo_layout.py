import os
import unittest

import QuditsOnQubits.benchmark_encoding_bases as benchmark_encoding_bases
from QuditsOnQubits.create_ame_circuit import create_ame_circuit


class RepoLayoutTests(unittest.TestCase):
    def test_create_ame_circuit_works_outside_repo_root(self):
        previous_cwd = os.getcwd()
        tmpdir = os.path.join(os.path.dirname(__file__), "_tmp_test_outputs", "repo_layout_cwd")
        os.makedirs(tmpdir, exist_ok=True)
        os.chdir(tmpdir)
        try:
            try:
                qc, graph = create_ame_circuit(n=2, dim=3, graph_type="star")
            except FileNotFoundError as exc:
                self.fail(f"create_ame_circuit should not depend on cwd: {exc}")
        finally:
            os.chdir(previous_cwd)

        self.assertEqual(qc.num_qubits, 4)
        self.assertEqual(graph.vcount(), 2)

    def test_benchmark_default_paths_live_under_data_benchmarks(self):
        results_path_fn = getattr(benchmark_encoding_bases, "benchmark_results_path", None)
        circuits_dir_fn = getattr(benchmark_encoding_bases, "benchmark_circuits_dir", None)

        self.assertTrue(callable(results_path_fn))
        self.assertTrue(callable(circuits_dir_fn))

        self.assertTrue(
            results_path_fn("full").endswith(
                os.path.join("data", "benchmarks", "benchmark_encoding_bases_full_results.csv")
            )
        )
        self.assertTrue(
            circuits_dir_fn().endswith(os.path.join("data", "benchmarks", "circuits"))
        )


if __name__ == "__main__":
    unittest.main()
