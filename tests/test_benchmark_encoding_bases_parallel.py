import os
import shutil
import unittest
from unittest.mock import patch

import pandas as pd

import QuditsOnQubits.benchmark_encoding_bases_parallel as parallel_bench


class _FakeExecutor:
    last_max_workers = None
    last_chunksize = None

    def __init__(self, max_workers=None):
        type(self).last_max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def map(self, fn, iterable, chunksize=1):
        type(self).last_chunksize = chunksize
        for item in iterable:
            yield fn(item)


class TestParallelBenchmark(unittest.TestCase):
    def tearDown(self):
        _FakeExecutor.last_max_workers = None
        _FakeExecutor.last_chunksize = None

    def test_run_single_state_parallel_benchmark_uses_requested_worker_count_and_preserves_order(self):
        tmpdir = parallel_bench._workspace_tempdir()
        csv_path = os.path.join(tmpdir, "parallel_results.csv")

        def fake_worker(task):
            return {
                "state_name": task["state_name"],
                "class_name": task["class_name"],
                "candidate_name": task["candidate_name"],
                "status": "ok",
                "best_depth": 1 if task["candidate_name"] == "first" else 2,
                "mean_depth": 1.0,
                "std_depth": 0.0,
                "best_size": 1,
                "mean_size": 1.0,
                "best_two_qubit_gate_count": 0,
                "mean_two_qubit_gate_count": 0.0,
                "uses_old_codespace_only": True,
                "avg_codeword_entanglement": 0.0,
                "overlap_with_old_codespace": 1.0,
            }

        try:
            with patch.object(parallel_bench, "ProcessPoolExecutor", _FakeExecutor), patch.object(
                parallel_bench, "_generate_candidates_for_mode",
                return_value=[("baseline", "first", None), ("baseline", "second", None)],
            ), patch.object(parallel_bench, "_benchmark_candidate_worker", side_effect=fake_worker), patch.object(
                parallel_bench, "_print_single_state_summary"
            ):
                df, saved_csv = parallel_bench.run_single_state_parallel_benchmark(
                    state_name="ghz3",
                    csv_path=csv_path,
                    max_workers=32,
                    circuits_output_dir=None,
                )

            self.assertEqual(saved_csv, csv_path)
            self.assertEqual(_FakeExecutor.last_max_workers, 32)
            self.assertEqual(_FakeExecutor.last_chunksize, 1)
            self.assertEqual(list(df["candidate_name"]), ["first", "second"])
            self.assertTrue(os.path.exists(csv_path))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_build_candidate_tasks_disables_export_when_requested(self):
        tasks = parallel_bench._build_candidate_tasks(
            candidates=[("baseline", "first", None)],
            state_name="ame43",
            n_transpile_runs=5,
            circuits_output_dir=None,
            approximation_values=(1.0,),
            fidelity_thresholds=(0.95,),
            approximation_seed=11,
            encoding_strategy="append_w",
        )

        self.assertEqual(len(tasks), 1)
        self.assertIsNone(tasks[0]["circuits_output_dir"])
        self.assertEqual(tasks[0]["state_name"], "ame43")
        self.assertEqual(tasks[0]["approximation_seed"], 11)


if __name__ == "__main__":
    unittest.main()
