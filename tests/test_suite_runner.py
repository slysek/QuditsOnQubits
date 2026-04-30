import io
import os
import shutil
import unittest
from unittest.mock import patch

import pandas as pd

from encoding_search_v2.suite import (
    COMBINED_RESULTS_BASENAME,
    SUITE_FILE_PREFIX,
    SuiteConfig,
    _build_state_tasks,
    _suite_combined_results_path,
    _suite_state_output_dir,
    run_benchmark_suite,
)
from encoding_search_v2.suite_cli import build_parser, main as suite_cli_main
from encoding_search_v2.states import resolve_benchmark_state


def _tmpdir():
    root = os.path.join(os.path.dirname(__file__), "_tmp_test_outputs")
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, "encoding_search_v2_suite_test")
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=False)
    return path


def _fake_benchmark_basis(E_new, class_name, candidate_name, **kwargs):
    return {
        "state_name": kwargs["state_name"],
        "class_name": class_name,
        "candidate_name": candidate_name,
        "status": "ok",
        "best_depth": 10,
        "mean_depth": 10.0,
        "std_depth": 0.0,
        "best_size": 10,
        "mean_size": 10.0,
        "best_two_qubit_gate_count": 5,
        "mean_two_qubit_gate_count": 5.0,
        "uses_old_codespace_only": class_name == "baseline",
        "avg_codeword_entanglement": 0.0,
        "overlap_with_old_codespace": 1.0,
    }


def _fake_candidates_for_mode(*args, **kwargs):
    return [
        ("baseline", "E_old", None),
        ("product", "U_H__V_H", None),
    ]


class TestSuiteOutputLayout(unittest.TestCase):
    def test_state_output_dir_lives_under_suite_root(self):
        config = SuiteConfig(
            suite_name="graph_states_extended",
            states=("path5",),
            output_root="/tmp/example",
        )
        self.assertEqual(
            _suite_state_output_dir(config, "path5"),
            os.path.join("/tmp/example", "graph_states_extended", "path5"),
        )

    def test_combined_csv_path_uses_canonical_basename(self):
        config = SuiteConfig(
            suite_name="graph_states_extended",
            states=("path5",),
            output_root="/tmp/example",
        )
        self.assertTrue(
            _suite_combined_results_path(config).endswith(COMBINED_RESULTS_BASENAME)
        )


class TestBuildStateTasks(unittest.TestCase):
    def test_tasks_use_resolved_state_id_and_request_strategy(self):
        config = SuiteConfig(
            suite_name="graph_states_extended",
            states=("path5",),
            jobs=1,
            output_root="/tmp/example",
            export_circuits=True,
        )
        spec = resolve_benchmark_state("path5")
        tasks = _build_state_tasks(
            spec,
            [("baseline", "E_old", None)],
            config,
        )
        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        self.assertEqual(task["state_name"], "path5")
        self.assertEqual(task["state_family"], "path")
        self.assertEqual(task["n_qutrits"], 5)
        self.assertEqual(task["encoding_strategy"], "append_w")
        self.assertIn(
            os.path.join("graph_states_extended", "path5", "circuits"),
            task["circuits_output_dir"],
        )
        self.assertEqual(task["suite_name"], "graph_states_extended")

    def test_tasks_skip_circuit_export_by_default(self):
        config = SuiteConfig(
            suite_name="graph_states_extended",
            states=("path5",),
        )
        spec = resolve_benchmark_state("path5")
        tasks = _build_state_tasks(
            spec,
            [("baseline", "E_old", None)],
            config,
        )
        self.assertIsNone(tasks[0]["circuits_output_dir"])


class TestSuiteRunnerEndToEnd(unittest.TestCase):
    def test_runs_each_state_writes_per_state_and_combined_csv(self):
        tmpdir = _tmpdir()
        try:
            config = SuiteConfig(
                suite_name="graph_states_extended",
                states=("path4", "cycle4"),
                jobs=1,
                output_root=tmpdir,
                n_transpile_runs=1,
            )
            log = io.StringIO()
            with patch(
                "encoding_search_v2.suite.benchmark_basis",
                side_effect=_fake_benchmark_basis,
            ), patch(
                "encoding_search_v2.suite.generate_all_class_candidates",
                side_effect=_fake_candidates_for_mode,
            ):
                summary = run_benchmark_suite(config, log_stream=log)

            self.assertEqual(summary["states"], ["path4", "cycle4"])
            for state_id in ("path4", "cycle4"):
                state_dir = os.path.join(tmpdir, "graph_states_extended", state_id)
                main_csv = os.path.join(
                    state_dir,
                    f"{SUITE_FILE_PREFIX}_{state_id}_results.csv",
                )
                self.assertTrue(os.path.exists(main_csv), main_csv)
                df = pd.read_csv(main_csv)
                self.assertIn("state_name", df.columns)
                self.assertIn("state_family", df.columns)
                self.assertIn("n_qutrits", df.columns)
                self.assertEqual(set(df["state_name"]), {state_id})

            combined_path = os.path.join(
                tmpdir, "graph_states_extended", COMBINED_RESULTS_BASENAME
            )
            self.assertTrue(os.path.exists(combined_path))
            combined = pd.read_csv(combined_path)
            self.assertEqual(set(combined["state_name"]), {"path4", "cycle4"})

            log_text = log.getvalue()
            self.assertIn("starting", log_text)
            self.assertIn("DONE", log_text)
            self.assertIn("path4", log_text)
            self.assertIn("cycle4", log_text)

            log_path = os.path.join(
                tmpdir, "graph_states_extended", "suite_run.log"
            )
            self.assertTrue(os.path.exists(log_path))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_skip_existing_reuses_csv_without_recomputation(self):
        tmpdir = _tmpdir()
        try:
            state_dir = os.path.join(tmpdir, "graph_states_extended", "path4")
            os.makedirs(state_dir, exist_ok=True)
            csv_path = os.path.join(
                state_dir,
                f"{SUITE_FILE_PREFIX}_path4_results.csv",
            )
            pd.DataFrame(
                [
                    {
                        "state_name": "path4",
                        "state_family": "path",
                        "n_qutrits": 4,
                        "class_name": "baseline",
                        "candidate_name": "E_old",
                        "status": "ok",
                        "best_depth": 1,
                        "mean_depth": 1.0,
                        "std_depth": 0.0,
                        "best_two_qubit_gate_count": 0,
                    }
                ]
            ).to_csv(csv_path, index=False)

            config = SuiteConfig(
                suite_name="graph_states_extended",
                states=("path4",),
                jobs=1,
                output_root=tmpdir,
                skip_existing=True,
            )

            calls = {"count": 0}

            def _exploding_benchmark(*args, **kwargs):
                calls["count"] += 1
                raise RuntimeError("benchmark must not be called when skipping")

            with patch(
                "encoding_search_v2.suite.benchmark_basis",
                side_effect=_exploding_benchmark,
            ), patch(
                "encoding_search_v2.suite.generate_all_class_candidates",
                side_effect=_fake_candidates_for_mode,
            ):
                summary = run_benchmark_suite(config, log_stream=io.StringIO())

            self.assertEqual(calls["count"], 0)
            self.assertEqual(summary["states"], ["path4"])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestSuiteCli(unittest.TestCase):
    def test_dry_run_prints_resolved_states(self):
        stream = io.StringIO()
        with patch("sys.stdout", stream):
            code = suite_cli_main(
                [
                    "--suite",
                    "graph_states_extended",
                    "--jobs",
                    "4",
                    "--dry-run",
                ]
            )
        self.assertEqual(code, 0)
        text = stream.getvalue()
        self.assertIn("graph_states_extended", text)
        self.assertIn("path9", text)
        self.assertIn("cluster2x3", text)

    def test_states_override_replaces_suite_choice(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--suite",
                "graph_states_extended",
                "--states",
                "path4,cycle4",
                "--dry-run",
            ]
        )
        self.assertEqual(args.states, "path4,cycle4")


if __name__ == "__main__":
    unittest.main()
