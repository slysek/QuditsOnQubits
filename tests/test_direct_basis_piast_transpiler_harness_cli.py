from __future__ import annotations

import contextlib
import io
import os
import unittest
from unittest.mock import patch

from scripts.run_piast_transpiler_harness import (
    _default_results_prefix,
    _load_candidates,
    _output_dir_from_args,
    build_parser,
    main,
)


class PiastTranspilerHarnessCliTests(unittest.TestCase):
    def test_parser_defaults(self):
        args = build_parser().parse_args(
            ["--state", "two_qutrit", "--candidate-set", "sanity", "--piast"]
        )

        self.assertEqual(args.state, "two_qutrit")
        self.assertEqual(args.candidate_set, "sanity")
        self.assertTrue(args.piast)
        self.assertEqual(args.n_transpile_runs, 1)
        self.assertEqual(args.max_depth_warning, 100)
        self.assertEqual(args.max_rxx_warning, 50)
        self.assertEqual(args.strategy, [])
        self.assertIsNone(args.quantum_circuits_dir)
        self.assertFalse(args.no_export_quantum_circuits)

    def test_from_old_csv_requires_old_csv(self):
        args = build_parser().parse_args(
            ["--state", "two_qutrit", "--candidate-set", "from-old-csv", "--piast"]
        )

        with self.assertRaisesRegex(ValueError, "--old-csv is required"):
            _load_candidates(args)

    def test_default_results_prefix_contains_piast_state_and_candidate_set(self):
        args = build_parser().parse_args(
            [
                "--state",
                "two_qutrit",
                "--candidate-set",
                "from-old-csv",
                "--old-csv",
                "old.csv",
                "--piast",
                "--n-transpile-runs",
                "3",
            ]
        )

        prefix = _default_results_prefix(args)

        self.assertEqual(prefix, "piast_transpiler_harness_piast_two_qutrit_from_old_csv_runs3")

    def test_output_dir_default_uses_short_timestamp_unless_run_id_is_explicit(self):
        args = build_parser().parse_args(
            ["--state", "two_qutrit", "--candidate-set", "sanity", "--piast"]
        )

        with (
            patch(
                "scripts.run_piast_transpiler_harness.default_piast_transpiler_harness_run_id",
                return_value="20260709_120000",
            ),
            patch(
                "scripts.run_piast_transpiler_harness.default_piast_transpiler_harness_output_dir",
                return_value="out",
            ) as default_output_dir,
        ):
            output_dir = _output_dir_from_args(args)

        self.assertEqual(output_dir, "out")
        default_output_dir.assert_called_once_with("20260709_120000")

        explicit_args = build_parser().parse_args(
            [
                "--state",
                "two_qutrit",
                "--candidate-set",
                "sanity",
                "--piast",
                "--run-id",
                "manual_run",
            ]
        )
        with patch(
            "scripts.run_piast_transpiler_harness.default_piast_transpiler_harness_output_dir",
            return_value="manual-out",
        ) as default_output_dir:
            output_dir = _output_dir_from_args(explicit_args)

        self.assertEqual(output_dir, "manual-out")
        default_output_dir.assert_called_once_with("manual_run")

    def test_main_requires_piast_flag_before_backend_load(self):
        with (
            patch("scripts.run_piast_transpiler_harness.load_piast_backend") as load_backend,
            contextlib.redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(SystemExit) as raised:
                main(["--state", "two_qutrit", "--candidate-set", "sanity"])

        self.assertEqual(raised.exception.code, 2)
        load_backend.assert_not_called()

    def test_main_rejects_zero_transpile_runs_before_backend_load(self):
        with patch("scripts.run_piast_transpiler_harness.load_piast_backend") as load_backend:
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    main(
                        [
                            "--state",
                            "two_qutrit",
                            "--candidate-set",
                            "sanity",
                            "--piast",
                            "--n-transpile-runs",
                            "0",
                        ]
                    )

        self.assertEqual(raised.exception.code, 2)
        load_backend.assert_not_called()

    def test_main_rejects_invalid_state_before_candidate_and_backend_load(self):
        with (
            patch("scripts.run_piast_transpiler_harness._load_candidates") as load_candidates,
            patch("scripts.run_piast_transpiler_harness.load_piast_backend") as load_backend,
            contextlib.redirect_stderr(io.StringIO()),
        ):
            with self.assertRaises(SystemExit) as raised:
                main(
                    [
                        "--state",
                        "missing_state",
                        "--candidate-set",
                        "sanity",
                        "--piast",
                    ]
                )

        self.assertEqual(raised.exception.code, 2)
        load_candidates.assert_not_called()
        load_backend.assert_not_called()

    def test_main_wires_backend_and_harness_without_network(self):
        backend = object()
        candidates = [object()]
        output_paths = {
            "all_trials_csv": os.path.join("out", "all_trials.csv"),
            "best_by_candidate_csv": os.path.join("out", "best_by_candidate.csv"),
            "summary_json": os.path.join("out", "summary.json"),
        }

        with (
            patch("scripts.run_piast_transpiler_harness._load_candidates", return_value=candidates),
            patch("scripts.run_piast_transpiler_harness.load_piast_backend", return_value=backend) as load_backend,
            patch(
                "scripts.run_piast_transpiler_harness.default_piast_transpiler_harness_output_dir",
                return_value="out",
            ),
            patch(
                "scripts.run_piast_transpiler_harness.run_piast_transpiler_harness",
                return_value=("all", "best", {"trial_count": 1}),
            ) as run_harness,
            patch(
                "scripts.run_piast_transpiler_harness.write_piast_transpiler_harness_outputs",
                return_value=output_paths,
            ) as write_outputs,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            return_code = main(
                [
                    "--state",
                    "two_qutrit",
                    "--candidate-set",
                    "sanity",
                    "--piast",
                    "--strategy",
                    "transpile_aqt_plugin",
                    "--strategy",
                    "preset_aqt_plugin",
                    "--n-transpile-runs",
                    "2",
                    "--max-depth-warning",
                    "80",
                    "--max-rxx-warning",
                    "40",
                ]
            )

        self.assertEqual(return_code, 0)
        load_backend.assert_called_once_with()
        config = run_harness.call_args.args[0]
        self.assertIs(config.backend, backend)
        self.assertEqual(config.strategy_names, ("transpile_aqt_plugin", "preset_aqt_plugin"))
        self.assertEqual(config.n_transpile_runs, 2)
        self.assertEqual(config.max_depth_warning, 80)
        self.assertEqual(config.max_rxx_warning, 40)
        self.assertEqual(config.quantum_circuits_dir, os.path.join("out", "quantum_circuits"))
        write_outputs.assert_called_once()

    def test_main_can_disable_quantum_circuit_exports(self):
        with (
            patch("scripts.run_piast_transpiler_harness._load_candidates", return_value=[]),
            patch("scripts.run_piast_transpiler_harness.load_piast_backend", return_value=object()),
            patch(
                "scripts.run_piast_transpiler_harness.run_piast_transpiler_harness",
                return_value=("all", "best", {}),
            ) as run_harness,
            patch(
                "scripts.run_piast_transpiler_harness.write_piast_transpiler_harness_outputs",
                return_value={
                    "all_trials_csv": "all.csv",
                    "best_by_candidate_csv": "best.csv",
                    "summary_json": "summary.json",
                },
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            main(
                [
                    "--state",
                    "two_qutrit",
                    "--candidate-set",
                    "sanity",
                    "--piast",
                    "--no-export-quantum-circuits",
                ]
            )

        config = run_harness.call_args.args[0]
        self.assertIsNone(config.quantum_circuits_dir)


if __name__ == "__main__":
    unittest.main()
