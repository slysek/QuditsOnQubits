from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from scripts.run_iqm_transpiler_harness import (
    _default_results_prefix,
    _load_candidates,
    build_parser,
    main,
)


class IqmTranspilerHarnessCliTests(unittest.TestCase):
    def test_parser_defaults(self):
        args = build_parser().parse_args(
            ["--state", "two_qutrit", "--candidate-set", "sanity", "--iqm-backend", "garnet"]
        )

        self.assertEqual(args.state, "two_qutrit")
        self.assertEqual(args.candidate_set, "sanity")
        self.assertEqual(args.iqm_backend, "garnet")
        self.assertEqual(args.n_transpile_runs, 1)
        self.assertEqual(args.max_depth_warning, 100)
        self.assertEqual(args.max_cz_warning, 50)
        self.assertEqual(args.strategy, [])

    def test_from_old_csv_requires_old_csv(self):
        args = build_parser().parse_args(
            ["--state", "two_qutrit", "--candidate-set", "from-old-csv", "--iqm-backend", "garnet"]
        )

        with self.assertRaisesRegex(ValueError, "--old-csv is required"):
            _load_candidates(args)

    def test_default_results_prefix_contains_backend_state_and_candidate_set(self):
        args = build_parser().parse_args(
            [
                "--state",
                "two_qutrit",
                "--candidate-set",
                "from-old-csv",
                "--old-csv",
                "old.csv",
                "--iqm-backend",
                "garnet",
                "--n-transpile-runs",
                "3",
            ]
        )

        prefix = _default_results_prefix(args)

        self.assertEqual(prefix, "iqm_transpiler_harness_garnet_two_qutrit_from_old_csv_runs3")

    def test_main_rejects_zero_transpile_runs_before_backend_load(self):
        with patch("scripts.run_iqm_transpiler_harness.load_iqm_backend") as load_backend:
            with self.assertRaises(SystemExit) as raised:
                main(
                    [
                        "--state",
                        "two_qutrit",
                        "--candidate-set",
                        "sanity",
                        "--iqm-backend",
                        "garnet",
                        "--n-transpile-runs",
                        "0",
                    ]
                )

        self.assertEqual(raised.exception.code, 2)
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
            patch("scripts.run_iqm_transpiler_harness._load_candidates", return_value=candidates),
            patch("scripts.run_iqm_transpiler_harness.load_iqm_backend", return_value=backend) as load_backend,
            patch(
                "scripts.run_iqm_transpiler_harness.default_iqm_transpiler_harness_output_dir",
                return_value="out",
            ),
            patch(
                "scripts.run_iqm_transpiler_harness.run_iqm_transpiler_harness",
                return_value=("all", "best", {"trial_count": 1}),
            ) as run_harness,
            patch(
                "scripts.run_iqm_transpiler_harness.write_iqm_transpiler_harness_outputs",
                return_value=output_paths,
            ) as write_outputs,
        ):
            return_code = main(
                [
                    "--state",
                    "two_qutrit",
                    "--candidate-set",
                    "sanity",
                    "--iqm-backend",
                    "garnet",
                    "--iqm-use-metrics",
                    "--strategy",
                    "preset_default",
                    "--strategy",
                    "transpile_to_iqm_default",
                    "--n-transpile-runs",
                    "2",
                    "--max-depth-warning",
                    "80",
                    "--max-cz-warning",
                    "40",
                ]
            )

        self.assertEqual(return_code, 0)
        load_backend.assert_called_once_with("garnet", use_metrics=True)
        config = run_harness.call_args.args[0]
        self.assertIs(config.backend, backend)
        self.assertEqual(config.iqm_backend_name, "garnet")
        self.assertEqual(config.strategy_names, ("preset_default", "transpile_to_iqm_default"))
        self.assertEqual(config.n_transpile_runs, 2)
        self.assertEqual(config.max_depth_warning, 80)
        self.assertEqual(config.max_cz_warning, 40)
        write_outputs.assert_called_once()


if __name__ == "__main__":
    unittest.main()
