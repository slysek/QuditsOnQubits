from __future__ import annotations

import contextlib
import io
import os
import unittest
from unittest.mock import patch

from scripts.run_direct_basis_benchmarks import (
    _default_results_prefix,
    _iqm_quantum_circuits_dir_from_args,
    build_parser,
    main,
)


class DirectBasisIqmCliTests(unittest.TestCase):
    def test_optimal_f3_comparison_is_opt_in(self):
        default = build_parser().parse_args(["--state", "two_qutrit"])
        enabled = build_parser().parse_args(
            ["--state", "two_qutrit", "--compare-optimal-f3-leakage"]
        )
        self.assertFalse(default.compare_optimal_f3_leakage)
        self.assertTrue(enabled.compare_optimal_f3_leakage)

    def test_iqm_backend_flag_defaults_iqm_options(self):
        args = build_parser().parse_args(["--state", "two_qutrit", "--iqm-backend", "garnet"])

        self.assertEqual(args.iqm_backend, "garnet")
        self.assertIs(args.iqm_use_metrics, False)
        self.assertIsNone(args.layout_method)
        self.assertIsNone(args.routing_method)
        self.assertEqual(args.iqm_strategy, [])
        self.assertFalse(args.iqm_legacy_pass_manager)
        self.assertEqual(args.ranking_workload, "state_preparation")

    def test_ranking_workload_accepts_bell_measurements(self):
        args = build_parser().parse_args(
            ["--state", "ghz3", "--ranking-workload", "bell_measurements"]
        )

        self.assertEqual(args.ranking_workload, "bell_measurements")

    def test_ranking_workload_rejects_unknown_value(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                build_parser().parse_args(
                    ["--state", "ghz3", "--ranking-workload", "unknown"]
                )

    def test_iqm_use_metrics_parses_true(self):
        args = build_parser().parse_args(
            ["--state", "two_qutrit", "--iqm-backend", "garnet", "--iqm-use-metrics"]
        )

        self.assertIs(args.iqm_use_metrics, True)

    def test_default_results_prefix_includes_iqm_backend(self):
        args = build_parser().parse_args(
            [
                "--state",
                "two_qutrit",
                "--candidate-set",
                "sanity",
                "--n-transpile-runs",
                "1",
                "--iqm-backend",
                "garnet",
            ]
        )

        prefix = _default_results_prefix(args)

        self.assertTrue(prefix.startswith("direct_basis_iqm_garnet_two_qutrit_sanity"))
        self.assertTrue(prefix.endswith("runs1"))

    def test_iqm_quantum_circuits_dir_uses_backend_slug(self):
        args = build_parser().parse_args(["--state", "two_qutrit", "--iqm-backend", "garnet"])

        path = _iqm_quantum_circuits_dir_from_args(args)

        self.assertTrue(
            path.endswith(
                os.path.join(
                    "artifacts",
                    "iqm_runs",
                    "raw",
                    "quantum_circuits",
                    "garnet",
                )
            )
        )

    def test_main_wires_iqm_backend_without_network(self):
        backend = object()
        metadata = {
            "transpiler_backend": "iqm",
            "iqm_backend_name": "garnet",
            "iqm_use_metrics": True,
            "optimization_level": 3,
            "layout_method": "sabre",
            "routing_method": "sabre",
        }
        candidates = [object()]

        with (
            patch("scripts.run_direct_basis_benchmarks._load_candidates", return_value=candidates),
            patch(
                "scripts.run_direct_basis_benchmarks.load_iqm_backend",
                return_value=backend,
            ) as load_iqm_backend,
            patch(
                "scripts.run_direct_basis_benchmarks.backend_metadata",
                return_value=metadata,
            ) as backend_metadata,
            patch(
                "scripts.run_direct_basis_benchmarks.timestamped_results_path",
                return_value="out.csv",
            ),
            patch(
                "scripts.run_direct_basis_benchmarks.benchmark_direct_basis_candidates",
                return_value=(None, "out.csv"),
            ) as benchmark_direct_basis_candidates,
        ):
            return_code = main(
                [
                    "--state",
                    "two_qutrit",
                    "--iqm-backend",
                    "garnet",
                    "--iqm-use-metrics",
                    "--layout-method",
                    "sabre",
                    "--routing-method",
                    "sabre",
                    "--jobs",
                    "3",
                    "--compare-optimal-f3-leakage",
                    "--no-export-quantum-circuits",
                    "--no-fidelity",
                ]
            )

        self.assertEqual(return_code, 0)
        load_iqm_backend.assert_called_once_with("garnet", use_metrics=True)
        backend_metadata.assert_called_once_with(
            backend,
            iqm_backend_name="garnet",
            iqm_use_metrics=True,
            optimization_level=3,
            layout_method="sabre",
            routing_method="sabre",
        )
        benchmark_kwargs = benchmark_direct_basis_candidates.call_args.kwargs
        self.assertIs(benchmark_kwargs["transpiler_backend"], backend)
        self.assertIs(benchmark_kwargs["transpiler_metadata"], metadata)
        self.assertEqual(benchmark_kwargs["optimization_level"], 3)
        self.assertEqual(benchmark_kwargs["layout_method"], "sabre")
        self.assertEqual(benchmark_kwargs["routing_method"], "sabre")
        self.assertEqual(benchmark_kwargs["jobs"], 3)
        self.assertTrue(benchmark_kwargs["compare_optimal_f3_leakage"])
        self.assertIsNone(benchmark_kwargs["quantum_circuits_dir"])
        self.assertEqual(
            benchmark_kwargs["iqm_strategy_names"],
            (
                "preset_default",
                "preset_exact",
                "transpile_to_iqm_default",
                "transpile_to_iqm_exact",
            ),
        )
        self.assertEqual(benchmark_kwargs["ranking_workload"], "state_preparation")

    def test_main_uses_explicit_output_csv_override_for_iqm(self):
        backend = object()
        metadata = {
            "transpiler_backend": "iqm",
            "iqm_backend_name": "garnet",
        }
        candidates = [object()]

        with (
            patch("scripts.run_direct_basis_benchmarks._load_candidates", return_value=candidates),
            patch(
                "scripts.run_direct_basis_benchmarks.load_iqm_backend",
                return_value=backend,
            ),
            patch(
                "scripts.run_direct_basis_benchmarks.backend_metadata",
                return_value=metadata,
            ),
            patch(
                "scripts.run_direct_basis_benchmarks.timestamped_results_path",
                return_value="out.csv",
            ) as timestamped_results_path,
            patch(
                "scripts.run_direct_basis_benchmarks.benchmark_direct_basis_candidates",
                return_value=(None, "explicit.csv"),
            ) as benchmark_direct_basis_candidates,
        ):
            return_code = main(
                [
                    "--state",
                    "two_qutrit",
                    "--iqm-backend",
                    "garnet",
                    "--output-csv",
                    "explicit.csv",
                    "--ranking-workload",
                    "bell_measurements",
                    "--no-export-quantum-circuits",
                    "--no-fidelity",
                ]
            )

        self.assertEqual(return_code, 0)
        timestamped_results_path.assert_not_called()
        benchmark_kwargs = benchmark_direct_basis_candidates.call_args.kwargs
        self.assertEqual(benchmark_kwargs["output_csv"], "explicit.csv")
        self.assertEqual(benchmark_kwargs["ranking_workload"], "bell_measurements")
        self.assertFalse(benchmark_kwargs["compare_optimal_f3_leakage"])


if __name__ == "__main__":
    unittest.main()
