from __future__ import annotations

import os
import unittest

from scripts.run_direct_basis_benchmarks import (
    _default_results_prefix,
    _iqm_quantum_circuits_dir_from_args,
    build_parser,
)


class DirectBasisIqmCliTests(unittest.TestCase):
    def test_iqm_backend_flag_defaults_iqm_options(self):
        args = build_parser().parse_args(["--iqm-backend", "garnet"])

        self.assertEqual(args.iqm_backend, "garnet")
        self.assertIs(args.iqm_use_metrics, False)
        self.assertIsNone(args.layout_method)
        self.assertIsNone(args.routing_method)

    def test_iqm_use_metrics_parses_true(self):
        args = build_parser().parse_args(["--iqm-backend", "garnet", "--iqm-use-metrics"])

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
        args = build_parser().parse_args(["--iqm-backend", "garnet"])

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


if __name__ == "__main__":
    unittest.main()
