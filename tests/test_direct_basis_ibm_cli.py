import contextlib
import io
import unittest
from unittest.mock import patch

import pandas as pd

from scripts.run_direct_basis_benchmarks import main


class IBMCLItests(unittest.TestCase):
    def test_ibm_routes_to_complete_workload(self):
        with patch("scripts.run_direct_basis_benchmarks.load_ibm_backend", return_value=object()), patch(
            "scripts.run_direct_basis_benchmarks.ibm_backend_metadata", return_value={"transpiler_backend": "ibm"}
        ), patch("scripts.run_direct_basis_benchmarks._load_candidates", return_value=[object()]), patch(
            "scripts.run_direct_basis_benchmarks.benchmark_direct_basis_candidates", return_value=(pd.DataFrame(), "out.csv")
        ) as benchmark, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--state", "two_qutrit", "--ibm-backend", "ibm_test",
                                   "--output-csv", "out.csv", "--initial-layout", "4,3,2,1"]), 0)
        self.assertEqual(benchmark.call_args.kwargs["transpiler_provider"], "ibm")
        self.assertEqual(benchmark.call_args.kwargs["ranking_workload"], "bell_measurements")
        self.assertEqual(benchmark.call_args.kwargs["initial_layout"], (4, 3, 2, 1))

    def test_conflicting_targets_fail_before_loading(self):
        with patch("scripts.run_direct_basis_benchmarks._load_candidates") as candidates, contextlib.redirect_stderr(io.StringIO()):
            for extra in (["--iqm-backend", "garnet"], ["--local-line-coupling"], ["--iqm-use-metrics"]):
                with self.assertRaises(SystemExit):
                    main(["--state", "two_qutrit", "--ibm-backend", "ibm_test", *extra])
            candidates.assert_not_called()
