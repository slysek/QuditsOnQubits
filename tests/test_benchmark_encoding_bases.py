import os
import shutil
import tempfile
import unittest

import pandas as pd
from igraph import Graph
from unittest.mock import patch

from qiskit import qpy

from QuditsOnQubits.create_ame_circuit import create_ame_circuit
from QuditsOnQubits.benchmark_encoding_bases import (
    benchmark_basis,
    write_multi_state_benchmark_report,
    _run_single_state_benchmark,
    _build_state_circuit,
)
from QuditsOnQubits.project_paths import (
    benchmark_state_results_path,
    multi_state_benchmark_report_path,
    benchmark_results_path,
)


def _workspace_tempdir():
    return tempfile.mkdtemp(prefix="bench_test_")


def _sample_report_frame(state_name):
    return pd.DataFrame(
        [
            {
                "state_name": state_name,
                "class_name": "baseline",
                "candidate_name": "E_old",
                "status": "ok",
                "best_depth": 47,
                "mean_depth": 53.9,
                "std_depth": 3.2,
                "best_size": 141,
                "best_two_qubit_gate_count": 32,
                "mean_two_qubit_gate_count": 33.05,
                "uses_old_codespace_only": True,
                "avg_codeword_entanglement": 0.0,
                "overlap_with_old_codespace": 1.0,
                "fid085_best_approx_degree": 0.91,
                "fid085_best_fidelity": 0.8786,
                "fid085_best_depth": 41,
                "fid085_best_two_qubit_gate_count": 28,
                "fid090_best_approx_degree": 0.95,
                "fid090_best_fidelity": 0.9098,
                "fid090_best_depth": 43,
                "fid090_best_two_qubit_gate_count": 30,
                "fid095_best_approx_degree": 0.99,
                "fid095_best_fidelity": 0.9909,
                "fid095_best_depth": 46,
                "fid095_best_two_qubit_gate_count": 31,
            }
        ]
    )


class TestStateAwarePaths(unittest.TestCase):
    def test_ghz3_path_matches_legacy_path(self):
        self.assertEqual(
            benchmark_state_results_path("ghz3", "full"),
            benchmark_results_path("full"),
        )

    def test_two_qutrit_path_contains_state_name(self):
        path = benchmark_state_results_path("two_qutrit", "full")
        self.assertTrue(
            path.endswith(
                os.path.join(
                    "data",
                    "benchmarks",
                    "benchmark_encoding_bases_two_qutrit_full_results.csv",
                )
            )
        )

    def test_ame43_path_contains_state_name(self):
        path = benchmark_state_results_path("ame43", "original")
        self.assertTrue(
            path.endswith(
                os.path.join(
                    "data",
                    "benchmarks",
                    "benchmark_encoding_bases_ame43_original_results.csv",
                )
            )
        )

    def test_multi_state_report_path(self):
        path = multi_state_benchmark_report_path()
        self.assertTrue(
            path.endswith(
                os.path.join(
                    "docs",
                    "benchmarks",
                    "benchmark_encoding_bases_multi_state_analysis.md",
                )
            )
        )

    def test_unknown_state_raises(self):
        with self.assertRaises(ValueError):
            benchmark_state_results_path("bogus", "full")


class TestBuildStateCircuit(unittest.TestCase):
    def test_two_qutrit_builds_4_qubit_circuit(self):
        qc, g = _build_state_circuit("two_qutrit", E_new=None)
        self.assertEqual(qc.num_qubits, 4)
        self.assertEqual(g.vcount(), 2)

    def test_ghz3_builds_6_qubit_circuit(self):
        qc, g = _build_state_circuit("ghz3", E_new=None)
        self.assertEqual(qc.num_qubits, 6)
        self.assertEqual(g.vcount(), 3)

    def test_ame43_builds_8_qubit_circuit(self):
        qc, g = _build_state_circuit("ame43", E_new=None)
        self.assertEqual(qc.num_qubits, 8)
        self.assertEqual(g.vcount(), 4)

    def test_unknown_state_raises(self):
        with self.assertRaises(ValueError):
            _build_state_circuit("nonsense", E_new=None)


class TestBenchmarkBasisStateAware(unittest.TestCase):
    def test_two_qutrit_saves_circuit_under_state_folder(self):
        tmpdir = _workspace_tempdir()
        try:
            row = benchmark_basis(
                E_new=None,
                class_name="baseline",
                candidate_name="E_old",
                state_name="two_qutrit",
                coupling_map=[[0, 1], [1, 2], [2, 3]],
                approximation_values=[1.0],
                fidelity_thresholds=(0.95,),
                n_transpile_runs=1,
                circuits_output_dir=tmpdir,
            )

            saved_path = os.path.join(tmpdir, "two_qutrit", "baseline", "E_old.qpy")
            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["state_name"], "two_qutrit")
            self.assertTrue(os.path.exists(saved_path))

            with open(saved_path, "rb") as fd:
                saved_qc = qpy.load(fd)[0]
            self.assertEqual(saved_qc.num_qubits, 4)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_ame43_saves_circuit_under_state_folder(self):
        tmpdir = _workspace_tempdir()
        try:
            row = benchmark_basis(
                E_new=None,
                class_name="baseline",
                candidate_name="E_old",
                state_name="ame43",
                coupling_map=[[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]],
                approximation_values=[1.0],
                fidelity_thresholds=(0.95,),
                n_transpile_runs=1,
                circuits_output_dir=tmpdir,
            )

            saved_path = os.path.join(tmpdir, "ame43", "baseline", "E_old.qpy")
            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["state_name"], "ame43")
            self.assertTrue(os.path.exists(saved_path))

            with open(saved_path, "rb") as fd:
                saved_qc = qpy.load(fd)[0]
            self.assertEqual(saved_qc.num_qubits, 8)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_ghz3_backward_compat(self):
        tmpdir = _workspace_tempdir()
        try:
            row = benchmark_basis(
                E_new=None,
                class_name="baseline",
                candidate_name="E_old",
                state_name="ghz3",
                coupling_map=[[i, i + 1] for i in range(5)],
                approximation_values=[1.0],
                fidelity_thresholds=(0.95,),
                n_transpile_runs=1,
                circuits_output_dir=tmpdir,
            )
            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["state_name"], "ghz3")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestMarkdownReport(unittest.TestCase):
    def test_report_contains_all_sections_and_fidelity_columns(self):
        frames = {
            "two_qutrit": _sample_report_frame("two_qutrit"),
            "ghz3": _sample_report_frame("ghz3"),
            "ame43": _sample_report_frame("ame43"),
        }

        tmpdir = _workspace_tempdir()
        try:
            report_path = os.path.join(tmpdir, "subdir", "report.md")
            write_multi_state_benchmark_report(frames, report_path)

            with open(report_path, "r", encoding="utf-8") as fd:
                content = fd.read()

            self.assertIn("# Multi-State Encoding Benchmark Analysis", content)
            self.assertIn("## two_qutrit", content)
            self.assertIn("## ghz3", content)
            self.assertIn("## ame43", content)
            self.assertIn("## Cross-state comparison", content)
            self.assertIn("fid085 depth", content)
            self.assertIn("fid090 depth", content)
            self.assertIn("fid095 depth", content)
            self.assertIn("fid085 2Q", content)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
