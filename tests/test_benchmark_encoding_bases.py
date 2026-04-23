import os
import shutil
import tempfile
import unittest
import uuid

import numpy as np
import pandas as pd
from igraph import Graph
from unittest.mock import patch

from qiskit import qpy

from QuditsOnQubits.create_ame_circuit import create_ame_circuit
from QuditsOnQubits.benchmark_encoding_bases import (
    E_OLD,
    P_OLD_CODESPACE,
    benchmark_basis,
    generate_monomial_full_bases,
    generate_monomial_old_codespace_bases,
    generate_product_bases,
    write_multi_state_benchmark_report,
    _single_qubit_product_library,
    _run_single_state_benchmark,
    _build_state_circuit,
    _load_preselected_candidates,
    _filter_candidates_by_preselection,
    _validate_preselection_coverage,
    _write_topk_tables_to_output_dir,
    _get_ame43_graph,
    _get_cached_approximation_pass_manager,
)
from QuditsOnQubits.project_paths import (
    benchmark_state_results_path,
    multi_state_benchmark_report_path,
    benchmark_results_path,
    prepared_w_benchmark_data_dir,
    prepared_w_benchmark_results_path,
)


def _workspace_tempdir():
    root = os.path.join(os.path.dirname(__file__), "_tmp_test_outputs")
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, f"bench_test_{uuid.uuid4().hex}")
    os.makedirs(path, exist_ok=False)
    return path


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

    def test_ame43_reuses_cached_graph_instance(self):
        _, graph_a = _build_state_circuit("ame43", E_new=None)
        _, graph_b = _build_state_circuit("ame43", E_new=None)

        self.assertIs(graph_a, graph_b)
        self.assertIs(graph_a, _get_ame43_graph())

    def test_unknown_state_raises(self):
        with self.assertRaises(ValueError):
            _build_state_circuit("nonsense", E_new=None)


class TestBenchmarkCachingHelpers(unittest.TestCase):
    def test_cached_approximation_pass_manager_reuses_same_instance(self):
        pm_a = _get_cached_approximation_pass_manager(tuple(["cz", "rx", "rz"]), 1.0, 7)
        pm_b = _get_cached_approximation_pass_manager(tuple(["cz", "rx", "rz"]), 1.0, 7)

        self.assertIs(pm_a, pm_b)


class TestBenchmarkBasisStateAware(unittest.TestCase):
    def test_two_qutrit_saves_raw_and_top3_transpiled_circuits_under_state_folder(self):
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
                n_transpile_runs=4,
                circuits_output_dir=tmpdir,
            )

            class_dir = os.path.join(tmpdir, "two_qutrit", "baseline")
            saved_path = os.path.join(class_dir, "E_old.qpy")
            transpiled_paths = [
                os.path.join(class_dir, f"E_old__transpiled_{rank}.qpy")
                for rank in range(1, 4)
            ]
            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["state_name"], "two_qutrit")
            self.assertTrue(os.path.exists(saved_path))
            for path in transpiled_paths:
                self.assertTrue(os.path.exists(path))

            with open(saved_path, "rb") as fd:
                saved_qc = qpy.load(fd)[0]
            self.assertEqual(saved_qc.num_qubits, 4)

            with open(transpiled_paths[0], "rb") as fd:
                saved_transpiled_qc = qpy.load(fd)[0]
            self.assertEqual(saved_transpiled_qc.num_qubits, 4)
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

    def test_nonbaseline_candidate_saves_w_circuit_next_to_exported_benchmarks(self):
        tmpdir = _workspace_tempdir()
        try:
            class_name, candidate_name, e_new = generate_monomial_old_codespace_bases(
                max_candidates=1
            )[0]

            row = benchmark_basis(
                E_new=e_new,
                class_name=class_name,
                candidate_name=candidate_name,
                state_name="two_qutrit",
                coupling_map=[[0, 1], [1, 2], [2, 3]],
                approximation_values=[1.0],
                fidelity_thresholds=(0.95,),
                n_transpile_runs=3,
                circuits_output_dir=tmpdir,
            )

            class_dir = os.path.join(tmpdir, "two_qutrit", class_name)
            w_path = os.path.join(class_dir, f"{candidate_name}__W.qpy")

            self.assertEqual(row["status"], "ok")
            self.assertTrue(os.path.exists(w_path))

            with open(w_path, "rb") as fd:
                w_qc = qpy.load(fd)[0]

            self.assertEqual(w_qc.num_qubits, 2)
            self.assertEqual([instruction.operation.label for instruction in w_qc.data], ["W"])
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


class TestMonomialGenerators(unittest.TestCase):
    def test_old_codespace_candidates_stay_in_old_codespace(self):
        candidates = generate_monomial_old_codespace_bases(max_candidates=None)

        self.assertEqual(len(candidates), 162)
        self.assertEqual(
            {class_name for class_name, _, _ in candidates},
            {"monomial_old_codespace"},
        )

        for _, _, E_new in candidates:
            self.assertTrue(np.allclose(P_OLD_CODESPACE @ E_new, E_new))

    def test_full_monomial_candidates_include_supports_with_state_11(self):
        candidates = generate_monomial_full_bases(max_candidates=None)

        self.assertEqual(
            {class_name for class_name, _, _ in candidates},
            {"monomial_full"},
        )

        supports = {
            tuple(np.flatnonzero(np.linalg.norm(E_new, axis=1) > 1e-10))
            for _, _, E_new in candidates
        }

        self.assertTrue(any(3 in support for support in supports))

    def test_all_generated_monomial_candidates_are_isometries(self):
        candidates = (
            generate_monomial_old_codespace_bases(max_candidates=None)
            + generate_monomial_full_bases(max_candidates=None)
        )

        for _, _, E_new in candidates:
            gram = E_new.conj().T @ E_new
            self.assertTrue(np.allclose(gram, np.eye(3), atol=1e-12))

    def test_full_monomial_candidate_count_without_limit(self):
        candidates = generate_monomial_full_bases(max_candidates=None)
        self.assertEqual(len(candidates), 4 * 6 * 27)

    def test_old_codespace_monomial_candidate_count_without_limit(self):
        candidates = generate_monomial_old_codespace_bases(max_candidates=None)
        self.assertEqual(len(candidates), 6 * 27)


class TestProductGenerators(unittest.TestCase):
    def test_discrete_product_candidates_have_expected_shape_and_are_isometries(self):
        candidates = generate_product_bases(max_candidates=None)

        self.assertEqual(
            {class_name for class_name, _, _ in candidates},
            {"product"},
        )

        for _, _, E_new in candidates:
            self.assertEqual(E_new.shape, (4, 3))
            gram = E_new.conj().T @ E_new
            self.assertTrue(np.allclose(gram, np.eye(3), atol=1e-12))

    def test_discrete_product_includes_identity_candidate_equal_to_e_old(self):
        candidates = generate_product_bases(max_candidates=None)
        identity_candidates = [
            E_new
            for _, candidate_name, E_new in candidates
            if candidate_name == "U_I__V_I"
        ]

        self.assertEqual(len(identity_candidates), 1)
        self.assertTrue(np.allclose(identity_candidates[0], E_OLD))

    def test_discrete_product_candidate_count_matches_library_squared(self):
        library_size = len(_single_qubit_product_library())
        candidates = generate_product_bases(max_candidates=None)
        limited_candidates = generate_product_bases(max_candidates=7)

        self.assertEqual(len(candidates), library_size * library_size)
        self.assertEqual(len(limited_candidates), 7)

    def test_grid_product_mode_accepts_numpy_angle_grid(self):
        candidates = generate_product_bases(
            max_candidates=1,
            mode="grid",
            angle_grid={
                "phase_angles": np.array([0.0]),
                "polar_angles": np.array([0.0]),
            },
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0][0], "product")
        self.assertEqual(candidates[0][1], "grid__U_a0.00_b0.00_g0.00__V_a0.00_b0.00_g0.00")
        self.assertTrue(np.allclose(candidates[0][2], E_OLD))

    def test_single_state_benchmark_can_filter_only_product_class(self):
        tmpdir = _workspace_tempdir()
        csv_path = os.path.join(tmpdir, "product_only.csv")
        circuits_output_dir = os.path.join(tmpdir, "circuits")

        def fake_benchmark_basis(E_new, class_name, candidate_name, state_name, **kwargs):
            return {
                "state_name": state_name,
                "class_name": class_name,
                "candidate_name": candidate_name,
                "status": "ok",
                "best_depth": 1,
                "mean_depth": 1.0,
                "best_two_qubit_gate_count": 0,
                "avg_codeword_entanglement": 0.0,
                "uses_old_codespace_only": bool(
                    np.allclose(P_OLD_CODESPACE @ E_new, E_new, atol=1e-12)
                ),
            }

        try:
            with patch(
                "QuditsOnQubits.benchmark_encoding_bases.benchmark_basis",
                side_effect=fake_benchmark_basis,
            ), patch(
                "QuditsOnQubits.benchmark_encoding_bases._print_single_state_summary"
            ):
                df, saved_csv_path = _run_single_state_benchmark(
                    state_name="two_qutrit",
                    n_transpile_runs=1,
                    csv_path=csv_path,
                    mode="extended",
                    circuits_output_dir=circuits_output_dir,
                    approximation_values=[1.0],
                    fidelity_thresholds=(0.95,),
                    class_filter="product",
                )

            self.assertEqual(saved_csv_path, csv_path)
            self.assertTrue(os.path.exists(csv_path))
            self.assertFalse(df.empty)
            self.assertEqual(set(df["class_name"]), {"product"})
            self.assertEqual(
                len(df),
                len(generate_product_bases(max_candidates=None)),
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestPreparedWPaths(unittest.TestCase):
    def test_prepared_w_data_dir_is_under_benchmarks(self):
        path = prepared_w_benchmark_data_dir()
        self.assertIn("prepared_w_then_conjugated_entanglers_results", path)
        self.assertIn(os.path.join("data", "benchmarks"), path)

    def test_prepared_w_results_path_contains_state_and_mode(self):
        path = prepared_w_benchmark_results_path("ghz3", "full")
        self.assertIn("ghz3", path)
        self.assertIn("full", path)
        self.assertIn("prepared_w", path)


class TestLoadPreselectedCandidates(unittest.TestCase):
    def test_load_valid_csv(self):
        tmpdir = _workspace_tempdir()
        try:
            csv_path = os.path.join(tmpdir, "top3.csv")
            pd.DataFrame([
                {"class_name": "baseline", "candidate_name": "E_old"},
                {"class_name": "product", "candidate_name": "U_I__V_I"},
            ]).to_csv(csv_path, index=False)

            result = _load_preselected_candidates(csv_path)
            self.assertEqual(result, {
                ("baseline", "E_old"),
                ("product", "U_I__V_I"),
            })
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_load_csv_with_whitespace_in_columns(self):
        tmpdir = _workspace_tempdir()
        try:
            csv_path = os.path.join(tmpdir, "top3_ws.csv")
            with open(csv_path, "w") as f:
                f.write("class_name            , candidate_name        , best_depth\n")
                f.write("baseline              , E_old                 , 47\n")

            result = _load_preselected_candidates(csv_path)
            self.assertEqual(result, {("baseline", "E_old")})
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_load_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            _load_preselected_candidates("/nonexistent/path/file.csv")

    def test_load_csv_missing_column_raises(self):
        tmpdir = _workspace_tempdir()
        try:
            csv_path = os.path.join(tmpdir, "bad.csv")
            pd.DataFrame([{"class_name": "baseline", "score": 42}]).to_csv(csv_path, index=False)

            with self.assertRaises(ValueError):
                _load_preselected_candidates(csv_path)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestFilterCandidatesByPreselection(unittest.TestCase):
    def test_filters_correctly(self):
        candidates = [
            ("baseline", "E_old", None),
            ("product", "U_I__V_I", None),
            ("product", "U_SX__V_SXdg", None),
        ]
        preselected = {("baseline", "E_old"), ("product", "U_SX__V_SXdg")}
        filtered = _filter_candidates_by_preselection(candidates, preselected)

        self.assertEqual(len(filtered), 2)
        self.assertEqual(
            {(cls, name) for cls, name, _ in filtered},
            preselected,
        )

    def test_empty_preselection_gives_empty(self):
        candidates = [("baseline", "E_old", None)]
        filtered = _filter_candidates_by_preselection(candidates, set())
        self.assertEqual(len(filtered), 0)


class TestValidatePreselectionCoverage(unittest.TestCase):
    def test_warns_on_missing_candidates(self):
        preselected = {("baseline", "E_old"), ("mystery", "unknown")}
        filtered = [("baseline", "E_old", None)]

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _validate_preselection_coverage(preselected, filtered, "fake.csv")
            self.assertEqual(len(w), 1)
            self.assertIn("mystery", str(w[0].message))

    def test_no_warning_when_all_found(self):
        preselected = {("baseline", "E_old")}
        filtered = [("baseline", "E_old", None)]

        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _validate_preselection_coverage(preselected, filtered, "fake.csv")
            self.assertEqual(len(w), 0)


class TestWriteTopkTables(unittest.TestCase):
    def test_creates_output_files(self):
        tmpdir = _workspace_tempdir()
        try:
            df = _sample_report_frame("ghz3")
            _write_topk_tables_to_output_dir(
                df, tmpdir, "test_prefix", fidelity_thresholds=(0.95,),
            )

            expected_main = os.path.join(tmpdir, "test_prefix_results.csv")
            self.assertTrue(os.path.exists(expected_main))

            top3_exact = os.path.join(tmpdir, "test_prefix_results_top3_exact.csv")
            self.assertTrue(os.path.exists(top3_exact))

            top3_depth = os.path.join(tmpdir, "test_prefix_results_top3_by_depth.csv")
            self.assertTrue(os.path.exists(top3_depth))

            top3_twoq = os.path.join(tmpdir, "test_prefix_results_top3_by_2q.csv")
            self.assertTrue(os.path.exists(top3_twoq))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestPreparedWRequiresPreselection(unittest.TestCase):
    def test_run_benchmark_prepared_w_without_file_raises(self):
        from QuditsOnQubits.benchmark_encoding_bases import run_benchmark

        with self.assertRaises(ValueError) as ctx:
            run_benchmark(
                encoding_strategy="prepared_w_then_conjugated_entanglers",
                preselected_candidates_file=None,
                state_name="ghz3",
            )
        self.assertIn("preselected", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
