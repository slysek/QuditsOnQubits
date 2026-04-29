import contextlib
import io
import os
import shutil
import unittest
import warnings
from unittest.mock import patch

import numpy as np
import pandas as pd

from encoding_search_v2.candidates import (
    CandidateSearchConfig,
    E_OLD,
    filter_candidates_for_stage2,
    generate_stage1_candidates,
)
from encoding_search_v2.cli import main as cli_main
from encoding_search_v2.paths import stage_output_dir
from encoding_search_v2.preselection import load_preselected_candidates
from encoding_search_v2.reports import write_state_report
from encoding_search_v2.results import write_result_bundle
from encoding_search_v2.runner import (
    PipelineConfig,
    _build_benchmark_tasks,
    run_candidate_benchmarks,
    run_stage1,
    run_stage2,
)
from encoding_search_v2.states import parse_n_values, resolve_benchmark_state, star_graph_edges
from encoding_search_v2.triviality import (
    _filter_trivial_candidates,
    _is_baseline_equivalent_candidate,
    _is_embedding_equal_up_to_global_phase,
)


def _tmpdir():
    root = os.path.join(os.path.dirname(__file__), "_tmp_test_outputs")
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, "encoding_search_v2_test")
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=False)
    return path


def _sample_row(state_name, class_name, candidate_name, depth, twoq):
    return {
        "state_name": state_name,
        "class_name": class_name,
        "candidate_name": candidate_name,
        "status": "ok",
        "best_depth": depth,
        "mean_depth": float(depth),
        "std_depth": 0.0,
        "best_size": depth + twoq,
        "mean_size": float(depth + twoq),
        "best_two_qubit_gate_count": twoq,
        "mean_two_qubit_gate_count": float(twoq),
        "uses_old_codespace_only": class_name == "baseline",
        "avg_codeword_entanglement": 0.0,
        "overlap_with_old_codespace": 1.0,
        "fid085_best_fidelity": 0.91,
        "fid085_best_depth": max(depth - 1, 0),
        "fid085_best_two_qubit_gate_count": max(twoq - 1, 0),
        "fid090_best_fidelity": 0.93,
        "fid090_best_depth": depth,
        "fid090_best_two_qubit_gate_count": twoq,
        "fid095_best_fidelity": 0.97,
        "fid095_best_depth": depth + 1,
        "fid095_best_two_qubit_gate_count": twoq + 1,
    }


class TestEncodingSearchV2Candidates(unittest.TestCase):
    def test_stage1_default_pool_is_layered_and_bounded_when_requested(self):
        config = CandidateSearchConfig(
            max_monomial_full=3,
            max_product=2,
            include_product_grid=True,
            max_product_grid=1,
            include_near_identity=True,
            near_identity_samples_per_eps=1,
        )

        candidates = generate_stage1_candidates(config)
        names_by_class = {}
        for class_name, candidate_name, _ in candidates:
            names_by_class.setdefault(class_name, []).append(candidate_name)

        self.assertEqual(names_by_class["baseline"], ["E_old"])
        self.assertEqual(len(names_by_class["monomial_full"]), 3)
        self.assertEqual(len([n for n in names_by_class["product"] if not n.startswith("grid__")]), 2)
        self.assertEqual(len([n for n in names_by_class["product"] if n.startswith("grid__")]), 1)
        self.assertEqual(len(names_by_class["near_identity"]), 4)

    def test_stage2_filter_uses_stable_class_and_candidate_names(self):
        candidates = [
            ("baseline", "E_old", None),
            ("product", "U_I__V_I", None),
            ("product", "U_H__V_H", None),
        ]
        filtered = filter_candidates_for_stage2(
            candidates,
            {("baseline", "E_old"), ("product", "U_H__V_H")},
        )

        self.assertEqual(
            [(class_name, name) for class_name, name, _ in filtered],
            [("baseline", "E_old"), ("product", "U_H__V_H")],
        )


class TestEncodingSearchV2Triviality(unittest.TestCase):
    def test_embedding_equal_up_to_global_phase_detects_baseline_phase(self):
        phased = np.exp(0.37j) * E_OLD

        self.assertTrue(_is_embedding_equal_up_to_global_phase(phased, E_OLD))

    def test_baseline_equivalent_candidate_marks_exact_duplicate(self):
        meta = _is_baseline_equivalent_candidate(
            "product",
            "U_I__V_I",
            E_OLD.copy(),
        )

        self.assertFalse(meta.is_baseline_reference)
        self.assertTrue(meta.is_trivial_identity)
        self.assertTrue(meta.is_baseline_equivalent)
        self.assertIn("embedding", meta.skip_reason)

    def test_filter_trivial_candidates_preserves_baseline_and_skips_duplicate_only(self):
        nontrivial = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
                [0, 0, 1],
            ],
            dtype=complex,
        )
        candidates = [
            ("baseline", "E_old", None),
            ("product", "U_I__V_I", E_OLD.copy()),
            ("product", "global_phase", np.exp(0.2j) * E_OLD),
            ("monomial_full", "uses_11", nontrivial),
        ]

        kept, skipped = _filter_trivial_candidates(candidates, state_name="ghz3", stage=1)

        self.assertEqual(
            [(class_name, name) for class_name, name, _ in kept],
            [("baseline", "E_old"), ("monomial_full", "uses_11")],
        )
        self.assertEqual(
            {(row["class_name"], row["candidate_name"]) for row in skipped},
            {("product", "U_I__V_I"), ("product", "global_phase")},
        )
        self.assertTrue(all(row["is_baseline_equivalent"] for row in skipped))
        self.assertTrue(all(row["status"] == "skipped_baseline_equivalent" for row in skipped))


class TestEncodingSearchV2States(unittest.TestCase):
    def test_star_graph_edges_are_center_zero_to_all_leaves(self):
        self.assertEqual(star_graph_edges(5), [(0, 1), (0, 2), (0, 3), (0, 4)])

    def test_ghz3_spec_matches_star_graph_n3_edges(self):
        ghz3 = resolve_benchmark_state("ghz3")
        star3 = resolve_benchmark_state("ghz_star", n_qutrits=3)

        self.assertEqual(ghz3.num_qutrits, 3)
        self.assertEqual(ghz3.edges, [(0, 1), (0, 2)])
        self.assertEqual(star3.edges, ghz3.edges)
        self.assertEqual(star3.state_id, "ghz_star_3")

    def test_ghz_star_state_uses_n_in_stable_result_id(self):
        spec = resolve_benchmark_state("ghz_star", n_qutrits=5)

        self.assertEqual(spec.state_name, "ghz_star")
        self.assertEqual(spec.state_id, "ghz_star_5")
        self.assertEqual(spec.num_qutrits, 5)
        self.assertEqual(spec.graph_type, "star")

    def test_ghz_star_requires_valid_n(self):
        with self.assertRaises(ValueError):
            resolve_benchmark_state("ghz_star")
        with self.assertRaises(ValueError):
            resolve_benchmark_state("ghz_star", n_qutrits=1)

    def test_parse_n_values_accepts_comma_separated_ints(self):
        self.assertEqual(parse_n_values("3, 4,5"), (3, 4, 5))


class TestEncodingSearchV2Preselection(unittest.TestCase):
    def test_load_preselection_sorts_by_requested_metric_and_top_k(self):
        tmpdir = _tmpdir()
        try:
            csv_path = os.path.join(tmpdir, "stage1_results.csv")
            pd.DataFrame(
                [
                    _sample_row("ghz3", "baseline", "E_old", 50, 20),
                    _sample_row("ghz3", "product", "slow", 80, 10),
                    _sample_row("ghz3", "product", "fast", 40, 30),
                ]
            ).to_csv(csv_path, index=False)

            selected = load_preselected_candidates(
                csv_path,
                state_name="ghz3",
                top_k=2,
                rank_by="exact_depth",
            )

            self.assertEqual(
                selected,
                {("product", "fast"), ("baseline", "E_old")},
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_preselection_rejects_csv_for_wrong_state(self):
        tmpdir = _tmpdir()
        try:
            csv_path = os.path.join(tmpdir, "ghz3_only.csv")
            pd.DataFrame([_sample_row("ghz3", "baseline", "E_old", 50, 20)]).to_csv(
                csv_path,
                index=False,
            )

            with self.assertRaises(ValueError):
                load_preselected_candidates(
                    csv_path,
                    state_name="ame43",
                    top_k=1,
                    rank_by="exact_depth",
                )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_preselection_without_state_column_warns_but_loads(self):
        tmpdir = _tmpdir()
        try:
            csv_path = os.path.join(tmpdir, "top3_no_state.csv")
            pd.DataFrame(
                [{"class_name": "baseline", "candidate_name": "E_old", "best_depth": 1}]
            ).to_csv(csv_path, index=False)

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                selected = load_preselected_candidates(
                    csv_path,
                    state_name="two_qutrit",
                    top_k=1,
                    rank_by="exact_depth",
                )

            self.assertEqual(selected, {("baseline", "E_old")})
            self.assertIn("state_name", str(caught[0].message))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestEncodingSearchV2Results(unittest.TestCase):
    def test_result_bundle_writes_full_csv_rankings_and_top3_tables(self):
        tmpdir = _tmpdir()
        try:
            df = pd.DataFrame(
                [
                    _sample_row("two_qutrit", "baseline", "E_old", 31, 19),
                    _sample_row("two_qutrit", "product", "U_H__V_H", 28, 17),
                ]
            )

            paths = write_result_bundle(
                df,
                output_dir=tmpdir,
                file_prefix="stage1_two_qutrit",
                top_k=1,
                fidelity_thresholds=(0.85, 0.90, 0.95),
            )

            expected = {
                "results_csv",
                "top_by_depth_csv",
                "top_by_2q_csv",
                "baseline_reference_csv",
                "top_nontrivial_by_depth_csv",
                "skipped_baseline_equivalent_csv",
                "top3_by_depth_csv",
                "top3_by_2q_csv",
                "top3_exact_csv",
                "top3_fid085_csv",
                "top3_fid090_csv",
                "top3_fid095_csv",
            }
            self.assertTrue(expected.issubset(paths))
            for key in expected:
                self.assertTrue(os.path.exists(paths[key]), key)

            nontrivial = pd.read_csv(paths["top_nontrivial_by_depth_csv"])
            self.assertEqual(list(nontrivial["candidate_name"]), ["U_H__V_H"])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class _FakeExecutor:
    last_max_workers = None
    submitted = []

    def __init__(self, max_workers=None):
        type(self).last_max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def submit(self, fn, task):
        type(self).submitted.append(task["candidate_name"])
        return _FakeFuture(fn(task))


class _FakeFuture:
    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result


def _fake_as_completed(futures):
    for future in reversed(list(futures)):
        yield future


class TestEncodingSearchV2Runner(unittest.TestCase):
    def tearDown(self):
        _FakeExecutor.last_max_workers = None
        _FakeExecutor.submitted = []

    def test_stage_output_dir_is_inside_new_pipeline_folder(self):
        tmpdir = _tmpdir()
        try:
            path = stage_output_dir("ame43", 1, output_root=tmpdir)
            self.assertEqual(path, os.path.join(tmpdir, "ame43", "stage1"))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_stage1_writes_new_folder_results_without_old_pipeline_paths(self):
        tmpdir = _tmpdir()
        calls = []

        def fake_benchmark_basis(E_new, class_name, candidate_name, **kwargs):
            calls.append((class_name, candidate_name, kwargs["encoding_strategy"]))
            return _sample_row(kwargs["state_name"], class_name, candidate_name, len(calls), len(calls))

        try:
            config = PipelineConfig(
                state_name="two_qutrit",
                stage=1,
                output_root=tmpdir,
                jobs=1,
                n_transpile_runs=1,
                max_monomial_full=1,
                max_product=1,
            )

            with patch("encoding_search_v2.runner.benchmark_basis", side_effect=fake_benchmark_basis):
                df, paths = run_stage1(config)

            self.assertFalse(df.empty)
            self.assertEqual(
                [(class_name, candidate_name) for class_name, candidate_name, _ in calls],
                [("baseline", "E_old")],
            )
            self.assertEqual({strategy for _, _, strategy in calls}, {"append_w"})
            skipped = df[df["status"] == "skipped_baseline_equivalent"]
            self.assertEqual(
                set(skipped["candidate_name"]),
                {"sup012_P012_ph000", "U_I__V_I"},
            )
            self.assertTrue(bool(df.loc[df["candidate_name"] == "E_old", "is_baseline_reference"].iloc[0]))
            self.assertTrue(paths["results_csv"].startswith(os.path.join(tmpdir, "two_qutrit", "stage1")))
            self.assertTrue(os.path.exists(paths["results_csv"]))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_stage1_for_ghz_star_uses_state_id_and_passes_n_to_benchmark(self):
        tmpdir = _tmpdir()
        calls = []

        def fake_benchmark_basis(E_new, class_name, candidate_name, **kwargs):
            calls.append(kwargs)
            return _sample_row(kwargs["state_name"], class_name, candidate_name, 10, 5)

        try:
            config = PipelineConfig(
                state_name="ghz_star",
                n_qutrits=5,
                stage=1,
                output_root=tmpdir,
                jobs=1,
                n_transpile_runs=1,
                limit_candidates=1,
            )

            with patch("encoding_search_v2.runner.benchmark_basis", side_effect=fake_benchmark_basis):
                df, paths = run_stage1(config)

            self.assertEqual(calls[0]["state_name"], "ghz_star_5")
            self.assertEqual(calls[0]["n_qutrits"], 5)
            self.assertEqual(set(df["state_name"]), {"ghz_star_5"})
            self.assertTrue(paths["results_csv"].startswith(os.path.join(tmpdir, "ghz_star_5", "stage1")))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_stage2_requires_ranking_csv(self):
        tmpdir = _tmpdir()
        try:
            config = PipelineConfig(state_name="ghz3", stage=2, output_root=tmpdir)
            with self.assertRaises(ValueError):
                run_stage2(config)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_stage2_warns_when_preselected_candidate_is_not_regenerated(self):
        tmpdir = _tmpdir()
        try:
            ranking_csv = os.path.join(tmpdir, "stage1_top.csv")
            pd.DataFrame(
                [
                    _sample_row("ghz3", "mystery_class", "not_in_generator", 1, 1),
                ]
            ).to_csv(ranking_csv, index=False)
            config = PipelineConfig(
                state_name="ghz3",
                stage=2,
                output_root=tmpdir,
                ranking_csv=ranking_csv,
                top_k=1,
            )

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                df, paths = run_stage2(config)

            self.assertTrue(df.empty)
            self.assertTrue(os.path.exists(paths["results_csv"]))
            self.assertTrue(any("not regenerated" in str(item.message) for item in caught))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_stage2_does_not_benchmark_preselected_baseline_equivalent_candidate(self):
        tmpdir = _tmpdir()
        calls = []

        def fake_benchmark_basis(E_new, class_name, candidate_name, **kwargs):
            calls.append((class_name, candidate_name))
            return _sample_row(kwargs["state_name"], class_name, candidate_name, 10, 5)

        try:
            ranking_csv = os.path.join(tmpdir, "stage1_results.csv")
            pd.DataFrame(
                [
                    _sample_row("ghz3", "baseline", "E_old", 50, 20),
                    _sample_row("ghz3", "monomial_full", "sup012_P012_ph000", 40, 10),
                ]
            ).to_csv(ranking_csv, index=False)

            config = PipelineConfig(
                state_name="ghz3",
                stage=2,
                output_root=tmpdir,
                ranking_csv=ranking_csv,
                top_k=2,
                max_monomial_full=1,
            )

            with patch("encoding_search_v2.runner.benchmark_basis", side_effect=fake_benchmark_basis):
                df, paths = run_stage2(config)

            self.assertEqual(calls, [("baseline", "E_old")])
            skipped = df[df["status"] == "skipped_baseline_equivalent"]
            self.assertEqual(list(skipped["candidate_name"]), ["sup012_P012_ph000"])
            self.assertTrue(os.path.exists(paths["skipped_baseline_equivalent_csv"]))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_stage2_tasks_use_prepared_w_strategy(self):
        tmpdir = _tmpdir()
        try:
            config = PipelineConfig(
                state_name="ame43",
                stage=2,
                output_root=tmpdir,
                export_circuits=True,
            )
            tasks = _build_benchmark_tasks(
                [("baseline", "E_old", None)],
                config,
                encoding_strategy="prepared_w_then_conjugated_entanglers",
            )

            self.assertEqual(tasks[0]["state_name"], "ame43")
            self.assertEqual(
                tasks[0]["encoding_strategy"],
                "prepared_w_then_conjugated_entanglers",
            )
            self.assertIn(os.path.join("ame43", "stage2", "circuits"), tasks[0]["circuits_output_dir"])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_benchmark_tasks_for_ghz_star_include_resolved_state_and_n(self):
        tmpdir = _tmpdir()
        try:
            config = PipelineConfig(
                state_name="ghz_star",
                n_qutrits=4,
                stage=2,
                output_root=tmpdir,
                export_circuits=True,
            )
            tasks = _build_benchmark_tasks(
                [("baseline", "E_old", None)],
                config,
                encoding_strategy="prepared_w_then_conjugated_entanglers",
            )

            self.assertEqual(tasks[0]["state_name"], "ghz_star_4")
            self.assertEqual(tasks[0]["n_qutrits"], 4)
            self.assertEqual(tasks[0]["state_family"], "ghz_star")
            self.assertIn(os.path.join("ghz_star_4", "stage2", "circuits"), tasks[0]["circuits_output_dir"])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestEncodingSearchV2Report(unittest.TestCase):
    def test_report_compares_baseline_against_nontrivial_candidates_only(self):
        tmpdir = _tmpdir()
        try:
            stage1_dir = os.path.join(tmpdir, "ghz3", "stage1")
            os.makedirs(stage1_dir, exist_ok=True)
            csv_path = os.path.join(stage1_dir, "encoding_search_v2_ghz3_stage1_results.csv")
            skipped = _sample_row("ghz3", "product", "U_I__V_I", 1, 1)
            skipped.update(
                {
                    "status": "skipped_baseline_equivalent",
                    "is_baseline_reference": False,
                    "is_baseline_equivalent": True,
                    "is_trivial_identity": True,
                    "skip_reason": "same embedding as baseline",
                }
            )
            baseline = _sample_row("ghz3", "baseline", "E_old", 50, 20)
            baseline.update(
                {
                    "is_baseline_reference": True,
                    "is_baseline_equivalent": True,
                    "is_trivial_identity": True,
                    "skip_reason": "",
                }
            )
            nontrivial = _sample_row("ghz3", "product", "U_H__V_H", 60, 25)
            nontrivial.update(
                {
                    "is_baseline_reference": False,
                    "is_baseline_equivalent": False,
                    "is_trivial_identity": False,
                    "skip_reason": "",
                }
            )
            pd.DataFrame([skipped, baseline, nontrivial]).to_csv(csv_path, index=False)

            report_path = write_state_report("ghz3", output_root=tmpdir)

            with open(report_path, "r", encoding="utf-8") as handle:
                content = handle.read()

            self.assertIn("Best nontrivial stage 1: product / U_H__V_H", content)
            self.assertIn("Stage 1 nontrivial better than baseline: no better basis found", content)
            self.assertIn("Skipped baseline-equivalent candidates: 1", content)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_parallel_runner_uses_requested_job_count_and_preserves_result_order(self):
        tmpdir = _tmpdir()

        def fake_worker(task):
            return _sample_row(
                task["state_name"],
                task["class_name"],
                task["candidate_name"],
                1 if task["candidate_name"] == "first" else 2,
                1,
            )

        try:
            config = PipelineConfig(
                state_name="ghz3",
                stage=1,
                output_root=tmpdir,
                jobs=32,
                n_transpile_runs=1,
            )
            candidates = [
                ("baseline", "first", None),
                ("baseline", "second", None),
            ]

            with patch("encoding_search_v2.runner.ProcessPoolExecutor", _FakeExecutor), patch(
                "encoding_search_v2.runner.as_completed",
                side_effect=_fake_as_completed,
            ), patch("encoding_search_v2.runner._benchmark_candidate_worker", side_effect=fake_worker):
                rows = run_candidate_benchmarks(candidates, config, encoding_strategy="append_w")

            self.assertEqual(_FakeExecutor.last_max_workers, 32)
            self.assertEqual(_FakeExecutor.submitted, ["first", "second"])
            self.assertEqual([row["candidate_name"] for row in rows], ["first", "second"])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestEncodingSearchV2Cli(unittest.TestCase):
    def test_dry_run_accepts_ghz_star_n_values_range(self):
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            code = cli_main(
                [
                    "--state",
                    "ghz_star",
                    "--stage",
                    "1",
                    "--n-values",
                    "3,4",
                    "--limit-candidates",
                    "1",
                    "--dry-run",
                ]
            )

        self.assertEqual(code, 0)
        output = stream.getvalue()
        self.assertIn("Dry run [ghz_star_3, stage 1]", output)
        self.assertIn("Dry run [ghz_star_4, stage 1]", output)


if __name__ == "__main__":
    unittest.main()
