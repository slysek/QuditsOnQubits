from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from qudits_on_qubits.benchmarks.direct_basis.benchmark import benchmark_direct_basis_candidates
from qudits_on_qubits.benchmarks.direct_basis.candidates import DirectBasisCandidate
from qudits_on_qubits.benchmarks.direct_basis.selection import (
    DEFAULT_APPROXIMATION_THRESHOLDS,
    SUPPORTED_BELL_STATES,
    SelectionConfig,
    materialize_selected_artifacts,
    parse_approximation_thresholds,
    require_supported_bell_state,
    select_top_k,
    selection_label,
    threshold_from_label,
    transpiled_qpy_filename,
)
from scripts.run_direct_basis_benchmarks import build_parser, _validate_cli_selection_args


class DirectBasisSelectionLabelTests(unittest.TestCase):
    def test_default_thresholds_match_pipeline_decision(self):
        self.assertEqual(DEFAULT_APPROXIMATION_THRESHOLDS, (0.99, 0.95, 0.90))

    def test_supported_states_are_bell_pipeline_states(self):
        self.assertEqual(SUPPORTED_BELL_STATES, ("two_qutrit", "ghz3", "ame43"))
        self.assertEqual(require_supported_bell_state("ghz3"), "ghz3")
        with self.assertRaisesRegex(ValueError, "Bell selected-circuit pipeline"):
            require_supported_bell_state("path5")

    def test_selection_label_formats_exact_and_thresholds(self):
        self.assertEqual(selection_label(None), "exact")
        self.assertEqual(selection_label(0.99), "fid099")
        self.assertEqual(selection_label(0.95), "fid095")
        self.assertEqual(selection_label(0.90), "fid090")

    def test_threshold_from_label_round_trips(self):
        self.assertIsNone(threshold_from_label("exact"))
        self.assertAlmostEqual(threshold_from_label("fid099"), 0.99)
        self.assertAlmostEqual(threshold_from_label("fid095"), 0.95)
        self.assertAlmostEqual(threshold_from_label("fid090"), 0.90)
        with self.assertRaisesRegex(ValueError, "selection label"):
            threshold_from_label("bad")

    def test_parse_approximation_thresholds(self):
        self.assertEqual(parse_approximation_thresholds(None), ())
        self.assertEqual(parse_approximation_thresholds(""), ())
        self.assertEqual(parse_approximation_thresholds("0.99,0.95,0.90"), (0.99, 0.95, 0.90))
        self.assertEqual(parse_approximation_thresholds(" 0.99 , 0.95 "), (0.99, 0.95))
        with self.assertRaisesRegex(ValueError, "between 0 and 1"):
            parse_approximation_thresholds("1.2")

    def test_transpiled_qpy_filenames(self):
        self.assertEqual(
            transpiled_qpy_filename("exact", legacy_exact=True),
            "graph_state_direct_basis_transpiled.qpy",
        )
        self.assertEqual(
            transpiled_qpy_filename("exact", legacy_exact=False),
            "graph_state_direct_basis_transpiled_exact.qpy",
        )
        self.assertEqual(
            transpiled_qpy_filename("fid095", legacy_exact=False),
            "graph_state_direct_basis_transpiled_fid095.qpy",
        )


class DirectBasisSelectionRankingTests(unittest.TestCase):
    def test_threshold_selection_filters_success_status_and_fidelity_then_sorts_by_depth(self):
        df = pd.DataFrame(
            [
                {
                    "selection_label": "fid099",
                    "status": "ok",
                    "success": True,
                    "class_name": "product",
                    "candidate_name": "slow",
                    "best_depth": 20,
                    "best_two_qubit_gate_count": 4,
                    "best_size": 40,
                    "fidelity": 0.995,
                },
                {
                    "selection_label": "fid099",
                    "status": "ok",
                    "success": True,
                    "class_name": "product",
                    "candidate_name": "fast",
                    "best_depth": 10,
                    "best_two_qubit_gate_count": 8,
                    "best_size": 44,
                    "fidelity": 0.991,
                },
                {
                    "selection_label": "fid099",
                    "status": "ok",
                    "success": True,
                    "class_name": "product",
                    "candidate_name": "low_fidelity",
                    "best_depth": 1,
                    "best_two_qubit_gate_count": 1,
                    "best_size": 1,
                    "fidelity": 0.980,
                },
                {
                    "selection_label": "fid099",
                    "status": "build_error",
                    "success": False,
                    "class_name": "product",
                    "candidate_name": "failed",
                    "best_depth": 0,
                    "best_two_qubit_gate_count": 0,
                    "best_size": 0,
                    "fidelity": 1.0,
                },
            ]
        )

        selected = select_top_k(
            df,
            label="fid099",
            top_k=2,
            fidelity_threshold=0.99,
        )

        self.assertEqual(selected["candidate_name"].tolist(), ["fast", "slow"])
        self.assertEqual(selected["selection_rank"].tolist(), [1, 2])

    def test_exact_selection_does_not_filter_by_fidelity(self):
        df = pd.DataFrame(
            [
                {
                    "selection_label": "exact",
                    "status": "ok",
                    "success": True,
                    "class_name": "baseline",
                    "candidate_name": "low_fid_but_shallow",
                    "best_depth": 2,
                    "best_two_qubit_gate_count": 1,
                    "best_size": 5,
                    "fidelity": 0.50,
                },
                {
                    "selection_label": "exact",
                    "status": "ok",
                    "success": True,
                    "class_name": "baseline",
                    "candidate_name": "high_fid_but_deep",
                    "best_depth": 9,
                    "best_two_qubit_gate_count": 1,
                    "best_size": 5,
                    "fidelity": 1.0,
                },
            ]
        )

        selected = select_top_k(df, label="exact", top_k=1, fidelity_threshold=None)

        self.assertEqual(selected["candidate_name"].tolist(), ["low_fid_but_shallow"])
        self.assertEqual(selected["selection_rank"].tolist(), [1])


class DirectBasisSelectionArtifactTests(unittest.TestCase):
    def test_materialize_selected_artifacts_copies_files_and_writes_manifest_with_relative_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            raw_dir = repo / "artifacts" / "direct_basis_runs" / "raw" / "quantum_circuits" / "ghz3" / "product__p001"
            raw_dir.mkdir(parents=True)
            for name in (
                "graph_state_direct_basis.qpy",
                "graph_state_direct_basis_transpiled_fid099.qpy",
                "F3_W.qpy",
                "CZ3_W.qpy",
            ):
                (raw_dir / name).write_bytes(f"{name}\n".encode("ascii"))
            np.save(raw_dir / "W.npy", np.eye(3, dtype=complex))

            df = pd.DataFrame(
                [
                    {
                        "selection_label": "fid099",
                        "state_name": "ghz3",
                        "class_name": "product",
                        "candidate_name": "p001",
                        "status": "ok",
                        "success": True,
                        "best_depth": 7,
                        "best_two_qubit_gate_count": 3,
                        "best_size": 12,
                        "fidelity": 0.995,
                        "approximation_degree": 0.99,
                        "graph_state_qpy": str(raw_dir / "graph_state_direct_basis.qpy"),
                        "graph_state_transpiled_qpy": str(raw_dir / "graph_state_direct_basis_transpiled_fid099.qpy"),
                        "f3_w_qpy": str(raw_dir / "F3_W.qpy"),
                        "cz3_w_qpy": str(raw_dir / "CZ3_W.qpy"),
                        "basis_change_matrix_npy": str(raw_dir / "W.npy"),
                    }
                ]
            )

            output = materialize_selected_artifacts(
                df,
                SelectionConfig(
                    repo_root=repo,
                    state_name="ghz3",
                    run_id="20260703_153000",
                    top_k=5,
                    labels=("fid099", "fid095"),
                    processed_dir=repo / "artifacts" / "direct_basis_runs" / "processed",
                    selected_root=repo / "artifacts" / "direct_basis_runs" / "selected_best",
                ),
            )

            selected_dir = (
                repo
                / "artifacts"
                / "direct_basis_runs"
                / "selected_best"
                / "ghz3"
                / "20260703_153000"
                / "fid099"
                / "rank01_product__p001"
            )
            self.assertTrue((selected_dir / "graph_state_direct_basis.qpy").is_file())
            self.assertTrue((selected_dir / "graph_state_direct_basis_transpiled.qpy").is_file())
            self.assertTrue((selected_dir / "F3_W.qpy").is_file())
            self.assertTrue((selected_dir / "CZ3_W.qpy").is_file())
            self.assertTrue((selected_dir / "W.npy").is_file())
            E = np.load(selected_dir / "E.npy")
            self.assertEqual(E.shape, (4, 3))
            np.testing.assert_allclose(E[:3, :], np.eye(3))
            np.testing.assert_allclose(E[3, :], np.zeros(3))

            manifest = pd.read_csv(output.manifest_csv)
            self.assertEqual(len(manifest), 1)
            self.assertEqual(manifest.loc[0, "selection_label"], "fid099")
            self.assertEqual(manifest.loc[0, "rank"], 1)
            self.assertFalse(Path(manifest.loc[0, "transpiled_qpy"]).is_absolute())
            self.assertEqual(
                manifest.loc[0, "transpiled_qpy"],
                "artifacts/direct_basis_runs/selected_best/ghz3/20260703_153000/fid099/rank01_product__p001/graph_state_direct_basis_transpiled.qpy",
            )
            self.assertTrue(output.top_csvs["fid099"].is_file())
            self.assertTrue(output.top_csvs["fid095"].is_file())
            self.assertTrue(
                (
                    repo
                    / "artifacts"
                    / "direct_basis_runs"
                    / "selected_best"
                    / "ghz3"
                    / "20260703_153000"
                    / "fid095"
                ).is_dir()
            )
            self.assertTrue(output.processed_manifest_csv.is_file())
            self.assertEqual(
                output.warnings,
                (
                    "fid099: selected 1 rows, requested 5",
                    "fid095: selected 0 rows, requested 5",
                ),
            )


class DirectBasisBenchmarkApproximationTests(unittest.TestCase):
    def test_benchmark_candidates_runs_exact_plus_each_threshold_in_long_format(self):
        candidate = DirectBasisCandidate(
            name="I",
            candidate_type="identity",
            matrix=np.eye(3, dtype=complex),
            source_class_name="baseline",
            source_candidate_name="I",
        )

        def fake_benchmark_direct_basis(**kwargs):
            label = kwargs["selection_label"]
            degree = kwargs["approximation_degree"]
            return {
                "selection_label": label,
                "approximation_degree": "" if degree is None else degree,
                "state_name": kwargs["state_name"],
                "class_name": kwargs["source_class_name"],
                "candidate_name": kwargs["source_candidate_name"],
                "status": "ok",
                "success": True,
                "best_depth": 1,
                "best_two_qubit_gate_count": 1,
                "best_size": 1,
            }

        with patch(
            "qudits_on_qubits.benchmarks.direct_basis.benchmark.benchmark_direct_basis",
            side_effect=fake_benchmark_direct_basis,
        ) as mocked:
            df, _ = benchmark_direct_basis_candidates(
                state_name="ghz3",
                candidates=[candidate],
                n_transpile_runs=1,
                compute_fidelity=True,
                approximation_degrees=(0.99, 0.95, 0.90),
            )

        self.assertEqual(df["selection_label"].tolist(), ["exact", "fid099", "fid095", "fid090"])
        self.assertEqual(mocked.call_count, 4)
        self.assertEqual(
            [call.kwargs["approximation_degree"] for call in mocked.call_args_list],
            [None, 0.99, 0.95, 0.90],
        )
        self.assertEqual(
            [call.kwargs["legacy_exact_transpiled_filename"] for call in mocked.call_args_list],
            [False, False, False, False],
        )

    def test_legacy_mode_runs_only_exact_with_legacy_transpiled_filename(self):
        candidate = DirectBasisCandidate(
            name="I",
            candidate_type="identity",
            matrix=np.eye(3, dtype=complex),
            source_class_name="baseline",
            source_candidate_name="I",
        )

        with patch(
            "qudits_on_qubits.benchmarks.direct_basis.benchmark.benchmark_direct_basis",
            return_value={
                "selection_label": "exact",
                "approximation_degree": "",
                "state_name": "ghz3",
                "class_name": "baseline",
                "candidate_name": "I",
                "status": "ok",
                "success": True,
            },
        ) as mocked:
            df, _ = benchmark_direct_basis_candidates(
                state_name="ghz3",
                candidates=[candidate],
                n_transpile_runs=1,
                compute_fidelity=True,
            )

        self.assertEqual(df["selection_label"].tolist(), ["exact"])
        self.assertIsNone(mocked.call_args.kwargs["approximation_degree"])
        self.assertTrue(mocked.call_args.kwargs["legacy_exact_transpiled_filename"])


class DirectBasisCliSelectionTests(unittest.TestCase):
    def test_parser_requires_state_and_defaults_candidate_set_to_all_qutrit_u3(self):
        parser = build_parser()
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args([])
        args = parser.parse_args(["--state", "ghz3"])
        self.assertEqual(args.state, "ghz3")
        self.assertEqual(args.candidate_set, "all-qutrit-u3")

    def test_no_fidelity_rejects_thresholds(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--state",
                "ghz3",
                "--no-fidelity",
                "--approximation-thresholds",
                "0.99,0.95,0.90",
            ]
        )
        with self.assertRaisesRegex(ValueError, "--no-fidelity cannot be combined"):
            _validate_cli_selection_args(args)

    def test_select_top_k_is_allowed_with_no_fidelity_exact_mode(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--state",
                "ghz3",
                "--no-fidelity",
                "--select-top-k",
                "5",
            ]
        )
        _validate_cli_selection_args(args)

    def test_bell_state_restriction_applies_only_to_selected_artifacts(self):
        parser = build_parser()
        args = parser.parse_args(["--state", "path5"])
        _validate_cli_selection_args(args)

        selecting_args = parser.parse_args(["--state", "path5", "--select-top-k", "1"])
        with self.assertRaisesRegex(ValueError, "Bell selected-circuit pipeline"):
            _validate_cli_selection_args(selecting_args)


if __name__ == "__main__":
    unittest.main()
