from __future__ import annotations

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


from qudits_on_qubits.benchmarks.direct_basis.candidates import (
    DirectBasisCandidate,
    candidates_from_old_csv,
)
from qudits_on_qubits.benchmarks.direct_basis.rerun_selection import (
    RerunSelectionConfig,
    annotate_baseline_equivalence,
    load_input_csvs,
    select_state_rerun_rows,
)


def _candidate(class_name: str, candidate_name: str) -> DirectBasisCandidate:
    return DirectBasisCandidate(
        name=candidate_name,
        candidate_type=class_name,
        matrix=np.eye(3, dtype=complex),
        source_class_name=class_name,
        source_candidate_name=candidate_name,
    )


def _unsupported_candidate(class_name: str, candidate_name: str) -> DirectBasisCandidate:
    return DirectBasisCandidate(
        name=candidate_name,
        candidate_type=class_name,
        matrix=None,
        source_class_name=class_name,
        source_candidate_name=candidate_name,
        error_message="candidate unavailable",
    )


class DirectBasisFromOldCsvRoleTests(unittest.TestCase):
    def test_selector_csv_reruns_only_baseline_and_candidate_roles(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "two_qutrit_top10_plus_baseline.csv"
            pd.DataFrame(
                [
                    {
                        "selection_role": "baseline",
                        "class_name": "baseline",
                        "candidate_name": "E_old",
                    },
                    {
                        "selection_role": "candidate",
                        "class_name": "monomial_full",
                        "candidate_name": "good",
                    },
                    {
                        "selection_role": "baseline_equivalent_excluded",
                        "class_name": "monomial_full",
                        "candidate_name": "equiv",
                    },
                    {
                        "selection_role": "unresolved_candidate",
                        "class_name": "monomial_full",
                        "candidate_name": "missing",
                    },
                ]
            ).to_csv(csv_path, index=False)

            with (
                patch(
                    "qudits_on_qubits.benchmarks.direct_basis.candidates.generate_legacy_qutrit_u3_candidates",
                    return_value=[_candidate("baseline", "E_old")],
                ),
                patch(
                    "qudits_on_qubits.benchmarks.direct_basis.candidates.generate_v2_stage1_direct_candidates",
                    return_value=[
                        _candidate("monomial_full", "good"),
                        _candidate("monomial_full", "equiv"),
                        _candidate("monomial_full", "missing"),
                    ],
                ),
            ):
                selected = candidates_from_old_csv(str(csv_path), include_unsupported=True)

        self.assertEqual(
            [(candidate.class_name, candidate.candidate_name) for candidate in selected],
            [("baseline", "E_old"), ("monomial_full", "good")],
        )

    def test_legacy_csv_without_selection_role_keeps_all_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "legacy.csv"
            pd.DataFrame(
                [
                    {"class_name": "baseline", "candidate_name": "E_old"},
                    {"class_name": "monomial_full", "candidate_name": "good"},
                    {"class_name": "monomial_full", "candidate_name": "equiv"},
                ]
            ).to_csv(csv_path, index=False)

            with (
                patch(
                    "qudits_on_qubits.benchmarks.direct_basis.candidates.generate_legacy_qutrit_u3_candidates",
                    return_value=[_candidate("baseline", "E_old")],
                ),
                patch(
                    "qudits_on_qubits.benchmarks.direct_basis.candidates.generate_v2_stage1_direct_candidates",
                    return_value=[
                        _candidate("monomial_full", "good"),
                        _candidate("monomial_full", "equiv"),
                    ],
                ),
            ):
                selected = candidates_from_old_csv(str(csv_path), include_unsupported=True)

        self.assertEqual(
            [(candidate.class_name, candidate.candidate_name) for candidate in selected],
            [
                ("baseline", "E_old"),
                ("monomial_full", "good"),
                ("monomial_full", "equiv"),
            ],
        )


class RerunSelectionValidationTests(unittest.TestCase):
    def test_config_rejects_invalid_top_k(self):
        invalid_values = (0, -1, True, False, 1.5, "1.5", "not-a-number")
        for top_k in invalid_values:
            with self.subTest(top_k=top_k):
                with self.assertRaisesRegex(ValueError, "--top-k must be positive"):
                    RerunSelectionConfig(
                        input_csvs=(Path("input.csv"),),
                        output_root=Path("out"),
                        run_id="run",
                        top_k=top_k,
                    )

    def test_config_rejects_empty_input_csvs(self):
        with self.assertRaisesRegex(ValueError, "at least one --input-csv is required"):
            RerunSelectionConfig(
                input_csvs=(),
                output_root=Path("out"),
                run_id="run",
            )

    def test_config_rejects_blank_run_id(self):
        with self.assertRaisesRegex(ValueError, "--run-id must not be empty"):
            RerunSelectionConfig(
                input_csvs=(Path("input.csv"),),
                output_root=Path("out"),
                run_id="  ",
            )

    def test_load_input_csvs_requires_core_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "bad.csv"
            pd.DataFrame([{"state_name": "ghz3"}]).to_csv(csv_path, index=False)

            with self.assertRaisesRegex(ValueError, "class_name"):
                load_input_csvs((csv_path,))

    def test_load_input_csvs_adds_source_csv_and_filters_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "raw.csv"
            pd.DataFrame(
                [
                    {
                        "selection_label": "exact",
                        "state_name": "ghz3",
                        "class_name": "baseline",
                        "candidate_name": "E_old",
                        "best_depth": 10,
                    },
                    {
                        "selection_label": "fid099",
                        "state_name": "ghz3",
                        "class_name": "baseline",
                        "candidate_name": "E_old",
                        "best_depth": 11,
                    },
                ]
            ).to_csv(csv_path, index=False)

            df = load_input_csvs((csv_path,), include_label="exact")

        self.assertEqual(len(df), 1)
        self.assertEqual(df.loc[0, "selection_label"], "exact")
        self.assertEqual(df.loc[0, "source_csv"], str(csv_path))

    def test_load_input_csvs_rejects_empty_inputs(self):
        with self.assertRaisesRegex(ValueError, "at least one --input-csv is required"):
            load_input_csvs(())

    def test_load_input_csvs_concatenates_multiple_csvs_with_reset_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            first_path = Path(tmp) / "first.csv"
            second_path = Path(tmp) / "second.csv"
            pd.DataFrame(
                [
                    {
                        "state_name": "ghz3",
                        "class_name": "baseline",
                        "candidate_name": "E_old",
                        "best_depth": 10,
                    }
                ],
                index=[7],
            ).to_csv(first_path, index=False)
            pd.DataFrame(
                [
                    {
                        "state_name": "w3",
                        "class_name": "baseline",
                        "candidate_name": "E_old",
                        "best_depth": 12,
                    }
                ],
                index=[9],
            ).to_csv(second_path, index=False)

            df = load_input_csvs((first_path, second_path))

        self.assertEqual(list(df.index), [0, 1])
        self.assertEqual(list(df["state_name"]), ["ghz3", "w3"])
        self.assertEqual(list(df["source_csv"]), [str(first_path), str(second_path)])

    def test_load_input_csvs_ignores_include_label_when_selection_label_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "raw.csv"
            pd.DataFrame(
                [
                    {
                        "state_name": "ghz3",
                        "class_name": "baseline",
                        "candidate_name": "E_old",
                        "best_depth": 10,
                    },
                    {
                        "state_name": "w3",
                        "class_name": "baseline",
                        "candidate_name": "E_old",
                        "best_depth": 11,
                    },
                ]
            ).to_csv(csv_path, index=False)

            df = load_input_csvs((csv_path,), include_label="exact")

        self.assertEqual(len(df), 2)
        self.assertEqual(list(df["state_name"]), ["ghz3", "w3"])
        self.assertEqual(list(df["source_csv"]), [str(csv_path), str(csv_path)])


class RerunSelectionRankingTests(unittest.TestCase):
    def test_numeric_one_baseline_equivalence_is_excluded_from_top_k(self):
        df = pd.DataFrame(
            [
                {
                    "state_name": "ghz3",
                    "class_name": "baseline",
                    "candidate_name": "E_old",
                    "status": "ok",
                    "success": True,
                    "best_depth": 100,
                    "is_baseline_reference": 1.0,
                    "is_baseline_equivalent": 1.0,
                },
                {
                    "state_name": "ghz3",
                    "class_name": "monomial_full",
                    "candidate_name": "numeric_equiv",
                    "status": "ok",
                    "success": True,
                    "best_depth": 1,
                    "is_baseline_reference": 0.0,
                    "is_baseline_equivalent": 1.0,
                    "skip_reason": "same embedding as baseline within tolerance",
                },
                {
                    "state_name": "ghz3",
                    "class_name": "monomial_full",
                    "candidate_name": "candidate",
                    "status": "ok",
                    "success": True,
                    "best_depth": 80,
                    "is_baseline_reference": 0.0,
                    "is_baseline_equivalent": 0.0,
                },
            ]
        )

        selected, warnings = select_state_rerun_rows(df, "ghz3", top_k=1)

        self.assertEqual(warnings, ())
        self.assertEqual(
            selected[["selection_role", "candidate_name"]].values.tolist(),
            [
                ["baseline", "E_old"],
                ["candidate", "candidate"],
                ["baseline_equivalent_excluded", "numeric_equiv"],
            ],
        )

    def test_numeric_zero_baseline_equivalence_can_be_selected(self):
        df = pd.DataFrame(
            [
                {
                    "state_name": "ghz3",
                    "class_name": "baseline",
                    "candidate_name": "E_old",
                    "status": "ok",
                    "success": True,
                    "best_depth": 100,
                    "is_baseline_reference": 1.0,
                    "is_baseline_equivalent": 1.0,
                },
                {
                    "state_name": "ghz3",
                    "class_name": "monomial_full",
                    "candidate_name": "numeric_candidate",
                    "status": "ok",
                    "success": True,
                    "best_depth": 80,
                    "is_baseline_reference": 0.0,
                    "is_baseline_equivalent": 0.0,
                },
            ]
        )

        selected, warnings = select_state_rerun_rows(df, "ghz3", top_k=1)

        self.assertEqual(warnings, ())
        self.assertEqual(
            selected[["selection_role", "candidate_name"]].values.tolist(),
            [["baseline", "E_old"], ["candidate", "numeric_candidate"]],
        )

    def test_select_state_excludes_baseline_equivalent_from_top_k_but_writes_diagnostic(self):
        df = pd.DataFrame(
            [
                {
                    "state_name": "ghz3",
                    "class_name": "baseline",
                    "candidate_name": "E_old",
                    "status": "ok",
                    "success": True,
                    "best_depth": 100,
                    "mean_depth": 105,
                    "std_depth": 2,
                    "best_two_qubit_gate_count": 50,
                    "best_one_qubit_gate_count": 100,
                    "best_size": 150,
                    "is_baseline_reference": True,
                    "is_baseline_equivalent": True,
                    "skip_reason": "",
                    "source_csv": "raw.csv",
                },
                {
                    "state_name": "ghz3",
                    "class_name": "monomial_full",
                    "candidate_name": "equiv",
                    "status": "ok",
                    "success": True,
                    "best_depth": 1,
                    "mean_depth": 1,
                    "std_depth": 0,
                    "best_two_qubit_gate_count": 1,
                    "best_one_qubit_gate_count": 1,
                    "best_size": 2,
                    "is_baseline_reference": False,
                    "is_baseline_equivalent": True,
                    "skip_reason": "same embedding as baseline within tolerance",
                    "source_csv": "raw.csv",
                },
                {
                    "state_name": "ghz3",
                    "class_name": "monomial_full",
                    "candidate_name": "better",
                    "status": "ok",
                    "success": True,
                    "best_depth": 80,
                    "mean_depth": 85,
                    "std_depth": 3,
                    "best_two_qubit_gate_count": 40,
                    "best_one_qubit_gate_count": 90,
                    "best_size": 130,
                    "is_baseline_reference": False,
                    "is_baseline_equivalent": False,
                    "skip_reason": "",
                    "source_csv": "raw.csv",
                },
                {
                    "state_name": "ghz3",
                    "class_name": "monomial_full",
                    "candidate_name": "equal",
                    "status": "ok",
                    "success": True,
                    "best_depth": 100,
                    "mean_depth": 100,
                    "std_depth": 1,
                    "best_two_qubit_gate_count": 45,
                    "best_one_qubit_gate_count": 90,
                    "best_size": 135,
                    "is_baseline_reference": False,
                    "is_baseline_equivalent": False,
                    "skip_reason": "",
                    "source_csv": "raw.csv",
                },
                {
                    "state_name": "ghz3",
                    "class_name": "product",
                    "candidate_name": "failed",
                    "status": "build_error",
                    "success": False,
                    "best_depth": 0,
                    "mean_depth": 0,
                    "std_depth": 0,
                    "best_two_qubit_gate_count": 0,
                    "best_one_qubit_gate_count": 0,
                    "best_size": 0,
                    "is_baseline_reference": False,
                    "is_baseline_equivalent": False,
                    "skip_reason": "",
                    "source_csv": "raw.csv",
                },
            ]
        )

        selected, warnings = select_state_rerun_rows(df, "ghz3", top_k=2)

        self.assertEqual(warnings, ())
        selection_rows = selected[
            ["selection_role", "selection_rank", "candidate_name"]
        ].values.tolist()
        self.assertEqual(selection_rows[:3], [
            ["baseline", 0, "E_old"],
            ["candidate", 1, "better"],
            ["candidate", 2, "equal"],
        ])
        self.assertEqual(selection_rows[3][0], "baseline_equivalent_excluded")
        self.assertTrue(pd.isna(selection_rows[3][1]))
        self.assertEqual(selection_rows[3][2], "equiv")
        self.assertEqual(selected.loc[1, "baseline_relation"], "better")
        self.assertEqual(selected.loc[2, "baseline_relation"], "equal")
        self.assertEqual(
            selected.loc[3, "baseline_relation"], "excluded_baseline_equivalent"
        )
        self.assertNotIn("failed", selected["candidate_name"].tolist())

    def test_missing_baseline_raises_clear_error(self):
        df = pd.DataFrame(
            [
                {
                    "state_name": "ame43",
                    "class_name": "monomial_full",
                    "candidate_name": "x",
                    "status": "ok",
                    "success": True,
                    "best_depth": 10,
                }
            ]
        )

        with self.assertRaisesRegex(ValueError, "ame43 has no baseline row"):
            select_state_rerun_rows(df, "ame43", top_k=10)

    def test_e_old_baseline_is_selected_over_better_ranked_alternate_baseline(self):
        df = pd.DataFrame(
            [
                {
                    "state_name": "ghz3",
                    "class_name": "baseline",
                    "candidate_name": "E_old",
                    "status": "ok",
                    "success": True,
                    "best_depth": 100,
                    "best_two_qubit_gate_count": 50,
                    "mean_depth": 105,
                    "std_depth": 2,
                    "best_one_qubit_gate_count": 100,
                    "best_size": 150,
                    "is_baseline_reference": True,
                    "is_baseline_equivalent": True,
                },
                {
                    "state_name": "ghz3",
                    "class_name": "baseline",
                    "candidate_name": "alternate",
                    "status": "ok",
                    "success": True,
                    "best_depth": 1,
                    "best_two_qubit_gate_count": 1,
                    "mean_depth": 1,
                    "std_depth": 0,
                    "best_one_qubit_gate_count": 1,
                    "best_size": 2,
                    "is_baseline_reference": False,
                    "is_baseline_equivalent": True,
                },
                {
                    "state_name": "ghz3",
                    "class_name": "monomial_full",
                    "candidate_name": "candidate",
                    "status": "ok",
                    "success": True,
                    "best_depth": 80,
                    "best_two_qubit_gate_count": 40,
                    "mean_depth": 85,
                    "std_depth": 3,
                    "best_one_qubit_gate_count": 90,
                    "best_size": 130,
                    "is_baseline_reference": False,
                    "is_baseline_equivalent": False,
                },
            ]
        )

        selected, warnings = select_state_rerun_rows(df, "ghz3", top_k=1)

        self.assertEqual(
            warnings,
            ("ghz3: multiple baseline rows found; selected best ranked baseline",),
        )
        baseline_rows = selected[selected["selection_role"] == "baseline"]
        self.assertEqual(baseline_rows["candidate_name"].tolist(), ["E_old"])

    def test_alternate_baseline_is_not_selected_as_candidate(self):
        df = pd.DataFrame(
            [
                {
                    "state_name": "ghz3",
                    "class_name": "baseline",
                    "candidate_name": "E_old",
                    "status": "ok",
                    "success": True,
                    "best_depth": 100,
                    "best_two_qubit_gate_count": 50,
                    "mean_depth": 105,
                    "std_depth": 2,
                    "best_one_qubit_gate_count": 100,
                    "best_size": 150,
                    "is_baseline_reference": True,
                    "is_baseline_equivalent": True,
                },
                {
                    "state_name": "ghz3",
                    "class_name": "baseline",
                    "candidate_name": "alternate",
                    "status": "ok",
                    "success": True,
                    "best_depth": 1,
                    "best_two_qubit_gate_count": 1,
                    "mean_depth": 1,
                    "std_depth": 0,
                    "best_one_qubit_gate_count": 1,
                    "best_size": 2,
                    "is_baseline_reference": False,
                    "is_baseline_equivalent": False,
                },
                {
                    "state_name": "ghz3",
                    "class_name": "monomial_full",
                    "candidate_name": "real_candidate",
                    "status": "ok",
                    "success": True,
                    "best_depth": 80,
                    "best_two_qubit_gate_count": 40,
                    "mean_depth": 85,
                    "std_depth": 3,
                    "best_one_qubit_gate_count": 90,
                    "best_size": 130,
                    "is_baseline_reference": False,
                    "is_baseline_equivalent": False,
                },
            ]
        )

        selected, warnings = select_state_rerun_rows(df, "ghz3", top_k=1)

        self.assertEqual(
            warnings,
            ("ghz3: multiple baseline rows found; selected best ranked baseline",),
        )
        self.assertNotIn(
            ["candidate", "alternate"],
            selected[["selection_role", "candidate_name"]].values.tolist(),
        )
        candidate_rows = selected[selected["selection_role"] == "candidate"]
        self.assertEqual(candidate_rows["candidate_name"].tolist(), ["real_candidate"])

    def test_multiple_alternate_baselines_select_best_ranked_with_warning(self):
        df = pd.DataFrame(
            [
                {
                    "state_name": "ghz3",
                    "class_name": "baseline",
                    "candidate_name": "slower",
                    "status": "ok",
                    "success": True,
                    "best_depth": 100,
                    "best_two_qubit_gate_count": 50,
                    "mean_depth": 105,
                    "std_depth": 2,
                    "best_one_qubit_gate_count": 100,
                    "best_size": 150,
                    "is_baseline_reference": False,
                    "is_baseline_equivalent": True,
                },
                {
                    "state_name": "ghz3",
                    "class_name": "baseline",
                    "candidate_name": "faster",
                    "status": "ok",
                    "success": True,
                    "best_depth": 80,
                    "best_two_qubit_gate_count": 40,
                    "mean_depth": 85,
                    "std_depth": 3,
                    "best_one_qubit_gate_count": 90,
                    "best_size": 130,
                    "is_baseline_reference": False,
                    "is_baseline_equivalent": True,
                },
                {
                    "state_name": "ghz3",
                    "class_name": "monomial_full",
                    "candidate_name": "candidate",
                    "status": "ok",
                    "success": True,
                    "best_depth": 70,
                    "best_two_qubit_gate_count": 35,
                    "mean_depth": 75,
                    "std_depth": 4,
                    "best_one_qubit_gate_count": 80,
                    "best_size": 115,
                    "is_baseline_reference": False,
                    "is_baseline_equivalent": False,
                },
            ]
        )

        selected, warnings = select_state_rerun_rows(df, "ghz3", top_k=1)

        self.assertEqual(
            warnings,
            ("ghz3: multiple baseline rows found; selected best ranked baseline",),
        )
        baseline_rows = selected[selected["selection_role"] == "baseline"]
        self.assertEqual(baseline_rows["candidate_name"].tolist(), ["faster"])

    def test_mixed_failed_and_runnable_baselines_emit_only_runnable_baseline(self):
        df = pd.DataFrame(
            [
                {
                    "state_name": "ghz3",
                    "class_name": "baseline",
                    "candidate_name": "failed",
                    "status": "build_error",
                    "success": False,
                    "best_depth": 1,
                    "is_baseline_reference": True,
                    "is_baseline_equivalent": True,
                },
                {
                    "state_name": "ghz3",
                    "class_name": "baseline",
                    "candidate_name": "runnable",
                    "status": "ok",
                    "success": True,
                    "best_depth": 100,
                    "is_baseline_reference": False,
                    "is_baseline_equivalent": True,
                },
                {
                    "state_name": "ghz3",
                    "class_name": "monomial_full",
                    "candidate_name": "candidate",
                    "status": "ok",
                    "success": True,
                    "best_depth": 80,
                    "is_baseline_reference": False,
                    "is_baseline_equivalent": False,
                },
            ]
        )

        selected, warnings = select_state_rerun_rows(df, "ghz3", top_k=1)

        self.assertEqual(warnings, ())
        baseline_rows = selected[selected["selection_role"] == "baseline"]
        self.assertEqual(baseline_rows["candidate_name"].tolist(), ["runnable"])
        self.assertNotIn("failed", selected["candidate_name"].tolist())

    def test_output_column_order_and_baseline_comparison_values(self):
        df = pd.DataFrame(
            [
                {
                    "state_name": "ghz3",
                    "class_name": "baseline",
                    "candidate_name": "E_old",
                    "status": "ok",
                    "success": True,
                    "best_depth": 100,
                    "mean_depth": 105,
                    "std_depth": 2,
                    "best_two_qubit_gate_count": 50,
                    "best_one_qubit_gate_count": 100,
                    "best_size": 150,
                    "is_baseline_reference": True,
                    "is_baseline_equivalent": True,
                    "baseline_equivalence_reason": "",
                    "source_csv": "raw.csv",
                },
                {
                    "state_name": "ghz3",
                    "class_name": "monomial_full",
                    "candidate_name": "better",
                    "status": "ok",
                    "success": True,
                    "best_depth": 80,
                    "mean_depth": 85,
                    "std_depth": 3,
                    "best_two_qubit_gate_count": 40,
                    "best_one_qubit_gate_count": 90,
                    "best_size": 130,
                    "is_baseline_reference": False,
                    "is_baseline_equivalent": False,
                    "baseline_equivalence_reason": "",
                    "source_csv": "raw.csv",
                },
                {
                    "state_name": "ghz3",
                    "class_name": "monomial_full",
                    "candidate_name": "equal",
                    "status": "ok",
                    "success": True,
                    "best_depth": 100,
                    "mean_depth": 100,
                    "std_depth": 1,
                    "best_two_qubit_gate_count": 45,
                    "best_one_qubit_gate_count": 90,
                    "best_size": 135,
                    "is_baseline_reference": False,
                    "is_baseline_equivalent": False,
                    "baseline_equivalence_reason": "",
                    "source_csv": "raw.csv",
                },
                {
                    "state_name": "ghz3",
                    "class_name": "product",
                    "candidate_name": "worse",
                    "status": "ok",
                    "success": True,
                    "best_depth": 120,
                    "mean_depth": 125,
                    "std_depth": 4,
                    "best_two_qubit_gate_count": 55,
                    "best_one_qubit_gate_count": 110,
                    "best_size": 165,
                    "is_baseline_reference": False,
                    "is_baseline_equivalent": False,
                    "baseline_equivalence_reason": "",
                    "source_csv": "raw.csv",
                },
            ]
        )

        selected, warnings = select_state_rerun_rows(df, "ghz3", top_k=3)

        self.assertEqual(warnings, ())
        expected_first = [
            "state_name",
            "selection_role",
            "selection_rank",
            "class_name",
            "candidate_name",
            "is_baseline_reference",
            "is_baseline_equivalent",
            "baseline_equivalence_reason",
            "best_depth",
            "mean_depth",
            "std_depth",
            "best_two_qubit_gate_count",
            "best_one_qubit_gate_count",
            "best_size",
            "baseline_best_depth",
            "depth_delta_vs_baseline",
            "depth_ratio_vs_baseline",
            "baseline_relation",
            "source_csv",
        ]
        self.assertEqual(
            list(selected.columns[: len(expected_first)]),
            expected_first,
        )
        by_candidate = {
            row["candidate_name"]: row for _, row in selected.iterrows()
        }
        self.assertEqual(by_candidate["better"]["baseline_best_depth"], 100.0)
        self.assertEqual(by_candidate["better"]["depth_delta_vs_baseline"], -20.0)
        self.assertAlmostEqual(
            by_candidate["better"]["depth_ratio_vs_baseline"], 0.8
        )
        self.assertEqual(by_candidate["equal"]["baseline_best_depth"], 100.0)
        self.assertEqual(by_candidate["equal"]["depth_delta_vs_baseline"], 0.0)
        self.assertAlmostEqual(by_candidate["equal"]["depth_ratio_vs_baseline"], 1.0)
        self.assertEqual(by_candidate["worse"]["baseline_best_depth"], 100.0)
        self.assertEqual(by_candidate["worse"]["depth_delta_vs_baseline"], 20.0)
        self.assertAlmostEqual(by_candidate["worse"]["depth_ratio_vs_baseline"], 1.2)
        self.assertEqual(by_candidate["worse"]["baseline_relation"], "worse")

    def test_selected_alternate_baseline_is_not_duplicated_as_excluded_diagnostic(self):
        df = pd.DataFrame(
            [
                {
                    "state_name": "ghz3",
                    "class_name": "baseline",
                    "candidate_name": "alternate",
                    "status": "ok",
                    "success": True,
                    "best_depth": 100,
                    "is_baseline_reference": False,
                    "is_baseline_equivalent": True,
                }
            ]
        )

        selected, warnings = select_state_rerun_rows(df, "ghz3", top_k=1)

        self.assertEqual(warnings, ("ghz3: selected 0 candidates, requested 1",))
        self.assertEqual(selected["candidate_name"].tolist(), ["alternate"])
        self.assertEqual(selected["selection_role"].tolist(), ["baseline"])
        self.assertEqual(selected.loc[0, "baseline_relation"], "baseline")

    def test_skipped_baseline_equivalent_row_is_written_as_diagnostic(self):
        df = pd.DataFrame(
            [
                {
                    "state_name": "ghz3",
                    "class_name": "baseline",
                    "candidate_name": "E_old",
                    "status": "ok",
                    "success": True,
                    "best_depth": 100,
                    "is_baseline_reference": True,
                    "is_baseline_equivalent": True,
                },
                {
                    "state_name": "ghz3",
                    "class_name": "monomial_full",
                    "candidate_name": "skipped_equiv",
                    "status": "skipped_baseline_equivalent",
                    "success": False,
                    "best_depth": 1,
                    "is_baseline_reference": False,
                    "is_baseline_equivalent": True,
                    "skip_reason": "same embedding as baseline within tolerance",
                },
            ]
        )

        selected, warnings = select_state_rerun_rows(df, "ghz3", top_k=1)

        self.assertEqual(warnings, ("ghz3: selected 0 candidates, requested 1",))
        self.assertEqual(
            selected[["selection_role", "candidate_name"]].values.tolist(),
            [
                ["baseline", "E_old"],
                ["baseline_equivalent_excluded", "skipped_equiv"],
            ],
        )
        self.assertTrue(pd.isna(selected.loc[1, "selection_rank"]))
        self.assertEqual(
            selected.loc[1, "baseline_relation"], "excluded_baseline_equivalent"
        )

    def test_candidate_without_equivalence_metadata_is_not_selected_for_top_k(self):
        df = pd.DataFrame(
            [
                {
                    "state_name": "ghz3",
                    "class_name": "baseline",
                    "candidate_name": "E_old",
                    "status": "ok",
                    "success": True,
                    "best_depth": 100,
                    "is_baseline_reference": True,
                },
                {
                    "state_name": "ghz3",
                    "class_name": "monomial_full",
                    "candidate_name": "missing_metadata",
                    "status": "ok",
                    "success": True,
                    "best_depth": 1,
                    "is_baseline_reference": False,
                },
            ]
        )

        selected, warnings = select_state_rerun_rows(df, "ghz3", top_k=1)

        self.assertEqual(warnings, ("ghz3: selected 0 candidates, requested 1",))
        self.assertNotIn("candidate", selected["selection_role"].tolist())
        missing_metadata = selected[
            selected["candidate_name"].astype(str) == "missing_metadata"
        ]
        self.assertEqual(len(missing_metadata), 1)
        self.assertEqual(
            missing_metadata.iloc[0]["selection_role"], "unresolved_candidate"
        )
        self.assertTrue(pd.isna(missing_metadata.iloc[0]["selection_rank"]))
        self.assertEqual(missing_metadata.iloc[0]["baseline_relation"], "unresolved")

    def test_failed_baseline_is_not_emitted(self):
        df = pd.DataFrame(
            [
                {
                    "state_name": "ghz3",
                    "class_name": "baseline",
                    "candidate_name": "E_old",
                    "status": "build_error",
                    "success": False,
                    "best_depth": 100,
                    "is_baseline_reference": True,
                    "is_baseline_equivalent": True,
                },
                {
                    "state_name": "ghz3",
                    "class_name": "monomial_full",
                    "candidate_name": "better",
                    "status": "ok",
                    "success": True,
                    "best_depth": 80,
                    "is_baseline_reference": False,
                    "is_baseline_equivalent": False,
                },
            ]
        )

        with self.assertRaisesRegex(ValueError, "ghz3 has no runnable baseline row"):
            select_state_rerun_rows(df, "ghz3", top_k=1)


class RerunSelectionEquivalenceTests(unittest.TestCase):
    def test_select_state_infers_row_level_missing_equivalence_metadata(self):
        df = pd.DataFrame(
            [
                {
                    "state_name": "two_qutrit",
                    "class_name": "baseline",
                    "candidate_name": "E_old",
                    "status": "ok",
                    "success": True,
                    "best_depth": 100,
                    "is_baseline_reference": True,
                    "is_baseline_equivalent": True,
                },
                {
                    "state_name": "two_qutrit",
                    "class_name": "monomial_full",
                    "candidate_name": "non_equiv",
                    "status": "ok",
                    "success": True,
                    "best_depth": 80,
                    "is_baseline_reference": pd.NA,
                    "is_baseline_equivalent": pd.NA,
                },
            ]
        )
        lookup = {
            ("baseline", "E_old"): _candidate("baseline", "E_old"),
            ("monomial_full", "non_equiv"): DirectBasisCandidate(
                name="non_equiv",
                candidate_type="monomial_full",
                matrix=np.array(
                    [
                        [0, 1, 0],
                        [1, 0, 0],
                        [0, 0, 1],
                    ],
                    dtype=complex,
                ),
                source_class_name="monomial_full",
                source_candidate_name="non_equiv",
            ),
        }

        selected, warnings = select_state_rerun_rows(
            df,
            "two_qutrit",
            top_k=1,
            candidate_lookup=lookup,
        )

        self.assertEqual(warnings, ())
        self.assertEqual(
            selected[["selection_role", "candidate_name"]].values.tolist(),
            [["baseline", "E_old"], ["candidate", "non_equiv"]],
        )

    def test_select_state_infers_missing_equivalence_metadata_from_lookup(self):
        df = pd.DataFrame(
            [
                {
                    "state_name": "two_qutrit",
                    "class_name": "baseline",
                    "candidate_name": "E_old",
                    "status": "ok",
                    "success": True,
                    "best_depth": 100,
                },
                {
                    "state_name": "two_qutrit",
                    "class_name": "monomial_full",
                    "candidate_name": "identity_like",
                    "status": "ok",
                    "success": True,
                    "best_depth": 1,
                },
                {
                    "state_name": "two_qutrit",
                    "class_name": "monomial_full",
                    "candidate_name": "non_equiv",
                    "status": "ok",
                    "success": True,
                    "best_depth": 80,
                },
            ]
        )
        lookup = {
            ("baseline", "E_old"): _candidate("baseline", "E_old"),
            ("monomial_full", "identity_like"): _candidate(
                "monomial_full", "identity_like"
            ),
            ("monomial_full", "non_equiv"): DirectBasisCandidate(
                name="non_equiv",
                candidate_type="monomial_full",
                matrix=np.array(
                    [
                        [0, 1, 0],
                        [1, 0, 0],
                        [0, 0, 1],
                    ],
                    dtype=complex,
                ),
                source_class_name="monomial_full",
                source_candidate_name="non_equiv",
            ),
        }

        selected, warnings = select_state_rerun_rows(
            df,
            "two_qutrit",
            top_k=1,
            candidate_lookup=lookup,
        )

        self.assertEqual(warnings, ())
        self.assertEqual(
            selected[["selection_role", "candidate_name"]].values.tolist(),
            [
                ["baseline", "E_old"],
                ["candidate", "non_equiv"],
                ["baseline_equivalent_excluded", "identity_like"],
            ],
        )

    def test_partial_equivalence_metadata_is_preserved(self):
        df = pd.DataFrame(
            [
                {
                    "state_name": "two_qutrit",
                    "class_name": "baseline",
                    "candidate_name": "E_old",
                    "best_depth": 100,
                    "is_baseline_equivalent": True,
                    "skip_reason": "",
                },
                {
                    "state_name": "two_qutrit",
                    "class_name": "monomial_full",
                    "candidate_name": "existing_equiv",
                    "best_depth": 1,
                    "is_baseline_equivalent": True,
                    "skip_reason": "already known equivalent",
                },
                {
                    "state_name": "two_qutrit",
                    "class_name": "monomial_full",
                    "candidate_name": "existing_candidate",
                    "best_depth": 80,
                    "is_baseline_equivalent": False,
                    "skip_reason": "",
                },
            ]
        )

        annotated = annotate_baseline_equivalence(df, candidate_lookup={})

        self.assertEqual(
            annotated["is_baseline_equivalent"].tolist(),
            [True, True, False],
        )
        self.assertEqual(
            annotated["is_baseline_reference"].tolist(),
            [True, False, False],
        )
        self.assertEqual(
            annotated["baseline_equivalence_reason"].tolist(),
            ["", "already known equivalent", ""],
        )

    def test_complete_metadata_selection_does_not_build_default_lookup(self):
        df = pd.DataFrame(
            [
                {
                    "state_name": "two_qutrit",
                    "class_name": "baseline",
                    "candidate_name": "E_old",
                    "status": "ok",
                    "success": True,
                    "best_depth": 100,
                    "is_baseline_reference": True,
                    "is_baseline_equivalent": True,
                },
                {
                    "state_name": "two_qutrit",
                    "class_name": "monomial_full",
                    "candidate_name": "candidate",
                    "status": "ok",
                    "success": True,
                    "best_depth": 80,
                    "is_baseline_reference": False,
                    "is_baseline_equivalent": False,
                },
            ]
        )

        with patch(
            "qudits_on_qubits.benchmarks.direct_basis.rerun_selection."
            "build_default_candidate_lookup",
            side_effect=AssertionError("default lookup should not be built"),
        ):
            selected, warnings = select_state_rerun_rows(
                df,
                "two_qutrit",
                top_k=1,
            )

        self.assertEqual(warnings, ())
        self.assertEqual(
            selected[["selection_role", "candidate_name"]].values.tolist(),
            [["baseline", "E_old"], ["candidate", "candidate"]],
        )

    def test_existing_equivalence_columns_without_reason_get_blank_reason(self):
        df = pd.DataFrame(
            [
                {
                    "state_name": "two_qutrit",
                    "class_name": "baseline",
                    "candidate_name": "E_old",
                    "best_depth": 100,
                    "is_baseline_reference": True,
                    "is_baseline_equivalent": True,
                }
            ]
        )

        annotated = annotate_baseline_equivalence(df, candidate_lookup={})

        self.assertEqual(annotated.loc[0, "baseline_equivalence_reason"], "")

    def test_missing_equivalence_columns_are_inferred_from_candidate_lookup(self):
        df = pd.DataFrame(
            [
                {
                    "state_name": "two_qutrit",
                    "class_name": "baseline",
                    "candidate_name": "E_old",
                    "status": "ok",
                    "success": True,
                    "best_depth": 100,
                },
                {
                    "state_name": "two_qutrit",
                    "class_name": "monomial_full",
                    "candidate_name": "identity_like",
                    "status": "ok",
                    "success": True,
                    "best_depth": 1,
                },
                {
                    "state_name": "two_qutrit",
                    "class_name": "monomial_full",
                    "candidate_name": "non_equiv",
                    "status": "ok",
                    "success": True,
                    "best_depth": 80,
                },
            ]
        )
        lookup = {
            ("baseline", "E_old"): _candidate("baseline", "E_old"),
            ("monomial_full", "identity_like"): _candidate(
                "monomial_full", "identity_like"
            ),
            ("monomial_full", "non_equiv"): DirectBasisCandidate(
                name="non_equiv",
                candidate_type="monomial_full",
                matrix=np.array(
                    [
                        [0, 1, 0],
                        [1, 0, 0],
                        [0, 0, 1],
                    ],
                    dtype=complex,
                ),
                source_class_name="monomial_full",
                source_candidate_name="non_equiv",
            ),
        }

        annotated = annotate_baseline_equivalence(df, candidate_lookup=lookup)
        selected, _ = select_state_rerun_rows(annotated, "two_qutrit", top_k=1)

        self.assertTrue(
            bool(
                annotated.loc[
                    annotated["candidate_name"].eq("identity_like"),
                    "is_baseline_equivalent",
                ].iloc[0]
            )
        )
        self.assertEqual(
            selected[["selection_role", "candidate_name"]].values.tolist(),
            [
                ["baseline", "E_old"],
                ["candidate", "non_equiv"],
                ["baseline_equivalent_excluded", "identity_like"],
            ],
        )

    def test_unresolved_candidate_is_kept_out_of_top_k(self):
        df = pd.DataFrame(
            [
                {
                    "state_name": "two_qutrit",
                    "class_name": "baseline",
                    "candidate_name": "E_old",
                    "status": "ok",
                    "success": True,
                    "best_depth": 100,
                },
                {
                    "state_name": "two_qutrit",
                    "class_name": "unknown",
                    "candidate_name": "missing",
                    "status": "ok",
                    "success": True,
                    "best_depth": 1,
                },
            ]
        )
        lookup = {("baseline", "E_old"): _candidate("baseline", "E_old")}

        annotated = annotate_baseline_equivalence(df, candidate_lookup=lookup)
        selected, warnings = select_state_rerun_rows(annotated, "two_qutrit", top_k=1)

        self.assertEqual(
            selected[
                ["selection_role", "candidate_name", "baseline_relation"]
            ].values.tolist(),
            [
                ["baseline", "E_old", "baseline"],
                ["unresolved_candidate", "missing", "unresolved"],
            ],
        )
        self.assertIn("two_qutrit: selected 0 candidates, requested 1", warnings)

    def test_unsupported_lookup_candidate_is_unresolved_and_helper_column_is_private(self):
        df = pd.DataFrame(
            [
                {
                    "state_name": "two_qutrit",
                    "class_name": "baseline",
                    "candidate_name": "E_old",
                    "status": "ok",
                    "success": True,
                    "best_depth": 100,
                },
                {
                    "state_name": "two_qutrit",
                    "class_name": "monomial_full",
                    "candidate_name": "unavailable",
                    "status": "ok",
                    "success": True,
                    "best_depth": 1,
                },
            ]
        )
        lookup = {
            ("baseline", "E_old"): _candidate("baseline", "E_old"),
            ("monomial_full", "unavailable"): _unsupported_candidate(
                "monomial_full",
                "unavailable",
            ),
        }

        annotated = annotate_baseline_equivalence(df, candidate_lookup=lookup)
        selected, warnings = select_state_rerun_rows(annotated, "two_qutrit", top_k=1)

        self.assertFalse(
            bool(
                annotated.loc[
                    annotated["candidate_name"].eq("unavailable"),
                    "is_baseline_equivalent",
                ].iloc[0]
            )
        )
        self.assertEqual(
            selected[["selection_role", "candidate_name", "baseline_relation"]]
            .values
            .tolist(),
            [
                ["baseline", "E_old", "baseline"],
                ["unresolved_candidate", "unavailable", "unresolved"],
            ],
        )
        self.assertIn("two_qutrit: selected 0 candidates, requested 1", warnings)
        self.assertNotIn("is_unresolved_candidate", selected.columns)


if __name__ == "__main__":
    unittest.main()
