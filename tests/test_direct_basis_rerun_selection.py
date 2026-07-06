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
    load_input_csvs,
)


def _candidate(class_name: str, candidate_name: str) -> DirectBasisCandidate:
    return DirectBasisCandidate(
        name=candidate_name,
        candidate_type=class_name,
        matrix=np.eye(3, dtype=complex),
        source_class_name=class_name,
        source_candidate_name=candidate_name,
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
        with self.assertRaisesRegex(ValueError, "--top-k must be positive"):
            RerunSelectionConfig(
                input_csvs=(Path("input.csv"),),
                output_root=Path("out"),
                run_id="run",
                top_k=0,
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


if __name__ == "__main__":
    unittest.main()
