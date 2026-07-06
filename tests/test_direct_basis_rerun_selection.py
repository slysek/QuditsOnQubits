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


if __name__ == "__main__":
    unittest.main()
