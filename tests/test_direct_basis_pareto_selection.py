from __future__ import annotations

import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from qudits_on_qubits.benchmarks.direct_basis.pareto_selection import (
    BEST_TRIAL_ORDER,
    IDENTITY_COLUMNS,
    OBJECTIVE_COLUMNS,
    OPTIONAL_BOUNDARY_COLUMNS,
    TRIAL_ID_COLUMNS,
    aggregate_strategy_statistics,
)


def _trial(
    *,
    state_name: str = "ghz3",
    class_name: str = "product",
    candidate_name: str = "p001",
    strategy_name: str = "default",
    seed_transpiler: int = 1,
    success: object = True,
    status: str = "ok",
    depth: object = 10,
    two_qubit_gate_count: object = 4,
    one_qubit_gate_count: object = 8,
    size: object = 20,
    **extra: object,
) -> dict[str, object]:
    return {
        "state_name": state_name,
        "class_name": class_name,
        "candidate_name": candidate_name,
        "strategy_name": strategy_name,
        "seed_transpiler": seed_transpiler,
        "success": success,
        "status": status,
        "depth": depth,
        "two_qubit_gate_count": two_qubit_gate_count,
        "one_qubit_gate_count": one_qubit_gate_count,
        "size": size,
        "graph_state_transpiled_qpy": f"{strategy_name}-{seed_transpiler}.qpy",
        "n_qutrits": 3,
        **extra,
    }


class AggregateStrategyStatisticsTests(unittest.TestCase):
    def test_contract_constants_are_exact(self) -> None:
        self.assertEqual(IDENTITY_COLUMNS, ("state_name", "class_name", "candidate_name", "strategy_name"))
        self.assertEqual(OPTIONAL_BOUNDARY_COLUMNS, ("iqm_backend_name", "backend_calibration_set_id", "selection_label"))
        self.assertEqual(TRIAL_ID_COLUMNS, (*IDENTITY_COLUMNS, "seed_transpiler"))
        self.assertEqual(OBJECTIVE_COLUMNS, ("mean_two_qubit_gate_count", "mean_depth", "std_depth"))
        self.assertEqual(BEST_TRIAL_ORDER, ("two_qubit_gate_count", "depth", "one_qubit_gate_count", "size", "seed_transpiler"))

    def test_empty_input_has_documented_schema(self) -> None:
        result = aggregate_strategy_statistics(pd.DataFrame())
        self.assertTrue(result.empty)
        self.assertEqual(
            result.columns.tolist(),
            [
                *IDENTITY_COLUMNS,
                "total_trial_count", "successful_trial_count", "failed_trial_count", "success_rate",
                "mean_depth", "min_depth", "max_depth", "std_depth",
                "mean_two_qubit_gate_count", "min_two_qubit_gate_count", "max_two_qubit_gate_count", "std_two_qubit_gate_count",
                "insufficient_stability_samples", "best_seed_transpiler", "best_graph_state_transpiled_qpy",
                "best_depth", "best_two_qubit_gate_count", "best_one_qubit_gate_count", "best_size",
                "pareto_eligible", "analysis_status",
            ],
        )

    def test_strategies_counts_statistics_and_metadata_stay_separate(self) -> None:
        rows = [
            _trial(strategy_name="a", seed_transpiler=1, depth=10, two_qubit_gate_count=5),
            _trial(strategy_name="a", seed_transpiler=2, depth=14, two_qubit_gate_count=3),
            _trial(strategy_name="a", seed_transpiler=3, success=False, status="failed"),
            _trial(strategy_name="b", seed_transpiler=1, depth=2, two_qubit_gate_count=1, n_qutrits=4),
        ]
        result = aggregate_strategy_statistics(pd.DataFrame(rows)).set_index("strategy_name")

        self.assertEqual(result.index.tolist(), ["a", "b"])
        self.assertEqual(result.loc["a", "total_trial_count"], 3)
        self.assertEqual(result.loc["a", "successful_trial_count"], 2)
        self.assertEqual(result.loc["a", "failed_trial_count"], 1)
        self.assertAlmostEqual(result.loc["a", "success_rate"], 2 / 3)
        self.assertEqual(result.loc["a", "mean_depth"], 12.0)
        self.assertEqual(result.loc["a", "min_depth"], 10.0)
        self.assertEqual(result.loc["a", "max_depth"], 14.0)
        self.assertEqual(result.loc["a", "std_depth"], 2.0)
        self.assertEqual(result.loc["a", "mean_two_qubit_gate_count"], 4.0)
        self.assertEqual(result.loc["a", "min_two_qubit_gate_count"], 3.0)
        self.assertEqual(result.loc["a", "max_two_qubit_gate_count"], 5.0)
        self.assertEqual(result.loc["a", "std_two_qubit_gate_count"], 1.0)
        self.assertFalse(result.loc["a", "insufficient_stability_samples"])
        self.assertEqual(result.loc["a", "best_seed_transpiler"], 2)
        self.assertEqual(result.loc["a", "n_qutrits"], 3)
        self.assertEqual(result.loc["b", "n_qutrits"], 4)

    def test_best_trial_uses_full_contract_order(self) -> None:
        rows = [
            _trial(seed_transpiler=8, depth=1, two_qubit_gate_count=2, one_qubit_gate_count=1, size=1),
            _trial(seed_transpiler=7, depth=100, two_qubit_gate_count=1, one_qubit_gate_count=99, size=99),
            _trial(seed_transpiler=6, depth=100, two_qubit_gate_count=1, one_qubit_gate_count=2, size=99),
            _trial(seed_transpiler=5, depth=100, two_qubit_gate_count=1, one_qubit_gate_count=2, size=3),
            _trial(seed_transpiler=4, depth=100, two_qubit_gate_count=1, one_qubit_gate_count=2, size=3),
        ]
        result = aggregate_strategy_statistics(pd.DataFrame(rows)).iloc[0]
        self.assertEqual(result["best_seed_transpiler"], 4)
        self.assertEqual(result["best_two_qubit_gate_count"], 1.0)
        self.assertEqual(result["best_graph_state_transpiled_qpy"], "default-4.qpy")

    def test_best_trial_breaks_each_remaining_contract_tie(self) -> None:
        rows = [
            _trial(strategy_name="depth", seed_transpiler=1, two_qubit_gate_count=1, depth=20, one_qubit_gate_count=1, size=1),
            _trial(strategy_name="depth", seed_transpiler=2, two_qubit_gate_count=1, depth=10, one_qubit_gate_count=99, size=99),
            _trial(strategy_name="one_qubit", seed_transpiler=1, two_qubit_gate_count=1, depth=10, one_qubit_gate_count=5, size=1),
            _trial(strategy_name="one_qubit", seed_transpiler=2, two_qubit_gate_count=1, depth=10, one_qubit_gate_count=4, size=99),
            _trial(strategy_name="size", seed_transpiler=1, two_qubit_gate_count=1, depth=10, one_qubit_gate_count=4, size=8),
            _trial(strategy_name="size", seed_transpiler=2, two_qubit_gate_count=1, depth=10, one_qubit_gate_count=4, size=7),
            _trial(strategy_name="seed", seed_transpiler=2, two_qubit_gate_count=1, depth=10, one_qubit_gate_count=4, size=7),
            _trial(strategy_name="seed", seed_transpiler=1, two_qubit_gate_count=1, depth=10, one_qubit_gate_count=4, size=7),
        ]
        result = aggregate_strategy_statistics(pd.DataFrame(rows)).set_index("strategy_name")
        self.assertEqual(result.loc["depth", "best_seed_transpiler"], 2)
        self.assertEqual(result.loc["one_qubit", "best_seed_transpiler"], 2)
        self.assertEqual(result.loc["size", "best_seed_transpiler"], 2)
        self.assertEqual(result.loc["seed", "best_seed_transpiler"], 1)

    def test_n_qutrits_comes_from_selected_successful_trial(self) -> None:
        result = aggregate_strategy_statistics(
            pd.DataFrame(
                [
                    _trial(seed_transpiler=1, success=False, status="failed", n_qutrits=99),
                    _trial(
                        seed_transpiler=2,
                        depth=10,
                        two_qubit_gate_count=1,
                        n_qutrits=3,
                        graph_state_transpiled_qpy="best.qpy",
                    ),
                ]
            )
        ).iloc[0]
        self.assertEqual(result["n_qutrits"], 3)
        self.assertEqual(result["best_seed_transpiler"], 2)
        self.assertEqual(result["best_graph_state_transpiled_qpy"], "best.qpy")

    def test_single_success_is_flagged_for_insufficient_stability(self) -> None:
        result = aggregate_strategy_statistics(pd.DataFrame([_trial()])).iloc[0]
        self.assertEqual(result["std_depth"], 0.0)
        self.assertEqual(result["std_two_qubit_gate_count"], 0.0)
        self.assertTrue(result["insufficient_stability_samples"])
        self.assertTrue(result["pareto_eligible"])
        self.assertEqual(result["analysis_status"], "eligible")

    def test_zero_successes_are_retained_as_diagnostics(self) -> None:
        result = aggregate_strategy_statistics(pd.DataFrame([_trial(success="false", status="failed")])).iloc[0]
        self.assertEqual(result["total_trial_count"], 1)
        self.assertEqual(result["successful_trial_count"], 0)
        self.assertEqual(result["failed_trial_count"], 1)
        self.assertEqual(result["success_rate"], 0.0)
        self.assertTrue(pd.isna(result["mean_depth"]))
        self.assertTrue(pd.isna(result["best_seed_transpiler"]))
        self.assertTrue(result["insufficient_stability_samples"])
        self.assertFalse(result["pareto_eligible"])
        self.assertEqual(result["analysis_status"], "no_successful_trials")

    def test_duplicate_trial_identity_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate.*state_name.*seed_transpiler"):
            aggregate_strategy_statistics(pd.DataFrame([_trial(), _trial(depth=11)]))

    def test_invalid_successful_metrics_name_identity_and_column(self) -> None:
        invalid_values = (-1, np.nan, np.inf, -np.inf, "bad")
        for column in ("depth", "two_qubit_gate_count", "one_qubit_gate_count", "size"):
            for value in invalid_values:
                with self.subTest(column=column, value=value):
                    with self.assertRaisesRegex(ValueError, rf"ghz3.*p001.*default.*{column}"):
                        aggregate_strategy_statistics(pd.DataFrame([_trial(**{column: value})]))

    def test_complex_successful_metrics_are_rejected_without_complex_warning(self) -> None:
        for column in ("depth", "two_qubit_gate_count", "one_qubit_gate_count", "size"):
            for value in (1 + 2j, complex(1, 0)):
                with self.subTest(column=column, value=value):
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        with self.assertRaisesRegex(ValueError, rf"ghz3.*p001.*default.*{column}"):
                            aggregate_strategy_statistics(pd.DataFrame([_trial(**{column: value})]))
                    self.assertFalse(any(issubclass(item.category, np.ComplexWarning) for item in caught))

    def test_invalid_success_encoding_names_success_and_identity(self) -> None:
        with self.assertRaisesRegex(ValueError, r"success.*ghz3.*p001.*default"):
            aggregate_strategy_statistics(pd.DataFrame([_trial(success="maybe")]))

    def test_true_success_with_non_ok_status_is_a_failed_diagnostic(self) -> None:
        result = aggregate_strategy_statistics(
            pd.DataFrame([_trial(success=True, status="failed")])
        ).iloc[0]
        self.assertEqual(result["successful_trial_count"], 0)
        self.assertEqual(result["failed_trial_count"], 1)
        self.assertFalse(result["pareto_eligible"])
        self.assertEqual(result["analysis_status"], "no_successful_trials")

    def test_optional_boundaries_partition_each_group(self) -> None:
        rows = [
            _trial(state_name="ghz3", seed_transpiler=1, iqm_backend_name="a", backend_calibration_set_id="c1", selection_label="exact"),
            _trial(state_name="ame43", seed_transpiler=1, iqm_backend_name="a", backend_calibration_set_id="c1", selection_label="exact"),
            _trial(state_name="ghz3", seed_transpiler=1, iqm_backend_name="b", backend_calibration_set_id="c1", selection_label="exact"),
            _trial(state_name="ghz3", seed_transpiler=1, iqm_backend_name="a", backend_calibration_set_id="c2", selection_label="exact"),
            _trial(state_name="ghz3", seed_transpiler=1, iqm_backend_name="a", backend_calibration_set_id="c1", selection_label="fid099"),
        ]
        result = aggregate_strategy_statistics(pd.DataFrame(rows))
        self.assertEqual(len(result), 5)
        self.assertEqual(result.groupby([*OPTIONAL_BOUNDARY_COLUMNS, "state_name"]).size().sum(), 5)

    def test_missing_required_columns_are_named(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing required.*depth"):
            aggregate_strategy_statistics(pd.DataFrame([{"state_name": "ghz3"}]))

    def test_csv_boolean_encodings_are_normalized_without_truthiness_bugs(self) -> None:
        rows = [
            _trial(seed_transpiler=1, success="False", status="ok"),
            _trial(seed_transpiler=2, success="yes", status="ok"),
            _trial(seed_transpiler=3, success=0, status="ok"),
            _trial(seed_transpiler=4, success="1", status="ok"),
        ]
        result = aggregate_strategy_statistics(pd.DataFrame(rows)).iloc[0]
        self.assertEqual(result["successful_trial_count"], 2)
        self.assertEqual(result["failed_trial_count"], 2)

    def test_numeric_string_metrics_are_accepted(self) -> None:
        result = aggregate_strategy_statistics(
            pd.DataFrame(
                [
                    _trial(
                        depth="10",
                        two_qubit_gate_count="4",
                        one_qubit_gate_count="8",
                        size="20",
                    )
                ]
            )
        ).iloc[0]
        self.assertEqual(result["mean_depth"], 10.0)
        self.assertEqual(result["best_two_qubit_gate_count"], 4.0)

    def test_na_boundary_values_form_one_group_and_duplicate_identity_is_rejected(self) -> None:
        rows = [
            _trial(seed_transpiler=1, iqm_backend_name=np.nan),
            _trial(seed_transpiler=2, iqm_backend_name=np.nan),
        ]
        result = aggregate_strategy_statistics(pd.DataFrame(rows))
        self.assertEqual(len(result), 1)
        self.assertTrue(pd.isna(result.iloc[0]["iqm_backend_name"]))

        with self.assertRaisesRegex(ValueError, "duplicate concrete trial identity"):
            aggregate_strategy_statistics(pd.DataFrame([rows[0], rows[0].copy()]))

    def test_output_order_is_stable_across_shuffled_input(self) -> None:
        rows = [
            _trial(state_name="ghz3", strategy_name="z", seed_transpiler=1, iqm_backend_name="b", backend_calibration_set_id="c2", selection_label="fid099"),
            _trial(state_name="ame43", strategy_name="a", seed_transpiler=1, iqm_backend_name="a", backend_calibration_set_id="c1", selection_label="exact"),
            _trial(state_name="ghz3", strategy_name="a", seed_transpiler=1, iqm_backend_name="a", backend_calibration_set_id="c1", selection_label="exact"),
        ]
        original = aggregate_strategy_statistics(pd.DataFrame(rows))
        shuffled = aggregate_strategy_statistics(pd.DataFrame(rows).sample(frac=1, random_state=7))
        pd.testing.assert_frame_equal(original, shuffled)


if __name__ == "__main__":
    unittest.main()
