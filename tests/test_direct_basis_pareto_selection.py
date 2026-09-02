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
    DEFAULT_OBJECTIVE_WEIGHTS,
    IDENTITY_COLUMNS,
    OBJECTIVE_COLUMNS,
    OPTIONAL_BOUNDARY_COLUMNS,
    TRIAL_ID_COLUMNS,
    _dominates,
    aggregate_strategy_statistics,
    assign_pareto_ranks,
    rank_pareto_candidates,
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


def _statistics_row(
    candidate_name: str,
    metrics: tuple[object, object, object],
    *,
    state_name: str = "ghz3",
    class_name: str = "product",
    strategy_name: str = "default",
    pareto_eligible: object = True,
    **boundary: object,
) -> dict[str, object]:
    return {
        "state_name": state_name,
        "class_name": class_name,
        "candidate_name": candidate_name,
        "strategy_name": strategy_name,
        "mean_two_qubit_gate_count": metrics[0],
        "mean_depth": metrics[1],
        "std_depth": metrics[2],
        "pareto_eligible": pareto_eligible,
        **boundary,
    }


class ParetoCandidateRankingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            _statistics_row("balanced", (3, 10, 2)),
            _statistics_row("low_2q", (2, 14, 3)),
            _statistics_row("stable", (4, 12, 1)),
            _statistics_row("dominated", (5, 16, 4)),
            _statistics_row("deeply_dominated", (6, 18, 5)),
            _statistics_row("balanced_copy", (3, 10, 2)),
        ]

    def test_default_weights_and_multiple_pareto_layers(self) -> None:
        self.assertEqual(
            DEFAULT_OBJECTIVE_WEIGHTS,
            {"mean_two_qubit_gate_count": 0.50, "mean_depth": 0.30, "std_depth": 0.20},
        )
        result = assign_pareto_ranks(pd.DataFrame(self.rows)).set_index("candidate_name")
        for candidate in ("balanced", "low_2q", "stable", "balanced_copy"):
            self.assertEqual(result.loc[candidate, "pareto_rank"], 1)
        self.assertEqual(result.loc["dominated", "pareto_rank"], 2)
        self.assertEqual(result.loc["deeply_dominated", "pareto_rank"], 3)
        self.assertEqual(
            result.loc["balanced", "pareto_metric_group_id"],
            result.loc["balanced_copy", "pareto_metric_group_id"],
        )
        self.assertNotEqual(result.loc["balanced", "pareto_metric_group_id"], result.loc["stable", "pareto_metric_group_id"])

    def test_dominance_is_strict_and_minimizes_all_objectives(self) -> None:
        self.assertTrue(_dominates((1, 2, 3), (1, 3, 4)))
        self.assertFalse(_dominates((1, 3, 4), (1, 2, 3)))
        self.assertFalse(_dominates((1, 4, 2), (2, 3, 3)))
        self.assertFalse(_dominates((1, 2, 3), (1, 2, 3)))

    def test_scores_are_global_per_partition_and_use_exact_formula(self) -> None:
        result = rank_pareto_candidates(pd.DataFrame(self.rows)).set_index("candidate_name")
        balanced = result.loc["balanced"]
        self.assertAlmostEqual(balanced["normalized_mean_two_qubit_gate_count"], 1 / 4)
        self.assertAlmostEqual(balanced["normalized_mean_depth"], 0.0)
        self.assertAlmostEqual(balanced["normalized_std_depth"], 1 / 4)
        self.assertAlmostEqual(balanced["ideal_score"], 0.50 / 4 + 0.20 / 4)
        self.assertLess(result.loc["balanced", "recommendation_order"], result.loc["dominated", "recommendation_order"])

    def test_constant_objectives_are_zero_and_equivalent_custom_weights_normalize(self) -> None:
        rows = [
            _statistics_row("a", (1, 10, 2)),
            _statistics_row("b", (3, 10, 2)),
        ]
        default = rank_pareto_candidates(pd.DataFrame(rows))
        custom = rank_pareto_candidates(
            pd.DataFrame(rows),
            objective_weights={"mean_two_qubit_gate_count": 5, "mean_depth": 3, "std_depth": 2},
        )
        self.assertTrue((default[["normalized_mean_depth", "normalized_std_depth"]] == 0.0).all().all())
        pd.testing.assert_series_equal(default["ideal_score"], custom["ideal_score"])

    def test_invalid_objective_weights_are_rejected(self) -> None:
        invalid = [
            {},
            {"mean_two_qubit_gate_count": 1, "mean_depth": 1},
            {**DEFAULT_OBJECTIVE_WEIGHTS, "extra": 1},
            {"mean_two_qubit_gate_count": -1, "mean_depth": 1, "std_depth": 1},
            {"mean_two_qubit_gate_count": np.nan, "mean_depth": 1, "std_depth": 1},
            {"mean_two_qubit_gate_count": np.inf, "mean_depth": 1, "std_depth": 1},
            {"mean_two_qubit_gate_count": 1 + 0j, "mean_depth": 1, "std_depth": 1},
            {"mean_two_qubit_gate_count": 0, "mean_depth": 0, "std_depth": 0},
        ]
        for weights in invalid:
            with self.subTest(weights=weights):
                with self.assertRaises(ValueError):
                    rank_pareto_candidates(pd.DataFrame(self.rows), objective_weights=weights)

    def test_boundaries_states_and_missing_boundary_values_are_independent(self) -> None:
        rows = [
            _statistics_row("first", (1, 9, 1), iqm_backend_name="a", backend_calibration_set_id="c1", selection_label="exact"),
            _statistics_row("second", (2, 10, 2), iqm_backend_name="a", backend_calibration_set_id="c1", selection_label="exact"),
            _statistics_row("first", (2, 10, 2), iqm_backend_name="b", backend_calibration_set_id="c1", selection_label="exact"),
            _statistics_row("second", (1, 9, 1), iqm_backend_name="b", backend_calibration_set_id="c1", selection_label="exact"),
            _statistics_row("first", (2, 10, 2), state_name="other", iqm_backend_name="a", backend_calibration_set_id="c1", selection_label="exact"),
            _statistics_row("second", (1, 9, 1), state_name="other", iqm_backend_name="a", backend_calibration_set_id="c1", selection_label="exact"),
            _statistics_row("na_first", (1, 9, 1), iqm_backend_name=np.nan, backend_calibration_set_id="c1", selection_label="exact"),
            _statistics_row("na_second", (2, 10, 2), iqm_backend_name=np.nan, backend_calibration_set_id="c1", selection_label="exact"),
        ]
        result = rank_pareto_candidates(pd.DataFrame(rows))

        def rank(state_name: str, backend: object, candidate_name: str) -> int:
            matching = result[
                (result["state_name"] == state_name)
                & (result["candidate_name"] == candidate_name)
                & (
                    result["iqm_backend_name"].isna()
                    if pd.isna(backend)
                    else result["iqm_backend_name"] == backend
                )
            ]
            return int(matching["pareto_rank"].item())

        self.assertEqual(rank("ghz3", "a", "first"), 1)
        self.assertEqual(rank("ghz3", "a", "second"), 2)
        self.assertEqual(rank("other", "a", "second"), 1)
        self.assertEqual(rank("other", "a", "first"), 2)
        self.assertEqual(rank("ghz3", np.nan, "na_first"), 1)
        self.assertEqual(rank("ghz3", np.nan, "na_second"), 2)

    def test_order_is_deterministic_after_identity_ties_and_shuffled_input(self) -> None:
        rows = [
            _statistics_row("z", (3, 10, 2), class_name="z", strategy_name="z"),
            _statistics_row("a", (3, 10, 2), class_name="a", strategy_name="a"),
        ]
        original = rank_pareto_candidates(pd.DataFrame(rows))
        shuffled = rank_pareto_candidates(pd.DataFrame(rows).sample(frac=1, random_state=4))
        pd.testing.assert_frame_equal(original, shuffled)
        self.assertEqual(original["candidate_name"].tolist(), ["a", "z"])

    def test_ineligible_diagnostics_follow_eligible_with_null_ranking_fields(self) -> None:
        rows = [
            _statistics_row("eligible", (1, 1, 1)),
            _statistics_row("z_diagnostic", ("not-a-number", None, np.nan), pareto_eligible=False),
            _statistics_row("a_diagnostic", ("not-a-number", None, np.nan), pareto_eligible=False),
        ]
        result = rank_pareto_candidates(pd.DataFrame(rows))
        self.assertEqual(result["candidate_name"].tolist(), ["eligible", "a_diagnostic", "z_diagnostic"])
        diagnostics = result.iloc[1:]
        self.assertTrue(diagnostics[["pareto_rank", "pareto_metric_group_id", "ideal_score", "normalized_mean_two_qubit_gate_count", "normalized_mean_depth", "normalized_std_depth"]].isna().all().all())
        self.assertEqual(result["recommendation_order"].tolist(), [1, 2, 3])

    def test_invalid_eligible_objectives_are_rejected_and_empty_input_has_stable_schema(self) -> None:
        for value in (-1, np.nan, np.inf, "bad", 1 + 0j):
            with self.subTest(value=value):
                rows = [_statistics_row("bad", (value, 1, 1))]
                with self.assertRaises(ValueError):
                    assign_pareto_ranks(pd.DataFrame(rows))
        ranked = rank_pareto_candidates(pd.DataFrame())
        self.assertTrue(ranked.empty)
        self.assertEqual(
            ranked.columns.tolist(),
            [
                "pareto_rank", "pareto_metric_group_id",
                "normalized_mean_two_qubit_gate_count", "normalized_mean_depth", "normalized_std_depth",
                "ideal_score", "recommendation_order",
            ],
        )


if __name__ == "__main__":
    unittest.main()
