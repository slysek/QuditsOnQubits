from __future__ import annotations

import sys
import time
import unittest
import warnings
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit.quantum_info import Statevector


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
    ParetoAnalysisResult,
    TRIAL_ID_COLUMNS,
    _dominates,
    analyze_iqm_trials,
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
        trials = pd.DataFrame(
            [
                _trial(
                    depth="10",
                    two_qubit_gate_count="4",
                    one_qubit_gate_count="8",
                    size="20",
                )
            ]
        )
        for column in ("depth", "two_qubit_gate_count", "one_qubit_gate_count", "size"):
            trials[column] = trials[column].astype("string")

        result = aggregate_strategy_statistics(trials).iloc[0]
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

    def test_more_than_nine_pareto_layers_sort_numerically(self) -> None:
        rows = [
            _statistics_row(f"layer_{number:02d}", (number, number, number))
            for number in range(1, 13)
        ]
        result = rank_pareto_candidates(pd.DataFrame(rows))
        self.assertEqual(result["candidate_name"].tolist(), [f"layer_{number:02d}" for number in range(1, 13)])
        self.assertEqual(result["pareto_rank"].tolist(), list(range(1, 13)))
        self.assertEqual(result["recommendation_order"].tolist(), list(range(1, 13)))

    def test_large_finite_weights_normalize_without_overflow(self) -> None:
        rows = [
            _statistics_row("first", (1, 2, 3)),
            _statistics_row("second", (3, 2, 1)),
        ]
        equal_weight_scores = rank_pareto_candidates(
            pd.DataFrame(rows),
            objective_weights={column: 1 for column in OBJECTIVE_COLUMNS},
        )["ideal_score"]
        large_weight_scores = rank_pareto_candidates(
            pd.DataFrame(rows),
            objective_weights={column: 1e308 for column in OBJECTIVE_COLUMNS},
        )["ideal_score"]
        pd.testing.assert_series_equal(equal_weight_scores, large_weight_scores)

    def test_every_optional_boundary_and_na_value_partition_ranks_and_normalization(self) -> None:
        rows = [
            _statistics_row("state_a", (2, 2, 2), state_name="state_a", iqm_backend_name="state", backend_calibration_set_id="state", selection_label="state"),
            _statistics_row("state_b", (1, 1, 1), state_name="state_b", iqm_backend_name="state", backend_calibration_set_id="state", selection_label="state"),
        ]

        def add_pair(prefix: str, metrics: tuple[int, int], **boundary: object) -> None:
            rows.extend(
                [
                    _statistics_row(f"{prefix}_best", (metrics[0], metrics[0], metrics[0]), **boundary),
                    _statistics_row(f"{prefix}_worse", (metrics[1], metrics[1], metrics[1]), **boundary),
                ]
            )

        add_pair("backend_a", (1, 2), iqm_backend_name="backend_a", backend_calibration_set_id="backend", selection_label="backend")
        add_pair("backend_b", (10, 20), iqm_backend_name="backend_b", backend_calibration_set_id="backend", selection_label="backend")
        add_pair("calibration_one", (100, 200), iqm_backend_name="calibration", backend_calibration_set_id="calibration_one", selection_label="calibration")
        add_pair("calibration_two", (1000, 2000), iqm_backend_name="calibration", backend_calibration_set_id="calibration_two", selection_label="calibration")
        add_pair("selection_one", (3, 6), iqm_backend_name="selection", backend_calibration_set_id="selection", selection_label="selection_one")
        add_pair("selection_two", (30, 60), iqm_backend_name="selection", backend_calibration_set_id="selection", selection_label="selection_two")
        add_pair("na_backend", (7, 14), iqm_backend_name=np.nan, backend_calibration_set_id="na_backend", selection_label="na_backend")
        add_pair("na_calibration", (8, 16), iqm_backend_name="na_calibration", backend_calibration_set_id=np.nan, selection_label="na_calibration")
        add_pair("na_selection", (9, 18), iqm_backend_name="na_selection", backend_calibration_set_id="na_selection", selection_label=np.nan)

        result = rank_pareto_candidates(pd.DataFrame(rows)).set_index("candidate_name")
        for prefix in (
            "backend_a", "backend_b", "calibration_one", "calibration_two",
            "selection_one", "selection_two", "na_backend", "na_calibration", "na_selection",
        ):
            with self.subTest(prefix=prefix):
                self.assertEqual(result.loc[f"{prefix}_best", "pareto_rank"], 1)
                self.assertEqual(result.loc[f"{prefix}_worse", "pareto_rank"], 2)
                self.assertEqual(result.loc[f"{prefix}_best", "normalized_mean_depth"], 0.0)
                self.assertEqual(result.loc[f"{prefix}_worse", "normalized_mean_depth"], 1.0)
        self.assertEqual(result.loc["state_a", "pareto_rank"], 1)
        self.assertEqual(result.loc["state_b", "pareto_rank"], 1)

    def test_dominance_chain_uses_quadratic_layer_assignment(self) -> None:
        count = 400
        rows = [
            _statistics_row(f"chain_{number:04d}", (number, number, number))
            for number in range(count)
        ]
        started = time.perf_counter()
        result = assign_pareto_ranks(pd.DataFrame(rows))
        elapsed = time.perf_counter() - started
        self.assertEqual(result["pareto_rank"].tolist(), list(range(1, count + 1)))
        self.assertLess(elapsed, 10.0)

    def test_weight_conversion_errors_use_descriptive_value_error(self) -> None:
        weights = {column: 1 for column in OBJECTIVE_COLUMNS}
        weights["mean_depth"] = 10**10000
        with self.assertRaisesRegex(ValueError, "objective_weights.*finite nonnegative"):
            rank_pareto_candidates(pd.DataFrame(self.rows), objective_weights=weights)

    def test_ranking_columns_use_nullable_contract_dtypes(self) -> None:
        rows = [
            _statistics_row("eligible", (1, 2, 3)),
            _statistics_row("diagnostic", (np.nan, np.nan, np.nan), pareto_eligible=False),
        ]
        result = rank_pareto_candidates(pd.DataFrame(rows))
        self.assertEqual(str(result["pareto_rank"].dtype), "Int64")
        self.assertEqual(str(result["recommendation_order"].dtype), "Int64")
        for column in (
            "normalized_mean_two_qubit_gate_count",
            "normalized_mean_depth",
            "normalized_std_depth",
            "ideal_score",
        ):
            self.assertEqual(str(result[column].dtype), "Float64")

        empty = rank_pareto_candidates(pd.DataFrame())
        diagnostics = rank_pareto_candidates(pd.DataFrame([rows[1]]))
        for output in (empty, diagnostics):
            self.assertEqual(str(output["pareto_rank"].dtype), "Int64")
            self.assertEqual(str(output["recommendation_order"].dtype), "Int64")
            self.assertEqual(str(output["ideal_score"].dtype), "Float64")


class IqmTrialAnalysisTests(unittest.TestCase):
    @staticmethod
    def _loader(states, calls=None):
        def load(path, logical_qubit_count, *, max_qubits):
            if calls is not None:
                calls.append((path, logical_qubit_count, max_qubits))
            return states[Path(path).name], ""

        return load

    def test_end_to_end_groups_equivalent_pareto_tradeoffs_and_reports_counts(self) -> None:
        trials = pd.DataFrame(
            [
                _trial(candidate_name="low_2q", strategy_name="a", seed_transpiler=seed,
                       depth=10, two_qubit_gate_count=2, graph_state_transpiled_qpy="a.qpy")
                for seed in (1, 2)
            ]
            + [
                _trial(candidate_name="low_depth", strategy_name="b", seed_transpiler=seed,
                       depth=5, two_qubit_gate_count=4, graph_state_transpiled_qpy="b.qpy")
                for seed in (1, 2)
            ]
        )
        result = analyze_iqm_trials(
            trials,
            state_loader=self._loader({"a.qpy": Statevector([1, 0]), "b.qpy": Statevector([-1j, 0])}),
        )

        self.assertIsInstance(result, ParetoAnalysisResult)
        self.assertEqual(len(result.strategy_statistics), 2)
        self.assertEqual(result.pareto_ranked["pareto_rank"].min(), 1)
        self.assertEqual(len(result.state_equivalence_groups), 2)
        self.assertEqual(len(result.recommended_circuits), 1)
        self.assertEqual(
            result.summary_counts,
            {
                "analyzed_strategy_combination_count": 2,
                "pareto_front_count": 2,
                "state_equivalence_group_count": 1,
                "recommended_circuit_count": 1,
            },
        )
        with self.assertRaises(FrozenInstanceError):
            result.summary_counts = {}
        self.assertEqual(
            result.pareto_ranked.sort_values("recommendation_order")["candidate_name"].tolist(),
            ["low_2q", "low_depth"],
        )

        custom = analyze_iqm_trials(
            trials,
            objective_weights={
                "mean_two_qubit_gate_count": 0.1,
                "mean_depth": 0.9,
                "std_depth": 0.0,
            },
            state_loader=self._loader({"a.qpy": Statevector([1, 0]), "b.qpy": Statevector([-1j, 0])}),
        ).pareto_ranked.set_index("candidate_name")
        self.assertAlmostEqual(custom.loc["low_2q", "ideal_score"], 0.9)
        self.assertAlmostEqual(custom.loc["low_depth", "ideal_score"], 0.1)
        self.assertEqual(custom.sort_values("recommendation_order").index.tolist(), ["low_depth", "low_2q"])

    def test_no_success_diagnostic_missing_qpy_and_custom_options_flow_through(self) -> None:
        trials = pd.DataFrame(
            [
                _trial(candidate_name="failed", strategy_name="failed", success=False, status="failed"),
                _trial(candidate_name="ok", strategy_name="ok", seed_transpiler=1,
                       graph_state_transpiled_qpy="missing.qpy"),
            ]
        )
        calls = []
        def missing_loader(path, logical_qubit_count, *, max_qubits):
            calls.append((path, logical_qubit_count, max_qubits))
            return None, "Missing QPY file: injected for test"

        result = analyze_iqm_trials(
            trials,
            objective_weights={
                "mean_two_qubit_gate_count": 1,
                "mean_depth": 0,
                "std_depth": 0,
            },
            max_state_qubits=7,
            state_loader=missing_loader,
        )

        self.assertEqual(len(result.strategy_statistics), 2)
        self.assertFalse(result.strategy_statistics.loc[
            result.strategy_statistics["candidate_name"] == "failed", "pareto_eligible"
        ].item())
        self.assertEqual(result.state_equivalence_groups.loc[
            result.state_equivalence_groups["candidate_name"] == "ok", "state_equivalence_status"
        ].item(), "missing_qpy")
        detailed = result.state_equivalence_groups.loc[
            result.state_equivalence_groups["candidate_name"] == "ok"
        ]
        compact = result.recommended_circuits
        self.assertTrue(pd.notna(detailed["state_equivalence_group_id"].item()))
        self.assertEqual(len(compact), 1)
        self.assertEqual(
            detailed.loc[:, list(IDENTITY_COLUMNS)].iloc[0].tolist(),
            compact.loc[:, list(IDENTITY_COLUMNS)].iloc[0].tolist(),
        )
        self.assertEqual(result.summary_counts["state_equivalence_group_count"], 1)
        self.assertEqual(result.summary_counts["recommended_circuit_count"], 1)
        self.assertEqual(calls[0][2], 7)

    def test_state_loader_cache_is_preserved_for_duplicate_qpy_paths(self) -> None:
        trials = pd.DataFrame(
            [
                _trial(candidate_name=name, strategy_name=name, seed_transpiler=seed,
                       graph_state_transpiled_qpy="same.qpy", depth=10 + index)
                for index, name in enumerate(("a", "b"))
                for seed in (1, 2)
            ]
        )
        calls = []
        analyze_iqm_trials(
            trials,
            state_loader=self._loader({"same.qpy": Statevector([1, 0])}, calls),
        )

        self.assertEqual(len(calls), 1)

    def test_empty_input_has_complete_empty_analysis_schema_and_zero_counts(self) -> None:
        result = analyze_iqm_trials(pd.DataFrame())

        self.assertTrue(result.strategy_statistics.empty)
        self.assertTrue(result.pareto_ranked.empty)
        self.assertTrue(result.state_equivalence_groups.empty)
        self.assertTrue(result.recommended_circuits.empty)
        self.assertIn("pareto_rank", result.pareto_ranked.columns)
        self.assertIn("state_equivalence_group_id", result.state_equivalence_groups.columns)
        self.assertIn("is_state_equivalence_recommendation", result.recommended_circuits.columns)
        self.assertEqual(
            result.summary_counts,
            {
                "analyzed_strategy_combination_count": 0,
                "pareto_front_count": 0,
                "state_equivalence_group_count": 0,
                "recommended_circuit_count": 0,
            },
        )

    def test_invalid_trials_and_multiple_boundaries_are_delegated_without_cross_boundary_grouping(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            analyze_iqm_trials(pd.DataFrame([{"state_name": "ghz3"}]))

        trials = pd.DataFrame(
            [
                _trial(candidate_name="a", strategy_name="a", seed_transpiler=seed,
                       graph_state_transpiled_qpy="same.qpy", iqm_backend_name=backend,
                       backend_calibration_set_id=calibration, selection_label=label)
                for backend, calibration, label in (
                    ("one", "calibration_one", "exact"),
                    ("two", "calibration_two", "fid099"),
                )
                for seed in (1, 2)
            ]
        )
        result = analyze_iqm_trials(
            trials,
            state_loader=self._loader({"same.qpy": Statevector([1, 0])}),
        )
        self.assertEqual(result.summary_counts["state_equivalence_group_count"], 2)


if __name__ == "__main__":
    unittest.main()
