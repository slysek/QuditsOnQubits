"""Post-transpilation, per-strategy trial aggregation for IQM benchmarks."""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .state_equivalence import group_state_equivalent_candidates, load_logical_state_from_qpy


IDENTITY_COLUMNS = ("state_name", "class_name", "candidate_name", "strategy_name")
OPTIONAL_BOUNDARY_COLUMNS = (
    "iqm_backend_name",
    "backend_calibration_set_id",
    "selection_label",
)
TRIAL_ID_COLUMNS = (*IDENTITY_COLUMNS, "seed_transpiler")
OBJECTIVE_COLUMNS = (
    "mean_two_qubit_gate_count",
    "mean_depth",
    "std_depth",
)
DEFAULT_OBJECTIVE_WEIGHTS = {
    "mean_two_qubit_gate_count": 0.50,
    "mean_depth": 0.30,
    "std_depth": 0.20,
}
BEST_TRIAL_ORDER = (
    "two_qubit_gate_count",
    "depth",
    "one_qubit_gate_count",
    "size",
    "seed_transpiler",
)

_REQUIRED_INPUT_COLUMNS = (
    *TRIAL_ID_COLUMNS,
    "success",
    "status",
    *BEST_TRIAL_ORDER[:-1],
    "graph_state_transpiled_qpy",
)
_METRIC_COLUMNS = BEST_TRIAL_ORDER[:-1]
_PARETO_COLUMNS = ("pareto_rank", "pareto_metric_group_id")
_NORMALIZED_OBJECTIVE_COLUMNS = tuple(f"normalized_{column}" for column in OBJECTIVE_COLUMNS)
_RANKING_COLUMNS = (*_PARETO_COLUMNS, *_NORMALIZED_OBJECTIVE_COLUMNS, "ideal_score", "recommendation_order")
_OUTPUT_COLUMNS = (
    *IDENTITY_COLUMNS,
    "total_trial_count",
    "successful_trial_count",
    "failed_trial_count",
    "success_rate",
    "mean_depth",
    "min_depth",
    "max_depth",
    "std_depth",
    "mean_two_qubit_gate_count",
    "min_two_qubit_gate_count",
    "max_two_qubit_gate_count",
    "std_two_qubit_gate_count",
    "insufficient_stability_samples",
    "best_seed_transpiler",
    "best_graph_state_transpiled_qpy",
    "best_depth",
    "best_two_qubit_gate_count",
    "best_one_qubit_gate_count",
    "best_size",
    "pareto_eligible",
    "analysis_status",
)


@dataclass(frozen=True)
class ParetoAnalysisResult:
    """Post-transpilation Pareto analysis artifacts for one IQM trial table."""

    strategy_statistics: pd.DataFrame
    pareto_ranked: pd.DataFrame
    state_equivalence_groups: pd.DataFrame
    recommended_circuits: pd.DataFrame
    summary_counts: dict[str, int]


def write_pareto_analysis_outputs(
    output_dir: str | Path, analysis: ParetoAnalysisResult
) -> dict[str, str]:
    """Write the four CSV artifacts produced by Pareto post-processing."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    outputs = {
        "strategy_statistics_csv": output_path / "strategy_statistics.csv",
        "pareto_ranked_csv": output_path / "pareto_ranked.csv",
        "state_equivalence_groups_csv": output_path / "state_equivalence_groups.csv",
        "recommended_circuits_csv": output_path / "recommended_circuits.csv",
    }
    analysis.strategy_statistics.to_csv(outputs["strategy_statistics_csv"], index=False)
    analysis.pareto_ranked.to_csv(outputs["pareto_ranked_csv"], index=False)
    analysis.state_equivalence_groups.to_csv(outputs["state_equivalence_groups_csv"], index=False)
    analysis.recommended_circuits.to_csv(outputs["recommended_circuits_csv"], index=False)
    return {key: str(path) for key, path in outputs.items()}


def _success_value(value: object) -> bool:
    """Normalize CSV-safe success encodings without Python truthiness."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (not isinstance(value, (list, tuple, dict, set)) and pd.isna(value)):
        raise ValueError("success must be a boolean or a recognized true/false encoding")
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False
    if isinstance(value, str):
        normalized = value.strip().casefold()
        if normalized in {"true", "1", "yes", "y", "t"}:
            return True
        if normalized in {"false", "0", "no", "n", "f"}:
            return False
    raise ValueError(
        f"invalid success value {value!r}; expected a boolean or a recognized true/false encoding"
    )


def _present_boundary_columns(columns: Iterable[str]) -> tuple[str, ...]:
    present = set(columns)
    return tuple(column for column in OPTIONAL_BOUNDARY_COLUMNS if column in present)


def _output_columns(boundary_columns: tuple[str, ...], include_n_qutrits: bool) -> list[str]:
    columns = [*boundary_columns, *_OUTPUT_COLUMNS]
    if include_n_qutrits:
        columns.insert(len(boundary_columns) + len(IDENTITY_COLUMNS), "n_qutrits")
    return columns


def _identity_description(row: pd.Series) -> str:
    return ", ".join(
        f"{column}={row[column]!r}" for column in ("state_name", "candidate_name", "strategy_name")
    )


def _is_real_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, (bool, np.bool_))


def _stable_value_key(value: object) -> tuple[int, str]:
    """Provide an order for display metadata, including nullable boundaries."""
    if value is None or (not isinstance(value, (list, tuple, dict, set)) and pd.isna(value)):
        return (1, "")
    return (0, f"{type(value).__name__}:{value}")


def _deterministic_sort(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    columns = tuple(dict.fromkeys(columns))
    if frame.empty or not columns:
        return frame.copy()
    sortable = frame.copy()
    key_columns: list[str] = []
    for position, column in enumerate(columns):
        key_column = f"__pareto_sort_{position}"
        values = sortable[column]
        non_missing = values.loc[
            values.map(
                lambda value: value is not None
                and (isinstance(value, (list, tuple, dict, set)) or not pd.isna(value))
            )
        ]
        if not non_missing.empty and non_missing.map(_is_real_number).all():
            sortable[key_column] = pd.to_numeric(values, errors="raise")
        else:
            sortable[key_column] = values.map(_stable_value_key)
        key_columns.append(key_column)
    return sortable.sort_values(key_columns, kind="mergesort", na_position="last").drop(columns=key_columns)


def _present_partition_columns(columns: Iterable[str]) -> tuple[str, ...]:
    return ("state_name", *_present_boundary_columns(columns))


def _eligible_mask(statistics: pd.DataFrame) -> pd.Series:
    return statistics["pareto_eligible"].map(
        lambda value: isinstance(value, (bool, np.bool_)) and bool(value)
    )


def _validate_statistics_schema(statistics: pd.DataFrame) -> None:
    required = (*IDENTITY_COLUMNS, "pareto_eligible", *OBJECTIVE_COLUMNS)
    missing = [column for column in required if column not in statistics.columns]
    if missing:
        raise ValueError(f"statistics is missing required columns: {', '.join(missing)}")


def _validate_eligible_objectives(statistics: pd.DataFrame, eligible: pd.Series) -> None:
    for column in OBJECTIVE_COLUMNS:
        for position, value in statistics.loc[eligible, column].items():
            if not _is_real_number(value) or not np.isfinite(float(value)) or float(value) < 0:
                row = statistics.loc[position]
                raise ValueError(
                    f"invalid eligible objective {_identity_description(row)}, {column}={value!r}; "
                    "expected a finite nonnegative real number"
                )


def _dominates(left: Iterable[float], right: Iterable[float]) -> bool:
    """Return whether ``left`` is strictly Pareto-better than ``right``."""
    left_values = tuple(left)
    right_values = tuple(right)
    return all(left_value <= right_value for left_value, right_value in zip(left_values, right_values, strict=True)) and any(
        left_value < right_value for left_value, right_value in zip(left_values, right_values, strict=True)
    )


def assign_pareto_ranks(statistics: pd.DataFrame) -> pd.DataFrame:
    """Assign nondominated-front ranks to eligible per-state benchmark statistics.

    Optional backend, calibration, and selection columns, when present, form
    additional independent ranking boundaries.  Diagnostic rows are retained
    but never participate in metric validation or front construction.
    """
    if statistics.empty:
        result = statistics.copy()
        result["pareto_rank"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
        result["pareto_metric_group_id"] = pd.Series(pd.NA, index=result.index, dtype="string")
        return result

    _validate_statistics_schema(statistics)
    result = statistics.copy().reset_index(drop=True)
    boundary_columns = _present_boundary_columns(result.columns)
    partition_columns = _present_partition_columns(result.columns)
    eligible = _eligible_mask(result)
    _validate_eligible_objectives(result, eligible)
    result.loc[eligible, list(OBJECTIVE_COLUMNS)] = result.loc[eligible, list(OBJECTIVE_COLUMNS)].astype(float)
    result["pareto_rank"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    result["pareto_metric_group_id"] = pd.Series(pd.NA, index=result.index, dtype="string")

    ordered = _deterministic_sort(result, [*boundary_columns, "state_name", *IDENTITY_COLUMNS])
    for _, partition in ordered.groupby(list(partition_columns), dropna=False, sort=False):
        positions = partition.index[eligible.loc[partition.index]].tolist()
        if not positions:
            continue
        metrics = {
            position: tuple(float(result.at[position, column]) for column in OBJECTIVE_COLUMNS)
            for position in positions
        }
        metric_group_ids = {
            metric: f"pareto_metrics_{number:04d}"
            for number, metric in enumerate(sorted(set(metrics.values())), start=1)
        }
        for position, metric in metrics.items():
            result.at[position, "pareto_metric_group_id"] = metric_group_ids[metric]

        values = np.array([metrics[position] for position in positions], dtype=float)
        dominance = np.all(values[:, np.newaxis, :] <= values[np.newaxis, :, :], axis=2) & np.any(
            values[:, np.newaxis, :] < values[np.newaxis, :, :], axis=2
        )
        domination_count = dominance.sum(axis=0)
        remaining = np.ones(len(positions), dtype=bool)
        rank = 1
        while remaining.any():
            front = np.flatnonzero(remaining & (domination_count == 0))
            result.loc[[positions[index] for index in front], "pareto_rank"] = rank
            domination_count -= dominance[front].sum(axis=0)
            remaining[front] = False
            domination_count[~remaining] = -1
            rank += 1

    return _deterministic_sort(
        result, [*boundary_columns, "state_name", "pareto_rank", *IDENTITY_COLUMNS]
    ).reset_index(drop=True)


def _normalized_objective_weights(objective_weights: Mapping[str, object] | None) -> dict[str, float]:
    weights = DEFAULT_OBJECTIVE_WEIGHTS if objective_weights is None else objective_weights
    if not isinstance(weights, Mapping) or set(weights) != set(OBJECTIVE_COLUMNS):
        raise ValueError(f"objective_weights must contain exactly: {', '.join(OBJECTIVE_COLUMNS)}")
    normalized: dict[str, float] = {}
    for column in OBJECTIVE_COLUMNS:
        value = weights[column]
        if not _is_real_number(value):
            raise ValueError("objective_weights must be finite nonnegative real numbers")
        try:
            numeric_value = float(value)
        except (OverflowError, TypeError, ValueError) as error:
            raise ValueError("objective_weights must be finite nonnegative real numbers") from error
        if not np.isfinite(numeric_value) or numeric_value < 0:
            raise ValueError("objective_weights must be finite nonnegative real numbers")
        normalized[column] = numeric_value
    maximum = max(normalized.values())
    if maximum <= 0:
        raise ValueError("objective_weights must have a positive total")
    scaled = {column: value / maximum for column, value in normalized.items()}
    total = sum(scaled.values())
    return {column: value / total for column, value in scaled.items()}


def rank_pareto_candidates(
    statistics: pd.DataFrame, *, objective_weights: Mapping[str, object] | None = None
) -> pd.DataFrame:
    """Score candidates within Pareto layers without changing Pareto dominance."""
    weights = _normalized_objective_weights(objective_weights)
    result = assign_pareto_ranks(statistics)
    for column in _NORMALIZED_OBJECTIVE_COLUMNS:
        result[column] = pd.Series(pd.NA, index=result.index, dtype="Float64")
    result["ideal_score"] = pd.Series(pd.NA, index=result.index, dtype="Float64")
    result["recommendation_order"] = pd.Series(pd.NA, index=result.index, dtype="Int64")
    if result.empty:
        return result

    boundary_columns = _present_boundary_columns(result.columns)
    partition_columns = _present_partition_columns(result.columns)
    ordered = _deterministic_sort(result, [*boundary_columns, "state_name", *IDENTITY_COLUMNS])
    ranked_partitions: list[pd.DataFrame] = []
    for _, partition in ordered.groupby(list(partition_columns), dropna=False, sort=False):
        partition = partition.copy()
        eligible = _eligible_mask(partition)
        eligible_rows = partition.loc[eligible].copy()
        if not eligible_rows.empty:
            for objective, normalized_column in zip(
                OBJECTIVE_COLUMNS, _NORMALIZED_OBJECTIVE_COLUMNS, strict=True
            ):
                minimum = eligible_rows[objective].min()
                maximum = eligible_rows[objective].max()
                if maximum == minimum:
                    eligible_rows[normalized_column] = 0.0
                else:
                    eligible_rows[normalized_column] = (
                        eligible_rows[objective] - minimum
                    ) / (maximum - minimum)
            eligible_rows["ideal_score"] = sum(
                eligible_rows[f"normalized_{objective}"] * weights[objective]
                for objective in OBJECTIVE_COLUMNS
            )
            eligible_rows = _deterministic_sort(
                eligible_rows,
                ["pareto_rank", "ideal_score", *OBJECTIVE_COLUMNS, *IDENTITY_COLUMNS[1:]],
            )
        diagnostics = _deterministic_sort(partition.loc[~eligible], IDENTITY_COLUMNS[1:])
        scoring_columns = [*_NORMALIZED_OBJECTIVE_COLUMNS, "ideal_score"]
        partition.loc[eligible_rows.index, scoring_columns] = eligible_rows[scoring_columns]
        ranked = partition.loc[[*eligible_rows.index, *diagnostics.index]].copy()
        ranked["recommendation_order"] = pd.array(range(1, len(ranked) + 1), dtype="Int64")
        ranked_partitions.append(ranked)
    return pd.concat(ranked_partitions, ignore_index=True)


def _validate_successful_metrics(trials: pd.DataFrame, successful: pd.Series) -> None:
    for column in _METRIC_COLUMNS:
        successful_values = trials.loc[successful, column]
        complex_values = successful_values.map(
            lambda value: isinstance(value, (complex, np.complexfloating))
        )
        if complex_values.any():
            index = complex_values[complex_values].index[0]
            row = trials.loc[index]
            raise ValueError(
                f"invalid successful metric {_identity_description(row)}, {column}={row[column]!r}; "
                "expected a finite nonnegative real number"
            )
        numeric = pd.to_numeric(successful_values, errors="coerce")
        numeric_values = numeric.astype(float)
        original_is_bool = successful_values.map(lambda value: isinstance(value, (bool, np.bool_)))
        invalid = (~np.isfinite(numeric_values)) | (numeric_values < 0) | original_is_bool
        if invalid.any():
            index = invalid[invalid].index[0]
            row = trials.loc[index]
            raise ValueError(
                f"invalid successful metric {_identity_description(row)}, {column}={row[column]!r}; "
                "expected a finite nonnegative number"
            )
        converted_values = trials[column].astype(object)
        converted_values.loc[successful] = numeric_values
        trials[column] = converted_values


def _best_successful_trial(successful_rows: pd.DataFrame) -> pd.Series:
    ranked = successful_rows.copy()
    numeric_seed = pd.to_numeric(ranked["seed_transpiler"], errors="coerce")
    if numeric_seed.notna().all():
        ranked["_seed_order"] = numeric_seed
    else:
        ranked["_seed_order"] = ranked["seed_transpiler"].astype(str)
    return ranked.sort_values(
        [*_METRIC_COLUMNS, "_seed_order"], kind="mergesort", na_position="last"
    ).iloc[0]


def aggregate_strategy_statistics(all_trials: pd.DataFrame) -> pd.DataFrame:
    """Aggregate concrete transpiler trials without combining strategies.

    The result has one row for every present boundary plus
    ``(state_name, class_name, candidate_name, strategy_name)``.  Empty input
    uses the stable ``_OUTPUT_COLUMNS`` schema; nonempty inputs additionally
    retain any present boundary columns and ``n_qutrits`` metadata.
    """
    if all_trials.empty:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)

    missing = [column for column in _REQUIRED_INPUT_COLUMNS if column not in all_trials.columns]
    if missing:
        raise ValueError(f"all_trials is missing required columns: {', '.join(missing)}")

    trials = all_trials.copy()
    boundary_columns = _present_boundary_columns(trials.columns)
    group_columns = [*boundary_columns, *IDENTITY_COLUMNS]
    duplicate_columns = [*boundary_columns, *TRIAL_ID_COLUMNS]
    duplicated = trials.duplicated(duplicate_columns, keep=False)
    if duplicated.any():
        identity = trials.loc[duplicated, duplicate_columns].iloc[0].to_dict()
        raise ValueError(
            f"duplicate concrete trial identity for columns {duplicate_columns}: {identity}"
        )

    normalized_success: list[bool] = []
    for position, value in enumerate(trials["success"].tolist()):
        try:
            normalized_success.append(_success_value(value))
        except ValueError as error:
            raise ValueError(
                f"invalid success value for {_identity_description(trials.iloc[position])}: {error}"
            ) from error
    trials["_success_value"] = normalized_success
    successful = trials["_success_value"] & (
        trials["status"].astype("string").str.strip().str.casefold() == "ok"
    )
    successful = successful.fillna(False).astype(bool)
    trials["_successful"] = successful
    _validate_successful_metrics(trials, successful)

    include_n_qutrits = "n_qutrits" in trials.columns
    rows: list[dict[str, object]] = []
    for group_key, group in trials.groupby(group_columns, dropna=False, sort=False):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row: dict[str, object] = dict(zip(group_columns, group_key, strict=True))
        total = len(group)
        group_successful = group["_successful"]
        successful_rows = group.loc[group_successful]
        success_count = len(successful_rows)
        row.update(
            total_trial_count=total,
            successful_trial_count=success_count,
            failed_trial_count=total - success_count,
            success_rate=success_count / total,
        )
        if include_n_qutrits:
            row["n_qutrits"] = np.nan

        if not success_count:
            row.update(
                mean_depth=np.nan,
                min_depth=np.nan,
                max_depth=np.nan,
                std_depth=np.nan,
                mean_two_qubit_gate_count=np.nan,
                min_two_qubit_gate_count=np.nan,
                max_two_qubit_gate_count=np.nan,
                std_two_qubit_gate_count=np.nan,
                insufficient_stability_samples=True,
                best_seed_transpiler=pd.NA,
                best_graph_state_transpiled_qpy=pd.NA,
                best_depth=np.nan,
                best_two_qubit_gate_count=np.nan,
                best_one_qubit_gate_count=np.nan,
                best_size=np.nan,
                pareto_eligible=False,
                analysis_status="no_successful_trials",
            )
        else:
            depths = successful_rows["depth"].astype(float)
            two_qubit = successful_rows["two_qubit_gate_count"].astype(float)
            best = _best_successful_trial(successful_rows)
            row.update(
                mean_depth=depths.mean(),
                min_depth=depths.min(),
                max_depth=depths.max(),
                std_depth=depths.std(ddof=0),
                mean_two_qubit_gate_count=two_qubit.mean(),
                min_two_qubit_gate_count=two_qubit.min(),
                max_two_qubit_gate_count=two_qubit.max(),
                std_two_qubit_gate_count=two_qubit.std(ddof=0),
                insufficient_stability_samples=success_count == 1,
                best_seed_transpiler=best["seed_transpiler"],
                best_graph_state_transpiled_qpy=best["graph_state_transpiled_qpy"],
                best_depth=float(best["depth"]),
                best_two_qubit_gate_count=float(best["two_qubit_gate_count"]),
                best_one_qubit_gate_count=float(best["one_qubit_gate_count"]),
                best_size=float(best["size"]),
                pareto_eligible=True,
                analysis_status="eligible",
            )
            if include_n_qutrits:
                row["n_qutrits"] = best["n_qutrits"]
        rows.append(row)

    result = pd.DataFrame(rows, columns=_output_columns(boundary_columns, include_n_qutrits))
    return result.sort_values(group_columns, kind="mergesort", na_position="last").reset_index(drop=True)


def analyze_iqm_trials(
    all_trials: pd.DataFrame,
    *,
    objective_weights: Mapping[str, object] | None = None,
    max_state_qubits: int = 12,
    state_loader: Callable[..., object] = load_logical_state_from_qpy,
) -> ParetoAnalysisResult:
    """Run the post-transpilation IQM strategy-selection analysis pipeline."""
    strategy_statistics = aggregate_strategy_statistics(all_trials)
    pareto_ranked = rank_pareto_candidates(
        strategy_statistics,
        objective_weights=objective_weights,
    )
    state_equivalence_groups, recommended_circuits = group_state_equivalent_candidates(
        pareto_ranked,
        max_qubits=max_state_qubits,
        state_loader=state_loader,
    )
    eligible = _eligible_mask(state_equivalence_groups)
    state_equivalence_group_count = int(
        state_equivalence_groups.loc[eligible, "state_equivalence_group_id"].dropna().nunique()
    )
    pareto_front_count = int(
        (
            _eligible_mask(pareto_ranked)
            & (pareto_ranked["pareto_rank"] == 1).fillna(False)
        ).sum()
    )
    return ParetoAnalysisResult(
        strategy_statistics=strategy_statistics,
        pareto_ranked=pareto_ranked,
        state_equivalence_groups=state_equivalence_groups,
        recommended_circuits=recommended_circuits,
        summary_counts={
            "analyzed_strategy_combination_count": int(len(strategy_statistics)),
            "pareto_front_count": pareto_front_count,
            "state_equivalence_group_count": state_equivalence_group_count,
            "recommended_circuit_count": int(len(recommended_circuits)),
        },
    )
