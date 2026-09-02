"""Post-transpilation, per-strategy trial aggregation for IQM benchmarks."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


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


def _validate_successful_metrics(trials: pd.DataFrame, successful: pd.Series) -> None:
    for column in _METRIC_COLUMNS:
        successful_values = trials.loc[successful, column]
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
        trials.loc[successful, column] = numeric_values


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
                insufficient_stability_samples=pd.NA,
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

    return pd.DataFrame(rows, columns=_output_columns(boundary_columns, include_n_qutrits))
