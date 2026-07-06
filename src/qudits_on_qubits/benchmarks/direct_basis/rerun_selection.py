from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from qudits_on_qubits.benchmarks.direct_basis.candidates import (
    DirectBasisCandidate,
    generate_all_qutrit_u3_candidates,
    generate_legacy_qutrit_u3_candidates,
)
from qudits_on_qubits.benchmarks.direct_basis.math_utils import encoding_embedding
from qudits_on_qubits.benchmarks.direct_basis.selection import safe_path_part
from qudits_on_qubits.encoding_search.triviality import candidate_metadata_fields


REQUIRED_INPUT_COLUMNS = ("state_name", "class_name", "candidate_name", "best_depth")
EQUIVALENCE_METADATA_PRESENT_COLUMN = "_has_baseline_equivalence_metadata"
INTERNAL_OUTPUT_COLUMNS = {
    EQUIVALENCE_METADATA_PRESENT_COLUMN,
    "is_unresolved_candidate",
}
OUTPUT_FIRST_COLUMNS = (
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
)
RANK_COLUMNS = (
    "best_depth",
    "best_two_qubit_gate_count",
    "mean_depth",
    "std_depth",
    "best_one_qubit_gate_count",
    "best_size",
    "candidate_name",
)


def _validate_top_k(top_k: int) -> int:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("--top-k must be positive")
    return top_k


@dataclass(frozen=True)
class RerunSelectionConfig:
    input_csvs: tuple[Path, ...]
    output_root: Path
    run_id: str
    top_k: int = 10
    include_label: str | None = None

    def __post_init__(self) -> None:
        if not self.input_csvs:
            raise ValueError("at least one --input-csv is required")
        _validate_top_k(self.top_k)
        run_id = str(self.run_id)
        if not run_id.strip():
            raise ValueError("--run-id must not be empty")
        if run_id != safe_path_part(run_id):
            raise ValueError("--run-id must be filesystem-safe")


@dataclass(frozen=True)
class StateSelectionOutput:
    state_name: str
    csv_path: Path
    selected_count: int
    baseline_equivalent_excluded_count: int
    unresolved_count: int
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class RerunSelectionOutput:
    run_id: str
    output_dir: Path
    state_outputs: tuple[StateSelectionOutput, ...]
    warnings: tuple[str, ...]


def _validate_input_columns(df: pd.DataFrame, source: Path) -> None:
    missing = [column for column in REQUIRED_INPUT_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {', '.join(missing)}")


def _truthy_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    values = df[column]
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    string_truth = (
        values.astype("string").str.strip().str.lower().isin({"true", "1", "yes", "ok"})
    )
    numeric_truth = pd.to_numeric(values, errors="coerce").eq(1)
    return (string_truth | numeric_truth).fillna(False).astype(bool)


def _numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(float("nan"), index=df.index)
    return pd.to_numeric(df[column], errors="coerce")


def _ensure_selection_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if _has_complete_equivalence_columns(result):
        result[EQUIVALENCE_METADATA_PRESENT_COLUMN] = _equivalence_metadata_present(
            result
        )
    else:
        result[EQUIVALENCE_METADATA_PRESENT_COLUMN] = False
    if "status" not in result.columns:
        result["status"] = "ok"
    if "success" not in result.columns:
        result["success"] = True
    if "is_baseline_reference" not in result.columns:
        result["is_baseline_reference"] = (
            result["class_name"].astype(str).eq("baseline")
            & result["candidate_name"].astype(str).eq("E_old")
        )
    if "is_baseline_equivalent" not in result.columns:
        result["is_baseline_equivalent"] = False
    if "baseline_equivalence_reason" not in result.columns:
        if "skip_reason" in result.columns:
            result["baseline_equivalence_reason"] = result["skip_reason"].fillna("")
        else:
            result["baseline_equivalence_reason"] = ""
    for column in RANK_COLUMNS:
        if column != "candidate_name":
            result[column] = _numeric_column(result, column)
    return result


def _sort_for_selection(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(list(RANK_COLUMNS), ascending=True, na_position="last")


def _dedupe_ranked_candidate_rows(df: pd.DataFrame) -> pd.DataFrame:
    ranked = _sort_for_selection(df).copy()
    unique_keys = pd.DataFrame(
        {
            "class_name": ranked["class_name"].astype(str),
            "candidate_name": ranked["candidate_name"].astype(str),
        },
        index=ranked.index,
    )
    return ranked.loc[~unique_keys.duplicated()].copy()


def _choose_baseline(
    state_df: pd.DataFrame, state_name: str
) -> tuple[pd.Series, tuple[str, ...]]:
    baseline_rows = state_df[state_df["class_name"].astype(str) == "baseline"].copy()
    if baseline_rows.empty:
        raise ValueError(f"{state_name} has no baseline row")
    baseline_rows = baseline_rows[
        (baseline_rows["status"].astype(str) == "ok")
        & _truthy_series(baseline_rows, "success")
    ].copy()
    if baseline_rows.empty:
        raise ValueError(f"{state_name} has no runnable baseline row")
    warnings: list[str] = []
    if len(baseline_rows) > 1:
        warnings.append(
            f"{state_name}: multiple baseline rows found; selected best ranked baseline"
        )
    e_old = baseline_rows[baseline_rows["candidate_name"].astype(str) == "E_old"].copy()
    if not e_old.empty:
        baseline_rows = e_old
    return _sort_for_selection(baseline_rows).iloc[0], tuple(warnings)


def _relation(depth: float, baseline_depth: float, role: str) -> str:
    if role == "baseline":
        return "baseline"
    if role == "baseline_equivalent_excluded":
        return "excluded_baseline_equivalent"
    if role == "unresolved_candidate":
        return "unresolved"
    if pd.isna(depth) or pd.isna(baseline_depth):
        return "unresolved"
    if float(depth) < float(baseline_depth):
        return "better"
    if float(depth) == float(baseline_depth):
        return "equal"
    return "worse"


def _with_baseline_comparison(
    df: pd.DataFrame, baseline_depth: float
) -> pd.DataFrame:
    result = df.copy()
    result["baseline_best_depth"] = baseline_depth
    result["depth_delta_vs_baseline"] = _numeric_column(result, "best_depth") - float(
        baseline_depth
    )
    result["depth_ratio_vs_baseline"] = _numeric_column(result, "best_depth") / float(
        baseline_depth
    )
    result["baseline_relation"] = [
        _relation(depth, baseline_depth, role)
        for depth, role in zip(result["best_depth"], result["selection_role"])
    ]
    return result


def _order_columns(df: pd.DataFrame) -> pd.DataFrame:
    first = [column for column in OUTPUT_FIRST_COLUMNS if column in df.columns]
    rest = [
        column
        for column in df.columns
        if column not in first and column not in INTERNAL_OUTPUT_COLUMNS
    ]
    return df[first + rest]


def _equivalence_metadata_present(df: pd.DataFrame) -> pd.Series:
    return (
        df[["is_baseline_equivalent", "is_baseline_reference"]].notna().all(axis=1)
    )


def _has_complete_equivalence_columns(df: pd.DataFrame) -> bool:
    if (
        "is_baseline_equivalent" not in df.columns
        or "is_baseline_reference" not in df.columns
    ):
        return False
    return bool(
        df[["is_baseline_equivalent", "is_baseline_reference"]]
        .notna()
        .all()
        .all()
    )


def load_input_csvs(
    input_csvs: Iterable[str | Path],
    *,
    include_label: str | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for value in input_csvs:
        path = Path(value)
        df = pd.read_csv(path)
        _validate_input_columns(df, path)
        if include_label is not None and "selection_label" in df.columns:
            df = df[df["selection_label"].astype(str) == str(include_label)].copy()
        df["source_csv"] = str(path)
        frames.append(df)
    if not frames:
        raise ValueError("at least one --input-csv is required")
    return pd.concat(frames, ignore_index=True)


def build_default_candidate_lookup() -> dict[tuple[str, str], DirectBasisCandidate]:
    lookup: dict[tuple[str, str], DirectBasisCandidate] = {}
    for candidate in generate_all_qutrit_u3_candidates():
        lookup[(candidate.class_name, candidate.candidate_name)] = candidate
    for candidate in generate_legacy_qutrit_u3_candidates("original"):
        lookup.setdefault((candidate.class_name, candidate.candidate_name), candidate)
    return lookup


def _candidate_embedding(candidate: DirectBasisCandidate):
    if candidate.matrix is None:
        return None
    matrix = np.asarray(candidate.matrix, dtype=complex)
    if matrix.shape == (3, 3) or matrix.shape == (4, 3):
        return encoding_embedding(matrix)
    return matrix


def annotate_baseline_equivalence(
    df: pd.DataFrame,
    *,
    candidate_lookup: dict[tuple[str, str], DirectBasisCandidate] | None = None,
) -> pd.DataFrame:
    result = df.copy()
    if "baseline_equivalence_reason" not in result.columns:
        result["baseline_equivalence_reason"] = pd.NA
    if "skip_reason" in result.columns:
        result["baseline_equivalence_reason"] = result[
            "baseline_equivalence_reason"
        ].where(result["baseline_equivalence_reason"].notna(), result["skip_reason"])
    if _has_complete_equivalence_columns(result):
        result["baseline_equivalence_reason"] = result[
            "baseline_equivalence_reason"
        ].fillna("")
        return result

    if "is_baseline_reference" not in result.columns:
        result["is_baseline_reference"] = pd.NA
    if "is_baseline_equivalent" not in result.columns:
        result["is_baseline_equivalent"] = pd.NA

    needs_lookup = result["is_baseline_equivalent"].isna()
    lookup = candidate_lookup
    if lookup is None and bool(needs_lookup.any()):
        lookup = build_default_candidate_lookup()
    unresolved: list[bool] = []
    for index, row in result.iterrows():
        class_name = str(row["class_name"])
        candidate_name = str(row["candidate_name"])
        is_baseline_reference = class_name == "baseline" and candidate_name == "E_old"
        reference_missing = pd.isna(row["is_baseline_reference"])
        equivalent_missing = pd.isna(row["is_baseline_equivalent"])
        reason_missing = pd.isna(row["baseline_equivalence_reason"])
        if not equivalent_missing:
            if reference_missing:
                result.at[index, "is_baseline_reference"] = is_baseline_reference
            unresolved.append(False)
            continue

        candidate = None if lookup is None else lookup.get((class_name, candidate_name))
        if candidate is None:
            if reference_missing:
                result.at[index, "is_baseline_reference"] = is_baseline_reference
            result.at[index, "is_baseline_equivalent"] = is_baseline_reference
            if reason_missing and not is_baseline_reference:
                result.at[
                    index, "baseline_equivalence_reason"
                ] = "candidate not found in direct-basis candidate lookup"
            unresolved.append(not is_baseline_reference)
            continue

        embedding = _candidate_embedding(candidate)
        if embedding is None:
            if reference_missing:
                result.at[index, "is_baseline_reference"] = is_baseline_reference
            result.at[index, "is_baseline_equivalent"] = is_baseline_reference
            if reason_missing and not is_baseline_reference:
                result.at[index, "baseline_equivalence_reason"] = (
                    candidate.error_message
                    or "candidate matrix unavailable for baseline-equivalence inference"
                )
            unresolved.append(not is_baseline_reference)
            continue

        metadata = candidate_metadata_fields(
            class_name,
            candidate_name,
            embedding,
        )
        if reference_missing:
            result.at[index, "is_baseline_reference"] = bool(
                metadata["is_baseline_reference"]
            )
        result.at[index, "is_baseline_equivalent"] = bool(
            metadata["is_baseline_equivalent"]
        )
        if reason_missing:
            result.at[index, "baseline_equivalence_reason"] = str(
                metadata.get("skip_reason", "")
            )
        unresolved.append(False)

    result["baseline_equivalence_reason"] = result[
        "baseline_equivalence_reason"
    ].fillna("")
    result["is_unresolved_candidate"] = unresolved
    return result


def select_state_rerun_rows(
    df: pd.DataFrame,
    state_name: str,
    *,
    top_k: int,
    candidate_lookup: dict[tuple[str, str], DirectBasisCandidate] | None = None,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    top_k = _validate_top_k(top_k)
    state_df = df[df["state_name"].astype(str) == str(state_name)].copy()
    if not _has_complete_equivalence_columns(state_df):
        state_df = annotate_baseline_equivalence(
            state_df,
            candidate_lookup=candidate_lookup,
        )
    state_df = _ensure_selection_columns(state_df)
    baseline, warnings = _choose_baseline(state_df, state_name)
    baseline_depth = float(baseline["best_depth"])

    status_ok = state_df["status"].astype(str) == "ok"
    success_ok = _truthy_series(state_df, "success")
    reference_ok = _truthy_series(state_df, "is_baseline_reference")
    equivalent_ok = _truthy_series(state_df, "is_baseline_equivalent")
    equivalence_metadata_present = _truthy_series(
        state_df, EQUIVALENCE_METADATA_PRESENT_COLUMN
    )
    unresolved_ok = _truthy_series(state_df, "is_unresolved_candidate")
    baseline_class = state_df["class_name"].astype(str).eq("baseline")
    baseline_key = (
        str(baseline["class_name"]),
        str(baseline["candidate_name"]),
    )
    baseline_match = (
        state_df["class_name"].astype(str).eq(baseline_key[0])
        & state_df["candidate_name"].astype(str).eq(baseline_key[1])
    )

    candidate_pool = state_df[
        status_ok
        & success_ok
        & ~baseline_match
        & ~baseline_class
        & ~equivalent_ok
        & ~unresolved_ok
        & equivalence_metadata_present
    ].copy()
    selected = _dedupe_ranked_candidate_rows(candidate_pool).head(top_k).copy()
    if len(selected) < top_k:
        warnings = warnings + (
            f"{state_name}: selected {len(selected)} candidates, requested {top_k}",
        )

    excluded = state_df[
        (
            (status_ok & success_ok)
            | state_df["status"].astype(str).eq("skipped_baseline_equivalent")
        )
        & equivalence_metadata_present
        & equivalent_ok
        & ~baseline_match
        & ~reference_ok
    ].copy()
    unresolved = state_df[
        status_ok
        & success_ok
        & ~baseline_class
        & (unresolved_ok | ~equivalence_metadata_present)
    ].copy()

    baseline_df = pd.DataFrame([baseline])
    baseline_df["selection_role"] = "baseline"
    baseline_df["selection_rank"] = 0
    selected["selection_role"] = "candidate"
    selected["selection_rank"] = range(1, len(selected) + 1)
    excluded["selection_role"] = "baseline_equivalent_excluded"
    excluded["selection_rank"] = pd.NA
    unresolved["selection_role"] = "unresolved_candidate"
    unresolved["selection_rank"] = pd.NA

    output = pd.concat(
        [
            baseline_df,
            selected,
            _sort_for_selection(excluded),
            _sort_for_selection(unresolved),
        ],
        ignore_index=True,
    )
    output = _with_baseline_comparison(output, baseline_depth)
    return _order_columns(output), warnings


def _selection_role_count(df: pd.DataFrame, role: str) -> int:
    if "selection_role" not in df.columns:
        return 0
    return int(df["selection_role"].astype(str).eq(role).sum())


def _state_path_parts(state_names: Iterable[str]) -> dict[str, str]:
    by_part: dict[str, str] = {}
    path_parts: dict[str, str] = {}
    for state_name in state_names:
        path_part = safe_path_part(state_name)
        previous = by_part.get(path_part)
        if previous is not None and previous != state_name:
            raise ValueError(
                "state filename collision: "
                f"{previous!r} and {state_name!r} both map to {path_part!r}"
            )
        by_part[path_part] = state_name
        path_parts[state_name] = path_part
    return path_parts


def write_rerun_selection_files(
    config: RerunSelectionConfig,
) -> RerunSelectionOutput:
    df = load_input_csvs(
        config.input_csvs,
        include_label=config.include_label,
    )
    annotated = annotate_baseline_equivalence(df)
    output_dir = Path(config.output_root) / str(config.run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    state_outputs: list[StateSelectionOutput] = []
    warnings: list[str] = []
    state_names = sorted(annotated["state_name"].dropna().astype(str).unique())
    state_path_parts = _state_path_parts(state_names)
    for state_name in state_names:
        selected, state_warnings = select_state_rerun_rows(
            annotated,
            state_name,
            top_k=config.top_k,
        )
        csv_path = (
            output_dir
            / (
                f"direct_basis_{state_path_parts[state_name]}_{config.run_id}"
                f"_top{int(config.top_k)}_rerun_candidates.csv"
            )
        )
        selected.to_csv(csv_path, index=False)
        state_outputs.append(
            StateSelectionOutput(
                state_name=state_name,
                csv_path=csv_path,
                selected_count=_selection_role_count(selected, "candidate"),
                baseline_equivalent_excluded_count=_selection_role_count(
                    selected,
                    "baseline_equivalent_excluded",
                ),
                unresolved_count=_selection_role_count(
                    selected,
                    "unresolved_candidate",
                ),
                warnings=tuple(state_warnings),
            )
        )
        warnings.extend(state_warnings)

    return RerunSelectionOutput(
        run_id=str(config.run_id),
        output_dir=output_dir,
        state_outputs=tuple(state_outputs),
        warnings=tuple(warnings),
    )
