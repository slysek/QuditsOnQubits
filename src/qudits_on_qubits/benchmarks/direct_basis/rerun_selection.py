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
from qudits_on_qubits.encoding_search.triviality import candidate_metadata_fields


REQUIRED_INPUT_COLUMNS = ("state_name", "class_name", "candidate_name", "best_depth")
EQUIVALENCE_METADATA_PRESENT_COLUMN = "_has_baseline_equivalence_metadata"
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
        if not str(self.run_id).strip():
            raise ValueError("--run-id must not be empty")


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
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.lower().isin({"true", "1", "yes", "ok"})


def _numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(float("nan"), index=df.index)
    return pd.to_numeric(df[column], errors="coerce")


def _ensure_selection_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    if "is_baseline_equivalent" in result.columns:
        result[EQUIVALENCE_METADATA_PRESENT_COLUMN] = result[
            "is_baseline_equivalent"
        ].notna()
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
        if column not in first and column != EQUIVALENCE_METADATA_PRESENT_COLUMN
    ]
    return df[first + rest]


def _has_complete_equivalence_columns(df: pd.DataFrame) -> bool:
    return (
        "is_baseline_equivalent" in df.columns
        and "is_baseline_reference" in df.columns
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
    if "is_baseline_equivalent" in result.columns:
        if "is_baseline_reference" not in result.columns:
            result["is_baseline_reference"] = (
                result["class_name"].astype(str).eq("baseline")
                & result["candidate_name"].astype(str).eq("E_old")
            )
        if "baseline_equivalence_reason" not in result.columns:
            if "skip_reason" in result.columns:
                result["baseline_equivalence_reason"] = result[
                    "skip_reason"
                ].fillna("")
            else:
                result["baseline_equivalence_reason"] = ""
        return result

    lookup = candidate_lookup
    if lookup is None:
        lookup = build_default_candidate_lookup()
    references: list[bool] = []
    equivalents: list[bool] = []
    reasons: list[str] = []
    unresolved: list[bool] = []
    for _, row in result.iterrows():
        class_name = str(row["class_name"])
        candidate_name = str(row["candidate_name"])
        candidate = lookup.get((class_name, candidate_name))
        if candidate is None:
            references.append(False)
            equivalents.append(False)
            reasons.append("candidate not found in direct-basis candidate lookup")
            unresolved.append(True)
            continue
        metadata = candidate_metadata_fields(
            class_name,
            candidate_name,
            _candidate_embedding(candidate),
        )
        references.append(bool(metadata["is_baseline_reference"]))
        equivalents.append(bool(metadata["is_baseline_equivalent"]))
        reasons.append(str(metadata.get("skip_reason", "")))
        unresolved.append(False)

    result["is_baseline_reference"] = references
    result["is_baseline_equivalent"] = equivalents
    result["baseline_equivalence_reason"] = reasons
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
    selected = _sort_for_selection(candidate_pool).head(top_k).copy()
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
