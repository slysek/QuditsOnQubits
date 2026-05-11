from __future__ import annotations

import os
from datetime import datetime
from typing import Optional

import pandas as pd

from basis_direct_encoding_benchmarks.benchmark import default_results_dir


def timestamped_comparison_path(
    *,
    output_dir: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> str:
    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    directory = output_dir or default_results_dir()
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, f"direct_vs_old_comparison_{stamp}.csv")


def _first_existing(df: pd.DataFrame, names: tuple[str, ...], default=None):
    for name in names:
        if name in df.columns:
            return df[name]
    return default


def _series_or_default(df: pd.DataFrame, names: tuple[str, ...], default="") -> pd.Series:
    value = _first_existing(df, names, None)
    if value is None:
        return pd.Series(default, index=df.index)
    return value


def _normalize_bool_success(df: pd.DataFrame, status_column: str = "status") -> pd.Series:
    if "success" in df.columns:
        values = df["success"]
        if values.dtype == bool:
            return values.fillna(False)
        return values.astype(str).str.lower().isin({"true", "1", "yes", "ok"})
    if status_column in df.columns:
        return df[status_column].astype(str).str.lower().isin({"ok", "true", "1", "yes"})
    return pd.Series(True, index=df.index)


def _normalize_old(df: pd.DataFrame) -> pd.DataFrame:
    normalized = pd.DataFrame()
    normalized["state_name"] = df["state_name"].astype(str)
    if "graph_name" in df.columns:
        normalized["graph_name"] = df["graph_name"].astype(str)
    normalized["basis_candidate_name"] = _first_existing(
        df, ("basis_candidate_name", "candidate_name")
    ).astype(str)
    normalized["old_class_name"] = _series_or_default(df, ("class_name",), "").astype(str)
    normalized["old_candidate_name"] = _series_or_default(df, ("candidate_name",), "").astype(str)
    normalized["old_two_qubit_gate_count"] = pd.to_numeric(
        _first_existing(df, ("two_qubit_gate_count", "best_two_qubit_gate_count")),
        errors="coerce",
    )
    normalized["old_depth"] = pd.to_numeric(
        _first_existing(df, ("circuit_depth", "best_depth")),
        errors="coerce",
    )
    normalized["old_total_gate_count"] = pd.to_numeric(
        _first_existing(df, ("total_gate_count", "best_size")),
        errors="coerce",
    )
    normalized["old_status"] = _series_or_default(df, ("status",), "").astype(str)
    normalized["old_success"] = _normalize_bool_success(df)
    return normalized


def _normalize_direct(df: pd.DataFrame) -> pd.DataFrame:
    normalized = pd.DataFrame()
    normalized["state_name"] = df["state_name"].astype(str)
    if "graph_name" in df.columns:
        normalized["graph_name"] = df["graph_name"].astype(str)
    normalized["basis_candidate_name"] = _first_existing(
        df, ("basis_candidate_name", "candidate_name")
    ).astype(str)
    normalized["direct_basis_candidate_type"] = _series_or_default(
        df, ("basis_candidate_type", "class_name"), ""
    ).astype(str)
    normalized["direct_two_qubit_gate_count"] = pd.to_numeric(
        _first_existing(df, ("two_qubit_gate_count", "best_two_qubit_gate_count")),
        errors="coerce",
    )
    normalized["direct_depth"] = pd.to_numeric(
        _first_existing(df, ("circuit_depth", "best_depth")),
        errors="coerce",
    )
    normalized["direct_total_gate_count"] = pd.to_numeric(
        _first_existing(df, ("total_gate_count", "best_size")),
        errors="coerce",
    )
    normalized["direct_status"] = _series_or_default(df, ("status",), "").astype(str)
    normalized["direct_success"] = _normalize_bool_success(df)
    return normalized


def _join_keys(old: pd.DataFrame, direct: pd.DataFrame) -> list[str]:
    keys = ["state_name", "basis_candidate_name"]
    if "graph_name" in old.columns and "graph_name" in direct.columns:
        keys.insert(1, "graph_name")
    return keys


def _comparison_class(row) -> str:
    old_tuple = (
        row["old_two_qubit_gate_count"],
        row["old_depth"],
        row["old_total_gate_count"],
    )
    direct_tuple = (
        row["direct_two_qubit_gate_count"],
        row["direct_depth"],
        row["direct_total_gate_count"],
    )
    if pd.isna(old_tuple[0]) or pd.isna(direct_tuple[0]):
        return "unscored"
    if direct_tuple < old_tuple:
        return "better"
    if direct_tuple > old_tuple:
        return "worse"
    return "tie"


def compare_old_vs_direct(
    old_csv: str,
    direct_csv: str,
    output_csv: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    old = _normalize_old(pd.read_csv(old_csv))
    direct = _normalize_direct(pd.read_csv(direct_csv))
    keys = _join_keys(old, direct)
    comparison = old.merge(direct, on=keys, how="inner")

    comparison["delta_two_qubit_gate_count"] = (
        comparison["direct_two_qubit_gate_count"] - comparison["old_two_qubit_gate_count"]
    )
    comparison["delta_depth"] = comparison["direct_depth"] - comparison["old_depth"]
    comparison["delta_total_gate_count"] = (
        comparison["direct_total_gate_count"] - comparison["old_total_gate_count"]
    )
    comparison["comparison"] = comparison.apply(_comparison_class, axis=1)

    summary = {
        "matched_rows": int(len(comparison)),
        "better": int((comparison["comparison"] == "better").sum()),
        "worse": int((comparison["comparison"] == "worse").sum()),
        "tie": int((comparison["comparison"] == "tie").sum()),
        "unscored": int((comparison["comparison"] == "unscored").sum()),
    }

    if output_csv is not None:
        directory = os.path.dirname(output_csv)
        if directory:
            os.makedirs(directory, exist_ok=True)
        comparison.to_csv(output_csv, index=False)

    return comparison, summary


def print_comparison_summary(comparison: pd.DataFrame, summary: dict) -> None:
    print(
        "Summary: "
        f"matched={summary['matched_rows']} "
        f"better={summary['better']} "
        f"worse={summary['worse']} "
        f"tie={summary['tie']} "
        f"unscored={summary['unscored']}"
    )
    if comparison.empty:
        return

    ranked = comparison.dropna(subset=["delta_two_qubit_gate_count", "delta_depth"]).copy()
    if ranked.empty:
        return

    display_columns = [
        "state_name",
        "basis_candidate_name",
        "old_two_qubit_gate_count",
        "direct_two_qubit_gate_count",
        "delta_two_qubit_gate_count",
        "old_depth",
        "direct_depth",
        "delta_depth",
        "delta_total_gate_count",
    ]
    existing = [column for column in display_columns if column in ranked.columns]

    print("\nTop 10 best improvements:")
    best = ranked.sort_values(
        ["delta_two_qubit_gate_count", "delta_depth", "delta_total_gate_count"],
        ascending=True,
    ).head(10)
    print(best[existing].to_string(index=False))

    print("\nTop 10 largest regressions:")
    worst = ranked.sort_values(
        ["delta_two_qubit_gate_count", "delta_depth", "delta_total_gate_count"],
        ascending=False,
    ).head(10)
    print(worst[existing].to_string(index=False))
