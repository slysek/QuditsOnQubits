import os
from typing import Iterable

import pandas as pd


def fidelity_label(threshold: float) -> str:
    return f"fid{int(round(float(threshold) * 100)):03d}"


def _ok_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "status" not in df.columns:
        return df.copy()
    return df[df["status"].astype(str) == "ok"].copy()


def _sort_existing(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    existing = [column for column in columns if column in df.columns]
    if not existing:
        return df.copy()
    ranked = df.copy()
    for column in existing:
        ranked[column] = pd.to_numeric(ranked[column], errors="coerce")
    return ranked.sort_values(by=existing, ascending=[True] * len(existing), na_position="last")


def _top_per_class(df: pd.DataFrame, sort_columns: list[str], limit: int = 3) -> pd.DataFrame:
    df_ok = _ok_rows(df)
    if df_ok.empty or "class_name" not in df_ok.columns:
        return df_ok.head(0).copy()

    parts = []
    for _, group in df_ok.groupby("class_name", sort=True):
        parts.append(_sort_existing(group, sort_columns).head(limit))
    if not parts:
        return df_ok.head(0).copy()
    return pd.concat(parts, ignore_index=True)


def _write_csv(df: pd.DataFrame, path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _select_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    keep = [column for column in columns if column in df.columns]
    if not keep:
        return df
    return df[keep]


def write_result_bundle(
    df: pd.DataFrame,
    output_dir: str,
    file_prefix: str,
    top_k: int = 30,
    fidelity_thresholds=(0.85, 0.90, 0.95),
) -> dict[str, str]:
    """Write full results plus v2 ranking and top-3 compatibility tables."""
    os.makedirs(output_dir, exist_ok=True)
    paths: dict[str, str] = {}

    results_csv = os.path.join(output_dir, f"{file_prefix}_results.csv")
    paths["results_csv"] = _write_csv(df, results_csv)

    df_ok = _ok_rows(df)
    by_depth = _sort_existing(
        df_ok,
        ["best_depth", "best_two_qubit_gate_count", "best_size"],
    ).head(int(top_k))
    by_2q = _sort_existing(
        df_ok,
        ["best_two_qubit_gate_count", "best_depth", "best_size"],
    ).head(int(top_k))

    paths["top_by_depth_csv"] = _write_csv(
        by_depth,
        os.path.join(output_dir, f"{file_prefix}_top{int(top_k)}_by_depth.csv"),
    )
    paths["top_by_2q_csv"] = _write_csv(
        by_2q,
        os.path.join(output_dir, f"{file_prefix}_top{int(top_k)}_by_2q.csv"),
    )

    top3_depth = _top_per_class(
        df,
        ["best_depth", "best_two_qubit_gate_count", "best_size"],
        limit=3,
    )
    top3_2q = _top_per_class(
        df,
        ["best_two_qubit_gate_count", "best_depth", "best_size"],
        limit=3,
    )
    core_cols = [
        "state_name",
        "class_name",
        "candidate_name",
        "best_depth",
        "best_size",
        "best_two_qubit_gate_count",
        "mean_depth",
        "std_depth",
        "mean_two_qubit_gate_count",
    ]
    paths["top3_by_depth_csv"] = _write_csv(
        _select_columns(top3_depth, core_cols),
        os.path.join(output_dir, f"{file_prefix}_top3_by_depth.csv"),
    )
    paths["top3_by_2q_csv"] = _write_csv(
        _select_columns(top3_2q, core_cols),
        os.path.join(output_dir, f"{file_prefix}_top3_by_2q.csv"),
    )
    paths["top3_exact_csv"] = _write_csv(
        _select_columns(top3_2q, core_cols),
        os.path.join(output_dir, f"{file_prefix}_top3_exact.csv"),
    )

    for threshold in fidelity_thresholds:
        label = fidelity_label(threshold)
        depth_col = f"{label}_best_depth"
        twoq_col = f"{label}_best_two_qubit_gate_count"
        fid_col = f"{label}_best_fidelity"

        if depth_col in df.columns and twoq_col in df.columns:
            fid_df = _ok_rows(df).dropna(subset=[depth_col, twoq_col])
            top3_fid = _top_per_class(fid_df, [twoq_col, depth_col], limit=3)
        else:
            top3_fid = df.head(0).copy()

        keep_cols = ["state_name", "class_name", "candidate_name", fid_col, depth_col, twoq_col]
        paths[f"top3_{label}_csv"] = _write_csv(
            _select_columns(top3_fid, keep_cols),
            os.path.join(output_dir, f"{file_prefix}_top3_{label}.csv"),
        )

    return paths
