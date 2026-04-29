import os
from typing import Optional

import pandas as pd

from encoding_search_v2.paths import stage_output_dir, stage_results_csv_path
from encoding_search_v2.results import fidelity_label


def _read_if_exists(path: Optional[str]) -> Optional[pd.DataFrame]:
    if not path or not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _ok(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    if "status" in df.columns:
        return df[df["status"].astype(str) == "ok"].copy()
    return df.copy()


def _truthy_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    values = df[column]
    if values.dtype == bool:
        return values.fillna(False)
    return values.astype(str).str.lower().isin({"true", "1", "yes"})


def _nontrivial_ok(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    df_ok = _ok(df)
    if df_ok.empty:
        return df_ok
    if "is_baseline_equivalent" not in df_ok.columns:
        return df_ok[
            ~(
                (df_ok["class_name"].astype(str) == "baseline")
                & (df_ok["candidate_name"].astype(str) == "E_old")
            )
        ].copy()
    return df_ok[
        ~_truthy_series(df_ok, "is_baseline_equivalent")
        & ~_truthy_series(df_ok, "is_baseline_reference")
    ].copy()


def _skipped_count(df: Optional[pd.DataFrame]) -> int:
    if df is None or df.empty or "status" not in df.columns:
        return 0
    return int((df["status"].astype(str) == "skipped_baseline_equivalent").sum())


def _best_exact(df: pd.DataFrame) -> Optional[pd.Series]:
    if df.empty:
        return None
    ranked = df.copy()
    for column in ("best_depth", "best_two_qubit_gate_count", "best_size"):
        if column in ranked.columns:
            ranked[column] = pd.to_numeric(ranked[column], errors="coerce")
    return ranked.sort_values(
        by=["best_depth", "best_two_qubit_gate_count", "best_size"],
        ascending=True,
        na_position="last",
    ).iloc[0]


def _baseline(df: pd.DataFrame) -> Optional[pd.Series]:
    if df.empty:
        return None
    baseline = df[
        (df["class_name"].astype(str) == "baseline")
        & (df["candidate_name"].astype(str) == "E_old")
    ]
    if baseline.empty:
        return None
    return baseline.iloc[0]


def _fmt_row(row: Optional[pd.Series]) -> str:
    if row is None:
        return "not available"
    return (
        f"{row['class_name']} / {row['candidate_name']} "
        f"(depth={row.get('best_depth')}, 2q={row.get('best_two_qubit_gate_count')})"
    )


def _improvement(baseline: Optional[pd.Series], best: Optional[pd.Series]) -> str:
    if baseline is None or best is None:
        return "not available"
    base_depth = float(baseline["best_depth"])
    best_depth = float(best["best_depth"])
    base_2q = float(baseline["best_two_qubit_gate_count"])
    best_2q = float(best["best_two_qubit_gate_count"])
    if (best_depth, best_2q) < (base_depth, base_2q):
        return (
            f"yes: depth {base_depth:g} -> {best_depth:g} "
            f"({base_depth - best_depth:+g}), 2q {base_2q:g} -> {best_2q:g} "
            f"({base_2q - best_2q:+g})"
        )
    return "no better basis found in the tested classes"


def _fidelity_lines(df: pd.DataFrame, thresholds) -> list[str]:
    lines = []
    if df.empty:
        return lines
    for threshold in thresholds:
        label = fidelity_label(threshold)
        depth_col = f"{label}_best_depth"
        twoq_col = f"{label}_best_two_qubit_gate_count"
        if depth_col not in df.columns or twoq_col not in df.columns:
            continue
        sub = df.dropna(subset=[depth_col, twoq_col]).copy()
        if sub.empty:
            continue
        for column in (depth_col, twoq_col):
            sub[column] = pd.to_numeric(sub[column], errors="coerce")
        best = sub.sort_values(by=[depth_col, twoq_col], ascending=True).iloc[0]
        lines.append(
            f"- {label}: {best['class_name']} / {best['candidate_name']} "
            f"depth={best[depth_col]}, 2q={best[twoq_col]}"
        )
    return lines


def write_state_report(
    state_name: str,
    output_root: Optional[str] = None,
    stage1_csv: Optional[str] = None,
    stage2_csv: Optional[str] = None,
    fidelity_thresholds=(0.85, 0.90, 0.95),
) -> str:
    stage1_csv = stage1_csv or stage_results_csv_path(state_name, 1, output_root=output_root)
    stage2_csv = stage2_csv or stage_results_csv_path(state_name, 2, output_root=output_root)
    stage1_full = _read_if_exists(stage1_csv)
    stage2_full = _read_if_exists(stage2_csv)
    stage1 = _ok(stage1_full)
    stage2 = _ok(stage2_full)
    stage1_nontrivial = _nontrivial_ok(stage1_full)
    stage2_nontrivial = _nontrivial_ok(stage2_full)

    baseline = _baseline(stage1)
    best_stage1 = _best_exact(stage1)
    best_stage2 = _best_exact(stage2)
    best_nontrivial_stage1 = _best_exact(stage1_nontrivial)
    best_nontrivial_stage2 = _best_exact(stage2_nontrivial)

    lines = [
        f"# Encoding Search v2 Report: {state_name}",
        "",
        f"Stage 1 CSV: `{stage1_csv}`",
        f"Stage 2 CSV: `{stage2_csv}`" if os.path.exists(stage2_csv) else "Stage 2 CSV: not available",
        "",
        "## Summary",
        "",
        f"- Baseline: {_fmt_row(baseline)}",
        f"- Best stage 1: {_fmt_row(best_stage1)}",
        f"- Best nontrivial stage 1: {_fmt_row(best_nontrivial_stage1)}",
        f"- Best stage 2: {_fmt_row(best_stage2)}",
        f"- Best nontrivial stage 2: {_fmt_row(best_nontrivial_stage2)}",
        f"- Stage 1 nontrivial better than baseline: {_improvement(baseline, best_nontrivial_stage1)}",
        f"- Stage 2 nontrivial better than baseline: {_improvement(baseline, best_nontrivial_stage2)}",
        f"- Skipped baseline-equivalent candidates: {_skipped_count(stage1_full) + _skipped_count(stage2_full)}",
        "",
        "## Fidelity Threshold Bests",
        "",
    ]
    lines.extend(_fidelity_lines(stage1_nontrivial, fidelity_thresholds) or ["- Stage 1 nontrivial: not available"])
    if not stage2_nontrivial.empty:
        lines.append("")
        lines.extend(_fidelity_lines(stage2_nontrivial, fidelity_thresholds) or ["- Stage 2 nontrivial: not available"])

    output_dir = stage_output_dir(state_name, "report", output_root=output_root)
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"encoding_search_v2_{state_name}_report.md")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).strip() + "\n")
    return path
