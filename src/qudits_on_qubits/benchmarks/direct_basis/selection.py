from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from qudits_on_qubits.benchmarks.direct_basis.math_utils import encoding_embedding


SUPPORTED_BELL_STATES = ("two_qutrit", "ghz3", "ame43")
DEFAULT_APPROXIMATION_THRESHOLDS = (0.99, 0.95, 0.90)
RANK_BY_DEPTH_COLUMNS = (
    "best_depth",
    "best_two_qubit_gate_count",
    "best_one_qubit_gate_count",
    "best_size",
)
WORKLOAD_SELECTION_COLUMNS = (
    "workload_max_two_qubit_gate_count",
    "workload_total_two_qubit_gate_count",
    "workload_max_depth",
    "workload_total_depth",
    "workload_max_size",
    "workload_total_size",
)


@dataclass(frozen=True)
class SelectionConfig:
    repo_root: Path
    state_name: str
    run_id: str
    top_k: int
    labels: tuple[str, ...]
    processed_dir: Path
    selected_root: Path


@dataclass(frozen=True)
class SelectionOutput:
    manifest_csv: Path
    processed_manifest_csv: Path
    top_csvs: dict[str, Path]
    selected_count: int
    warnings: tuple[str, ...]


def require_supported_bell_state(state_name: str) -> str:
    value = str(state_name)
    if value not in SUPPORTED_BELL_STATES:
        supported = ", ".join(SUPPORTED_BELL_STATES)
        raise ValueError(
            f"Bell selected-circuit pipeline supports only: {supported}. "
            f"Got {state_name!r}."
        )
    return value


def selection_label(approximation_degree: float | None) -> str:
    if approximation_degree is None:
        return "exact"
    value = float(approximation_degree)
    if value <= 0.0 or value > 1.0:
        raise ValueError("approximation_degree must be between 0 and 1")
    return f"fid{int(round(value * 100)):03d}"


def threshold_from_label(label: str) -> float | None:
    text = str(label)
    if text == "exact":
        return None
    match = re.fullmatch(r"fid(\d{3})", text)
    if match is None:
        raise ValueError(f"unknown selection label: {label!r}")
    return int(match.group(1)) / 100.0


def parse_approximation_thresholds(value: str | None) -> tuple[float, ...]:
    if value is None or not str(value).strip():
        return ()
    thresholds: list[float] = []
    for item in str(value).split(","):
        stripped = item.strip()
        if not stripped:
            continue
        threshold = float(stripped)
        if threshold <= 0.0 or threshold > 1.0:
            raise ValueError("approximation thresholds must be between 0 and 1")
        thresholds.append(threshold)
    return tuple(thresholds)


def transpiled_qpy_filename(label: str, *, legacy_exact: bool) -> str:
    if label == "exact" and legacy_exact:
        return "graph_state_direct_basis_transpiled.qpy"
    if label == "exact" or threshold_from_label(label) is not None:
        return f"graph_state_direct_basis_transpiled_{label}.qpy"
    raise ValueError(f"unknown selection label: {label!r}")


def safe_path_part(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("._") or "unnamed"


def repo_relative(path: str | Path, repo_root: str | Path) -> str:
    target = Path(path).resolve()
    root = Path(repo_root).resolve()
    return str(target.relative_to(root)).replace("\\", "/")


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


def select_top_k(
    df: pd.DataFrame,
    *,
    label: str,
    top_k: int,
    fidelity_threshold: float | None,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    ranked = df.copy()
    if "selection_label" in ranked.columns:
        ranked = ranked[ranked["selection_label"].astype(str) == str(label)].copy()
    if ranked.empty:
        return ranked

    status_ok = ranked.get("status", pd.Series("ok", index=ranked.index)).astype(str) == "ok"
    success_ok = _truthy_series(ranked, "success")
    ranked = ranked[status_ok & success_ok].copy()
    if ranked.empty:
        return ranked

    if fidelity_threshold is not None:
        ranked["fidelity"] = _numeric_column(ranked, "fidelity")
        ranked = ranked[ranked["fidelity"] >= float(fidelity_threshold)].copy()
        if ranked.empty:
            return ranked

    selection_columns = RANK_BY_DEPTH_COLUMNS
    has_complete_workload = (
        "ranking_workload" in ranked.columns
        and ranked["ranking_workload"].eq("bell_measurements").all()
        and all(column in ranked.columns for column in WORKLOAD_SELECTION_COLUMNS)
    )
    if has_complete_workload:
        workload_values = ranked.loc[:, WORKLOAD_SELECTION_COLUMNS].apply(
            pd.to_numeric,
            errors="coerce",
        )
        if np.isfinite(workload_values.to_numpy(dtype=float)).all():
            ranked.loc[:, WORKLOAD_SELECTION_COLUMNS] = workload_values
            selection_columns = WORKLOAD_SELECTION_COLUMNS

    for column in selection_columns:
        ranked[column] = _numeric_column(ranked, column)

    ranked = ranked.sort_values(
        list(selection_columns),
        ascending=True,
        na_position="last",
    ).head(int(top_k)).copy()
    ranked.insert(0, "selection_rank", range(1, len(ranked) + 1))
    return ranked.reset_index(drop=True)


def _path_from_row(row: pd.Series, column: str) -> Path:
    value = row.get(column, "")
    if value is None or str(value) == "" or str(value).lower() == "nan":
        raise ValueError(f"selected row is missing {column}")
    return Path(str(value))


def _copy_required(src: Path, dst: Path) -> Path:
    if not src.is_file():
        raise FileNotFoundError(f"required selected artifact does not exist: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _write_selected_encoding(raw_matrix_path: Path, selected_dir: Path) -> tuple[Path, Path | None]:
    matrix = np.load(raw_matrix_path)
    if matrix.shape == (3, 3):
        w_path = selected_dir / "W.npy"
        shutil.copy2(raw_matrix_path, w_path)
        e_matrix = encoding_embedding(matrix)
        e_path = selected_dir / "E.npy"
        np.save(e_path, e_matrix)
        return e_path, w_path
    if matrix.shape == (4, 3):
        e_path = selected_dir / "E.npy"
        shutil.copy2(raw_matrix_path, e_path)
        return e_path, None
    raise ValueError(f"selected basis matrix must have shape (3, 3) or (4, 3), got {matrix.shape}")


def _selected_candidate_dir(config: SelectionConfig, label: str, row: pd.Series) -> Path:
    rank = int(row["selection_rank"])
    class_name = safe_path_part(row.get("class_name", "class"))
    candidate_name = safe_path_part(row.get("candidate_name", "candidate"))
    return (
        Path(config.selected_root)
        / config.state_name
        / config.run_id
        / label
        / f"rank{rank:02d}_{class_name}__{candidate_name}"
    )


def _manifest_row(config: SelectionConfig, label: str, row: pd.Series, selected_dir: Path) -> dict[str, object]:
    graph_state_qpy = selected_dir / "graph_state_direct_basis.qpy"
    transpiled_qpy = selected_dir / "graph_state_direct_basis_transpiled.qpy"
    e_npy = selected_dir / "E.npy"
    w_npy = selected_dir / "W.npy"
    f3_qpy = selected_dir / "F3_W.qpy"
    cz_qpy = selected_dir / "CZ3_W.qpy"
    return {
        "state": config.state_name,
        "run_id": config.run_id,
        "selection_label": label,
        "approximation_degree": row.get("approximation_degree", ""),
        "rank": int(row["selection_rank"]),
        "class_name": row.get("class_name", ""),
        "candidate_name": row.get("candidate_name", ""),
        "best_depth": row.get("best_depth", ""),
        "best_two_qubit_gate_count": row.get("best_two_qubit_gate_count", ""),
        "best_one_qubit_gate_count": row.get("best_one_qubit_gate_count", ""),
        "best_size": row.get("best_size", ""),
        "fidelity": row.get("fidelity", ""),
        "graph_state_qpy": repo_relative(graph_state_qpy, config.repo_root),
        "transpiled_qpy": repo_relative(transpiled_qpy, config.repo_root),
        "E_npy": repo_relative(e_npy, config.repo_root),
        "W_npy": repo_relative(w_npy, config.repo_root) if w_npy.is_file() else "",
        "F_qpy": repo_relative(f3_qpy, config.repo_root),
        "CZ_qpy": repo_relative(cz_qpy, config.repo_root),
    }


def _copy_selected_row(config: SelectionConfig, label: str, row: pd.Series) -> dict[str, object]:
    selected_dir = _selected_candidate_dir(config, label, row)
    selected_dir.mkdir(parents=True, exist_ok=True)
    _copy_required(_path_from_row(row, "graph_state_qpy"), selected_dir / "graph_state_direct_basis.qpy")
    _copy_required(_path_from_row(row, "graph_state_transpiled_qpy"), selected_dir / "graph_state_direct_basis_transpiled.qpy")
    _copy_required(_path_from_row(row, "f3_w_qpy"), selected_dir / "F3_W.qpy")
    _copy_required(_path_from_row(row, "cz3_w_qpy"), selected_dir / "CZ3_W.qpy")
    _write_selected_encoding(_path_from_row(row, "basis_change_matrix_npy"), selected_dir)
    return _manifest_row(config, label, row, selected_dir)


def materialize_selected_artifacts(df: pd.DataFrame, config: SelectionConfig) -> SelectionOutput:
    require_supported_bell_state(config.state_name)
    selected_root = Path(config.selected_root) / config.state_name / config.run_id
    selected_root.mkdir(parents=True, exist_ok=True)
    Path(config.processed_dir).mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    top_csvs: dict[str, Path] = {}
    warnings: list[str] = []

    for label in config.labels:
        (selected_root / label).mkdir(parents=True, exist_ok=True)
        threshold = threshold_from_label(label)
        top = select_top_k(
            df,
            label=label,
            top_k=config.top_k,
            fidelity_threshold=threshold,
        )
        if len(top) < int(config.top_k):
            warnings.append(f"{label}: selected {len(top)} rows, requested {int(config.top_k)}")

        top_csv = (
            Path(config.processed_dir)
            / f"direct_basis_{config.state_name}_{config.run_id}_top{int(config.top_k)}_{label}_by_depth.csv"
        )
        top.to_csv(top_csv, index=False)
        top_csvs[label] = top_csv

        for _, row in top.iterrows():
            manifest_rows.append(_copy_selected_row(config, label, row))

    manifest = pd.DataFrame(manifest_rows)
    manifest_csv = selected_root / "manifest.csv"
    manifest.to_csv(manifest_csv, index=False)
    processed_manifest_csv = (
        Path(config.processed_dir)
        / f"direct_basis_{config.state_name}_{config.run_id}_selected_manifest.csv"
    )
    manifest.to_csv(processed_manifest_csv, index=False)
    return SelectionOutput(
        manifest_csv=manifest_csv,
        processed_manifest_csv=processed_manifest_csv,
        top_csvs=top_csvs,
        selected_count=len(manifest_rows),
        warnings=tuple(warnings),
    )
