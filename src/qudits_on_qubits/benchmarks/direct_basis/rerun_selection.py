from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


REQUIRED_INPUT_COLUMNS = ("state_name", "class_name", "candidate_name", "best_depth")


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
        if (
            isinstance(self.top_k, bool)
            or not isinstance(self.top_k, int)
            or self.top_k < 1
        ):
            raise ValueError("--top-k must be positive")
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
