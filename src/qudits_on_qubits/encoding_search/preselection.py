import warnings

import pandas as pd


_RANKING_COLUMNS = {
    "exact_depth": ["best_depth", "best_two_qubit_gate_count", "best_size"],
    "exact_2q": ["best_two_qubit_gate_count", "best_depth", "best_size"],
    "fid085": ["fid085_best_two_qubit_gate_count", "fid085_best_depth"],
    "fid090": ["fid090_best_two_qubit_gate_count", "fid090_best_depth"],
    "fid095": ["fid095_best_two_qubit_gate_count", "fid095_best_depth"],
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(column).strip() for column in df.columns]
    return df


def _validate_required_columns(df: pd.DataFrame, csv_path: str):
    missing = [column for column in ("class_name", "candidate_name") if column not in df.columns]
    if missing:
        raise ValueError(
            f"Preselection file {csv_path!r} is missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def _filter_to_state(df: pd.DataFrame, state_name: str, csv_path: str) -> pd.DataFrame:
    if "state_name" not in df.columns:
        warnings.warn(
            f"Preselection file {csv_path!r} has no state_name column; "
            f"cannot verify that it belongs to {state_name!r}.",
            stacklevel=2,
        )
        return df

    states = set(df["state_name"].astype(str).str.strip())
    if state_name not in states:
        raise ValueError(
            f"Preselection file {csv_path!r} contains state_name values "
            f"{sorted(states)}, not requested state {state_name!r}."
        )
    if states != {state_name}:
        warnings.warn(
            f"Preselection file {csv_path!r} contains mixed states {sorted(states)}; "
            f"using only rows for {state_name!r}.",
            stacklevel=2,
        )
    return df[df["state_name"].astype(str).str.strip() == state_name].copy()


def _sort_for_rank(df: pd.DataFrame, rank_by: str, csv_path: str) -> pd.DataFrame:
    if rank_by not in _RANKING_COLUMNS:
        raise ValueError(
            f"Unknown rank_by={rank_by!r}. Expected one of {sorted(_RANKING_COLUMNS)}."
        )

    columns = [column for column in _RANKING_COLUMNS[rank_by] if column in df.columns]
    if not columns:
        warnings.warn(
            f"Preselection file {csv_path!r} does not contain metric columns for "
            f"rank_by={rank_by!r}; preserving CSV order.",
            stacklevel=2,
        )
        return df

    ranked = df.copy()
    for column in columns:
        ranked[column] = pd.to_numeric(ranked[column], errors="coerce")
    return ranked.sort_values(by=columns, ascending=[True] * len(columns), na_position="last")


def select_preselected_rows(
    csv_path: str,
    state_name: str,
    top_k: int,
    rank_by: str = "exact_depth",
) -> pd.DataFrame:
    df = _normalize_columns(pd.read_csv(csv_path))
    _validate_required_columns(df, csv_path)
    df = _filter_to_state(df, state_name, csv_path)

    if "status" in df.columns:
        df = df[df["status"].astype(str).str.strip() == "ok"].copy()

    df["class_name"] = df["class_name"].astype(str).str.strip()
    df["candidate_name"] = df["candidate_name"].astype(str).str.strip()
    ranked = _sort_for_rank(df, rank_by, csv_path)
    ranked = ranked.drop_duplicates(["class_name", "candidate_name"], keep="first")

    if top_k is not None:
        ranked = ranked.head(int(top_k))
    return ranked.reset_index(drop=True)


def load_preselected_candidates(
    csv_path: str,
    state_name: str,
    top_k: int,
    rank_by: str = "exact_depth",
) -> set[tuple[str, str]]:
    selected = select_preselected_rows(
        csv_path=csv_path,
        state_name=state_name,
        top_k=top_k,
        rank_by=rank_by,
    )
    return set(zip(selected["class_name"], selected["candidate_name"]))
