"""Group compiled direct-basis candidates by reconstructed logical state."""

from __future__ import annotations

import os
from numbers import Real
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from qiskit import qpy
from qiskit.quantum_info import DensityMatrix, Statevector, state_fidelity

from .benchmark import logical_output_state


STATE_EQUIVALENCE_ATOL = 1e-9

_IDENTITY_COLUMNS = ("state_name", "class_name", "candidate_name", "strategy_name")
_OPTIONAL_BOUNDARY_COLUMNS = (
    "iqm_backend_name",
    "backend_calibration_set_id",
    "selection_label",
)
_RANKING_COLUMNS = (
    "pareto_rank",
    "ideal_score",
    "mean_two_qubit_gate_count",
    "mean_depth",
    "std_depth",
)
_QPY_COLUMN = "best_graph_state_transpiled_qpy"
_REQUIRED_COLUMNS = (
    *_IDENTITY_COLUMNS,
    "pareto_eligible",
    *_RANKING_COLUMNS,
    _QPY_COLUMN,
)
_ADDED_COLUMNS = (
    "state_equivalence_group_id",
    "state_equivalence_status",
    "state_equivalence_diagnostic",
    "recommended_class_name",
    "recommended_candidate_name",
    "recommended_strategy_name",
    "is_state_equivalence_recommendation",
)
_RECOMMENDATION_ORDER = (
    "pareto_rank",
    "ideal_score",
    "mean_two_qubit_gate_count",
    "mean_depth",
    "std_depth",
    "class_name",
    "candidate_name",
    "strategy_name",
)


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, dict, set)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _blank_path(path: object) -> bool:
    return _is_missing(path) or not str(path).strip()


def load_logical_state_from_qpy(
    path: str,
    logical_qubit_count: int | None,
    *,
    max_qubits: int,
) -> tuple[Statevector | DensityMatrix | None, str]:
    """Load one local QPY circuit and reconstruct its logical output state."""
    if _blank_path(path):
        return None, "Missing QPY path for logical state reconstruction."
    try:
        qpy_path = Path(os.fspath(path)).expanduser()
    except (TypeError, ValueError, OSError) as exc:
        return None, f"Invalid QPY path: {exc}"
    if not qpy_path.is_file():
        return None, f"Missing QPY file: {qpy_path}"
    try:
        with qpy_path.open("rb") as handle:
            circuits = qpy.load(handle)
    except Exception as exc:
        return None, f"QPY load failed: {type(exc).__name__}: {exc}"
    if len(circuits) != 1:
        return None, f"QPY payload must contain exactly one circuit; found {len(circuits)}."
    circuit = circuits[0]
    logical_width = logical_qubit_count
    if logical_width is None:
        logical_width = circuit.num_qubits
        layout = getattr(circuit, "layout", None)
        final_index_layout = getattr(layout, "final_index_layout", None)
        if callable(final_index_layout):
            try:
                logical_width = len(final_index_layout(filter_ancillas=True))
            except Exception:
                pass
    return logical_output_state(
        circuit,
        logical_qubit_count=logical_width,
        max_qubits=max_qubits,
    )


def _stable_value_key(value: object) -> tuple[int, str]:
    if _is_missing(value):
        return (1, "")
    return (0, f"{type(value).__name__}:{value}")


def _sort_frame(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    if frame.empty or not columns:
        return frame.copy()
    sortable = frame.copy()
    sort_columns: list[str] = []
    for index, column in enumerate(columns):
        temporary = f"__state_equivalence_sort_{index}"
        sortable[temporary] = sortable[column].map(_stable_value_key)
        sort_columns.append(temporary)
    return (
        sortable.sort_values(sort_columns, kind="mergesort")
        .drop(columns=sort_columns)
        .reset_index(drop=True)
    )


def _boundary_value(value: object) -> object:
    if _is_missing(value):
        return ("__state_equivalence_missing__",)
    try:
        hash(value)
    except TypeError:
        return (type(value).__name__, repr(value))
    return (type(value).__name__, value)


def _logical_width(row: pd.Series) -> int | None:
    if "n_qutrits" not in row or _is_missing(row["n_qutrits"]):
        return None
    value = row["n_qutrits"]
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        return None
    numeric = float(value)
    if not np.isfinite(numeric):
        return None
    width = numeric * 2
    return int(width) if width.is_integer() else None


def _resolved_path(path: object) -> str:
    return str(Path(os.fspath(path)).expanduser().resolve(strict=False))


def _eligible(row: pd.Series) -> bool:
    flag = row["pareto_eligible"]
    return isinstance(flag, (bool, np.bool_)) and bool(flag) and not _is_missing(row["pareto_rank"])


def _recommendation_value(value: object) -> tuple[int, float | str]:
    if _is_missing(value):
        return (1, "")
    if isinstance(value, Real) and not isinstance(value, (bool, np.bool_)):
        numeric = float(value)
        if np.isfinite(numeric):
            return (0, numeric)
    return (0, str(value))


def _recommendation_key(row: pd.Series) -> tuple[tuple[int, float | str], ...]:
    return tuple(_recommendation_value(row[column]) for column in _RECOMMENDATION_ORDER)


def _append_diagnostic(frame: pd.DataFrame, position: int, message: str) -> None:
    existing = str(frame.at[position, "state_equivalence_diagnostic"] or "").strip()
    if message not in existing:
        frame.at[position, "state_equivalence_diagnostic"] = " ".join(
            part for part in (existing, message) if part
        )


def _validate_schema(frame: pd.DataFrame) -> None:
    missing = [column for column in _REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"pareto_ranked is missing required columns: {', '.join(missing)}")


def _identity_display_value(value: object) -> str:
    if value is pd.NA or _is_missing(value):
        return "<NA>"
    return repr(value)


def _validate_unique_analysis_identities(frame: pd.DataFrame) -> None:
    identity_columns = (
        "state_name",
        *(column for column in _OPTIONAL_BOUNDARY_COLUMNS if column in frame.columns),
        "class_name",
        "candidate_name",
        "strategy_name",
    )
    duplicate_mask = frame.loc[:, identity_columns].duplicated(keep=False)
    if not duplicate_mask.any():
        return

    duplicate_rows = frame.loc[duplicate_mask, identity_columns]
    identities: dict[tuple[object, ...], tuple[tuple[object, ...], int]] = {}
    for values in duplicate_rows.itertuples(index=False, name=None):
        key = tuple(_boundary_value(value) for value in values)
        if key in identities:
            identity, count = identities[key]
            identities[key] = (identity, count + 1)
        else:
            identities[key] = (values, 1)
    ordered = sorted(
        identities.values(),
        key=lambda item: tuple(_stable_value_key(value) for value in item[0]),
    )
    descriptions = []
    for values, count in ordered:
        fields = ", ".join(
            f"{column}={_identity_display_value(value)}"
            for column, value in zip(identity_columns, values)
        )
        descriptions.append(f"({fields}; rows={count})")
    raise ValueError(
        "Duplicate state-equivalence analysis identity detected: "
        + "; ".join(descriptions)
    )


def group_state_equivalent_candidates(
    pareto_ranked: pd.DataFrame,
    *,
    max_qubits: int = 12,
    state_loader: Callable[
        ..., tuple[Statevector | DensityMatrix | None, str]
    ] = load_logical_state_from_qpy,
    atol: float = STATE_EQUIVALENCE_ATOL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return detailed rows and one recommendation per logical-state component."""
    if (
        isinstance(atol, (bool, np.bool_))
        or not isinstance(atol, Real)
        or not np.isfinite(float(atol))
        or float(atol) < 0
    ):
        raise ValueError("atol must be a finite non-negative real number")
    if pareto_ranked.empty:
        empty = pareto_ranked.copy()
        for column in _ADDED_COLUMNS:
            empty[column] = pd.Series(dtype="bool" if column == _ADDED_COLUMNS[-1] else "object")
        return empty, empty.copy()
    _validate_schema(pareto_ranked)
    _validate_unique_analysis_identities(pareto_ranked)

    boundary_columns = (
        "state_name",
        *(column for column in _OPTIONAL_BOUNDARY_COLUMNS if column in pareto_ranked.columns),
    )
    detailed = _sort_frame(
        pareto_ranked,
        (*boundary_columns, "class_name", "candidate_name", "strategy_name"),
    )
    detailed["state_equivalence_group_id"] = pd.NA
    detailed["state_equivalence_status"] = "ineligible"
    detailed["state_equivalence_diagnostic"] = "Not eligible for state-equivalence recommendation."
    detailed["recommended_class_name"] = pd.NA
    detailed["recommended_candidate_name"] = pd.NA
    detailed["recommended_strategy_name"] = pd.NA
    detailed["is_state_equivalence_recommendation"] = False

    partitions: dict[tuple[object, ...], list[int]] = {}
    for position, row in detailed.iterrows():
        if _eligible(row):
            key = tuple(_boundary_value(row[column]) for column in boundary_columns)
            partitions.setdefault(key, []).append(position)

    load_cache: dict[
        tuple[str, int | None, int],
        tuple[Statevector | DensityMatrix | None, str, str],
    ] = {}
    next_group_number = 1
    for positions in partitions.values():
        parent = {position: position for position in positions}
        states: dict[int, Statevector | DensityMatrix] = {}

        def find(position: int) -> int:
            while parent[position] != position:
                parent[position] = parent[parent[position]]
                position = parent[position]
            return position

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[max(left_root, right_root)] = min(left_root, right_root)

        for position in positions:
            row = detailed.loc[position]
            path = row[_QPY_COLUMN]
            if _blank_path(path):
                detailed.at[position, "state_equivalence_status"] = "missing_qpy"
                detailed.at[position, "state_equivalence_diagnostic"] = (
                    "Missing QPY path for logical state reconstruction."
                )
                continue
            logical_width = _logical_width(row)
            try:
                resolved = _resolved_path(path)
            except Exception as exc:
                detailed.at[position, "state_equivalence_status"] = "state_reconstruction_failed"
                detailed.at[position, "state_equivalence_diagnostic"] = (
                    f"State reconstruction failed: {type(exc).__name__}: {exc}"
                )
                continue
            cache_key = (resolved, logical_width, max_qubits)
            if cache_key not in load_cache:
                try:
                    state, diagnostic = state_loader(
                        resolved,
                        logical_width,
                        max_qubits=max_qubits,
                    )
                    diagnostic = str(diagnostic or "")
                    if state is None:
                        status = (
                            "missing_qpy"
                            if "missing" in diagnostic.casefold()
                            else "state_reconstruction_failed"
                        )
                    else:
                        status = "state_reconstructed"
                except Exception as exc:
                    state = None
                    diagnostic = f"State reconstruction failed: {type(exc).__name__}: {exc}"
                    status = "state_reconstruction_failed"
                load_cache[cache_key] = (state, diagnostic, status)
            state, diagnostic, status = load_cache[cache_key]
            detailed.at[position, "state_equivalence_status"] = status
            detailed.at[position, "state_equivalence_diagnostic"] = diagnostic
            if state is not None:
                states[position] = state

        state_positions = sorted(states)
        for left_offset, left in enumerate(state_positions):
            for right in state_positions[left_offset + 1 :]:
                try:
                    fidelity = float(state_fidelity(states[left], states[right]))
                    if not np.isfinite(fidelity):
                        raise ValueError(f"non-finite state fidelity {fidelity!r}")
                except Exception as exc:
                    message = f"State comparison failed: {type(exc).__name__}: {exc}"
                    detailed.at[left, "state_equivalence_status"] = "state_comparison_failed"
                    detailed.at[right, "state_equivalence_status"] = "state_comparison_failed"
                    _append_diagnostic(detailed, left, message)
                    _append_diagnostic(detailed, right, message)
                    continue
                if fidelity >= 1.0 - float(atol):
                    union(left, right)

        components: dict[int, list[int]] = {}
        for position in positions:
            components.setdefault(find(position), []).append(position)
        ordered_components = sorted(
            components.values(),
            key=lambda members: tuple(
                sorted(
                    tuple(str(detailed.at[position, column]) for column in _IDENTITY_COLUMNS[1:])
                    for position in members
                )
            ),
        )
        for members in ordered_components:
            group_id = f"state_equivalence_{next_group_number:04d}"
            next_group_number += 1
            recommended = min(members, key=lambda position: _recommendation_key(detailed.loc[position]))
            recommended_row = detailed.loc[recommended]
            for position in members:
                detailed.at[position, "state_equivalence_group_id"] = group_id
                detailed.at[position, "recommended_class_name"] = recommended_row["class_name"]
                detailed.at[position, "recommended_candidate_name"] = recommended_row["candidate_name"]
                detailed.at[position, "recommended_strategy_name"] = recommended_row["strategy_name"]
            detailed.at[recommended, "is_state_equivalence_recommendation"] = True

    compact = detailed.loc[detailed["is_state_equivalence_recommendation"]].copy()
    compact = _sort_frame(compact, (*boundary_columns, "state_equivalence_group_id"))
    return detailed, compact
