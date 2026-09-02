from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from qiskit import QuantumCircuit, qpy, transpile
from qiskit.transpiler import CouplingMap
from qiskit.quantum_info import DensityMatrix, Statevector, state_fidelity


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from qudits_on_qubits.benchmarks.direct_basis.state_equivalence import (
    STATE_EQUIVALENCE_ATOL,
    group_state_equivalent_candidates,
    load_logical_state_from_qpy,
)


ADDED_COLUMNS = [
    "state_equivalence_group_id",
    "state_equivalence_status",
    "state_equivalence_diagnostic",
    "recommended_class_name",
    "recommended_candidate_name",
    "recommended_strategy_name",
    "is_state_equivalence_recommendation",
]


def _row(
    candidate_name: str,
    path: object,
    *,
    state_name: str = "ghz3",
    class_name: str = "product",
    strategy_name: str = "default",
    pareto_eligible: object = True,
    pareto_rank: object = 1,
    ideal_score: object = 0.1,
    mean_two_qubit_gate_count: object = 2.0,
    mean_depth: object = 10.0,
    std_depth: object = 1.0,
    n_qutrits: object = np.nan,
    **extra: object,
) -> dict[str, object]:
    return {
        "state_name": state_name,
        "class_name": class_name,
        "candidate_name": candidate_name,
        "strategy_name": strategy_name,
        "pareto_eligible": pareto_eligible,
        "pareto_rank": pareto_rank,
        "ideal_score": ideal_score,
        "mean_two_qubit_gate_count": mean_two_qubit_gate_count,
        "mean_depth": mean_depth,
        "std_depth": std_depth,
        "best_graph_state_transpiled_qpy": path,
        "n_qutrits": n_qutrits,
        **extra,
    }


def _density(amplitudes) -> DensityMatrix:
    return DensityMatrix(Statevector(np.asarray(amplitudes, dtype=complex)))


def _mapping_loader(states, calls=None):
    def load(path, logical_qubit_count, *, max_qubits):
        if calls is not None:
            calls.append((str(path), logical_qubit_count, max_qubits))
        return states[Path(path).name], ""

    return load


def test_global_phase_equivalent_states_group_and_best_row_is_retained():
    states = {
        "a.qpy": _density([1, 0]),
        "b.qpy": _density([-1j, 0]),
        "c.qpy": _density([0, 1]),
    }
    rows = [
        _row("b", "b.qpy", ideal_score=0.2),
        _row("c", "c.qpy", ideal_score=0.3),
        _row("a", "a.qpy", ideal_score=0.1),
    ]

    detailed, compact = group_state_equivalent_candidates(
        pd.DataFrame(rows), state_loader=_mapping_loader(states)
    )
    indexed = detailed.set_index("candidate_name")

    assert STATE_EQUIVALENCE_ATOL == 1e-9
    assert indexed.loc["a", "state_equivalence_group_id"] == indexed.loc["b", "state_equivalence_group_id"]
    assert indexed.loc["a", "state_equivalence_group_id"] != indexed.loc["c", "state_equivalence_group_id"]
    assert indexed.loc["a", "recommended_candidate_name"] == "a"
    assert indexed.loc["b", "recommended_candidate_name"] == "a"
    assert indexed.loc["a", "is_state_equivalence_recommendation"]
    assert not indexed.loc["b", "is_state_equivalence_recommendation"]
    assert compact["candidate_name"].tolist() == ["a", "c"]


def test_pairwise_transitive_equivalence_uses_union_find_connectivity():
    angles = {"a.qpy": 0.0, "b.qpy": 0.08, "c.qpy": 0.16}
    states = {
        name: _density([np.cos(angle), np.sin(angle)])
        for name, angle in angles.items()
    }
    assert state_fidelity(states["a.qpy"], states["b.qpy"]) >= 0.99
    assert state_fidelity(states["b.qpy"], states["c.qpy"]) >= 0.99
    assert state_fidelity(states["a.qpy"], states["c.qpy"]) < 0.99

    detailed, compact = group_state_equivalent_candidates(
        pd.DataFrame([_row(name[0], name) for name in states]),
        state_loader=_mapping_loader(states),
        atol=0.01,
    )

    assert detailed["state_equivalence_group_id"].nunique() == 1
    assert len(compact) == 1


def test_boundaries_never_compare_and_na_values_share_a_boundary():
    state = _density([1, 0])
    rows = [
        _row("a1", "a1.qpy", iqm_backend_name="a"),
        _row("a2", "a2.qpy", iqm_backend_name="a"),
        _row("b1", "b1.qpy", iqm_backend_name="b"),
        _row("n1", "n1.qpy", iqm_backend_name=np.nan),
        _row("n2", "n2.qpy", iqm_backend_name=np.nan),
    ]
    states = {Path(row["best_graph_state_transpiled_qpy"]).name: state for row in rows}

    detailed, compact = group_state_equivalent_candidates(
        pd.DataFrame(rows), state_loader=_mapping_loader(states)
    )
    ids = detailed.set_index("candidate_name")["state_equivalence_group_id"]

    assert ids["a1"] == ids["a2"]
    assert ids["n1"] == ids["n2"]
    assert len(compact) == 3
    assert compact["iqm_backend_name"].isna().sum() == 1


def test_all_optional_boundaries_and_state_name_are_respected():
    state = _density([1, 0])
    rows = [
        _row("base", "base.qpy", state_name="s1", iqm_backend_name="a", backend_calibration_set_id="c1", selection_label="exact"),
        _row("state", "state.qpy", state_name="s2", iqm_backend_name="a", backend_calibration_set_id="c1", selection_label="exact"),
        _row("cal", "cal.qpy", state_name="s1", iqm_backend_name="a", backend_calibration_set_id="c2", selection_label="exact"),
        _row("selection", "selection.qpy", state_name="s1", iqm_backend_name="a", backend_calibration_set_id="c1", selection_label="approx"),
    ]
    states = {Path(row["best_graph_state_transpiled_qpy"]).name: state for row in rows}
    _, compact = group_state_equivalent_candidates(
        pd.DataFrame(rows), state_loader=_mapping_loader(states)
    )
    assert len(compact) == 4


def test_output_and_group_ids_are_deterministic_for_shuffled_input():
    states = {"a.qpy": _density([1, 0]), "b.qpy": _density([1, 0]), "c.qpy": _density([0, 1])}
    frame = pd.DataFrame([_row("c", "c.qpy"), _row("b", "b.qpy"), _row("a", "a.qpy")])

    first = group_state_equivalent_candidates(frame, state_loader=_mapping_loader(states))
    second = group_state_equivalent_candidates(
        frame.sample(frac=1, random_state=17), state_loader=_mapping_loader(states)
    )

    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])


def test_recommendation_tie_breaking_uses_documented_stable_order():
    state = _density([1, 0])
    states = {f"{name}.qpy": state for name in ("rank", "score", "metric", "lexical")}
    rows = [
        _row("rank", "rank.qpy", pareto_rank=2, ideal_score=0, mean_two_qubit_gate_count=0),
        _row("score", "score.qpy", pareto_rank=1, ideal_score=0.2, mean_two_qubit_gate_count=0),
        _row("metric", "metric.qpy", pareto_rank=1, ideal_score=0.1, mean_two_qubit_gate_count=2),
        _row("lexical", "lexical.qpy", pareto_rank=1, ideal_score=0.1, mean_two_qubit_gate_count=1),
    ]
    detailed, compact = group_state_equivalent_candidates(
        pd.DataFrame(rows), state_loader=_mapping_loader(states)
    )
    assert detailed["recommended_candidate_name"].unique().tolist() == ["lexical"]
    assert compact["candidate_name"].tolist() == ["lexical"]


def test_ineligible_and_null_rank_rows_remain_detailed_but_not_compact():
    state = _density([1, 0])
    calls = []
    rows = [
        _row("eligible", "eligible.qpy"),
        _row("flagged", "flagged.qpy", pareto_eligible=False),
        _row("unranked", "unranked.qpy", pareto_rank=np.nan),
    ]
    detailed, compact = group_state_equivalent_candidates(
        pd.DataFrame(rows),
        state_loader=_mapping_loader({"eligible.qpy": state}, calls),
    )
    indexed = detailed.set_index("candidate_name")

    assert indexed.loc["flagged", "state_equivalence_status"] == "ineligible"
    assert indexed.loc["unranked", "state_equivalence_status"] == "ineligible"
    assert pd.isna(indexed.loc["flagged", "state_equivalence_group_id"])
    assert compact["candidate_name"].tolist() == ["eligible"]
    assert len(calls) == 1


def test_missing_exception_and_unsafe_reconstruction_are_diagnostic_singletons():
    def loader(path, logical_qubit_count, *, max_qubits):
        name = Path(path).name
        if name == "exception.qpy":
            raise RuntimeError("broken payload")
        if name == "unsafe.qpy":
            return None, "unsafe width exceeds max_qubits"
        raise AssertionError(name)

    rows = [
        _row("missing", "  "),
        _row("exception", "exception.qpy"),
        _row("unsafe", "unsafe.qpy"),
    ]
    detailed, compact = group_state_equivalent_candidates(
        pd.DataFrame(rows), state_loader=loader
    )
    indexed = detailed.set_index("candidate_name")

    assert indexed.loc["missing", "state_equivalence_status"] == "missing_qpy"
    assert indexed.loc["exception", "state_equivalence_status"] == "state_reconstruction_failed"
    assert indexed.loc["unsafe", "state_equivalence_status"] == "state_reconstruction_failed"
    assert all(indexed.loc[name, "state_equivalence_diagnostic"] for name in indexed.index)
    assert detailed["state_equivalence_group_id"].nunique() == 3
    assert len(compact) == 3


def test_loader_results_and_failures_are_cached_by_path_width_and_limit():
    calls = []

    def loader(path, logical_qubit_count, *, max_qubits):
        calls.append((path, logical_qubit_count, max_qubits))
        if Path(path).name == "bad.qpy":
            raise ValueError("bad qpy")
        return _density([1, 0]), ""

    rows = [
        _row("a", "same.qpy", n_qutrits=0.5),
        _row("b", "same.qpy", n_qutrits=0.5),
        _row("c", "bad.qpy", n_qutrits=0.5),
        _row("d", "bad.qpy", n_qutrits=0.5),
    ]
    group_state_equivalent_candidates(pd.DataFrame(rows), state_loader=loader, max_qubits=7)

    assert len(calls) == 2
    assert {call[1:] for call in calls} == {(1, 7)}


@pytest.mark.parametrize("atol", [-1, np.nan, np.inf, -np.inf])
def test_invalid_atol_is_rejected(atol):
    with pytest.raises(ValueError, match="atol"):
        group_state_equivalent_candidates(pd.DataFrame(), atol=atol)


def test_empty_input_has_added_columns_and_matching_compact_schema():
    detailed, compact = group_state_equivalent_candidates(pd.DataFrame())
    assert detailed.empty and compact.empty
    assert detailed.columns.tolist() == ADDED_COLUMNS
    assert compact.columns.tolist() == ADDED_COLUMNS


def test_missing_required_columns_are_named():
    with pytest.raises(ValueError, match="missing required.*pareto_rank.*ideal_score"):
        group_state_equivalent_candidates(pd.DataFrame([{"state_name": "ghz3"}]))


def test_comparison_errors_do_not_merge_states_and_are_reported():
    states = {"one.qpy": _density([1, 0]), "two.qpy": _density([1, 0, 0, 0])}
    detailed, compact = group_state_equivalent_candidates(
        pd.DataFrame([_row("one", "one.qpy"), _row("two", "two.qpy")]),
        state_loader=_mapping_loader(states),
    )
    assert detailed["state_equivalence_group_id"].nunique() == 2
    assert set(detailed["state_equivalence_status"]) == {"state_comparison_failed"}
    assert detailed["state_equivalence_diagnostic"].str.contains("comparison", case=False).all()
    assert len(compact) == 2


def test_default_qpy_loader_reconstructs_a_local_circuit(tmp_path):
    circuit = QuantumCircuit(1)
    circuit.h(0)
    path = tmp_path / "one.qpy"
    with path.open("wb") as handle:
        qpy.dump(circuit, handle)

    state, diagnostic = load_logical_state_from_qpy(
        str(path), None, max_qubits=1
    )

    assert state is not None, diagnostic
    assert state_fidelity(Statevector.from_instruction(circuit), state) > 1 - 1e-10


def test_default_qpy_loader_infers_pretranspile_input_width(tmp_path):
    source = QuantumCircuit(1)
    source.x(0)
    compiled = transpile(
        source,
        basis_gates=["u", "cx"],
        coupling_map=CouplingMap.from_line(3),
        initial_layout=[2],
        optimization_level=0,
    )
    path = tmp_path / "compiled.qpy"
    with path.open("wb") as handle:
        qpy.dump(compiled, handle)

    state, diagnostic = load_logical_state_from_qpy(
        str(path), None, max_qubits=1
    )

    assert state is not None, diagnostic
    assert state_fidelity(Statevector.from_instruction(source), state) > 1 - 1e-10


def test_default_qpy_loader_rejects_zero_and_multiple_circuits(tmp_path):
    circuit = QuantumCircuit(1)
    zero = tmp_path / "zero.qpy"
    multiple = tmp_path / "multiple.qpy"
    zero.touch()
    with multiple.open("wb") as handle:
        qpy.dump([circuit, circuit], handle)

    with patch(
        "qudits_on_qubits.benchmarks.direct_basis.state_equivalence.qpy.load",
        return_value=[],
    ):
        zero_state, zero_diagnostic = load_logical_state_from_qpy(
            str(zero), None, max_qubits=1
        )
    multiple_state, multiple_diagnostic = load_logical_state_from_qpy(
        str(multiple), None, max_qubits=1
    )

    assert zero_state is None
    assert "exactly one" in zero_diagnostic
    assert multiple_state is None
    assert "exactly one" in multiple_diagnostic


@pytest.mark.parametrize("path", [None, "", "   "])
def test_default_qpy_loader_reports_missing_paths(path):
    state, diagnostic = load_logical_state_from_qpy(path, None, max_qubits=1)
    assert state is None
    assert "missing" in diagnostic.casefold()
