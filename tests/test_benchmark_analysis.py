import pandas as pd
import pytest

from qudits_on_qubits.benchmarks.analysis import analyze_trials


def trial(candidate, seed, gates, depth, *, target="target-a", success=True):
    return dict(comparison_id="protocol", target_hash=target, workload_hash="work",
                state_name="custom", class_name="family", candidate_name=candidate,
                strategy_name="compiler", seed_transpiler=seed, success=success,
                status="ok" if success else "failed", two_qubit_gate_count=gates,
                depth=depth, one_qubit_gate_count=1, size=gates + 1,
                graph_state_transpiled_qpy=f"{candidate}/{seed}.qpy")


def test_actual_pareto_keeps_tradeoffs_and_excludes_incomplete_candidates():
    rows = [trial(name, seed, gates, depth)
            for name, gates, depth in [("baseline", 5, 10), ("cheap", 3, 12), ("short", 6, 8), ("bad", 7, 20)]
            for seed in (0, 1)]
    rows += [trial("partial", 0, 1, 1), trial("partial", 1, 0, 0, success=False)]
    result = analyze_trials(pd.DataFrame(rows), seeds=(0, 1)).set_index("candidate_name")
    assert set(result.index[result.pareto_rank == 1]) == {"baseline", "cheap", "short"}
    assert not result.loc["partial", "pareto_eligible"]
    assert pd.isna(result.loc["partial", "pareto_rank"])
    assert result.loc["bad", "pareto_rank"] == 2


def test_backend_boundaries_and_missing_seed_are_not_merged():
    rows = [trial("a", seed, 10, 10) for seed in (0, 1)]
    rows += [trial("b", seed, 1, 1, target="target-b") for seed in (0, 1)]
    rows += [trial("incomplete", 0, 0, 0)]
    result = analyze_trials(pd.DataFrame(rows), seeds=(0, 1)).set_index("candidate_name")
    assert result.loc["a", "pareto_rank"] == result.loc["b", "pareto_rank"] == 1
    assert not result.loc["incomplete", "pareto_eligible"]


def test_rejects_duplicate_or_unplanned_seeds():
    row = trial("a", 0, 1, 2)
    with pytest.raises(ValueError, match="duplicate"):
        analyze_trials(pd.DataFrame([row, row]), seeds=(0, 1))
    with pytest.raises(ValueError, match="seed"):
        analyze_trials(pd.DataFrame([trial("a", 9, 1, 2)]), seeds=(0, 1))
