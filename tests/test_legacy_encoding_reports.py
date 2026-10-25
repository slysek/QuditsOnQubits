import pandas as pd

from qudits_on_qubits.core.benchmark_encoding_bases import _save_top3_per_class_csvs


def test_legacy_rankings_keep_depth_gate_tradeoffs_and_compatibility_alias(tmp_path):
    rows = [
        dict(class_name="product", candidate_name=name, status="ok", best_depth=depth,
             best_two_qubit_gate_count=gates, best_size=gates + 1,
             fid095_best_depth=depth, fid095_best_two_qubit_gate_count=gates,
             fid095_best_fidelity=1.0)
        for name, depth, gates in [("short", 1, 9), ("cheap", 9, 1), ("middle", 5, 5), ("last", 10, 10)]
    ]
    rows.append({**rows[0], "candidate_name": "failed", "status": "build_error", "best_depth": 0})
    _save_top3_per_class_csvs(pd.DataFrame(rows), str(tmp_path / "run.csv"), fidelity_thresholds=(0.95,))

    depth = pd.read_csv(tmp_path / "run_top3_by_depth.csv")
    gates = pd.read_csv(tmp_path / "run_top3_by_2q.csv")
    fidelity = pd.read_csv(tmp_path / "run_top3_fid095.csv")
    assert depth.candidate_name.tolist() == ["short", "middle", "cheap"]
    assert gates.candidate_name.tolist() == ["cheap", "middle", "short"]
    assert fidelity.candidate_name.tolist() == ["cheap", "middle", "short"]
    assert (tmp_path / "run_top3_exact.csv").read_bytes() == (tmp_path / "run_top3_by_2q.csv").read_bytes()
