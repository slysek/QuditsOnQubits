from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest
from qiskit import QuantumCircuit, qpy
from qiskit.quantum_info import Operator, Statevector

from qudits_on_qubits.benchmarks.direct_basis.candidates import DirectBasisCandidate
from qudits_on_qubits.benchmarks.direct_basis.math_utils import qutrit_fourier


benchmark = importlib.import_module("qudits_on_qubits.benchmarks.direct_basis.benchmark")


def _compare(**overrides):
    options = dict(
        state_name="two_qutrit", basis_matrix=np.eye(4, 3), n_qutrits=2,
        coupling_map=[[0, 1], [1, 2], [2, 3]],
        basis_gates=("cz", "rz", "sx", "x"), n_transpile_runs=2,
        transpiler_backend=None, optimization_level=3,
        layout_method=None, routing_method=None,
        approximation_degree=None, iqm_strategy_names=None,
    )
    options.update(overrides)
    return benchmark.benchmark_optimal_f3_graph_comparison(**options)


def _compiled(circuit):
    output = QuantumCircuit(circuit.num_qubits)
    output.cz(0, 1)
    if circuit.name.endswith("_baseline"):
        output.x(0)
        output.cz(0, 1)
    return output


@pytest.mark.parametrize("use_backend", [False, True])
def test_comparison_pairs_all_options_and_reports_positive_deltas(monkeypatch, use_backend):
    calls = []

    def compiler(circuit, **options):
        calls.append((circuit.name.rsplit("_", 1)[-1], options))
        return _compiled(circuit)

    monkeypatch.setattr(benchmark, "_transpile_one_trial", compiler)
    backend = object() if use_backend else None
    strategies = ("first", "second") if use_backend else (None,)
    result = _compare(
        transpiler_backend=backend,
        iqm_strategy_names=(name for name in strategies if name),
        layout_method="sabre", routing_method="basic", approximation_degree=0.99,
    )

    expected = [(variant, seed, strategy) for seed in range(2)
                for strategy in strategies for variant in ("baseline", "optimal")]
    assert [(v, o["trial"], o["iqm_strategy_name"]) for v, o in calls] == expected
    for (_, left), (_, right) in zip(calls[::2], calls[1::2]):
        assert left == right
        assert left["transpiler_backend"] is backend
        assert left["layout_method"] == "sabre"
        assert left["routing_method"] == "basic"
        assert left["approximation_degree"] == 0.99
        assert left["optimization_level"] == 3
        assert left["basis_gates"] == ("cz", "rz", "sx", "x")
        assert left["coupling_map"] == [[0, 1], [1, 2], [2, 3]]
    assert result["f3_graph_comparison_status"] == "ok"
    assert result["f3_graph_successful_pairs"] == 2 * len(strategies)
    assert result["f3_graph_failed_pairs"] == 0
    assert result["f3_optimal_leakage_phase_over_pi"] == pytest.approx(11 / 6)
    assert result["f3_graph_depth_improvement"] == 2
    assert result["f3_graph_size_improvement"] == 2
    assert result["f3_graph_two_qubit_gate_count_improvement"] == 1
    assert result["f3_graph_one_qubit_gate_count_improvement"] == 1
    assert result["f3_graph_baseline_mean_size"] == 3
    assert result["f3_graph_optimal_mean_size"] == 1
    assert json.loads(result["f3_graph_baseline_best_count_ops"]) == {"cz": 2, "x": 1}
    assert result["f3_graph_optimal_is_better"] is True


@pytest.mark.parametrize("matrix,status", [
    (qutrit_fourier(), "not_monomial"),
    (np.zeros((4, 3)), "analysis_error"),
])
def test_inapplicable_comparison_does_not_compile(monkeypatch, matrix, status):
    calls = []
    monkeypatch.setattr(benchmark, "_transpile_one_trial", lambda *a, **kw: calls.append(kw))
    result = _compare(basis_matrix=matrix)
    assert calls == []
    assert result["f3_graph_comparison_status"] == status
    assert result["f3_graph_size_improvement"] is None
    assert result["f3_graph_comparison_error"]


def test_failed_optimal_discards_its_baseline_and_preserves_negative_delta(monkeypatch):
    def compiler(circuit, *, trial, **options):
        if trial == 0:
            if circuit.name.endswith("_optimal"):
                raise RuntimeError("only one arm failed")
            return QuantumCircuit(4)  # Would spuriously win if not discarded.
        output = _compiled(circuit)
        if circuit.name.endswith("_optimal"):
            for _ in range(5):
                output.x(0)
        return output

    monkeypatch.setattr(benchmark, "_transpile_one_trial", compiler)
    result = _compare()
    assert result["f3_graph_successful_pairs"] == 1
    assert result["f3_graph_failed_pairs"] == 1
    assert result["f3_graph_baseline_best_size"] == 3
    assert result["f3_graph_baseline_best_seed_transpiler"] == 1
    assert result["f3_graph_size_improvement"] == -3
    assert result["f3_graph_optimal_is_better"] is False


def test_all_failed_pairs_leave_metrics_empty(monkeypatch):
    def compiler(*args, **kwargs):
        raise RuntimeError("compile failed")

    monkeypatch.setattr(benchmark, "_transpile_one_trial", compiler)
    result = _compare()
    assert result["f3_graph_comparison_status"] == "all_transpile_failed"
    assert result["f3_graph_failed_pairs"] == 2
    assert result["f3_graph_successful_pairs"] == 0
    assert result["f3_graph_depth_improvement"] is None
    assert "compile failed" in result["f3_graph_comparison_error"]


@pytest.mark.parametrize("error", [KeyboardInterrupt, SystemExit, MemoryError])
def test_comparison_preserves_process_interruptions(monkeypatch, error):
    def compiler(*args, **kwargs):
        raise error("stop")

    monkeypatch.setattr(benchmark, "_transpile_one_trial", compiler)
    with pytest.raises(error):
        _compare()


def test_ranks_each_arm_independently_over_identical_successful_pairs(monkeypatch):
    def compiler(circuit, *, trial, **options):
        baseline = circuit.name.endswith("_baseline")
        count = ((1, 3) if baseline else (4, 2))[trial]
        output = QuantumCircuit(4)
        for _ in range(count):
            output.cz(0, 1)
        return output

    monkeypatch.setattr(benchmark, "_transpile_one_trial", compiler)
    result = _compare()
    assert result["f3_graph_baseline_best_seed_transpiler"] == 0
    assert result["f3_graph_optimal_best_seed_transpiler"] == 1
    assert result["f3_graph_depth_improvement"] == -1
    assert result["f3_graph_baseline_mean_depth"] == 2
    assert result["f3_graph_optimal_mean_depth"] == 3


@pytest.mark.parametrize("enabled", [False, True])
def test_candidate_flag_reaches_comparison_without_changing_primary_metrics(monkeypatch, enabled):
    calls = []

    def compiler(circuit, *, trial, **options):
        calls.append(circuit.name)
        return _compiled(circuit)

    monkeypatch.setattr(benchmark, "_transpile_one_trial", compiler)
    candidate = DirectBasisCandidate(
        name="canonical", candidate_type="monomial", matrix=np.eye(4, 3),
        source_class_name="monomial_full", source_candidate_name="canonical",
    )
    frame, _ = benchmark.benchmark_direct_basis_candidates(
        state_name="two_qutrit", candidates=[candidate],
        n_transpile_runs=1, compute_fidelity=False,
        compare_optimal_f3_leakage=enabled,
    )
    row = frame.iloc[0]
    assert row["success"]
    assert row["best_size"] == 1
    assert row["best_depth"] == 1
    assert len(calls) == (3 if enabled else 1)
    assert row["f3_graph_comparison_status"] == ("ok" if enabled else "not_requested")


def test_default_run_does_not_analyze_or_build_comparison(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("comparison was not requested")

    monkeypatch.setattr(benchmark, "benchmark_optimal_f3_graph_comparison", unexpected, raising=False)
    monkeypatch.setattr(benchmark, "_transpile_one_trial", lambda circuit, **kw: _compiled(circuit))
    row = benchmark.benchmark_direct_basis(
        state_name="two_qutrit", basis_matrix=np.eye(3),
        basis_candidate_name="I", basis_candidate_type="test",
        n_transpile_runs=1, compute_fidelity=False,
    )
    assert row["success"]
    assert row["f3_graph_comparison_status"] == "not_requested"


def test_exports_loadable_source_pairs_without_overwriting_historical_files(monkeypatch, tmp_path):
    monkeypatch.setattr(benchmark, "_transpile_one_trial", lambda circuit, **kw: _compiled(circuit))
    row = benchmark.benchmark_direct_basis(
        state_name="two_qutrit", basis_matrix=np.eye(4, 3),
        basis_candidate_name="I", basis_candidate_type="test",
        n_transpile_runs=1, compute_fidelity=False,
        compare_optimal_f3_leakage=True, quantum_circuits_dir=str(tmp_path),
    )
    expected = {
        "f3_graph_baseline_qpy": "graph_state_f3_baseline.qpy",
        "f3_graph_optimal_qpy": "graph_state_f3_optimal.qpy",
        "f3_baseline_w_qpy": "F3_W_phi0.qpy",
        "f3_optimal_w_qpy": "F3_W_phi_optimal.qpy",
        "f3_w_qpy": "F3_W.qpy",
        "graph_state_qpy": "graph_state_direct_basis.qpy",
    }
    loaded = {}
    for key, filename in expected.items():
        path = Path(row[key])
        assert path.name == filename
        with path.open("rb") as handle:
            circuits = qpy.load(handle)
        assert len(circuits) == 1
        loaded[key] = circuits[0]
    assert row["success"]
    assert row["f3_graph_comparison_status"] == "ok"
    assert Operator(loaded["f3_w_qpy"]).equiv(Operator(loaded["f3_baseline_w_qpy"]))
    assert not Operator(loaded["f3_w_qpy"]).equiv(Operator(loaded["f3_optimal_w_qpy"]))
    historical = Statevector.from_instruction(loaded["graph_state_qpy"])
    for key in ("f3_graph_baseline_qpy", "f3_graph_optimal_qpy"):
        assert historical.equiv(Statevector.from_instruction(loaded[key]))
    assert loaded["graph_state_qpy"].count_ops().get("F3_W", 0) == 0
    # QPY restores UnitaryGate names as "unitary", retaining their labels.
    assert sum(
        instruction.operation.label == "F3_W"
        for instruction in loaded["f3_graph_optimal_qpy"].data
    ) == 2


def test_export_error_does_not_discard_valid_comparison_metrics(monkeypatch, tmp_path):
    def cannot_write(*args, **kwargs):
        raise OSError("read-only artifacts")

    monkeypatch.setattr(benchmark, "_save_qpy", cannot_write)
    monkeypatch.setattr(benchmark, "_transpile_one_trial", lambda circuit, **kw: _compiled(circuit))
    result = _compare(quantum_circuits_dir=str(tmp_path))
    assert result["f3_graph_comparison_status"] == "ok"
    assert result["f3_graph_successful_pairs"] == 2
    assert "read-only artifacts" in result["f3_graph_export_error"]
    assert result["f3_graph_baseline_qpy"] == ""


def test_strategy_iterator_is_reused_for_every_candidate_and_approximation(monkeypatch):
    monkeypatch.setattr(benchmark, "_transpile_one_trial", lambda circuit, **kw: _compiled(circuit))
    candidates = [
        DirectBasisCandidate(name=str(index), candidate_type="test", matrix=np.eye(3))
        for index in range(2)
    ]
    frame, _ = benchmark.benchmark_direct_basis_candidates(
        state_name="two_qutrit", candidates=candidates,
        n_transpile_runs=1, compute_fidelity=False, transpiler_backend=object(),
        iqm_strategy_names=(name for name in ("first", "second")),
        approximation_degrees=[0.99], compare_optimal_f3_leakage=True,
    )
    assert frame["f3_graph_successful_pairs"].tolist() == [2, 2, 2, 2]
    assert frame["successful_trials"].tolist() == [2, 2, 2, 2]


@pytest.mark.parametrize("matrix", [None, np.zeros((4, 3))])
def test_requested_comparison_on_invalid_candidate_is_not_reported_unrequested(matrix):
    candidate = DirectBasisCandidate(name="invalid", candidate_type="test", matrix=matrix)
    frame, _ = benchmark.benchmark_direct_basis_candidates(
        state_name="two_qutrit", candidates=[candidate],
        compare_optimal_f3_leakage=True, n_transpile_runs=1,
    )
    row = frame.iloc[0]
    assert not row["success"]
    assert row["f3_graph_comparison_status"] == "analysis_error"
    assert row["f3_graph_comparison_error"]


def test_invalid_compiler_output_does_not_accept_half_a_pair(monkeypatch):
    def compiler(circuit, **options):
        return None if circuit.name.endswith("_optimal") else _compiled(circuit)

    monkeypatch.setattr(benchmark, "_transpile_one_trial", compiler)
    result = _compare()
    assert result["f3_graph_successful_pairs"] == 0
    assert result["f3_graph_failed_pairs"] == 2
    assert result["f3_graph_baseline_best_depth"] is None
