from dataclasses import replace

import numpy as np
import pytest

from qudits_on_qubits.benchmarks.theta_continuation import runner
from qudits_on_qubits.benchmarks.theta_continuation.artifacts import RunStore
from qudits_on_qubits.benchmarks.theta_continuation.models import ThetaBenchmarkConfig
from qudits_on_qubits.benchmarks.theta_continuation.cz3_continuation import ContinuationResult, ContinuationState, Cz3FitResult
from qudits_on_qubits.benchmarks.theta_continuation.compilation import CompilationResult, validate_cz3_candidate


@pytest.fixture
def config():
    return ThetaBenchmarkConfig(limit_points=1, transpiler_seeds=(0,), states=("two_qutrit",))


@pytest.fixture(autouse=True)
def no_bqskit(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("BQSKit must not run on successful continuation")
    monkeypatch.setattr(runner.optimized_gates, "synthesize_cz3", forbidden)


def test_theta_zero_real_run_persists_baseline_and_gate_artifacts(tmp_path, config):
    result = runner.run_theta_benchmark(config, tmp_path / "run", mode="gates")
    assert result["correct"] == result["total"] == 1
    assert result["fallback"] == result["failed"] == 0
    point = result["points"][0]
    assert point["selected_method"] == "baseline"
    assert point["cz3_n_cz"] <= point["baseline_n_cz"]
    store = RunStore.open(result["output_dir"])
    bundle = store.read_bundle("points/00000")
    assert bundle["arrays"]["cz3_parameters.npy"].shape == (33,)
    assert bundle["documents"]["branch_state.json"]["theta"] == 0.0
    assert len(store.read_bundle("baseline")["documents"]["template.json"]["parameter_metadata"]) == 33


def test_resume_recovers_point_written_before_checkpoint(tmp_path, config, monkeypatch):
    root = tmp_path / "run"
    original = RunStore.save_checkpoint
    monkeypatch.setattr(RunStore, "save_checkpoint", lambda *args: (_ for _ in ()).throw(RuntimeError("crash")))
    with pytest.raises(RuntimeError, match="crash"):
        runner.run_theta_benchmark(config, root, mode="gates")
    monkeypatch.setattr(RunStore, "save_checkpoint", original)
    monkeypatch.setattr(runner, "synthesize_theta_f3", lambda *args, **kwargs: pytest.fail("completed point resynthesized"))
    result = runner.run_theta_benchmark(config, root, mode="gates", resume=True)
    assert result["correct"] == 1


def test_resume_rejects_config_mismatch_and_corruption(tmp_path, config):
    root = tmp_path / "run"
    runner.run_theta_benchmark(config, root, mode="gates")
    with pytest.raises(ValueError, match="fingerprint"):
        runner.run_theta_benchmark(replace(config, max_nfev=5), root, mode="gates", resume=True)
    (root / "points/00000/branch_parameters.npy").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash"):
        runner.run_theta_benchmark(config, root, mode="gates", resume=True)


def test_circuits_mode_uses_saved_gates_without_synthesis(tmp_path, config, monkeypatch):
    root = tmp_path / "run"
    runner.run_theta_benchmark(config, root, mode="gates")
    monkeypatch.setattr(runner, "synthesize_theta_f3", lambda *args, **kwargs: pytest.fail("F3 synthesized"))
    monkeypatch.setattr(runner, "continue_cz3", lambda *args, **kwargs: pytest.fail("CZ3 fitted"))
    result = runner.run_theta_benchmark(config, root, mode="circuits")
    assert len(result["full_circuits"]) == 1
    row = result["full_circuits"][0]
    assert row["success"] is True
    assert row["fidelity"] > 1 - 1e-8
    assert row["gate_library"] == "theta_continuation_v1"
    monkeypatch.setattr(runner, "benchmark_direct_basis", lambda **kwargs: pytest.fail("completed full circuit rebuilt"))
    assert runner.run_theta_benchmark(config, root, mode="circuits")["full_circuits"] == result["full_circuits"]


def test_circuits_mode_requires_complete_gate_grid(tmp_path, config):
    with pytest.raises(FileNotFoundError):
        runner.run_theta_benchmark(config, tmp_path / "missing", mode="circuits")


def test_f3_failure_records_failed_point_without_skipping_baseline_cz3(tmp_path, config, monkeypatch):
    monkeypatch.setattr(runner, "synthesize_theta_f3", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("F3 failed")))
    result = runner.run_theta_benchmark(config, tmp_path / "run", mode="all")
    assert result["failed"] == 1
    point = result["points"][0]
    assert point["continuation_valid"] is True
    assert point["selected_method"] == "baseline"
    assert point["cz3_n_cz"] is not None
    assert result["full_circuits"] == []



def _controlled_result(template, state, theta, target_id, *, valid):
    circuit = template.bind(state.parameters)
    fit = Cz3FitResult(state.parameters.copy(), circuit, 0.0, valid, 1e-6, 1e-6, 1, 1, 1, "controlled", 0.01)
    attempt = {"id": target_id + "__attempt_001", "theta": theta, "requested_grid_point": True,
               "warm_start_parent_id": state.parent_id, "p_start": state.parameters.tolist(), "fit": fit.to_dict()}
    next_state = ContinuationState(theta, state.parameters.copy(), target_id) if valid else state
    return ContinuationResult(fit, next_state, [attempt])


def test_successful_continuation_never_uses_bqskit_and_resume_replays_branch(tmp_path, config, monkeypatch):
    config = replace(config, theta_points=1_000_001, limit_points=3)
    calls = []

    def continuation(template, state, theta, *, target_id, **kwargs):
        calls.append(state.parent_id)
        return _controlled_result(template, state, theta, target_id, valid=True)

    monkeypatch.setattr(runner, "continue_cz3", continuation)
    root = tmp_path / "run"
    result = runner.run_theta_benchmark(config, root, mode="gates")
    assert result["correct"] == 3
    assert calls == ["theta_00000", "theta_00001"]
    assert result["fallback"] == 0
    runner.run_theta_benchmark(config, root, mode="gates", resume=True)
    assert len(calls) == 2


def test_fallback_never_replaces_original_branch(tmp_path, config, monkeypatch):
    config = replace(config, theta_points=1_000_001, limit_points=3)
    calls = []
    baseline = runner.load_cz3_template(config.baseline_qpy).original_circuit

    def continuation(template, state, theta, *, target_id, **kwargs):
        calls.append((state.theta, state.parent_id))
        return _controlled_result(template, state, theta, target_id, valid=target_id == "theta_00002")

    monkeypatch.setattr(runner, "continue_cz3", continuation)
    monkeypatch.setattr(runner.optimized_gates, "synthesize_cz3", lambda encoding, **kwargs: baseline.copy())
    root = tmp_path / "run"
    result = runner.run_theta_benchmark(config, root, mode="gates")
    assert calls == [(0.0, "theta_00000"), (0.0, "theta_00000")]
    assert result["points"][1]["selected_method"] == "bqskit"
    assert result["points"][1]["continuation_valid"] is False
    assert result["points"][1]["branch_state"]["parent_id"] == "theta_00000"
    assert result["points"][2]["selected_method"] == "continuation"
    runner.run_theta_benchmark(config, root, mode="gates", resume=True)


@pytest.mark.parametrize("fallback_cost,expected_method,expected_cost", [(6, "bqskit", 6), (8, "continuation", 8), (10, "continuation", 8)])
def test_expensive_continuation_compares_fallback_cost_and_keeps_ties(config, monkeypatch, fallback_cost, expected_method, expected_cost):
    template = runner.load_cz3_template(config.baseline_qpy)
    baseline = runner._compile(template.original_circuit, np.eye(4, 3), config)
    prior = ContinuationState(0.0, template.initial_parameters.copy(), "theta_00000")
    monkeypatch.setattr(runner, "continue_cz3", lambda template, state, theta, target_id, **kwargs:
                        _controlled_result(template, state, theta, target_id, valid=True))
    real = runner._compile
    counts = iter((8, fallback_cost))

    def compile(circuit, encoding, config):
        result = real(circuit, encoding, config)
        padded = result.circuit.copy()
        for _ in range(next(counts) - result.metrics["N_CZ"]):
            padded.cz(0, 1)
        metrics = validate_cz3_candidate(padded, encoding, source=circuit, compiled=True)
        metrics["seed"] = 0
        return CompilationResult(padded, metrics, [])

    monkeypatch.setattr(runner, "_compile", compile)
    monkeypatch.setattr(runner.optimized_gates, "synthesize_cz3", lambda encoding, **kwargs: template.original_circuit.copy())
    metadata, _, _, _, current = runner._run_point(1, 1e-8, prior, template, baseline, config, None)
    assert metadata["fallback_used"] is True
    assert metadata["fallback_reason"] == "above_baseline_budget"
    assert metadata["selected_method"] == expected_method
    assert metadata["cz3_n_cz"] == expected_cost
    assert metadata["status"] == ("ok" if expected_cost == 6 else "above_baseline")
    assert current.parent_id == "theta_00001"


def test_invalid_fallback_cannot_replace_valid_expensive_continuation(config, monkeypatch):
    template = runner.load_cz3_template(config.baseline_qpy)
    baseline = runner._compile(template.original_circuit, np.eye(4, 3), config)
    prior = ContinuationState(0, template.initial_parameters.copy(), "theta_00000")
    monkeypatch.setattr(runner, "continue_cz3", lambda template, state, theta, target_id, **kwargs:
                        _controlled_result(template, state, theta, target_id, valid=True))
    real = runner._compile

    def compile(circuit, encoding, config):
        result = real(circuit, encoding, config)
        padded = result.circuit.copy()
        padded.cz(0, 1)
        padded.cz(0, 1)
        metrics = {**validate_cz3_candidate(padded, encoding, source=circuit, compiled=True), "seed": 0}
        return CompilationResult(padded, metrics, [])

    monkeypatch.setattr(runner, "_compile", compile)
    bad = template.original_circuit.copy()
    bad.u(0.3, 0, 0, 0)
    monkeypatch.setattr(runner.optimized_gates, "synthesize_cz3", lambda encoding, **kwargs: bad)
    metadata, _, _, _, current = runner._run_point(1, 1e-8, prior, template, baseline, config, None)
    assert metadata["selected_method"] == "continuation"
    assert metadata["status"] == "above_baseline"
    assert any("Fallback:" in error for error in metadata["errors"])
    assert current.parent_id == "theta_00001"


def test_runner_rejects_forged_valid_branch_parent_before_using_it(config, monkeypatch):
    template = runner.load_cz3_template(config.baseline_qpy)
    baseline = runner._compile(template.original_circuit, np.eye(4, 3), config)
    prior = ContinuationState(0, template.initial_parameters.copy(), "theta_00000")

    def continuation(template, state, theta, *, target_id, **kwargs):
        result = _controlled_result(template, state, theta, target_id, valid=True)
        return replace(result, state=replace(result.state, parent_id="unrelated_branch"))

    monkeypatch.setattr(runner, "continue_cz3", continuation)
    monkeypatch.setattr(runner.optimized_gates, "synthesize_cz3", lambda encoding, **kwargs: template.original_circuit.copy())
    metadata, _, _, _, current = runner._run_point(1, 1e-8, prior, template, baseline, config, None)
    assert current.parent_id == prior.parent_id
    assert metadata["continuation_valid"] is False
    assert metadata["selected_method"] == "bqskit"


def test_resume_rejects_forged_flattened_metrics(tmp_path, config):
    import hashlib
    import json
    root = tmp_path / "run"
    runner.run_theta_benchmark(config, root, mode="gates")
    metadata_path = root / "points/00000/metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["cz3_n_cz"] = 0
    metadata_path.write_text(json.dumps(metadata))
    complete_path = metadata_path.with_name("complete.json")
    complete = json.loads(complete_path.read_text())
    complete["files"]["metadata.json"] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    complete_path.write_text(json.dumps(complete))
    with pytest.raises(ValueError, match="metric"):
        runner.run_theta_benchmark(config, root, mode="gates", resume=True)


@pytest.mark.parametrize("state", ["ghz3", "ame43"])
def test_theta_zero_real_full_circuits_include_each_state(tmp_path, config, state):
    result = runner.run_theta_benchmark(replace(config, states=(state,)), tmp_path / state, mode="all")
    row = result["full_circuits"][0]
    assert row["success"] is True
    assert row["fidelity"] > 1 - 1e-8
    assert row["num_qubits"] == {"ghz3": 6, "ame43": 8}[state]


def test_invalid_continuation_and_fallback_record_failure(tmp_path, config, monkeypatch):
    config = replace(config, theta_points=1_000_001, limit_points=2)
    monkeypatch.setattr(runner, "continue_cz3", lambda template, state, theta, target_id, **kwargs:
                        _controlled_result(template, state, theta, target_id, valid=False))
    def failed_fallback(encoding, **kwargs):
        raise ValueError("fallback did not converge")
    monkeypatch.setattr(runner.optimized_gates, "synthesize_cz3", failed_fallback)
    result = runner.run_theta_benchmark(config, tmp_path / "run", mode="all")
    assert result["failed"] == 1
    assert result["fallback"] == 1
    assert result["points"][1]["selected_method"] is None
    assert result["points"][1]["cz3_E_norm"] is None
    assert len(result["full_circuits"]) == 1
    resumed = runner.run_theta_benchmark(config, tmp_path / "run", mode="gates", resume=True)
    assert resumed["failed"] == 1


def test_continuation_compile_failure_preserves_valid_parameters(config, monkeypatch):
    template = runner.load_cz3_template(config.baseline_qpy)
    baseline = runner._compile(template.original_circuit, np.eye(4, 3), config)
    prior = ContinuationState(0, template.initial_parameters.copy(), "theta_00000")
    monkeypatch.setattr(runner, "continue_cz3", lambda template, state, theta, target_id, **kwargs:
                        _controlled_result(template, state, theta, target_id, valid=True))
    real = runner._compile
    calls = []
    def compilation(circuit, encoding, config):
        calls.append(True)
        if len(calls) == 1:
            raise ValueError("compile failed")
        return real(circuit, encoding, config)
    monkeypatch.setattr(runner, "_compile", compilation)
    monkeypatch.setattr(runner.optimized_gates, "synthesize_cz3", lambda encoding, **kwargs: template.original_circuit.copy())
    metadata, _, arrays, _, current = runner._run_point(1, 1e-8, prior, template, baseline, config, None)
    assert metadata["continuation_valid"] is True
    assert metadata["fallback_reason"] == "continuation_compile_failed"
    assert metadata["selected_method"] == "bqskit"
    assert current.parent_id == "theta_00001"
    np.testing.assert_array_equal(arrays["cz3_parameters.npy"], current.parameters)


@pytest.mark.parametrize("theta", [0.0, 0.2])
def test_zero_preparation_cost_tracks_entangled_encoded_zero(config, theta):
    from qiskit.quantum_info import Statevector, state_fidelity
    encoding = runner.theta_embedding(theta)
    prepared = runner._compile_zero_preparation(encoding, replace(config, transpiler_seeds=(2, 0)))
    assert prepared.metrics["N_CZ"] == prepared.circuit.count_ops().get("cz", 0)
    assert (prepared.metrics["N_CZ"] == 0) if theta == 0 else (prepared.metrics["N_CZ"] > 0)
    assert state_fidelity(Statevector.from_instruction(prepared.circuit), encoding[:, 0]) > 1 - 1e-12
    assert prepared.metrics["full_error_norm"] < 1e-10
    assert {trial["seed"] for trial in prepared.trials} == {2, 0}


def test_all_graph_component_counts_and_persisted_preparation(tmp_path, config, monkeypatch):
    from qiskit.quantum_info import Statevector, state_fidelity
    config = replace(config, states=("two_qutrit", "ghz3", "ame43"))
    calls = []
    compile_preparation = runner._compile_zero_preparation
    def counted(encoding, config):
        calls.append(encoding.copy())
        return compile_preparation(encoding, config)
    monkeypatch.setattr(runner, "_compile_zero_preparation", counted)
    root = tmp_path / "run"
    result = runner.run_theta_benchmark(config, root, mode="all")
    assert len(calls) == 1
    store = RunStore.open(root)
    gate_bundle = store.read_bundle("points/00000")
    f3 = gate_bundle["circuits"]["f3_optimal.qpy"]
    cz3 = gate_bundle["circuits"]["cz3_selected.qpy"]
    for row, (n_qutrits, n_edges) in zip(result["full_circuits"], ((2, 1), (3, 2), (4, 5)), strict=True):
        assert row["component_num_qutrits"] == n_qutrits
        assert row["component_num_cz3_blocks"] == n_edges
        assert row["preparation_n_cz"] == 0
        assert row["f3_blocks_n_cz"] == n_qutrits * f3.count_ops().get("cz", 0)
        assert row["cz3_blocks_n_cz"] == n_edges * cz3.count_ops().get("cz", 0)
        assert row["f3_blocks_n_1q"] == n_qutrits * f3.count_ops().get("u", 0)
        assert row["cz3_blocks_n_1q"] == n_edges * cz3.count_ops().get("u", 0)
        assert row["unfused_n_cz"] == row["preparation_n_cz"] + row["f3_blocks_n_cz"] + row["cz3_blocks_n_cz"]
        assert row["unfused_n_1q"] == row["preparation_n_1q"] + row["f3_blocks_n_1q"] + row["cz3_blocks_n_1q"]
        bundle = store.read_bundle(f"circuits/{row['state_name']}/00000")
        zero = bundle["circuits"]["zero_preparation.qpy"]
        assert state_fidelity(Statevector.from_instruction(zero), [1, 0, 0, 0]) > 1 - 1e-12
        assert row["preparation_n_1q"] == n_qutrits * zero.count_ops().get("u", 0)
    monkeypatch.setattr(runner, "_compile_zero_preparation", lambda *args: pytest.fail("saved preparation recompiled"))
    assert runner.run_theta_benchmark(config, root, mode="circuits")["full_circuits"] == result["full_circuits"]


@pytest.mark.parametrize("corruption", ["component_metric", "preparation_state", "preparation_operator"])
def test_resume_rejects_forged_preparation_and_component_metrics(tmp_path, config, corruption):
    import hashlib
    import json
    from qiskit import qpy
    root = tmp_path / "run"
    runner.run_theta_benchmark(config, root, mode="all")
    directory = root / "circuits/two_qutrit/00000"
    complete_path = directory / "complete.json"
    complete = json.loads(complete_path.read_text())
    if corruption == "component_metric":
        for filename in ("metadata.json", "row.json"):
            path = directory / filename
            row = json.loads(path.read_text())
            row["preparation_n_cz"] = 100
            path.write_text(json.dumps(row))
            complete["files"][filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    else:
        path = directory / "zero_preparation.qpy"
        with path.open("rb") as stream:
            preparation = qpy.load(stream)[0]
        if corruption == "preparation_state":
            preparation.u(0.3, 0, 0, 0)
        else:
            preparation.u(0, 0, 0.3, 0)
        with path.open("wb") as stream:
            qpy.dump(preparation, stream)
        complete["files"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    complete_path.write_text(json.dumps(complete))
    with pytest.raises(ValueError, match="preparation|component"):
        runner.run_theta_benchmark(config, root, mode="circuits")


def test_failed_point_progress_includes_norms_and_error(tmp_path, config, monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("test failure detail")
    monkeypatch.setattr(runner, "synthesize_theta_f3", fail)
    messages = []
    runner.run_theta_benchmark(config, tmp_path / "run", mode="gates", progress=messages.append)
    assert "E_norm=" in messages[-1]
    assert "L_norm=" in messages[-1]
    assert "test failure detail" in messages[-1]


def test_component_counts_skip_self_edges_and_keep_duplicates(config, monkeypatch):
    from types import SimpleNamespace
    from qiskit import QuantumCircuit
    preparation = QuantumCircuit(2)
    preparation.cz(0, 1)
    f3 = QuantumCircuit(2)
    f3.u(0.1, 0.2, 0.3, 0)
    cz3 = QuantumCircuit(4)
    cz3.cz(0, 2)
    monkeypatch.setattr(runner, "resolve_direct_state", lambda name: SimpleNamespace(num_qutrits=3, edges=[(0, 0), (0, 1), (0, 1), (1, 2)]))
    costs = runner._component_costs("custom", preparation, SimpleNamespace(f3=f3, cz3=cz3))
    assert costs["component_num_cz3_blocks"] == 3
    assert costs["preparation_n_cz"] == 3
    assert costs["f3_blocks_n_1q"] == 3
    assert costs["unfused_n_cz"] == 6


def test_preparation_compilation_ranks_exact_candidates_with_seed_tie_break(config, monkeypatch):
    config = replace(config, transpiler_seeds=(0, 7, 3))
    encoding = runner.theta_embedding(0.2)
    source = runner._zero_preparation_source(encoding)
    base = runner.transpile(source, basis_gates=["u", "cz"], optimization_level=3)
    costly = base.copy()
    costly.cz(0, 1)
    costly.cz(0, 1)
    calls = []
    def compile(source, **kwargs):
        calls.append(kwargs)
        return costly.copy() if kwargs["seed_transpiler"] == 0 else base.copy()
    monkeypatch.setattr(runner, "transpile", compile)
    result = runner._compile_zero_preparation(encoding, config)
    assert result.metrics["seed"] == 3
    assert result.metrics["N_CZ"] == base.count_ops().get("cz", 0)
    assert all(call["coupling_map"] is None and call["routing_method"] == "none" and call["approximation_degree"] == 1.0 for call in calls)


def test_preparation_compilation_rejects_all_invalid_seed_results(config, monkeypatch):
    from qiskit import QuantumCircuit
    monkeypatch.setattr(runner, "transpile", lambda *args, **kwargs: QuantumCircuit(2))
    with pytest.raises(ValueError, match="No exact zero-preparation"):
        runner._compile_zero_preparation(runner.theta_embedding(0.2), config)


def test_runner_propagates_explicit_cz3_tolerance_through_synthesis_and_graphs(tmp_path, config, monkeypatch):
    config = replace(config, theta_points=10_001, limit_points=2, cz3_tolerance=5e-4)
    baseline = runner.load_cz3_template(config.baseline_qpy).original_circuit
    baseline.u(1e-4, 0, 0, 0)
    synthesized_tolerances = []
    benchmark_tolerances = []
    real_benchmark = runner.benchmark_direct_basis
    def failed_continuation(*args, **kwargs):
        raise ValueError("force fallback")
    def fallback(encoding, *, tolerance):
        synthesized_tolerances.append(tolerance)
        return baseline.copy()
    def benchmark(**kwargs):
        benchmark_tolerances.append(kwargs["cz3_tolerance"])
        return real_benchmark(**kwargs)
    monkeypatch.setattr(runner, "continue_cz3", failed_continuation)
    monkeypatch.setattr(runner.optimized_gates, "synthesize_cz3", fallback)
    monkeypatch.setattr(runner, "benchmark_direct_basis", benchmark)
    result = runner.run_theta_benchmark(config, tmp_path / "relaxed", mode="all")
    assert result["correct"] == 2
    assert synthesized_tolerances == [5e-4]
    assert benchmark_tolerances == [5e-4, 5e-4]
    assert len(result["full_circuits"]) == 2
    assert all(row["success"] for row in result["full_circuits"])
    assert 1e-5 < result["points"][1]["cz3_E_norm"] < 5e-4
