from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from qiskit import QuantumCircuit, qpy, transpile
from qiskit.quantum_info import Operator, Statevector, state_fidelity

from qudits_on_qubits.benchmarks.direct_basis import benchmark
from qudits_on_qubits.benchmarks.direct_basis import optimized_gates
from qudits_on_qubits.benchmarks.direct_basis.circuits import (
    build_direct_basis_graph_state_circuit,
)
from qudits_on_qubits.benchmarks.direct_basis.math_utils import (
    qutrit_cz,
    qutrit_fourier,
)


@pytest.fixture
def supplied_library(exact_cz3_synthesis):
    source = optimized_gates.optimized_gate_library(np.eye(3))
    # The theta wrapper uses CZ for both blocks. Preserve a distinct global phase
    # to verify exports retain the supplied circuits rather than rebuilding them.
    f3 = transpile(source.f3, basis_gates=["u", "cz"], optimization_level=1)
    f3.global_phase += 0.17
    cz3 = source.cz3.copy()
    cz3.global_phase += 0.29
    metrics = {
        "gate_library": "supplied_theta",
        "f3_synthesis_method": "supplied_local_wrapper",
        "cz3_synthesis_method": "supplied_canonical",
        **{key: -99 for key in (
            "E_norm", "L_norm", "N_2q", "raw_error_norm",
            "f3_E_norm", "f3_L_norm", "f3_N_2q", "f3_raw_error_norm",
        )},
    }
    f3.metadata = dict(metrics)
    cz3.metadata = dict(metrics)
    return SimpleNamespace(f3=f3, cz3=cz3, benchmark_metrics=lambda: dict(metrics))


def _run(**kwargs):
    options = {
        "state_name": "two_qutrit",
        "basis_matrix": np.eye(3),
        "basis_candidate_name": "theta_test",
        "basis_candidate_type": "test",
        "n_transpile_runs": 1,
        "basis_gates": ["u", "cz"],
        "coupling_map": [[0, 1], [1, 2], [2, 3]],
    }
    return benchmark.benchmark_direct_basis(**{**options, **kwargs})


def _forbid_synthesis(*args, **kwargs):
    pytest.fail("A supplied gate library must not trigger synthesis")


def test_default_benchmark_still_synthesizes_and_validates(exact_cz3_synthesis):
    row = _run()

    assert row["success"]
    assert len(exact_cz3_synthesis) == 1
    assert row["gate_library"] == optimized_gates.GATE_LIBRARY
    assert row["f3_E_norm"] <= 1e-10
    assert row["E_norm"] <= 1e-5
    assert row["fidelity"] >= 1 - 1e-10


def test_supplied_library_reused_validated_and_exported(supplied_library, monkeypatch, tmp_path):
    monkeypatch.setattr(benchmark, "optimized_gate_library", _forbid_synthesis)
    seen = []
    for name in (
        "build_optimized_direct_basis_graph_state_circuit",
        "export_direct_basis_candidate_circuits",
    ):
        original = getattr(benchmark, name)

        def record(*args, _original=original, **kwargs):
            seen.append(kwargs["gate_library"])
            return _original(*args, **kwargs)

        monkeypatch.setattr(benchmark, name, record)

    row = _run(gate_library=supplied_library, quantum_circuits_dir=str(tmp_path))

    assert row["success"], row["error_message"]
    assert len(seen) == 2 and all(item is supplied_library for item in seen)
    assert row["gate_library"] == "supplied_theta"
    assert row["f3_synthesis_method"] == "supplied_local_wrapper"
    assert row["cz3_synthesis_method"] == "supplied_canonical"
    embedding = np.eye(4, 3)
    for circuit, path_key, prefix, code, target in (
        (supplied_library.f3, "f3_w_qpy", "f3_", embedding, qutrit_fourier()),
        (supplied_library.cz3, "cz3_w_qpy", "", np.kron(embedding, embedding), qutrit_cz()),
    ):
        actual = optimized_gates.validate_code_space_gate(circuit, code, target)
        for key in ("E_norm", "L_norm", "N_2q", "raw_error_norm"):
            assert row[prefix + key] == pytest.approx(getattr(actual, key))
        with open(row[path_key], "rb") as handle:
            exported = qpy.load(handle)[0]
        np.testing.assert_allclose(Operator(exported).data, Operator(circuit).data, atol=1e-12)
        assert exported.count_ops() == circuit.count_ops()
    with open(row["graph_state_qpy"], "rb") as handle:
        graph = qpy.load(handle)[0]
    reference = build_direct_basis_graph_state_circuit("two_qutrit", np.eye(3))
    assert state_fidelity(Statevector(graph), Statevector(reference)) >= 1 - 1e-10
    assert row["fidelity"] >= 1 - 1e-10
    assert supplied_library.benchmark_metrics()["E_norm"] == -99


@pytest.mark.parametrize("fault", ["encoding", "leakage", "column_phases"])
def test_wrong_supplied_code_action_fails_before_transpilation(
    fault, supplied_library, monkeypatch,
):
    monkeypatch.setattr(benchmark, "optimized_gate_library", _forbid_synthesis)
    monkeypatch.setattr(benchmark, "_transpile_one_trial", lambda *a, **k: pytest.fail("Invalid gates reached transpilation"))
    encoding = np.eye(3)
    if fault == "encoding":
        encoding = np.eye(4)[:, [1, 2, 3]]
    elif fault == "leakage":
        supplied_library.cz3.u(np.pi, 0, np.pi, 0)
    else:
        supplied_library.cz3.u(0, 0, np.pi, 0)

    row = _run(gate_library=supplied_library, basis_matrix=encoding)

    assert row["status"] == "gate_validation_failed"
    assert not row["success"] and row["successful_trials"] == 0
    if fault == "encoding":
        assert row["f3_E_norm"] > 1e-10
    else:
        assert row["E_norm"] > 1e-5
        assert row["N_2q"] == sum(item.operation.num_qubits == 2 for item in supplied_library.cz3.data)
        if fault == "leakage":
            assert row["L_norm"] > 1e-5
        else:
            assert row["L_norm"] < 1e-10


@pytest.mark.parametrize("fault", ["f3_width", "cz3_width", "classical_bits", "measurement", "opaque_gate", "renamed_opaque_gate", "cz3_cx", "f3_budget", "nonfinite"])
def test_supplied_structure_and_finite_metrics_are_required(
    fault, supplied_library, monkeypatch,
):
    monkeypatch.setattr(benchmark, "optimized_gate_library", _forbid_synthesis)
    monkeypatch.setattr(benchmark, "_transpile_one_trial", lambda *a, **k: pytest.fail("Invalid gates reached transpilation"))
    if fault == "f3_width":
        supplied_library.f3 = QuantumCircuit(1)
    elif fault == "cz3_width":
        supplied_library.cz3 = QuantumCircuit(3)
    elif fault in {"classical_bits", "measurement"}:
        circuit = QuantumCircuit(2, 1)
        circuit.compose(supplied_library.f3, inplace=True)
        if fault == "measurement":
            circuit.measure(0, 0)
        supplied_library.f3 = circuit
    elif fault == "opaque_gate":
        supplied_library.f3.unitary(np.eye(4), [0, 1])
    elif fault == "renamed_opaque_gate":
        supplied_library.f3.unitary(np.eye(2), [0])
        supplied_library.f3.data[-1].operation.name = "u"
    elif fault == "cz3_cx":
        supplied_library.cz3.cx(0, 1)
        supplied_library.cz3.cx(0, 1)
    elif fault == "f3_budget":
        supplied_library.f3.cz(0, 1)
        supplied_library.f3.cz(0, 1)
    else:
        supplied_library.f3.u(float("nan"), 0, 0, 0)

    row = _run(gate_library=supplied_library)

    assert row["status"] == "gate_validation_failed"
    assert not row["success"] and row["successful_trials"] == 0
    if fault == "f3_budget":
        assert row["f3_N_2q"] == 4
        assert row["f3_E_norm"] < 1e-10


def test_custom_seeds_drive_real_trials_and_override_run_count(supplied_library, monkeypatch):
    original = benchmark._transpile_one_trial
    seen = []

    def record(circuit, **kwargs):
        seen.append(kwargs['trial'])
        return original(circuit, **kwargs)

    monkeypatch.setattr(benchmark, '_transpile_one_trial', record)
    row = _run(
        gate_library=supplied_library,
        transpiler_seeds=(seed for seed in [19, 7]),
        n_transpile_runs=999,
    )

    assert row['success'], row['error_message']
    assert seen == [19, 7]
    assert row['n_transpile_runs'] == row['successful_trials'] == 2
    assert row['transpiler_seed'] in seen
    assert row['iqm_transpiler_seed'] == row['transpiler_seed']
    assert row['fidelity'] >= 1 - 1e-10


@pytest.mark.parametrize('seeds', [[], [1, 1], [True], [-1], [2**32], [1.5], ['1'], 1])
def test_invalid_custom_seeds_rejected_before_synthesis(seeds, monkeypatch):
    monkeypatch.setattr(benchmark, 'optimized_gate_library', _forbid_synthesis)

    with pytest.raises(ValueError, match='transpiler_seeds'):
        _run(transpiler_seeds=seeds)


@pytest.mark.parametrize('rank_by_cz, expected_seed', [(False, 7), (True, 19)])
def test_cz_ranking_prioritizes_entanglers_over_depth(
    rank_by_cz, expected_seed, supplied_library, monkeypatch,
):
    original = benchmark._transpile_one_trial
    compiled = None

    def identity_variants(circuit, **kwargs):
        nonlocal compiled
        if compiled is None:
            compiled = original(circuit, **{**kwargs, 'trial': 0})
        result = compiled.copy()
        if kwargs['trial'] == 19:
            for _ in range(compiled.depth() + 20):
                result.u(0, 0, 0, 0)
        else:
            result.cz(0, 1)
            result.cz(0, 1)
        return result

    monkeypatch.setattr(benchmark, '_transpile_one_trial', identity_variants)
    row = _run(
        gate_library=supplied_library, transpiler_seeds=[19, 7], rank_by_cz=rank_by_cz,
    )

    assert row['success'], row['error_message']
    assert row['transpiler_seed'] == expected_seed
    assert row['fidelity'] >= 1 - 1e-10


def test_cz_ranking_breaks_equal_circuit_ties_by_actual_seed(supplied_library, monkeypatch):
    original = benchmark._transpile_one_trial
    compiled = None

    def identical(circuit, **kwargs):
        nonlocal compiled
        if compiled is None:
            compiled = original(circuit, **{**kwargs, 'trial': 0})
        return compiled.copy()

    monkeypatch.setattr(benchmark, '_transpile_one_trial', identical)
    row = _run(gate_library=supplied_library, transpiler_seeds=[19, 7], rank_by_cz=True)

    assert row['success']
    assert row['transpiler_seed'] == 7
    assert row['fidelity'] >= 1 - 1e-10


@pytest.mark.parametrize("injected", [False, True])
@pytest.mark.parametrize("explicit_map", [False, True])
def test_coupling_map_defaults_depend_on_supplied_library(
    injected, explicit_map, supplied_library, monkeypatch,
):
    legacy_map = [[0, 1], [1, 2], [2, 3]]
    chosen_map = [[0, 1], [0, 2], [0, 3]] if explicit_map else None
    monkeypatch.setattr(benchmark, "COUPLING_MAP", legacy_map)
    original = benchmark._transpile_one_trial
    seen = []

    def record(circuit, **kwargs):
        seen.append(kwargs["coupling_map"])
        return original(circuit, **kwargs)

    monkeypatch.setattr(benchmark, "_transpile_one_trial", record)
    row = _run(
        gate_library=supplied_library if injected else None,
        coupling_map=chosen_map,
    )

    assert row["success"], row["error_message"]
    expected = chosen_map if explicit_map or injected else legacy_map
    assert len(seen) == 1 and seen[0] is expected
    assert row["fidelity"] >= 1 - 1e-10


@pytest.mark.parametrize("explicit_options", [False, True])
def test_manual_transpilation_forwards_explicit_layout_and_routing(
    explicit_options, monkeypatch,
):
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    original = benchmark.transpile
    seen = []
    options = (
        {"initial_layout": (1, 0), "layout_method": "trivial", "routing_method": "none"}
        if explicit_options else {}
    )

    def record(*args, **kwargs):
        seen.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(benchmark, "transpile", record)
    compiled = benchmark._transpile_one_trial(
        circuit, trial=7, basis_gates=["u", "cz"],
        coupling_map=[[0, 1], [1, 0]], **options,
    )

    assert len(seen) == 1
    for name in ("initial_layout", "layout_method", "routing_method"):
        if explicit_options:
            assert seen[0][name] == options[name]
        else:
            assert name not in seen[0]
    if explicit_options:
        assert compiled.layout.initial_index_layout() == [1, 0]
    fidelity, _ = benchmark._safe_fidelity(circuit, compiled, max_qubits=2)
    assert fidelity >= 1 - 1e-10
