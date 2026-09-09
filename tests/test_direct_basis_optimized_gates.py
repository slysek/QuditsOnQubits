from __future__ import annotations

import json
import os
from itertools import combinations, permutations
from types import SimpleNamespace

import numpy as np
import pytest
from qiskit import QuantumCircuit, qpy
from qiskit.quantum_info import Operator, Statevector, state_fidelity

from qudits_on_qubits.benchmarks.direct_basis import optimized_gates as gates
from qudits_on_qubits.benchmarks.direct_basis.benchmark import benchmark_direct_basis, benchmark_direct_basis_candidates
from qudits_on_qubits.benchmarks.direct_basis.candidates import DirectBasisCandidate
from qudits_on_qubits.benchmarks.direct_basis.circuits import (
    build_direct_basis_graph_state_circuit, build_optimized_direct_basis_graph_state_circuit,
)
from qudits_on_qubits.benchmarks.direct_basis.math_utils import qutrit_cz, qutrit_fourier


def _dense_basis(seed=13):
    rng = np.random.default_rng(seed)
    return np.linalg.qr(rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3)))[0]


@pytest.mark.parametrize("support", list(combinations(range(4), 3)))
@pytest.mark.parametrize("permutation", list(permutations(range(3))))
def test_monomial_f3_is_correct_with_at_most_two_cnot(support, permutation):
    embedding = np.eye(4)[:, support] @ np.diag(np.exp(1j * np.array([.2, -.7, 1.3]))) @ np.eye(3)[list(permutation)]
    circuit = gates.synthesize_f3(embedding)
    assert circuit.count_ops().get("cx", 0) <= 2
    assert circuit.metadata["synthesis_method"] == "analytic_monomial"
    np.testing.assert_allclose(Operator(circuit).data @ embedding, embedding @ qutrit_fourier(), atol=1e-10)


@pytest.mark.parametrize("support", list(combinations(range(4), 3)))
@pytest.mark.parametrize("seed", range(8))
def test_dense_bs_w_f3_is_correct_with_at_most_two_cnot(support, seed):
    embedding = np.eye(4)[:, support] @ _dense_basis(seed)
    circuit = gates.synthesize_f3(embedding)
    assert circuit.metadata["synthesis_method"] == "numerical_two_cnot_invariant"
    assert circuit.count_ops().get("cx", 0) <= 2
    assert circuit.metadata["E_norm"] < 1e-10
    assert circuit.metadata["L_norm"] < 1e-10
    np.testing.assert_allclose(Operator(circuit).data @ embedding, embedding @ qutrit_fourier(), atol=1e-10)


def test_dense_isometry_and_diagonal_fourier_degenerate_invariant():
    rng = np.random.default_rng(14)
    isometry = np.linalg.qr(rng.normal(size=(4, 3)) + 1j * rng.normal(size=(4, 3)))[0]
    # Eigenbasis makes W F W^dag diagonal; the monomial formula's numerator is zero.
    from scipy.linalg import schur
    _, eigenvectors = schur(qutrit_fourier(), output="complex")
    for embedding in (isometry, np.eye(4, 3) @ eigenvectors.conj().T):
        circuit = gates.synthesize_f3(embedding)
        assert circuit.metadata["N_2q"] <= 2
        assert circuit.metadata["E_norm"] < 1e-10


def test_validation_preserves_relative_phases_and_detects_leakage():
    embedding = np.kron(np.eye(4, 3), np.eye(4, 3))
    correct = np.eye(16, dtype=complex) + embedding @ (qutrit_cz() - np.eye(9)) @ embedding.conj().T
    circuit = QuantumCircuit(4)
    circuit.unitary(correct * np.exp(.7j), range(4))
    validation = gates.validate_code_space_gate(circuit, embedding, qutrit_cz())
    assert validation.E_norm < 1e-12
    assert validation.raw_error_norm > 1
    # Nine independent phase-insensitive state comparisons would miss this error.
    wrong_phase = correct.copy()
    wrong_phase[:, 0] *= -1
    wrong = QuantumCircuit(4)
    wrong.unitary(wrong_phase, range(4))
    assert gates.validate_code_space_gate(wrong, embedding, qutrit_cz()).E_norm > 1
    leaking = circuit.copy()
    leaking.x(0)
    assert gates.validate_code_space_gate(leaking, embedding, qutrit_cz()).L_norm > 1


@pytest.mark.parametrize("state", ["two_qutrit", "ghz3", "ame43"])
def test_graph_uses_explicit_gates_and_encoded_zero(exact_cz3_synthesis, state):
    # Physical |00> is outside the code space; catches missing encoded-zero prep.
    embedding = np.eye(4)[:, [1, 2, 3]] @ _dense_basis()
    library = gates.optimized_gate_library(embedding)
    circuit = build_optimized_direct_basis_graph_state_circuit(state, embedding, gate_library=library)
    reference = build_direct_basis_graph_state_circuit(state, embedding)
    assert state_fidelity(Statevector(circuit), Statevector(reference)) > 1 - 1e-10
    assert all(item.operation.num_qubits <= 2 for item in circuit.data)
    assert not any(item.operation.name in {"unitary", "CZ_W", "CZ3_W", "F3_W"} for item in circuit.data)
    assert circuit.metadata["N_2q"] == library.cz3.metadata["N_2q"]


def test_cache_validates_reused_gates_and_does_not_alias_encodings(exact_cz3_synthesis, tmp_path):
    cache = tmp_path / "cache"
    first = gates.optimized_gate_library(np.eye(3), cache_dir=cache)
    repeated = gates.optimized_gate_library(np.eye(4, 3), cache_dir=cache)
    assert not first.cache_hit and repeated.cache_hit
    assert len(exact_cz3_synthesis) == 1
    gates.optimized_gate_library(qutrit_fourier(), cache_dir=cache)
    assert len(exact_cz3_synthesis) == 2
    first.f3.x(0)
    assert gates.optimized_gate_library(np.eye(3), cache_dir=cache).f3.metadata["E_norm"] < 1e-10
    # Never trust cached metrics when the actual circuit was modified.
    path = next(p for p in cache.glob("*/F3_W.qpy") if json.loads((p.parent / "synthesis.json").read_text())["F3"]["synthesis_method"] == "analytic_monomial")
    with path.open("wb") as handle:
        qpy.dump(first.f3, handle)
    repaired = gates.optimized_gate_library(np.eye(3), cache_dir=cache)
    assert not repaired.cache_hit
    assert repaired.f3.metadata["E_norm"] < 1e-10
    assert len(exact_cz3_synthesis) == 3


def test_benchmark_exports_and_reuses_new_gates_for_thresholds(exact_cz3_synthesis, tmp_path):
    frame, _ = benchmark_direct_basis_candidates(
        state_name="two_qutrit", candidates=[DirectBasisCandidate("I", "test", np.eye(3))],
        n_transpile_runs=1, basis_gates=["u", "cz"], coupling_map=[[0, 1], [1, 2], [2, 3]],
        approximation_degrees=[.99], quantum_circuits_dir=str(tmp_path / "exports"),
        output_csv=str(tmp_path / "results.csv"),
    )
    assert frame.success.all()
    assert len(exact_cz3_synthesis) == 1
    assert (frame.E_norm <= 1e-5).all() and (frame.L_norm <= 1e-5).all()
    assert (frame.f3_N_2q <= 2).all()
    for row in frame.to_dict("records"):
        for key, count in (("f3_w_qpy", row["f3_N_2q"]), ("cz3_w_qpy", row["N_2q"])):
            with open(row[key], "rb") as handle:
                circuit = qpy.load(handle)[0]
            assert sum(item.operation.num_qubits == 2 for item in circuit.data) == count
            assert "unitary" not in circuit.count_ops()


def test_failed_cz3_is_recorded_and_not_transpiled(monkeypatch, exact_cz3_synthesis):
    monkeypatch.setattr(gates, "_compile_cz3", lambda _: QuantumCircuit(4))
    row = benchmark_direct_basis(
        state_name="two_qutrit", basis_matrix=np.eye(3), basis_candidate_name="I",
        basis_candidate_type="test", n_transpile_runs=1,
    )
    assert row["status"] == "gate_validation_failed"
    assert not row["success"] and row["successful_trials"] == 0
    assert row["E_norm"] > 1e-5
    assert row["N_2q"] == 0


@pytest.mark.parametrize("provider", ["iqm", "piast"])
@pytest.mark.parametrize("invalid", [False, True])
def test_harness_records_gate_metrics_and_rejects_bad_gates(provider, invalid, monkeypatch, exact_cz3_synthesis):
    from qudits_on_qubits.benchmarks.direct_basis.iqm_transpiler_harness import (
        IqmTranspilerHarnessConfig, run_iqm_transpiler_harness,
    )
    from qudits_on_qubits.benchmarks.direct_basis.piast_transpiler_harness import (
        PiastTranspilerHarnessConfig, run_piast_transpiler_harness,
    )
    if invalid:
        monkeypatch.setattr(gates, "_compile_cz3", lambda _: QuantumCircuit(4))
    calls = []

    def runner(strategy_name, circuit, *, seed_transpiler, **kwargs):
        calls.append(circuit)
        compiled = QuantumCircuit(4)
        if provider == "iqm":
            compiled.cz(0, 1)
        else:
            compiled.rxx(.2, 0, 1)
        return SimpleNamespace(
            strategy_name=strategy_name, seed_transpiler=seed_transpiler,
            success=True, circuit=compiled, compile_time_seconds=.1,
        )

    options = dict(
        state_name="two_qutrit", n_qutrits=2, backend=object(),
        candidates=[DirectBasisCandidate("I", "test", np.eye(3))],
        strategy_names=("test",), n_transpile_runs=2,
    )
    if provider == "iqm":
        config = IqmTranspilerHarnessConfig(**options, iqm_backend_name="garnet", iqm_use_metrics=False)
        run = run_iqm_transpiler_harness
    else:
        config = PiastTranspilerHarnessConfig(**options)
        run = run_piast_transpiler_harness
    frame, best, summary = run(config, strategy_runner=runner)
    if invalid:
        assert not calls
        assert frame.iloc[0]["status"] == best.iloc[0]["status"] == "gate_validation_failed"
        assert frame.iloc[0]["E_norm"] > 1e-5
        assert summary["gate_validation_failed_count"] == 1
    else:
        assert len(calls) == 2
        assert frame.success.all()
        assert (frame.f3_N_2q <= 2).all()
        assert (frame.E_norm <= 1e-5).all()
        assert (frame.L_norm <= 1e-5).all()
        assert frame.N_2q.tolist() == [calls[0].metadata["N_2q"]] * 2
        assert best.iloc[0]["N_2q"] == calls[0].metadata["N_2q"]


def test_f3_cnot_limit_is_an_acceptance_condition(monkeypatch):
    original = gates.TwoQubitBasisDecomposer

    def decomposer(*args, **kwargs):
        real = original(*args, **kwargs)

        def decompose(unitary):
            circuit = real(unitary)
            # An identity pair preserves correctness but violates the gate budget.
            circuit.cx(0, 1)
            circuit.cx(0, 1)
            return circuit
        return decompose

    monkeypatch.setattr(gates, "TwoQubitBasisDecomposer", decomposer)
    with pytest.raises(gates.GateSynthesisError) as error:
        gates.synthesize_f3(np.eye(3))
    assert error.value.metrics["f3_N_2q"] == 4
    assert error.value.metrics["f3_E_norm"] < 1e-10


@pytest.mark.parametrize("metric", ["E_norm", "L_norm"])
def test_cz3_thresholds_are_independent_and_inclusive(metric):
    values = dict(E_norm=0.0, L_norm=0.0, N_2q=0, raw_error_norm=0.0)
    values[metric] = 1e-5
    gates._accept(QuantumCircuit(4), gates.GateValidation(**values), name="CZ3", tolerance=1e-5)
    values[metric] = np.nextafter(1e-5, np.inf)
    with pytest.raises(gates.GateSynthesisError):
        gates._accept(QuantumCircuit(4), gates.GateValidation(**values), name="CZ3", tolerance=1e-5)


def test_bqskit_targets_all_code_columns_and_converts_bit_order(monkeypatch):
    import bqskit
    from bqskit.ir import Circuit
    from bqskit.ir.gates import U3Gate
    embedding = np.eye(4)[:, [0, 2, 3]] @ _dense_basis()
    native = Circuit(4)
    native.append_gate(U3Gate(), 0, [.2, .3, .4])

    def compiler(target, **options):
        b2 = np.kron(embedding, embedding)
        assert len(target) == 9
        for j, (source, output) in enumerate(target.items()):
            np.testing.assert_allclose(np.asarray(source), b2[:, j])
            np.testing.assert_allclose(np.asarray(output), (b2 @ qutrit_cz())[:, j])
        assert options["seed"] == 0
        assert options["synthesis_epsilon"] == 1e-8
        assert options["max_synthesis_size"] == 4
        assert options["optimization_level"] == 2
        assert {gate.name for gate in options["model"].gate_set} == {"U3Gate", "CZGate"}
        return native

    monkeypatch.setattr(bqskit, "compile", compiler)
    converted = gates._compile_cz3(embedding)
    np.testing.assert_allclose(Operator(converted).data, np.asarray(native.get_unitary()), atol=1e-12)


@pytest.mark.skipif(os.environ.get("QOQ_RUN_BQSKIT_TESTS") != "1", reason="Set QOQ_RUN_BQSKIT_TESTS=1 for real BQSKit synthesis")
def test_real_bqskit_cz3_synthesis():
    library = gates.optimized_gate_library(np.eye(4, 3))
    assert library.cz3.metadata["E_norm"] <= 1e-5
    assert library.cz3.metadata["L_norm"] <= 1e-5
    assert library.f3.count_ops().get("cx", 0) <= 2
    assert set(library.cz3.count_ops()) <= {"u", "u3", "cz"}
    for state in ("two_qutrit", "ghz3", "ame43"):
        optimized = build_optimized_direct_basis_graph_state_circuit(state, np.eye(3), gate_library=library)
        reference = build_direct_basis_graph_state_circuit(state, np.eye(3))
        assert state_fidelity(Statevector(optimized), Statevector(reference)) > 1 - 1e-10
