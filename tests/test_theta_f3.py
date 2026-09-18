from __future__ import annotations

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator

from qudits_on_qubits.benchmarks.direct_basis import optimized_gates
from qudits_on_qubits.benchmarks.direct_basis.math_utils import qutrit_fourier
from qudits_on_qubits.benchmarks.theta_continuation import f3


def _embedding(theta):
    return np.array([
        [np.cos(theta), 0, 0], [0, 1, 0], [0, 0, 1],
        [-np.sin(theta), 0, 0],
    ], dtype=complex)


def _align(actual, target):
    return actual * np.exp(1j * np.angle(np.vdot(actual, target)))


@pytest.mark.parametrize("theta", [0, np.pi / 160, np.pi / 8, np.pi / 4])
def test_real_f3_phase_changes_only_complement_and_uses_at_most_two_cz(theta):
    encoding = _embedding(theta)
    leakage = np.array([np.sin(theta), 0, 0, np.cos(theta)])
    result = f3.synthesize_theta_f3(encoding)
    assert isinstance(result, f3.F3Result)
    assert np.isfinite(result.synthesis_seconds) and result.synthesis_seconds >= 0
    assert 0 <= result.alpha < 2 * np.pi
    coded = encoding @ qutrit_fourier() @ encoding.conj().T
    complement = np.outer(leakage, leakage.conj())
    aligned = []
    for circuit, metrics, alpha in (
        (result.optimal, result.optimal_metrics, result.alpha),
        (result.alpha_zero, result.alpha_zero_metrics, 0.0),
    ):
        target = coded + np.exp(1j * alpha) * complement
        actual = _align(Operator(circuit).data, target)
        aligned.append(actual)
        np.testing.assert_allclose(actual, target, atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual @ encoding, encoding @ qutrit_fourier(), atol=1e-10, rtol=0)
        np.testing.assert_allclose(actual @ leakage, np.exp(1j * alpha) * leakage, atol=1e-10, rtol=0)
        assert set(circuit.count_ops()) <= {"u", "cz"}
        assert metrics["N_CZ"] == circuit.count_ops().get("cz", 0)
        assert metrics["N_2q"] == metrics["N_CZ"]
        assert metrics["N_1q"] == sum(item.operation.num_qubits == 1 for item in circuit.data)
        assert metrics["depth"] == circuit.depth()
        assert metrics["seed"] in (0, 1, 2)
        assert metrics["leakage_phase"] == alpha
        assert metrics["E_norm"] <= 1e-10 and metrics["L_norm"] <= 1e-10
        assert np.isfinite(metrics["raw_error_norm"])
        assert metrics["synthesis_method"]
        assert all(circuit.metadata[key] == metrics[key] for key in metrics)
    assert result.optimal_metrics["N_CZ"] <= 2
    np.testing.assert_allclose(
        aligned[0] - aligned[1], (np.exp(1j * result.alpha) - 1) * complement,
        atol=1e-10, rtol=0,
    )


def test_real_transpiler_seed_selection_matches_cost_ranking():
    encoding = _embedding(np.pi / 8)
    seeds = (7, 2, 5)
    result = f3.synthesize_theta_f3(encoding, transpiler_seeds=seeds)
    sources = [optimized_gates.synthesize_f3(encoding), QuantumCircuit(2)]
    sources[1].unitary(
        encoding @ qutrit_fourier() @ encoding.conj().T
        + np.eye(4) - encoding @ encoding.conj().T,
        [0, 1],
    )
    for source, metrics in zip(sources, (result.optimal_metrics, result.alpha_zero_metrics)):
        ranks = []
        for seed in seeds:
            candidate = transpile(
                source, basis_gates=["u", "cz"], optimization_level=3,
                seed_transpiler=seed, approximation_degree=1.0,
                coupling_map=None, routing_method="none",
            )
            ranks.append((candidate.count_ops().get("cz", 0), candidate.depth(),
                          sum(item.operation.num_qubits == 1 for item in candidate.data), seed))
        assert tuple(metrics[key] for key in ("N_CZ", "depth", "N_1q", "seed")) == min(ranks)


@pytest.mark.parametrize("encoding", [np.zeros((4, 3)), np.eye(4), np.eye(4, 2),
                                          np.full((4, 3), np.nan), np.full((4, 3), np.inf)])
def test_invalid_embedding_rejected(encoding):
    with pytest.raises(ValueError):
        f3.synthesize_theta_f3(encoding)


@pytest.mark.parametrize("options", [
    {"tolerance": value} for value in (True, 0, -1, np.nan, np.inf, 1j, "1e-10")
] + [
    {"transpiler_seeds": value} for value in ((), (0, 0), (-1,), (2**32,), (True,), (1.5,), ("0",), None)
] + [
    {"optimization_level": value} for value in (True, -1, 4, 3.0, "3")
])
def test_invalid_synthesis_configuration_rejected(options):
    with pytest.raises(ValueError):
        f3.synthesize_theta_f3(_embedding(0), **options)


@pytest.mark.parametrize("phase", [None, True, np.nan, np.inf, 1j, "0"])
def test_invalid_synthesized_phase_rejected(monkeypatch, phase):
    circuit = optimized_gates.synthesize_f3(_embedding(0))
    circuit.metadata["leakage_phase"] = phase
    monkeypatch.setattr(optimized_gates, "synthesize_f3", lambda _: circuit)
    with pytest.raises(optimized_gates.GateSynthesisError, match="phase"):
        f3.synthesize_theta_f3(_embedding(0))


@pytest.mark.parametrize("failure", ["phase", "logical", "leakage"])
def test_bad_source_synthesis_rejected(monkeypatch, failure):
    encoding = _embedding(0)
    circuit = optimized_gates.synthesize_f3(encoding)
    if failure == "phase":
        circuit.metadata["leakage_phase"] += 0.25
    elif failure == "logical":
        circuit.z(0)
    else:
        circuit.x(0)
    monkeypatch.setattr(optimized_gates, "synthesize_f3", lambda _: circuit)
    with pytest.raises(optimized_gates.GateSynthesisError):
        f3.synthesize_theta_f3(encoding)


@pytest.mark.parametrize("failure", ["extra_cz", "complement", "logical", "opaque"])
def test_bad_transpilation_rejected_from_actual_operator_and_gate_count(monkeypatch, failure):
    encoding = _embedding(0)
    source = optimized_gates.synthesize_f3(encoding)
    circuit = transpile(source, basis_gates=["u", "cz"], optimization_level=3)
    if failure == "extra_cz":
        circuit.cz(0, 1)
        circuit.cz(0, 1)
        circuit.metadata["N_2q"] = 0
    elif failure == "complement":
        # CZ is identity on the canonical code but changes its complement.
        circuit.cz(0, 1)
    elif failure == "logical":
        circuit.u(0, 0, np.pi, 0)
    else:
        circuit = QuantumCircuit(2)
        circuit.unitary(Operator(source), [0, 1])
    monkeypatch.setattr(f3, "transpile", lambda *args, **kwargs: circuit)
    with pytest.raises(optimized_gates.GateSynthesisError):
        f3.synthesize_theta_f3(encoding)


def test_nonfinite_recomputed_metric_rejected(monkeypatch):
    real_validation = optimized_gates.validate_code_space_gate

    def invalid_validation(*args):
        validation = real_validation(*args)
        return optimized_gates.GateValidation(
            validation.E_norm, validation.L_norm, validation.N_2q, float("nan"),
        )

    monkeypatch.setattr(optimized_gates, "validate_code_space_gate", invalid_validation)
    with pytest.raises(optimized_gates.GateSynthesisError):
        f3.synthesize_theta_f3(_embedding(0))


def test_f3_does_not_launch_bqskit(monkeypatch):
    def forbidden():
        pytest.fail("F3 must not launch BQSKit")

    monkeypatch.setattr(optimized_gates, "_create_bqskit_compiler", forbidden)
    assert f3.synthesize_theta_f3(_embedding(0)).optimal_metrics["N_CZ"] <= 2


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_valid_optimization_levels_use_exact_synthesis(level):
    result = f3.synthesize_theta_f3(_embedding(np.pi / 160), optimization_level=level)
    assert result.optimal_metrics["N_CZ"] <= 2
    assert result.alpha_zero_metrics["E_norm"] <= 1e-10


@pytest.mark.parametrize("bad_result", [None, 17, QuantumCircuit(1)])
def test_malformed_source_synthesis_raises_gate_synthesis_error(monkeypatch, bad_result):
    monkeypatch.setattr(optimized_gates, "synthesize_f3", lambda _: bad_result)
    with pytest.raises(optimized_gates.GateSynthesisError):
        f3.synthesize_theta_f3(_embedding(0))


def test_failed_seeds_are_discarded_and_valid_seed_selected(monkeypatch):
    from qiskit.exceptions import QiskitError

    real_transpile = f3.transpile

    def failing_transpile(source, **options):
        if options["seed_transpiler"] == 0:
            raise QiskitError("Injected transpiler failure")
        if options["seed_transpiler"] == 1:
            return QuantumCircuit(2)
        return real_transpile(source, **options)

    monkeypatch.setattr(f3, "transpile", failing_transpile)
    result = f3.synthesize_theta_f3(_embedding(0))
    assert result.optimal_metrics["seed"] == 2
    assert result.alpha_zero_metrics["seed"] == 2


def test_invalid_alpha_zero_control_rejects_entire_result(monkeypatch):
    real_transpile = f3.transpile

    def failing_control(source, **options):
        if source.name == "F3_theta_alpha_zero":
            return QuantumCircuit(2)
        return real_transpile(source, **options)

    monkeypatch.setattr(f3, "transpile", failing_control)
    with pytest.raises(optimized_gates.GateSynthesisError):
        f3.synthesize_theta_f3(_embedding(0))
