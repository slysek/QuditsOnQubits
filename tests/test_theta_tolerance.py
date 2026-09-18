from types import SimpleNamespace

import numpy as np
import pytest
from qiskit import qpy
from qiskit.quantum_info import Operator

from qudits_on_qubits.benchmarks.direct_basis import benchmark, optimized_gates
from qudits_on_qubits.benchmarks.direct_basis.math_utils import qutrit_cz
from qudits_on_qubits.benchmarks.theta_continuation.models import ThetaBenchmarkConfig


@pytest.fixture
def approximate_cz3():
    with open(ThetaBenchmarkConfig().baseline_qpy, "rb") as handle:
        circuit = qpy.load(handle)[0]
    circuit.u(1e-4, 0, 0, 0)
    validation = optimized_gates.validate_code_space_gate(
        circuit, np.kron(np.eye(4, 3), np.eye(4, 3)), qutrit_cz(),
    )
    assert 1e-5 < validation.E_norm < 5e-4
    assert 1e-5 < validation.L_norm < 5e-4
    return circuit


def _benchmark(**kwargs):
    return benchmark.benchmark_direct_basis(
        state_name="two_qutrit", basis_matrix=np.eye(4, 3),
        basis_candidate_name="threshold-test", basis_candidate_type="test",
        basis_gates=["u", "cz"], coupling_map=None, transpiler_seeds=(0,),
        **kwargs,
    )


def _library(circuit):
    return SimpleNamespace(
        f3=optimized_gates.synthesize_f3(np.eye(4, 3)), cz3=circuit,
        benchmark_metrics=lambda: {"gate_library": "threshold-test", "cz3_tolerance": 100.0},
    )


def test_config_allows_explicit_relaxed_cz3_and_preserves_defaults():
    assert ThetaBenchmarkConfig().cz3_tolerance == 1e-5
    config = ThetaBenchmarkConfig(cz3_tolerance=5e-4)
    assert config.cz3_tolerance == 5e-4
    assert config.f3_tolerance == 1e-10
    assert ThetaBenchmarkConfig.from_dict(config.to_dict()) == config


def test_synthesis_threshold_changes_acceptance_only(approximate_cz3, monkeypatch):
    calls = []
    def compile_cz3(embedding):
        calls.append(embedding.copy())
        return approximate_cz3.copy()
    monkeypatch.setattr(optimized_gates, "_compile_cz3", compile_cz3)
    with pytest.raises(optimized_gates.GateSynthesisError, match="CZ3 failed"):
        optimized_gates.synthesize_cz3(np.eye(4, 3))
    accepted = optimized_gates.synthesize_cz3(np.eye(4, 3), tolerance=5e-4)
    assert len(calls) == 2
    for embedding in calls:
        np.testing.assert_array_equal(embedding, np.eye(4, 3))
    np.testing.assert_array_equal(Operator(accepted).data, Operator(approximate_cz3).data)
    assert accepted.metadata["cz3_tolerance"] == 5e-4
    assert 1e-5 < accepted.metadata["E_norm"] < 5e-4
    assert 1e-5 < accepted.metadata["L_norm"] < 5e-4


def test_supplied_library_requires_explicit_relaxation_ignoring_metadata(approximate_cz3):
    library = _library(approximate_cz3)
    strict = _benchmark(gate_library=library)
    assert strict["status"] == "gate_validation_failed"
    assert strict["successful_trials"] == 0
    relaxed = _benchmark(gate_library=library, cz3_tolerance=5e-4)
    assert relaxed["success"], relaxed["error_message"]
    assert relaxed["cz3_tolerance"] == 5e-4
    assert 1e-5 < relaxed["E_norm"] < 5e-4
    assert 1e-5 < relaxed["L_norm"] < 5e-4
    assert relaxed["fidelity"] > 1 - 1e-7


def test_relaxed_cz3_does_not_relax_f3(approximate_cz3):
    library = _library(approximate_cz3)
    library.f3.u(1e-5, 0, 0, 0)
    row = _benchmark(gate_library=library, cz3_tolerance=5e-4)
    assert row["status"] == "gate_validation_failed"
    assert row["f3_E_norm"] > 1e-10


@pytest.mark.parametrize("entrypoint", ["synthesis", "benchmark", "validator"])
@pytest.mark.parametrize("value", [True, np.bool_(True), 0, -1e-6, np.inf, -np.inf, np.nan, "0.1", 1j, None])
def test_invalid_tolerance_rejected_before_any_work(entrypoint, value, monkeypatch):
    def forbid(*args, **kwargs):
        pytest.fail("invalid tolerance reached synthesis")
    monkeypatch.setattr(optimized_gates, "_compile_cz3", forbid)
    monkeypatch.setattr(benchmark, "optimized_gate_library", forbid)
    with pytest.raises((TypeError, ValueError), match="tolerance"):
        if entrypoint == "synthesis":
            optimized_gates.synthesize_cz3(np.eye(4, 3), tolerance=value)
        elif entrypoint == "benchmark":
            _benchmark(cz3_tolerance=value)
        else:
            benchmark._validate_supplied_gate_library(None, np.eye(4, 3), cz3_tolerance=value)


@pytest.mark.parametrize("entrypoint", ["synthesis", "benchmark", "validator"])
def test_overflowing_tolerance_rejected(entrypoint, monkeypatch):
    def forbid(*args, **kwargs):
        pytest.fail("invalid tolerance reached synthesis")
    monkeypatch.setattr(optimized_gates, "_compile_cz3", forbid)
    monkeypatch.setattr(benchmark, "optimized_gate_library", forbid)
    with pytest.raises(ValueError, match="tolerance"):
        if entrypoint == "synthesis":
            optimized_gates.synthesize_cz3(np.eye(4, 3), tolerance=10**1000)
        elif entrypoint == "benchmark":
            _benchmark(cz3_tolerance=10**1000)
        else:
            benchmark._validate_supplied_gate_library(None, np.eye(4, 3), cz3_tolerance=10**1000)
