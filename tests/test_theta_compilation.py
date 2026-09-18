from pathlib import Path

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Gate
from qiskit.circuit.library import U3Gate
from qiskit.quantum_info import Operator

from qudits_on_qubits.benchmarks.theta_continuation import compilation
from qudits_on_qubits.benchmarks.theta_continuation.cz3_template import load_cz3_template


BASELINE = Path(__file__).resolve().parents[1] / "experiment_inputs/iqm_randomized_bell/canonical_optimized_20260909/CZ3_W.qpy"


@pytest.fixture
def baseline():
    return load_cz3_template(BASELINE).original_circuit


def test_real_pinned_compilation_preserves_operator_and_counts(baseline):
    result = compilation.compile_cz3_candidate(baseline, np.eye(4, 3), seeds=(0, 1))
    assert result.metrics["N_CZ"] == result.circuit.count_ops()["cz"]
    assert result.metrics["N_CZ"] <= 6
    assert result.metrics["full_error_norm"] < 1e-10
    assert result.metrics["E_norm"] < 1e-5
    assert len(result.trials) == 2
    assert result.metrics["compilation_method"] == "transpile"
    assert baseline.count_ops() == {"cz": 6, "u3": 11}


def test_rank_prefers_cz_count_before_depth_and_seed(monkeypatch, baseline):
    base = transpile(baseline, basis_gates=["u", "cz"], optimization_level=0)
    costly = base.copy()
    costly.cz(0, 1)
    costly.cz(0, 1)
    calls = []

    def compile(source, **kwargs):
        calls.append(kwargs)
        return costly.copy() if kwargs["seed_transpiler"] == 0 else base.copy()

    monkeypatch.setattr(compilation, "transpile", compile)
    result = compilation.compile_cz3_candidate(baseline, np.eye(4, 3), seeds=(0, 8, 3))
    assert result.metrics["seed"] == 3
    assert result.metrics["N_CZ"] == 6
    assert all(call["routing_method"] == "none" and call["approximation_degree"] == 1.0 for call in calls)


def test_rejects_code_correct_but_changed_full_operator(baseline):
    changed = baseline.copy()
    changed.mcp(0.4, [0, 1, 2], 3)
    changed = transpile(changed, basis_gates=["u", "cz"], optimization_level=0)
    with pytest.raises(ValueError, match="full.operator"):
        compilation.validate_cz3_candidate(changed, np.eye(4, 3), source=baseline, compiled=True)


@pytest.mark.parametrize("kind", ["wrong_code", "classical", "opaque", "wrong_size", "nan"])
def test_rejects_invalid_physical_circuits(baseline, kind):
    circuit = baseline.copy()
    if kind == "wrong_code":
        circuit.u(0.3, 0, 0, 0)
    elif kind == "classical":
        circuit = QuantumCircuit(4, 1).compose(circuit)
    elif kind == "opaque":
        circuit.append(Gate("u", 1, [0.0, 0.0, 0.0]), [0])
    elif kind == "wrong_size":
        circuit = QuantumCircuit(3)
    else:
        circuit.u(float("nan"), 0, 0, 0)
    with pytest.raises(ValueError):
        compilation.compile_cz3_candidate(circuit, np.eye(4, 3), seeds=(0,))


def test_bad_seed_recorded_and_valid_seed_retained(monkeypatch, baseline):
    real = compilation.transpile

    def compile(circuit, **kwargs):
        if kwargs["seed_transpiler"] == 0:
            raise ValueError("seed failed")
        return real(circuit, **kwargs)

    monkeypatch.setattr(compilation, "transpile", compile)
    result = compilation.compile_cz3_candidate(baseline, np.eye(4, 3), seeds=(0, 1))
    assert result.trials[0]["error"] == "seed failed"
    assert result.metrics["seed"] == 1


def test_tiny_rotation_loss_uses_exact_conversion_and_preserves_global_phase(baseline):
    source = baseline.copy()
    source.append(U3Gate(1e-7, 0, 0), [0])
    source.global_phase = 0.371
    source.metadata = {"origin": "tiny_rotation_regression"}
    original = Operator(source).data.copy()

    result = compilation.compile_cz3_candidate(
        source, np.eye(4, 3), tolerance=5e-4, seeds=(2, 0, 1),
    )

    assert result.metrics["compilation_method"] == "exact_u3_to_u"
    assert result.metrics["seed"] == 0
    assert result.metrics["full_error_norm"] < 1e-10
    assert result.metrics["N_CZ"] == 6
    assert result.metrics["N_1q"] == 12
    assert len(result.trials) == 4
    for trial in result.trials[:-1]:
        assert trial["compilation_method"] == "transpile"
        assert not trial["valid"]
        assert trial["metrics"]["full_error_norm"] > 1e-10
    assert result.trials[-1]["compilation_method"] == "exact_u3_to_u"
    assert result.trials[-1]["valid"]
    assert set(result.circuit.count_ops()) == {"u", "cz"}
    assert result.circuit.global_phase == source.global_phase
    assert result.circuit.metadata["origin"] == "tiny_rotation_regression"
    np.testing.assert_allclose(Operator(result.circuit).data, original, atol=1e-14, rtol=0)
    np.testing.assert_array_equal(Operator(source).data, original)
    assert source.count_ops()["u3"] == 12


def test_exact_conversion_after_transpiler_errors_retains_existing_u(monkeypatch, baseline):
    source = transpile(baseline, basis_gates=["u", "cz"], optimization_level=0)

    def fail(*args, **kwargs):
        raise RuntimeError("transpilation failed")

    monkeypatch.setattr(compilation, "transpile", fail)
    result = compilation.compile_cz3_candidate(source, np.eye(4, 3), seeds=(4,))
    assert result.metrics["compilation_method"] == "exact_u3_to_u"
    assert result.metrics["seed"] == 4
    assert result.trials[0]["error"] == "transpilation failed"
    assert result.circuit.count_ops() == source.count_ops()
    np.testing.assert_allclose(Operator(result.circuit).data, Operator(source).data, atol=1e-14, rtol=0)


def test_exact_conversion_still_rejects_operator_change_at_looser_code_tolerance(monkeypatch, baseline):
    changed = transpile(baseline, basis_gates=["u", "cz"], optimization_level=0)
    changed.u(1e-7, 0, 0, 0)

    def fail(*args, **kwargs):
        raise RuntimeError("transpilation failed")

    monkeypatch.setattr(compilation, "transpile", fail)
    monkeypatch.setattr(compilation, "_exact_u_cz_copy", lambda source: changed.copy(), raising=False)
    with pytest.raises(compilation.GateSynthesisError, match="No exact U/CZ") as caught:
        compilation.compile_cz3_candidate(baseline, np.eye(4, 3), tolerance=5e-4, seeds=(0,))
    last = caught.value.metrics["trials"][-1]
    assert last["compilation_method"] == "exact_u3_to_u"
    assert not last["valid"]
    assert last["metrics"]["E_norm"] < 5e-4
    assert last["metrics"]["full_error_norm"] > 1e-10
