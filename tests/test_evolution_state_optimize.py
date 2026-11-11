"""Check bounded fitting for a full encoded graph state."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from qiskit import QuantumCircuit, qpy
from qiskit.quantum_info import Operator

from qudits_on_qubits.evolution.state_optimize import fit_state
from qudits_on_qubits.evolution.state_workloads import build_workload, evaluate_workload

REFERENCE = (Path(__file__).resolve().parents[1] / "experiment_inputs" / "iqm_randomized_bell"
             / "canonical_optimized_20260909")


@pytest.fixture(scope="module")
def ghz():
    with (REFERENCE / "CZ3_W.qpy").open("rb") as stream:
        cz3, = qpy.load(stream)
    encoding = np.eye(4, 3)
    return build_workload("ghz3", encoding, cz3), encoding


def test_saved_ghz_is_preserved_with_no_fit(ghz):
    circuit, encoding = ghz
    result = fit_state("ghz3", circuit, encoding, max_parameters=0, tolerance=1e-5)
    assert result.metrics["is_valid"]
    assert result.nfev == 0
    assert result.evaluations == 1
    np.testing.assert_allclose(Operator(result.circuit).data, Operator(circuit).data, atol=1e-14)
    assert result.metrics["cz_count"] == 15


def test_fit_is_bounded_and_checks_the_result(ghz):
    circuit, encoding = ghz
    initial = np.array(circuit.data[0].operation.params, dtype=float)
    initial[0] += 0.15
    baseline = circuit.copy()
    gate = baseline.data[0].operation.copy()
    gate.params = list(initial)
    baseline.data[0] = baseline.data[0].replace(operation=gate)
    start = evaluate_workload("ghz3", baseline, encoding)["error_norm"]
    result = fit_state("ghz3", circuit, encoding, max_nfev=12, max_parameters=3,
                       initial_parameters=initial, tolerance=1e-5)
    assert result.nfev <= 12
    assert result.evaluations <= 12
    assert result.jacobian_evaluations <= result.evaluations
    assert result.active_parameters == (0, 1, 2)
    assert result.metrics["error_norm"] < start
    independent = evaluate_workload("ghz3", result.circuit, encoding, tolerance=1e-5)
    assert independent["error_norm"] == pytest.approx(result.metrics["error_norm"], abs=1e-15)
    assert independent["cz_count"] == 15


def test_analytic_jacobian_matches_finite_difference(ghz, monkeypatch):
    circuit, encoding = ghz
    import qudits_on_qubits.evolution.state_optimize as module

    def check(residual, x0, *, jac, **kwargs):
        x = x0 + np.linspace(0.011, 0.029, len(x0))
        analytic = jac(x)
        numerical = np.empty_like(analytic)
        step = 1e-6
        for i in range(len(x)):
            delta = np.zeros_like(x)
            delta[i] = step
            numerical[:, i] = (residual(x + delta) - residual(x - delta)) / (2 * step)
        np.testing.assert_allclose(analytic, numerical, atol=2e-9, rtol=0)
        residual(x0)
        return SimpleNamespace(nfev=1, status=1, message="Done")

    monkeypatch.setattr(module, "least_squares", check)
    result = fit_state("ghz3", circuit, encoding, max_parameters=6)
    assert result.jacobian_evaluations == 1


def test_reject_invalid_inputs(ghz):
    circuit, encoding = ghz
    for keyword, value in [("max_nfev", 201), ("max_nfev", 0), ("max_parameters", 25),
                           ("max_parameters", -1), ("tolerance", float("nan"))]:
        with pytest.raises(ValueError):
            fit_state("ghz3", circuit, encoding, **{keyword: value})
    with pytest.raises(ValueError, match="selected size"):
        fit_state("ghz3", circuit, encoding, max_parameters=1, initial_parameters=[np.nan])
    with pytest.raises(ValueError):
        fit_state("ghz3", circuit, np.zeros((4, 3)))
    with pytest.raises(ValueError, match="6 qubits"):
        fit_state("ghz3", QuantumCircuit(4), encoding)
    invalid = circuit.copy()
    invalid.cx(0, 1)
    with pytest.raises(ValueError, match="only U, U3 and CZ"):
        fit_state("ghz3", invalid, encoding)


def test_ame_reference_with_global_phase_and_one_solver_step():
    with (REFERENCE / "CZ3_W.qpy").open("rb") as stream:
        cz3, = qpy.load(stream)
    encoding = np.eye(4, 3)
    circuit = build_workload("ame43", encoding, cz3)
    circuit.global_phase = 1.2
    result = fit_state("ame43", circuit, encoding, max_parameters=3, max_nfev=1, tolerance=1e-5)
    assert result.metrics["is_valid"]
    assert result.metrics["cz_count"] == 28
    assert result.circuit.global_phase == pytest.approx(1.2)
    assert result.nfev == 1
    assert result.evaluations == 1
    assert evaluate_workload("ame43", result.circuit, encoding, tolerance=1e-5)["is_valid"]
