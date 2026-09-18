from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from qudits_on_qubits.benchmarks.direct_basis.math_utils import qutrit_cz
from qudits_on_qubits.benchmarks.theta_continuation import cz3_continuation as continuation


def _embedding(theta=0.0):
    matrix = np.eye(4, 3, dtype=complex)
    matrix[:, 0] = [np.cos(theta), 0, 0, -np.sin(theta)]
    return matrix


def _exact_circuit(theta=0.0, phase=0.0):
    code = np.kron(_embedding(theta), _embedding(theta))
    matrix = np.eye(16) + code @ (qutrit_cz() - np.eye(9)) @ code.conj().T
    circuit = QuantumCircuit(4, global_phase=phase)
    circuit.unitary(matrix, range(4))
    return circuit


class StaticTemplate:
    initial_parameters = np.array([0.0])
    template_id = "static-test-template"

    def __init__(self, circuit):
        self.circuit = circuit

    def bind(self, parameters):
        return self.circuit.copy()


def _solver_result(x, *, success=True, status=1, nfev=1):
    return SimpleNamespace(x=x, success=success, status=status, nfev=nfev, message="controlled solver")


@pytest.mark.parametrize("phase", [0.0, 0.72, -2.3])
def test_fit_accepts_exact_gate_with_one_shared_phase_and_json_metadata(phase):
    result = continuation.fit_cz3(StaticTemplate(_exact_circuit(phase=phase)), _embedding(), [0.0], max_nfev=2)
    assert result.valid
    assert result.E_norm < 1e-12
    assert result.L_norm < 1e-12
    assert np.exp(1j * result.global_phase) == pytest.approx(np.exp(1j * phase))
    assert result.evaluations >= result.nfev >= 1
    assert result.elapsed_seconds >= 0
    document = result.to_dict()
    assert "circuit" not in document
    assert document["parameters"] == [0.0]
    json.dumps(document, allow_nan=False)


@pytest.mark.parametrize("kind", ["identity", "column_sign", "leakage"])
def test_fit_rejects_wrong_gate_even_when_solver_reports_success(monkeypatch, kind):
    circuit = _exact_circuit()
    if kind == "identity":
        circuit = QuantumCircuit(4)
    elif kind == "column_sign":
        signs = np.ones(16)
        signs[0] = -1
        circuit.unitary(np.diag(signs), range(4))
    else:
        swap = np.eye(16)
        swap[[0, 3]] = swap[[3, 0]]
        circuit.unitary(swap, range(4))

    def solve(fun, x0, **kwargs):
        residual = fun(x0)
        assert residual.shape == (288,)
        assert np.linalg.norm(residual) > 1
        return _solver_result(x0, success=True)

    monkeypatch.setattr(continuation, "least_squares", solve)
    result = continuation.fit_cz3(StaticTemplate(circuit), _embedding(), [0.0])
    assert not result.valid
    assert result.E_norm > 1
    if kind == "leakage":
        assert result.L_norm > 0.9


def test_fit_accepts_valid_best_probe_despite_solver_limit_and_invalid_final(monkeypatch):
    class ProbeTemplate(StaticTemplate):
        def bind(self, parameters):
            circuit = _exact_circuit()
            circuit.ry(float(parameters[0]), 0)
            return circuit

    def solve(fun, x0, **kwargs):
        assert x0[0] == 7.0
        assert kwargs == dict(method="trf", jac="3-point", loss="linear", max_nfev=7, ftol=1e-11, xtol=1e-11, gtol=1e-11)
        fun(x0)
        probe = np.array([0.0, 0.0])
        assert np.linalg.norm(fun(probe)) < 1e-12
        final = np.array([0.7, 0.0])
        fun(final)
        return _solver_result(final, success=False, status=0, nfev=7)

    monkeypatch.setattr(continuation, "least_squares", solve)
    result = continuation.fit_cz3(ProbeTemplate(_exact_circuit()), _embedding(), [7.0], max_nfev=7)
    assert result.valid
    np.testing.assert_array_equal(result.parameters, [0.0])
    assert result.solver_status == 0
    assert result.nfev == 7
    np.testing.assert_allclose(Operator(result.circuit).data, Operator(_exact_circuit()).data, atol=1e-14)


def test_fit_phase_initialization_and_residual_match_full_complex_action(monkeypatch):
    theta = 0.13
    circuit = _exact_circuit(theta, phase=0.37)
    circuit.ry(0.11, 3)
    code = np.kron(_embedding(theta), _embedding(theta))
    actual = Operator(circuit).data @ code
    target = code @ qutrit_cz()

    def solve(fun, x0, **kwargs):
        assert x0[-1] == pytest.approx(np.angle(np.vdot(target, actual)))
        vector = np.array([0.0, 0.22])
        error = actual - np.exp(1j * 0.22) * target
        np.testing.assert_allclose(fun(vector), np.r_[error.real.ravel(), error.imag.ravel()])
        return _solver_result(x0)

    monkeypatch.setattr(continuation, "least_squares", solve)
    continuation.fit_cz3(StaticTemplate(circuit), _embedding(theta), [0.0])


@pytest.mark.parametrize("parameters", [[np.nan], [np.inf], [1j], [True], ["0"], [], [[0.0]], None])
def test_fit_rejects_invalid_parameters(parameters):
    with pytest.raises(ValueError, match="parameters"):
        continuation.fit_cz3(StaticTemplate(_exact_circuit()), _embedding(), parameters)


@pytest.mark.parametrize("encoding", [np.eye(3), np.eye(4, 3) * 2, np.full((4, 3), np.nan), np.full((4, 3), np.inf)])
def test_fit_rejects_invalid_encoding(encoding):
    with pytest.raises(ValueError, match="encoding"):
        continuation.fit_cz3(StaticTemplate(_exact_circuit()), encoding, [0.0])


@pytest.mark.parametrize("settings", [{"max_nfev": 0}, {"max_nfev": 1.5}, {"max_nfev": True}, {"tolerance": 0}, {"tolerance": np.nan}, {"tolerance": np.inf}, {"tolerance": True}])
def test_fit_rejects_invalid_solver_settings(settings):
    with pytest.raises(ValueError):
        continuation.fit_cz3(StaticTemplate(_exact_circuit()), _embedding(), [0.0], **settings)


def _fit(theta, parameters, *, valid=True, E_norm=None):
    return continuation.Cz3FitResult(
        parameters=np.array(parameters, dtype=float), circuit=_exact_circuit(theta),
        global_phase=0.0, valid=valid, E_norm=(0.0 if valid else 1.0) if E_norm is None else E_norm,
        L_norm=0.0, nfev=1, evaluations=1, solver_status=1,
        message="controlled fit", elapsed_seconds=0.0,
    )


def test_continuation_propagates_exact_previous_parameters_and_parent_ids():
    calls = []

    def fit(template, encoding, parameters, **kwargs):
        calls.append(parameters.copy())
        theta = np.arctan2(-encoding[3, 0].real, encoding[0, 0].real)
        return _fit(theta, parameters + 2 * np.pi + 0.25)

    initial = continuation.ContinuationState(0.0, np.array([7.0]), "baseline")
    first = continuation.continue_cz3(StaticTemplate(_exact_circuit()), initial, 0.1, target_id="point-1", fit_function=fit)
    second = continuation.continue_cz3(StaticTemplate(_exact_circuit()), first.state, 0.2, target_id="point-2", fit_function=fit)
    np.testing.assert_array_equal(calls[0], initial.parameters)
    np.testing.assert_array_equal(calls[1], first.target_fit.parameters)
    assert second.state.theta == 0.2
    assert second.state.parent_id == "point-2"
    assert first.attempts[0]["warm_start_parent_id"] == "baseline"
    assert second.attempts[0]["warm_start_parent_id"] == "point-1"
    assert second.attempts[0]["p_start"] == calls[1].tolist()
    np.testing.assert_array_equal(initial.parameters, [7.0])
    json.dumps(second.attempts, allow_nan=False)


def test_failed_target_solves_midpoint_then_retries_target_from_midpoint():
    calls = []

    def fit(template, encoding, parameters, **kwargs):
        theta = float(np.arctan2(-encoding[3, 0].real, encoding[0, 0].real))
        calls.append((theta, parameters.copy()))
        return _fit(theta, parameters + 1, valid=len(calls) > 1)

    result = continuation.continue_cz3(StaticTemplate(_exact_circuit()), continuation.ContinuationState(0.0, np.array([4.0]), "baseline"), 0.2, target_id="point", max_subdivisions=1, fit_function=fit)
    np.testing.assert_allclose([entry[0] for entry in calls], [0.2, 0.1, 0.2])
    np.testing.assert_array_equal(calls[2][1], [5.0])
    assert result.target_fit.valid
    assert result.state.parent_id == "point"
    assert result.attempts[1]["requested_grid_point"] is False
    assert result.attempts[2]["warm_start_parent_id"] == result.attempts[1]["id"]
    assert len({attempt["id"] for attempt in result.attempts}) == 3


def test_failed_intermediates_preserve_original_state_and_stop_at_depth_limit():
    def fit(template, encoding, parameters, **kwargs):
        return _fit(0.0, [99.0], valid=False)

    initial = continuation.ContinuationState(0.0, np.array([4.0]), "baseline")
    result = continuation.continue_cz3(StaticTemplate(_exact_circuit()), initial, 0.2, target_id="point", max_subdivisions=4, fit_function=fit)
    assert len(result.attempts) == 5
    assert not result.target_fit.valid
    assert result.state.theta == 0.0
    assert result.state.parent_id == "baseline"
    np.testing.assert_array_equal(result.state.parameters, [4.0])
    assert all(attempt["p_start"] == [4.0] for attempt in result.attempts)


def test_target_failure_preserves_successful_auxiliary_without_claiming_target_success():
    def fit(template, encoding, parameters, **kwargs):
        theta = float(np.arctan2(-encoding[3, 0].real, encoding[0, 0].real))
        return _fit(theta, [8.0], valid=theta < 0.15)

    result = continuation.continue_cz3(StaticTemplate(_exact_circuit()), continuation.ContinuationState(0.0, np.array([4.0]), "baseline"), 0.2, target_id="point", max_subdivisions=1, fit_function=fit)
    assert not result.target_fit.valid
    assert result.state.theta == pytest.approx(0.1)
    assert result.state.parent_id == result.attempts[1]["id"]
    np.testing.assert_array_equal(result.state.parameters, [8.0])


@pytest.mark.parametrize("bad_result", ["parameters", "norm", "too_large_norm"])
def test_continuation_rejects_nonfinite_or_wrong_results_before_state_update(bad_result):
    def fit(template, encoding, parameters, **kwargs):
        return _fit(0.1, [np.nan if bad_result == "parameters" else 8.0], E_norm=np.nan if bad_result == "norm" else 1.0 if bad_result == "too_large_norm" else 0.0)

    result = continuation.continue_cz3(StaticTemplate(_exact_circuit()), continuation.ContinuationState(0.0, np.array([4.0]), "baseline"), 0.1, target_id="point", max_subdivisions=0, fit_function=fit)
    assert not result.target_fit.valid
    assert result.state.theta == 0.0
    np.testing.assert_array_equal(result.state.parameters, [4.0])
    json.dumps(result.attempts, allow_nan=False)


@pytest.mark.parametrize("target", [-0.1, np.pi / 4 + 0.01, np.nan, np.inf, True])
def test_continuation_rejects_invalid_target_theta(target):
    with pytest.raises(ValueError, match="theta"):
        continuation.continue_cz3(StaticTemplate(_exact_circuit()), continuation.ContinuationState(0.0, np.array([0.0]), "baseline"), target, target_id="point")


def test_pinned_baseline_real_fit_remains_valid_without_topology_changes():
    from qudits_on_qubits.benchmarks.theta_continuation.cz3_template import load_cz3_template

    path = Path(__file__).resolve().parents[1] / "experiment_inputs/iqm_randomized_bell/canonical_optimized_20260909/CZ3_W.qpy"
    template = load_cz3_template(path)
    result = continuation.fit_cz3(template, _embedding(), template.initial_parameters, max_nfev=2)
    assert result.valid
    assert result.parameters.shape == (33,)
    assert result.circuit.count_ops() == {"cz": 6, "u3": 11}
    assert result.E_norm <= 1e-5
    assert result.L_norm <= 1e-5


def test_cached_action_matches_qiskit_for_baseline_and_unwrapped_perturbations():
    from qudits_on_qubits.benchmarks.theta_continuation.cz3_template import load_cz3_template

    path = Path(__file__).resolve().parents[1] / "experiment_inputs/iqm_randomized_bell/canonical_optimized_20260909/CZ3_W.qpy"
    template = load_cz3_template(path)
    code = np.kron(_embedding(0.19), _embedding(0.19))
    action = continuation._template_action(template, code)
    for shift in [np.zeros(33), np.linspace(-0.17, 0.41, 33), np.arange(33) * 2 * np.pi + 0.11]:
        parameters = template.initial_parameters + shift
        np.testing.assert_allclose(action(parameters), Operator(template.bind(parameters)).data @ code, atol=1e-13, rtol=0)


def test_continuation_independently_rejects_falsely_valid_circuit():
    def fit(template, encoding, parameters, **kwargs):
        result = _fit(0.1, [8.0])
        return continuation.Cz3FitResult(**{**result.__dict__, "circuit": QuantumCircuit(4)})

    result = continuation.continue_cz3(StaticTemplate(_exact_circuit()), continuation.ContinuationState(0.0, np.array([4.0]), "baseline"), 0.1, target_id="point", max_subdivisions=0, fit_function=fit)
    assert not result.target_fit.valid
    assert result.target_fit.E_norm > 1
    assert result.state.parent_id == "baseline"


def test_continuation_full_recursive_tree_has_at_most_31_attempts():
    def outcomes(depth):
        return [True] if depth == 0 else [False] + outcomes(depth - 1) + outcomes(depth - 1)

    results = iter(outcomes(4))

    def fit(template, encoding, parameters, **kwargs):
        theta = float(np.arctan2(-encoding[3, 0].real, encoding[0, 0].real))
        return _fit(theta, parameters + 1, valid=next(results))

    result = continuation.continue_cz3(StaticTemplate(_exact_circuit()), continuation.ContinuationState(0.0, np.array([4.0]), "baseline"), 0.2, target_id="point", max_subdivisions=4, fit_function=fit)
    assert len(result.attempts) == 31
    assert result.target_fit.valid
    assert result.state.theta == 0.2


def test_nonfinite_solver_probe_and_final_cannot_replace_finite_best(monkeypatch):
    def solve(fun, x0, **kwargs):
        assert np.isfinite(fun(np.full_like(x0, np.nan))).all()
        return _solver_result(np.full_like(x0, np.inf), success=False, status=0)

    monkeypatch.setattr(continuation, "least_squares", solve)
    result = continuation.fit_cz3(StaticTemplate(_exact_circuit()), _embedding(), [0.0])
    assert result.valid
    np.testing.assert_array_equal(result.parameters, [0.0])


@pytest.mark.parametrize("max_subdivisions", [-1, 5, True, 1.5, np.inf])
def test_continuation_rejects_invalid_subdivision_limits(max_subdivisions):
    with pytest.raises(ValueError, match="max_subdivisions"):
        continuation.continue_cz3(StaticTemplate(_exact_circuit()), continuation.ContinuationState(0.0, np.array([4.0]), "baseline"), 0.1, target_id="point", max_subdivisions=max_subdivisions)


@pytest.mark.parametrize("theta", [0.0, 0.11, np.pi / 4])
def test_cached_action_matches_full_operator_with_mixed_u3_and_nonzero_phase(theta):
    from qiskit.circuit.library import U3Gate
    from qudits_on_qubits.benchmarks.theta_continuation.cz3_template import parameterize_cz3_circuit

    source = QuantumCircuit(4, global_phase=0.731)
    source.u(0.1, 0.2, 0.3, 3)
    source.cz(3, 0)
    source.append(U3Gate(-0.2, 0.4, 0.7), [0])
    source.cz(1, 2)
    source.u(-0.6, 0.8, -0.9, 1)
    source.cz(1, 3)
    source.append(U3Gate(1.3, -2.1, 0.7), [2])
    template = parameterize_cz3_circuit(source)
    parameters = template.initial_parameters + np.random.default_rng(927).normal(size=12)
    matrix = Operator(template.bind(parameters)).data
    full_action = continuation._template_action(template, np.eye(16))
    np.testing.assert_allclose(full_action(parameters), matrix, atol=1e-14, rtol=0)
    code = np.kron(_embedding(theta), _embedding(theta))
    np.testing.assert_allclose(continuation._template_action(template, code)(parameters), matrix @ code, atol=1e-14, rtol=0)


def test_fit_leaves_entire_seven_dimensional_complement_unrestricted():
    from scipy.linalg import null_space

    theta = 0.23
    code = np.kron(_embedding(theta), _embedding(theta))
    complement = null_space(code.conj().T)
    unitary = code @ qutrit_cz() @ code.conj().T + complement @ np.diag(np.exp(1j * np.linspace(-2.4, 1.7, 7))) @ complement.conj().T
    circuit = QuantumCircuit(4, global_phase=0.57)
    circuit.unitary(unitary, range(4))
    result = continuation.fit_cz3(StaticTemplate(circuit), _embedding(theta), [0.0], max_nfev=2)
    assert result.valid
    assert result.E_norm < 1e-12
    assert result.L_norm < 1e-12


@pytest.mark.parametrize("theta", [0.0, 0.137, np.pi / 4])
def test_analytic_residual_jacobian_matches_every_central_difference_column(theta):
    from qudits_on_qubits.benchmarks.theta_continuation.cz3_template import load_cz3_template

    path = Path(__file__).resolve().parents[1] / "experiment_inputs/iqm_randomized_bell/canonical_optimized_20260909/CZ3_W.qpy"
    template = load_cz3_template(path)
    code = np.kron(_embedding(theta), _embedding(theta))
    target = code @ qutrit_cz()
    action = continuation._template_action(template, code)
    parameters = template.initial_parameters + np.random.default_rng(421).normal(size=33)
    vector = np.r_[parameters, 0.371]
    jacobian = continuation._residual_jacobian(action, vector, target)
    assert jacobian.shape == (288, 34)

    def residual(value):
        error = Operator(template.bind(value[:-1])).data @ code - np.exp(1j * value[-1]) * target
        return np.r_[error.real.ravel(), error.imag.ravel()]

    for index in range(34):
        step = np.zeros(34)
        step[index] = 1e-6
        difference = (residual(vector + step) - residual(vector - step)) / 2e-6
        np.testing.assert_allclose(jacobian[:, index], difference, atol=2e-9, rtol=2e-7, err_msg=f"parameter {index}")


def test_fit_reports_analytic_jacobian_for_verified_template():
    from qudits_on_qubits.benchmarks.theta_continuation.cz3_template import load_cz3_template

    path = Path(__file__).resolve().parents[1] / "experiment_inputs/iqm_randomized_bell/canonical_optimized_20260909/CZ3_W.qpy"
    template = load_cz3_template(path)
    result = continuation.fit_cz3(template, _embedding(), template.initial_parameters, max_nfev=2)
    assert result.to_dict()["jacobian_method"] == "analytic_u3"
    assert result.to_dict()["jacobian_evaluations"] > 0
