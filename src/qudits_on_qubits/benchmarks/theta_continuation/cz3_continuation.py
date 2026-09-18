"""Fit a fixed CZ3 topology and continue its last valid parameter branch."""
from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Integral, Real
from time import perf_counter

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from scipy.optimize import least_squares

from qudits_on_qubits.benchmarks.direct_basis.math_utils import qutrit_cz
from qudits_on_qubits.benchmarks.direct_basis.optimized_gates import validate_code_space_gate
from .encoding import theta_embedding
from .cz3_template import Cz3Template


def _finite_real(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real number") from exc
    if not np.isfinite(number):
        raise ValueError(f"{name} must be a finite real number")
    return number


def _parameters(values: object, expected_shape: tuple[int, ...]) -> np.ndarray:
    try:
        vector = np.asarray(values, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError("parameters must be a finite real vector") from exc
    if vector.shape != expected_shape:
        raise ValueError(f"parameters must have shape {expected_shape}")
    return np.array([_finite_real(value, "parameters") for value in vector], dtype=float)


def _settings(max_nfev: int, tolerance: float) -> tuple[int, float]:
    if isinstance(max_nfev, (bool, np.bool_)) or not isinstance(max_nfev, Integral) or max_nfev < 1:
        raise ValueError("max_nfev must be a positive integer")
    tolerance = _finite_real(tolerance, "tolerance")
    if tolerance <= 0:
        raise ValueError("tolerance must be positive")
    return int(max_nfev), tolerance


def _json_number(value: float) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


@dataclass(frozen=True)
class Cz3FitResult:
    parameters: np.ndarray
    circuit: QuantumCircuit
    global_phase: float
    valid: bool
    E_norm: float
    L_norm: float
    nfev: int
    evaluations: int
    solver_status: int
    message: str
    elapsed_seconds: float
    jacobian_method: str = "3-point"
    evaluator_method: str = "qiskit"
    jacobian_evaluations: int = 0

    def to_dict(self) -> dict:
        """Return strict JSON data, retaining nonfinite diagnostics as null."""
        return {
            "parameters": [_json_number(value) for value in self.parameters],
            "global_phase": _json_number(self.global_phase),
            "valid": bool(self.valid),
            "E_norm": _json_number(self.E_norm),
            "L_norm": _json_number(self.L_norm),
            "nfev": int(self.nfev),
            "evaluations": int(self.evaluations),
            "solver_status": int(self.solver_status),
            "message": str(self.message),
            "elapsed_seconds": _json_number(self.elapsed_seconds),
            "jacobian_method": self.jacobian_method,
            "evaluator_method": self.evaluator_method,
            "jacobian_evaluations": int(self.jacobian_evaluations),
        }



def _template_action(template, code: np.ndarray):
    """Cache gate wiring; apply exact U/U3 and CZ matrices to code columns.

    Qiskit's qubit 0 is the least-significant matrix-index bit. Gate order,
    parameter order and the source global phase remain unchanged. Two independent
    Qiskit evaluations verify the kernel before an optimization uses it.
    """
    def qiskit_action(parameters):
        circuit = template.bind(parameters)
        if circuit.num_qubits != 4:
            raise ValueError("CZ3 template must bind a four-qubit circuit")
        return Operator(circuit).data @ code

    if not isinstance(template, Cz3Template):
        return qiskit_action
    operations = []
    indices = np.arange(16)
    slot = 0
    for instruction in template.original_circuit.data:
        qubits = [template.original_circuit.find_bit(qubit).index for qubit in instruction.qubits]
        if instruction.operation.name == "cz":
            mask = (1 << qubits[0]) | (1 << qubits[1])
            operations.append((None, indices[(indices & mask) == mask], None))
        else:
            mask = 1 << qubits[0]
            zero = indices[(indices & mask) == 0]
            operations.append((slot, zero, zero | mask))
            slot += 1
    source_phase = np.exp(1j * float(template.original_circuit.global_phase))

    def action(parameters):
        angles = np.asarray(parameters).reshape(-1, 3)
        cosines = np.cos(angles[:, 0] / 2)
        sines = np.sin(angles[:, 0] / 2)
        phi_phases = np.exp(1j * angles[:, 1])
        lambda_phases = np.exp(1j * angles[:, 2])
        coefficients = np.column_stack((cosines, -lambda_phases * sines,
                                        phi_phases * sines, phi_phases * lambda_phases * cosines))
        output = np.array(code, dtype=complex, copy=True)
        for gate_slot, zero, one in operations:
            if gate_slot is None:
                output[zero] *= -1
            else:
                before_zero, before_one = output[zero], output[one]
                a, b, c, d = coefficients[gate_slot]
                output[zero] = a * before_zero + b * before_one
                output[one] = c * before_zero + d * before_one
        return source_phase * output

    def jacobian(parameters):
        # The final axis contains the action and all parameter tangents. Each
        # gate applies its ordinary matrix to all previous tangents, then adds
        # the local derivative acting on the incoming, undifferentiated state.
        angles = np.asarray(parameters).reshape(-1, 3)
        cosines = np.cos(angles[:, 0] / 2)
        sines = np.sin(angles[:, 0] / 2)
        phi_phases = np.exp(1j * angles[:, 1])
        lambda_phases = np.exp(1j * angles[:, 2])
        output = np.zeros((*code.shape, len(parameters) + 1), dtype=complex)
        output[:, :, 0] = code
        for gate_slot, zero, one in operations:
            if gate_slot is None:
                output[zero] *= -1
                continue
            cosine, sine = cosines[gate_slot], sines[gate_slot]
            phi, lam = phi_phases[gate_slot], lambda_phases[gate_slot]
            a, b, c, d = cosine, -lam * sine, phi * sine, phi * lam * cosine
            before_zero, before_one = output[zero], output[one]
            output[zero] = a * before_zero + b * before_one
            output[one] = c * before_zero + d * before_one
            local_derivatives = (
                (-sine / 2, -lam * cosine / 2, phi * cosine / 2, -phi * lam * sine / 2),
                (0, 0, 1j * c, 1j * d),
                (0, 1j * b, 0, 1j * d),
            )
            for offset, (da, db, dc, dd) in enumerate(local_derivatives):
                column = 1 + 3 * gate_slot + offset
                output[zero, :, column] += da * before_zero[:, :, 0] + db * before_one[:, :, 0]
                output[one, :, column] += dc * before_zero[:, :, 0] + dd * before_one[:, :, 0]
        return source_phase * output[:, :, 1:]

    action.jacobian = jacobian
    initial = template.initial_parameters
    perturbation = np.linspace(0.007, 0.023, len(initial))
    for parameters in (initial, initial + perturbation):
        if not np.allclose(action(parameters), qiskit_action(parameters), atol=5e-13, rtol=0):
            raise ValueError("cached CZ3 evaluator disagrees with Qiskit")
    return action



def _residual_jacobian(action, vector: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Real residual Jacobian, including the single shared phase column."""
    parameter_columns = action.jacobian(vector[:-1])
    phase_column = -1j * np.exp(1j * vector[-1]) * target
    columns = np.concatenate((parameter_columns, phase_column[:, :, None]), axis=2)
    return np.concatenate((columns.real.reshape(-1, len(vector)),
                           columns.imag.reshape(-1, len(vector))), axis=0)


def fit_cz3(
    template,
    encoding: np.ndarray,
    initial_parameters: np.ndarray,
    *,
    max_nfev: int = 3000,
    tolerance: float = 1e-5,
) -> Cz3FitResult:
    """Fit nine code columns using one phase and an unrestricted complement.

    Every finite residual evaluation, including finite-difference probes, is
    eligible for selection. Solver termination never substitutes for independent
    code-space validation. The previous angles are used without wrapping.
    """
    started = perf_counter()
    max_nfev, tolerance = _settings(max_nfev, tolerance)
    initial = _parameters(initial_parameters, np.asarray(template.initial_parameters).shape)
    try:
        embedding = np.asarray(encoding, dtype=complex)
    except (ValueError, TypeError) as exc:
        raise ValueError("encoding must be a finite 4x3 isometry") from exc
    if (
        embedding.shape != (4, 3)
        or not np.isfinite(embedding).all()
        or not np.allclose(embedding.conj().T @ embedding, np.eye(3), atol=1e-12, rtol=0)
    ):
        raise ValueError("encoding must be a finite 4x3 isometry")
    code = np.kron(embedding, embedding)
    target = code @ qutrit_cz()
    leakage_projector = np.eye(16) - code @ code.conj().T

    action = _template_action(template, code)
    first_actual = action(initial)
    if not np.isfinite(first_actual).all():
        raise ValueError("initial parameters produce a nonfinite circuit")
    gamma = float(np.angle(np.vdot(target, first_actual)))
    x0 = np.r_[initial, gamma]
    best_x = x0.copy()
    best_cost = float("inf")
    best_valid_x = None
    best_valid_cost = float("inf")
    evaluations = 0
    jacobian_evaluations = 0

    def residual(vector):
        nonlocal best_x, best_cost, best_valid_x, best_valid_cost, evaluations
        evaluations += 1
        vector = np.asarray(vector, dtype=float)
        if vector.shape != x0.shape or not np.isfinite(vector).all():
            return np.full(288, 1e50)
        actual = action(vector[:-1])
        error = actual - np.exp(1j * vector[-1]) * target
        output = np.r_[error.real.ravel(), error.imag.ravel()]
        if not np.isfinite(output).all():
            return np.full(288, 1e50)
        cost = float(np.dot(output, output))
        if cost < best_cost:
            best_cost, best_x = cost, vector.copy()
        physical_phase = np.exp(1j * np.angle(np.vdot(target, actual)))
        error_norm = np.linalg.norm(actual - physical_phase * target, ord="fro")
        leakage_norm = np.linalg.norm(leakage_projector @ actual, ord="fro")
        if error_norm <= tolerance and leakage_norm <= tolerance and cost < best_valid_cost:
            best_valid_cost, best_valid_x = cost, vector.copy()
        return output

    def analytic_jacobian(vector):
        nonlocal jacobian_evaluations
        jacobian_evaluations += 1
        return _residual_jacobian(action, vector, target)

    analytic = hasattr(action, "jacobian")
    residual(x0)
    solution = least_squares(
        residual, x0, method="trf", jac=analytic_jacobian if analytic else "3-point", loss="linear",
        max_nfev=max_nfev, ftol=1e-11, xtol=1e-11, gtol=1e-11,
    )
    residual(solution.x)
    selected = best_valid_x if best_valid_x is not None else best_x
    circuit = template.bind(selected[:-1])
    validation = validate_code_space_gate(circuit, code, qutrit_cz())
    valid = bool(
        np.isfinite([validation.E_norm, validation.L_norm]).all()
        and validation.E_norm <= tolerance
        and validation.L_norm <= tolerance
    )
    return Cz3FitResult(
        parameters=selected[:-1].copy(), circuit=circuit,
        global_phase=float(selected[-1]), valid=valid,
        E_norm=float(validation.E_norm), L_norm=float(validation.L_norm),
        nfev=int(solution.nfev), evaluations=evaluations,
        solver_status=int(solution.status), message=str(solution.message),
        elapsed_seconds=perf_counter() - started,
        jacobian_method="analytic_u3" if analytic else "3-point",
        evaluator_method="cached_u3_cz" if analytic else "qiskit",
        jacobian_evaluations=jacobian_evaluations,
    )


@dataclass(frozen=True)
class ContinuationState:
    theta: float
    parameters: np.ndarray
    parent_id: str


@dataclass(frozen=True)
class ContinuationResult:
    target_fit: Cz3FitResult | None
    state: ContinuationState
    attempts: list[dict]


def continue_cz3(
    template,
    state: ContinuationState,
    target_theta: float,
    *,
    target_id: str,
    max_nfev: int = 3000,
    tolerance: float = 1e-5,
    max_subdivisions: int = 4,
    fit_function=None,
) -> ContinuationResult:
    """Try the target, then bounded midpoint rescue from the last valid state.

    A depth of d permits at most 2**(d+1)-1 fits. Successful auxiliary points
    remain valid warm starts even when the requested target is unreachable.
    """
    max_nfev, tolerance = _settings(max_nfev, tolerance)
    target_theta = _finite_real(target_theta, "target_theta")
    previous_theta = _finite_real(state.theta, "state.theta")
    if not 0 <= previous_theta <= target_theta <= np.pi / 4:
        raise ValueError("theta must increase within [0, pi/4]")
    if (
        isinstance(max_subdivisions, (bool, np.bool_))
        or not isinstance(max_subdivisions, Integral)
        or not 0 <= max_subdivisions <= 4
    ):
        raise ValueError("max_subdivisions must be an integer between 0 and 4")
    if not isinstance(target_id, str) or not target_id.strip():
        raise ValueError("target_id must be a nonempty string")
    if not isinstance(state.parent_id, str) or not state.parent_id.strip():
        raise ValueError("state.parent_id must be a nonempty string")
    shape = np.asarray(template.initial_parameters).shape
    current = ContinuationState(previous_theta, _parameters(state.parameters, shape), state.parent_id)
    fit_function = fit_cz3 if fit_function is None else fit_function
    attempts = []
    target_fit = None

    def advance(theta, depth):
        nonlocal current, target_fit
        requested = theta == target_theta
        sequence = len(attempts) + 1
        attempt_id = f"{target_id}__{'attempt' if requested else 'aux'}_{sequence:03d}"
        start = current
        fitted = fit_function(
            template, theta_embedding(theta), start.parameters.copy(),
            max_nfev=max_nfev, tolerance=tolerance,
        )
        try:
            parameters = _parameters(fitted.parameters, shape)
            finite_metrics = np.isfinite([fitted.global_phase, fitted.E_norm, fitted.L_norm]).all()
            accepted = bool(
                fitted.valid and finite_metrics
                and 0 <= fitted.E_norm <= tolerance
                and 0 <= fitted.L_norm <= tolerance
            )
        except (ValueError, TypeError):
            accepted = False
        if accepted:
            code = np.kron(theta_embedding(theta), theta_embedding(theta))
            validation = validate_code_space_gate(fitted.circuit, code, qutrit_cz())
            accepted = bool(
                np.isfinite([validation.E_norm, validation.L_norm]).all()
                and validation.E_norm <= tolerance and validation.L_norm <= tolerance
            )
            fitted = replace(fitted, E_norm=float(validation.E_norm), L_norm=float(validation.L_norm))
        if not accepted and fitted.valid:
            fitted = replace(fitted, valid=False, message=f"{fitted.message}; rejected invalid continuation result")
        attempts.append({
            "id": attempt_id,
            "theta": float(theta),
            "requested_grid_point": requested,
            "warm_start_parent_id": start.parent_id,
            "p_start": start.parameters.tolist(),
            "fit": fitted.to_dict(),
        })
        if requested:
            target_fit = fitted
        if accepted:
            current = ContinuationState(float(theta), parameters.copy(), target_id if requested else attempt_id)
            return True
        if depth == 0:
            return False
        midpoint = (current.theta + theta) / 2
        if midpoint <= current.theta or midpoint >= theta:
            return False
        if not advance(midpoint, depth - 1):
            return False
        return advance(theta, depth - 1)

    advance(target_theta, int(max_subdivisions))
    return ContinuationResult(target_fit, current, attempts)
