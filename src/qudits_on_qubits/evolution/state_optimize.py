"""Fit angles of a fixed circuit for one encoded qudit state."""
from dataclasses import dataclass
from numbers import Integral, Real
from time import perf_counter

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.optimize import least_squares

from .state_workloads import _check_circuit, encoded_target, evaluate_workload, layout


@dataclass(frozen=True)
class StateFitResult:
    """Keep the circuit, metrics, parameters and solver counts."""

    circuit: QuantumCircuit
    metrics: dict
    initial_parameters: np.ndarray
    parameters: np.ndarray
    active_parameters: tuple[int, ...]
    nfev: int
    evaluations: int
    jacobian_evaluations: int
    solver_status: int
    message: str
    elapsed_seconds: float


def fit_state(name, circuit, encoding, *, max_nfev=50, tolerance=1e-10,
              max_parameters=24, initial_parameters=None):
    """Fit at most 24 U angles and check the resulting state.

    The fit selects angles in circuit order. The circuit starts at physical zero.
    The input angle vector follows the same order as the selected angles.
    """
    if isinstance(max_nfev, bool) or not isinstance(max_nfev, Integral) or not 1 <= max_nfev <= 200:
        raise ValueError("max_nfev must be an integer from 1 to 200")
    if isinstance(max_parameters, bool) or not isinstance(max_parameters, Integral) or not 0 <= max_parameters <= 24:
        raise ValueError("max_parameters must be an integer from 0 to 24")
    if isinstance(tolerance, bool) or not isinstance(tolerance, Real) or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be a finite positive number")
    n, _, q = layout(name)
    _check_circuit(circuit, q * n)
    if len(circuit.data) > 512:
        raise ValueError("The circuit must contain at most 512 gates")
    if any(item.operation.name not in {"u", "u3", "cz"} for item in circuit.data):
        raise ValueError("The circuit must contain only U, U3 and CZ gates")
    target, _ = encoded_target(name, encoding)
    dimensions = 1 << circuit.num_qubits
    indices = np.arange(dimensions)
    operations = []
    gate_angles = []
    for instruction_index, item in enumerate(circuit.data):
        qubits = [circuit.find_bit(q).index for q in item.qubits]
        if item.operation.name == "cz":
            mask = (1 << qubits[0]) | (1 << qubits[1])
            operations.append((None, indices[(indices & mask) == mask], None, None))
        else:
            mask = 1 << qubits[0]
            zero = indices[(indices & mask) == 0]
            slot = len(gate_angles)
            gate_angles.append([float(value) for value in item.operation.params])
            operations.append((slot, zero, zero | mask, instruction_index))
    flat = np.asarray(gate_angles, dtype=float).reshape(-1)
    active = tuple(range(min(int(max_parameters), len(flat))))
    original = flat[:len(active)].copy()
    if initial_parameters is not None:
        try:
            initial = np.asarray(initial_parameters, dtype=float)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError("initial_parameters must be a finite angle vector") from exc
        if initial.shape != original.shape or not np.isfinite(initial).all():
            raise ValueError("initial_parameters must be a finite angle vector of the selected size")
        original = initial.copy()
    started = perf_counter()
    source_phase = np.exp(1j * float(circuit.global_phase))

    def action(parameters, derivative=False):
        angles = flat.copy()
        angles[:len(active)] = parameters
        angles = angles.reshape(-1, 3)
        columns = len(active) + 1 if derivative else 1
        state = np.zeros((dimensions, columns), dtype=complex)
        state[0, 0] = 1
        for slot, zero, one, _ in operations:
            if slot is None:
                state[zero] *= -1
                continue
            theta, phi, lam = angles[slot]
            cosine, sine = np.cos(theta / 2), np.sin(theta / 2)
            ep, el = np.exp(1j * phi), np.exp(1j * lam)
            a, b, c, d = cosine, -el * sine, ep * sine, ep * el * cosine
            old_zero, old_one = state[zero].copy(), state[one].copy()
            state[zero] = a * old_zero + b * old_one
            state[one] = c * old_zero + d * old_one
            if derivative:
                derivatives = ((-sine / 2, -el * cosine / 2, ep * cosine / 2, -ep * el * sine / 2),
                               (0, 0, 1j * c, 1j * d), (0, 1j * b, 0, 1j * d))
                for offset, (da, db, dc, dd) in enumerate(derivatives):
                    parameter = 3 * slot + offset
                    if parameter < len(active):
                        state[zero, parameter + 1] += da * old_zero[:, 0] + db * old_one[:, 0]
                        state[one, parameter + 1] += dc * old_zero[:, 0] + dd * old_one[:, 0]
        return source_phase * state

    def bind(parameters):
        updated = flat.copy()
        updated[:len(active)] = parameters
        output = circuit.copy()
        for slot, _, _, instruction_index in operations:
            if slot is not None:
                item = output.data[instruction_index]
                gate = item.operation.copy()
                gate.params = list(updated[3 * slot:3 * slot + 3])
                output.data[instruction_index] = item.replace(operation=gate)
        return output

    initial_circuit = bind(original)
    first = action(original)[:, 0]
    if not np.allclose(first, Statevector.from_instruction(initial_circuit).data, rtol=0, atol=1e-12):
        raise ValueError("The state calculation does not match Qiskit")
    if not active:
        metrics = evaluate_workload(name, initial_circuit, encoding, tolerance=tolerance)
        return StateFitResult(initial_circuit, metrics, original, original.copy(), active,
                              0, 1, 0, 0, "No angles to fit", perf_counter() - started)
    phase = float(np.angle(np.vdot(target, first)))
    x0 = np.r_[original, phase]
    best = x0.copy()
    best_cost = float("inf")
    evaluations = 0
    jacobian_evaluations = 0

    def residual(values):
        nonlocal best, best_cost, evaluations
        evaluations += 1
        difference = action(values[:-1])[:, 0] - np.exp(1j * values[-1]) * target
        result = np.r_[difference.real, difference.imag]
        cost = float(np.dot(result, result))
        if cost < best_cost:
            best, best_cost = values.copy(), cost
        return result

    def jacobian(values):
        nonlocal jacobian_evaluations
        jacobian_evaluations += 1
        columns = np.column_stack((action(values[:-1], derivative=True)[:, 1:],
                                   -1j * np.exp(1j * values[-1]) * target))
        return np.vstack((columns.real, columns.imag))

    lower = np.r_[original - 2 * np.pi, phase - np.pi]
    upper = np.r_[original + 2 * np.pi, phase + np.pi]
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower >= upper):
        raise ValueError("The initial angles cannot use the fit limits")
    solution = least_squares(residual, x0, jac=jacobian, method="trf", loss="linear",
                             bounds=(lower, upper), max_nfev=int(max_nfev),
                             ftol=1e-11, xtol=1e-11, gtol=1e-11)
    fitted = bind(best[:-1])
    metrics = evaluate_workload(name, fitted, encoding, tolerance=tolerance)
    return StateFitResult(fitted, metrics, original, best[:-1].copy(), active,
                          int(solution.nfev), evaluations, jacobian_evaluations,
                          int(solution.status), str(solution.message), perf_counter() - started)
