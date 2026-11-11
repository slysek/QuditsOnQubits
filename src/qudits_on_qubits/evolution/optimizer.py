"""Fit one fixed CZ3 circuit with two local encodings."""

from dataclasses import asdict, dataclass

import numpy as np
from qiskit import QuantumCircuit

from qudits_on_qubits.benchmarks.theta_continuation.cz3_continuation import (
    Cz3FitResult, fit_cz3,
)
from qudits_on_qubits.benchmarks.theta_continuation.cz3_template import (
    parameterize_cz3_circuit,
)

from .evaluator import evaluate
from .models import EvaluationResult, EvaluationTask
from .targets import controlled_z_d

ANGLE_BOUND_RADIUS = 2 * np.pi


@dataclass(frozen=True)
class FitResult:
    """Keep the fitted circuit and its independent evaluation."""

    solver: Cz3FitResult
    evaluation: EvaluationResult

    @property
    def circuit(self) -> QuantumCircuit:
        return self.solver.circuit

    @property
    def parameters(self) -> np.ndarray:
        return self.solver.parameters

    @property
    def nfev(self) -> int:
        return self.solver.nfev

    @property
    def evaluations(self) -> int:
        return self.solver.evaluations

    @property
    def elapsed_seconds(self) -> float:
        return self.solver.elapsed_seconds

    def to_dict(self) -> dict:
        """Return the solver and evaluator results as JSON data."""
        return {"solver": self.solver.to_dict(), "evaluation": asdict(self.evaluation),
                "solver_config": {"method": "scipy_least_squares_trf",
                                  "angle_bound_radius": ANGLE_BOUND_RADIUS,
                                  "phase_bound_radius": np.pi}}


def fit_fixed_cz3(
    circuit: QuantumCircuit,
    encodings: list[np.ndarray],
    initial_parameters: np.ndarray,
    *,
    max_nfev: int = 200,
    tolerance: float = 1e-10,
) -> FitResult:
    """Fit the U/U3 angles. Preserve the CZ gates and the encodings.

    The tolerance applies to the unnormalized error and leakage norms.
    Encoding A applies to qubits 0 and 1. Encoding B applies to qubits 2 and 3.
    """
    if len(encodings) != 2:
        raise ValueError("The input must contain two encodings")
    template = parameterize_cz3_circuit(circuit)
    task = EvaluationTask(3, 2, controlled_z_d(3),
                          logical_error_tolerance=tolerance / 3,
                          leakage_tolerance=(tolerance / 3) ** 2,
                          allowed_gates=("u", "u3", "cz"))
    evaluate(task, template.bind(initial_parameters), encodings)
    solver = fit_cz3(template, encodings[0], initial_parameters,
                     encoding_b=encodings[1], max_nfev=max_nfev, tolerance=tolerance,
                     angle_radius=ANGLE_BOUND_RADIUS)
    evaluation = evaluate(task, solver.circuit, encodings)
    return FitResult(solver, evaluation)
