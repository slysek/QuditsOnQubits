from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector

from .encoding import infer_num_qutrits_from_state, leakage_probability, statevector_data
from .operators import split_nonhermitian


@dataclass(frozen=True)
class BellResult:
    value: complex
    leakage_probability: float
    backend: str
    shots: int | None = None


def bell_value_estimator(
    state_or_circuit: Statevector | QuantumCircuit | np.ndarray,
    bell_operator: np.ndarray | Operator | SparsePauliOp,
    *,
    E: np.ndarray | None = None,
    estimator: Any | None = None,
    shots: int | None = None,
) -> BellResult:
    """Evaluate a Bell operator through an Estimator-compatible path.

    If an estimator and circuit are supplied, the non-Hermitian safety split is
    still applied before submission. Otherwise the function uses an exact
    Statevector expectation value with the same Hermitian split.
    """
    matrix = _operator_to_matrix(bell_operator)
    real, imag = split_nonhermitian(matrix)

    if estimator is not None and isinstance(state_or_circuit, QuantumCircuit):
        value = _run_estimator(estimator, state_or_circuit, real, imag, shots=shots)
        state = Statevector.from_instruction(state_or_circuit)
        backend_name = estimator.__class__.__name__
    else:
        state = _as_statevector(state_or_circuit)
        value = _exact_expectation(state, real) + 1j * _exact_expectation(state, imag)
        backend_name = "StatevectorEstimator"

    leak = 0.0
    if E is not None:
        leak = leakage_probability(state, E, infer_num_qutrits_from_state(state))
    return BellResult(value=complex(value), leakage_probability=leak, backend=backend_name, shots=shots)


def _as_statevector(state_or_circuit: Statevector | QuantumCircuit | np.ndarray) -> Statevector:
    if isinstance(state_or_circuit, Statevector):
        return state_or_circuit
    if isinstance(state_or_circuit, QuantumCircuit):
        return Statevector.from_instruction(state_or_circuit)
    return Statevector(statevector_data(state_or_circuit))


def _operator_to_matrix(operator: np.ndarray | Operator | SparsePauliOp) -> np.ndarray:
    if isinstance(operator, SparsePauliOp):
        return np.asarray(operator.to_matrix(), dtype=complex)
    if isinstance(operator, Operator):
        return np.asarray(operator.data, dtype=complex)
    return np.asarray(operator, dtype=complex)


def _exact_expectation(state: Statevector, operator: np.ndarray) -> complex:
    return complex(state.expectation_value(Operator(operator)))


def _run_estimator(
    estimator: Any,
    circuit: QuantumCircuit,
    real: np.ndarray,
    imag: np.ndarray,
    *,
    shots: int | None,
) -> complex:
    real_op = SparsePauliOp.from_operator(Operator(real))
    imag_op = SparsePauliOp.from_operator(Operator(imag))
    pubs = [(circuit, real_op), (circuit, imag_op)]
    if shots is not None and hasattr(estimator, "options"):
        try:
            estimator.options.default_shots = shots
        except Exception:
            pass
    result = estimator.run(pubs).result()
    ev_real = _extract_estimator_value(result, 0)
    ev_imag = _extract_estimator_value(result, 1)
    return complex(ev_real, ev_imag)


def _extract_estimator_value(result: Any, index: int) -> float:
    item = result[index]
    data = getattr(item, "data", item)
    evs = getattr(data, "evs", data)
    arr = np.asarray(evs)
    return float(arr.reshape(-1)[0])


def make_aer_estimator() -> Any | None:
    """Return Aer EstimatorV2 when qiskit-aer is installed."""
    try:
        from qiskit_aer.primitives import EstimatorV2 as AerEstimator
    except Exception:
        return None
    return AerEstimator()
