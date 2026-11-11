"""Evaluate the logical action of an elementary physical circuit."""

from collections.abc import Sequence
from time import perf_counter

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.circuit.library import CXGate, CZGate, U3Gate, UGate
from qiskit.quantum_info import Operator

from qudits_on_qubits.benchmarks.direct_basis.math_utils import validate_unitary

from .encodings import tensor_encoding
from .metrics import code_space_metrics, gate_costs
from .models import EvaluationResult, EvaluationTask, _finite_matrix

_BASIS = {"u": UGate, "u3": U3Gate, "cz": CZGate, "cx": CXGate}


def evaluate(
    task: EvaluationTask, circuit: QuantumCircuit, encodings: Sequence[np.ndarray],
) -> EvaluationResult:
    """Score a bound U/U3/CZ/CX circuit in physical wire order.

    Supply one local isometry per qudit in increasing order. Leakage is averaged
    over logical basis inputs after the complete circuit. Intermediate excursions
    into the complement are allowed. All matrices and metrics are dense.
    """
    started = perf_counter()
    if not isinstance(circuit, QuantumCircuit):
        raise TypeError("circuit must be a QuantumCircuit")
    if circuit.num_qubits != task.num_qubits:
        raise ValueError(f"circuit must have exactly {task.num_qubits} physical qubits")
    if circuit.num_ancillas or circuit.num_clbits or circuit.num_vars:
        raise ValueError("circuit must have no ancillas, classical bits or classical variables")
    if circuit.num_parameters or not np.isfinite(float(circuit.global_phase)):
        raise ValueError("circuit parameters and global phase must be bound and finite")
    if len(encodings) != task.n:
        raise ValueError(f"expected {task.n} local encodings")

    encoding = tensor_encoding(encodings, d=task.d)
    # Evolve only encoded inputs. Each composition left-multiplies the columns
    # by a local unitary on the Qiskit wires given by qargs.
    output = Operator(
        encoding, input_dims=(task.d,) * task.n, output_dims=(2,) * task.num_qubits,
    )
    for item in circuit.data:
        operation = item.operation
        expected_class = _BASIS.get(operation.name)
        if (operation.name not in task.allowed_gates
                or not isinstance(operation, Gate) or operation.base_class is not expected_class):
            raise ValueError(f"unsupported gate {operation.name!r} (expected U/U3, CZ or CX)")
        if not all(np.isfinite(float(p)) for p in operation.params):
            raise ValueError(f"{operation.name} gate parameters must be finite")
        matrix = _finite_matrix(operation.to_matrix(), operation.name)
        validate_unitary(matrix, 2**len(item.qubits), operation.name)
        wires = [circuit.find_bit(q).index for q in item.qubits]
        output = output.compose(Operator(matrix), qargs=wires)
    actual = np.exp(1j * float(circuit.global_phase)) * output.data
    metrics = code_space_metrics(encoding, task.target, actual)
    costs = gate_costs(circuit)
    return EvaluationResult(
        d=task.d, n=task.n, num_qubits=task.num_qubits,
        is_valid=metrics["eps"] <= task.logical_error_tolerance and metrics["L"] <= task.leakage_tolerance,
        evaluation_seconds=perf_counter() - started,
        **metrics, **costs,
    )
