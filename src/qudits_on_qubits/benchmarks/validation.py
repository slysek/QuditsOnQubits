"""Independent ideal validation of physical preparations after layout and routing."""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

_IGNORED = {"barrier", "delay"}


def circuit_metrics(circuit: QuantumCircuit) -> dict:
    operations = [item for item in circuit.data if item.operation.name not in _IGNORED]
    return {
        "two_qubit_gate_count": sum(len(item.qubits) == 2 for item in operations),
        "one_qubit_gate_count": sum(len(item.qubits) == 1 for item in operations),
        "depth": int(circuit.depth() or 0),
        "two_qubit_depth": int(circuit.depth(
            filter_function=lambda item: len(item.qubits) == 2 and item.operation.name not in _IGNORED
        ) or 0),
        "size": len(operations),
    }


def state_metrics(
    circuit: QuantumCircuit, expected: np.ndarray, encoding: np.ndarray,
    num_qutrits: int, *, max_qubits: int = 12,
) -> dict:
    """Trace routing ancillas without allocating a statevector for idle device wires."""
    logical_width = 2 * num_qutrits
    if logical_width > 12 or logical_width > max_qubits:
        raise ValueError("Logical state exceeds the validation qubit limit")
    if circuit.num_parameters or circuit.num_clbits:
        raise ValueError("Validation requires a bound unitary preparation circuit")
    mapping = (
        list(circuit.layout.final_index_layout(filter_ancillas=True))
        if circuit.layout is not None else list(range(logical_width))
    )
    if (len(mapping) != logical_width or len(set(mapping)) != logical_width
            or any(index < 0 or index >= circuit.num_qubits for index in mapping)):
        raise ValueError("Final layout does not match logical input width")
    active = set(mapping)
    for item in circuit.data:
        if item.operation.name in _IGNORED:
            continue
        if item.operation.name in {"measure", "reset", "initialize"} or item.clbits:
            raise ValueError("Validation requires a unitary preparation circuit")
        active.update(circuit.find_bit(bit).index for bit in item.qubits)
    physical = sorted(active)
    if len(physical) > max_qubits:
        raise ValueError(f"Active circuit exceeds validation qubit limit ({len(physical)} > {max_qubits})")
    positions = {wire: index for index, wire in enumerate(physical)}
    compact = QuantumCircuit(len(physical))
    compact.global_phase = circuit.global_phase
    for item in circuit.data:
        if item.operation.name not in _IGNORED:
            compact.append(item.operation, [positions[circuit.find_bit(bit).index] for bit in item.qubits])
    state = Statevector.from_instruction(compact).data
    output_axes = [len(physical) - 1 - positions[mapping[index]] for index in reversed(range(logical_width))]
    other_axes = [axis for axis in range(len(physical)) if axis not in output_axes]
    amplitudes = state.reshape([2] * len(physical)).transpose(output_axes + other_axes).reshape(2**logical_width, -1)
    expected = np.asarray(expected, dtype=complex)
    if expected.shape != (2**logical_width,) or not np.isfinite(expected).all() or not np.isclose(np.vdot(expected, expected), 1):
        raise ValueError("Expected state must be finite, normalized and match logical width")
    embedding = np.ones((1, 1), dtype=complex)
    for _ in range(num_qutrits):
        embedding = np.kron(embedding, encoding)
    fidelity = float(np.sum(np.abs(expected.conj() @ amplitudes) ** 2))
    retained = float(np.sum(np.abs(embedding.conj().T @ amplitudes) ** 2))
    return {
        "fidelity": float(np.clip(fidelity, 0, 1)),
        "leakage": float(np.clip(1 - retained, 0, 1)),
        "validation_qubits": len(physical),
        "active_physical_qubits": physical,
        "output_mapping": mapping,
    }
