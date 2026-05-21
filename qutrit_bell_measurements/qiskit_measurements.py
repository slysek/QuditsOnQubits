from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TYPE_CHECKING

import numpy as np

from .basis import local_measurement_basis_unitary

if TYPE_CHECKING:
    from qiskit import QuantumCircuit


def append_measurement_for_global_setting(
    state_circuit: "QuantumCircuit",
    global_setting: Sequence[object],
    qutrit_qubits: Sequence[Sequence[int]],
    E: np.ndarray,
    observable_from_label: Callable[[object], np.ndarray],
    d: int = 3,
    inplace: bool = False,
    add_measurements: bool = True,
    classical_register_name: str | None = None,
) -> tuple["QuantumCircuit", dict[str, Any]]:
    """Append local qutrit measurement-basis gates and optional measurements.

    The function works setting-by-setting and never constructs a global tensor
    product measurement operator. Each non-None label produces one local 4x4
    ``UnitaryGate(W_dag)`` on the corresponding pair of physical qubits.
    """
    ClassicalRegister, UnitaryGate = _load_qiskit_types()
    labels = tuple(global_setting)
    pairs = _normalize_qutrit_qubits(qutrit_qubits)
    if len(labels) != len(pairs):
        raise ValueError("global_setting and qutrit_qubits must have the same length")

    qc = state_circuit if inplace else state_circuit.copy()
    _validate_qubits_exist(qc, pairs)

    local_gammas: list[complex | None] = []
    local_basis_gates: list[dict[str, Any] | None] = []

    for label, qubits in zip(labels, pairs):
        if label is None:
            local_gammas.append(None)
            local_basis_gates.append(None)
            continue

        observable = observable_from_label(label)
        _, W_dag, _, gamma = local_measurement_basis_unitary(
            E,
            observable,
            d=d,
        )
        gate_label = f"meas_{label}"
        gate = UnitaryGate(np.asarray(W_dag, dtype=complex), label=gate_label)
        qc.append(gate, list(qubits))

        local_gammas.append(gamma)
        local_basis_gates.append(
            {
                "setting_label": label,
                "gate_label": gate_label,
                "qubits": qubits,
                "unitary": W_dag,
            }
        )

    classical_bits_by_qutrit: list[tuple[int, int]] = []
    if add_measurements:
        base_classical_index = len(qc.clbits)
        register_name = _unique_creg_name(
            qc,
            classical_register_name or "qutrit_meas",
        )
        creg = ClassicalRegister(2 * len(pairs), register_name)
        qc.add_register(creg)
        for qutrit_index, qubits in enumerate(pairs):
            local_indices: list[int] = []
            for offset, qubit in enumerate(qubits):
                creg_index = 2 * qutrit_index + offset
                qc.measure(qubit, creg[creg_index])
                local_indices.append(base_classical_index + creg_index)
            classical_bits_by_qutrit.append((local_indices[0], local_indices[1]))

    metadata: dict[str, Any] = {
        "global_setting": labels,
        "qutrit_qubits": pairs,
        "local_gammas": local_gammas,
        "local_basis_gates": local_basis_gates,
        "classical_bits_by_qutrit": classical_bits_by_qutrit,
    }
    return qc, metadata


def _load_qiskit_types() -> tuple[type, type]:
    try:
        from qiskit import ClassicalRegister
        from qiskit.circuit.library import UnitaryGate
    except Exception as exc:
        raise ImportError(
            "append_measurement_for_global_setting requires qiskit to be installed"
        ) from exc
    return ClassicalRegister, UnitaryGate


def _normalize_qutrit_qubits(
    qutrit_qubits: Sequence[Sequence[int]],
) -> tuple[tuple[int, int], ...]:
    pairs: list[tuple[int, int]] = []
    used: set[int] = set()
    for pair in qutrit_qubits:
        if len(pair) != 2:
            raise ValueError("each qutrit must be represented by exactly two qubits")
        q0, q1 = int(pair[0]), int(pair[1])
        if q0 == q1:
            raise ValueError("a qutrit pair cannot use the same qubit twice")
        if q0 in used or q1 in used:
            raise ValueError("qutrit qubit pairs must be disjoint")
        used.update((q0, q1))
        pairs.append((q0, q1))
    return tuple(pairs)


def _validate_qubits_exist(qc: "QuantumCircuit", pairs: Sequence[tuple[int, int]]) -> None:
    for pair in pairs:
        for qubit in pair:
            if qubit < 0 or qubit >= qc.num_qubits:
                raise ValueError(f"qubit index {qubit} is out of range")


def _unique_creg_name(qc: "QuantumCircuit", requested_name: str) -> str:
    existing = {register.name for register in qc.cregs}
    if requested_name not in existing:
        return requested_name
    suffix = 1
    while f"{requested_name}_{suffix}" in existing:
        suffix += 1
    return f"{requested_name}_{suffix}"


def _to_qiskit_two_qubit_order(matrix: np.ndarray) -> np.ndarray:
    """Convert |00>, |01>, |10>, |11> order to Qiskit's [q0, q1] order."""
    permutation = [0, 2, 1, 3]
    P = np.eye(4, dtype=complex)[permutation, :]
    return P @ np.asarray(matrix, dtype=complex) @ P.T
