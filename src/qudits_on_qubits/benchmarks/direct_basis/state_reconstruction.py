"""Logical-state reconstruction helpers independent of benchmark execution."""

from __future__ import annotations

import numpy as np
from qiskit.circuit import ControlFlowOp
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.quantum_info import DensityMatrix, Operator, Statevector


def _strip_idle_qubits_with_indices(qc):
    no_measurements = qc.remove_final_measurements(inplace=False)
    dag = circuit_to_dag(no_measurements)
    idle_qubits = [wire for wire in dag.idle_wires() if wire in dag.qubits]
    active_original_indices = [
        no_measurements.find_bit(qubit).index
        for qubit in dag.qubits
        if qubit not in idle_qubits
    ]
    if idle_qubits:
        dag.remove_qubits(*idle_qubits)
    return dag_to_circuit(dag), active_original_indices


def _restore_stripped_input_order(
    state: np.ndarray,
    candidate_qc,
    active_indices: list[int],
    reference_qubits: int,
) -> np.ndarray:
    input_to_output = _input_to_active_output_order(
        candidate_qc,
        active_indices,
        reference_qubits,
    )
    if input_to_output is None:
        return state
    return _restore_input_qubit_order(state, input_to_output)


def _input_to_active_output_order(
    candidate_qc,
    active_indices: list[int],
    reference_qubits: int,
) -> list[int] | None:
    layout = getattr(candidate_qc, "layout", None)
    if layout is None:
        return None
    try:
        final_index_layout = layout.final_index_layout(filter_ancillas=True)
    except Exception:
        return None
    if len(final_index_layout) != reference_qubits:
        return None
    if (
        any(
            not isinstance(index, (int, np.integer))
            or isinstance(index, (bool, np.bool_))
            or index < 0
            or index >= candidate_qc.num_qubits
            for index in final_index_layout
        )
        or len(set(final_index_layout)) != len(final_index_layout)
        or any(
            not isinstance(index, (int, np.integer))
            or isinstance(index, (bool, np.bool_))
            or index < 0
            or index >= candidate_qc.num_qubits
            for index in active_indices
        )
        or len(set(active_indices)) != len(active_indices)
    ):
        return None
    active_positions = {
        physical_index: index for index, physical_index in enumerate(active_indices)
    }
    if not all(physical_index in active_positions for physical_index in final_index_layout):
        return None
    translated = [active_positions[physical_index] for physical_index in final_index_layout]
    if len(set(translated)) != len(translated):
        return None
    return translated


def _density_matrix_in_input_order(
    state: np.ndarray,
    input_to_output: list[int],
) -> np.ndarray:
    total_qubits = int(np.log2(len(state)))
    if 2**total_qubits != len(state):
        raise ValueError("state length must be a power of two")
    kept_positions = tuple(int(position) for position in input_to_output)
    if len(set(kept_positions)) != len(kept_positions):
        raise ValueError("input_to_output contains duplicate output positions")
    traced_positions = tuple(
        position for position in range(total_qubits) if position not in set(kept_positions)
    )
    dim = 2 ** len(kept_positions)
    groups: dict[int, list[tuple[int, complex]]] = {}
    for basis_index, amplitude in enumerate(np.asarray(state, dtype=complex)):
        traced_index = _bits_to_index(basis_index, traced_positions)
        kept_index = _bits_to_index(basis_index, kept_positions)
        groups.setdefault(traced_index, []).append((kept_index, amplitude))

    density = np.zeros((dim, dim), dtype=complex)
    for amplitudes in groups.values():
        vector = np.zeros(dim, dtype=complex)
        for kept_index, amplitude in amplitudes:
            vector[kept_index] = amplitude
        density += np.outer(vector, vector.conj())
    return density


def _bits_to_index(source_index: int, positions: tuple[int, ...]) -> int:
    target_index = 0
    for target_bit, source_bit in enumerate(positions):
        target_index |= ((source_index >> source_bit) & 1) << target_bit
    return target_index


def _restore_input_qubit_order(state: np.ndarray, input_to_output: list[int]) -> np.ndarray:
    """Reorder a statevector from final physical order back to input logical order."""
    n_qubits = len(input_to_output)
    restored = np.zeros_like(state)
    for input_index in range(2**n_qubits):
        input_bits = [(input_index >> bit) & 1 for bit in range(n_qubits)]
        output_bits = [0] * n_qubits
        for input_bit, output_bit in enumerate(input_to_output):
            output_bits[output_bit] = input_bits[input_bit]
        output_index = sum(bit_value << bit for bit, bit_value in enumerate(output_bits))
        restored[input_index] = state[output_index]
    return restored


def _unsafe_statevector_operation(circuit) -> str | None:
    for instruction in circuit.data:
        operation = instruction.operation
        name = str(operation.name)
        if name == "measure":
            return "circuit contains measurement operations"
        if name == "reset":
            return (
                "circuit contains reset operations, which are non-unitary and "
                "unsafe for deterministic statevector reconstruction"
            )
        if isinstance(operation, ControlFlowOp):
            return f"circuit contains control-flow operation {name!r}"
        if getattr(operation, "_condition", None) is not None:
            return f"circuit contains classically controlled operation {name!r}"
        if instruction.clbits or getattr(operation, "num_clbits", 0):
            return f"circuit contains classical-data operation {name!r}"
        try:
            operation_is_unitary = Operator(operation).is_unitary()
        except Exception:
            operation_is_unitary = False
        if not operation_is_unitary:
            return f"circuit contains non-unitary or unsupported operation {name!r}"
    return None


def logical_output_state(
    circuit,
    *,
    logical_qubit_count: int,
    max_qubits: int,
) -> tuple[Statevector | DensityMatrix | None, str]:
    """Reconstruct a compiled circuit's logical output state without execution."""
    notes: list[str] = []
    if type(logical_qubit_count) is not int or logical_qubit_count < 0:
        return None, "Logical state reconstruction requires a non-negative logical_qubit_count."
    if type(max_qubits) is not int or max_qubits < 0:
        return None, "Logical state reconstruction requires a non-negative max_qubits."
    unsafe_operation = _unsafe_statevector_operation(circuit)
    if unsafe_operation is not None:
        return None, f"Fidelity skipped because {unsafe_operation}."

    fidelity_qc = circuit
    active_indices = list(range(circuit.num_qubits))
    if fidelity_qc.num_qubits > int(max_qubits):
        stripped_qc, stripped_active_indices = _strip_idle_qubits_with_indices(fidelity_qc)
        if stripped_qc.num_qubits < fidelity_qc.num_qubits:
            notes.append(
                f"Idle qubits stripped for fidelity: {fidelity_qc.num_qubits}->{stripped_qc.num_qubits}."
            )
            fidelity_qc = stripped_qc
            active_indices = stripped_active_indices
    if fidelity_qc.num_qubits > int(max_qubits):
        notes.append(f"Fidelity skipped because transpiled circuit has {fidelity_qc.num_qubits} qubits.")
        return None, " ".join(notes)
    if fidelity_qc.num_qubits < logical_qubit_count:
        notes.append(
            f"Fidelity skipped because active qubit count {fidelity_qc.num_qubits} "
            f"differs from reference {logical_qubit_count}."
        )
        return None, " ".join(notes)

    try:
        candidate = Statevector.from_instruction(fidelity_qc)
        logical_dims = (2,) * logical_qubit_count
        if fidelity_qc.num_qubits > logical_qubit_count:
            input_to_output = _input_to_active_output_order(
                circuit,
                active_indices,
                logical_qubit_count,
            )
            if input_to_output is None:
                notes.append(
                    f"Fidelity skipped because active qubit count {fidelity_qc.num_qubits} "
                    f"differs from reference {logical_qubit_count}."
                )
                return None, " ".join(notes)
            candidate_density = DensityMatrix(
                _density_matrix_in_input_order(candidate.data, input_to_output),
                dims=logical_dims,
            )
            notes.append(
                "Extra active qubits traced for fidelity: "
                f"{fidelity_qc.num_qubits}->{logical_qubit_count}."
            )
            return candidate_density, " ".join(notes)

        candidate_data = candidate.data
        if fidelity_qc is circuit:
            layout = getattr(circuit, "layout", None)
            if layout is not None:
                final_index_layout = layout.final_index_layout(filter_ancillas=True)
                valid_permutation = (
                    len(final_index_layout) == logical_qubit_count == circuit.num_qubits
                    and all(
                        isinstance(index, (int, np.integer))
                        and not isinstance(index, (bool, np.bool_))
                        for index in final_index_layout
                    )
                    and set(final_index_layout) == set(range(logical_qubit_count))
                )
                if valid_permutation:
                    candidate_data = _restore_input_qubit_order(candidate_data, final_index_layout)
        else:
            candidate_data = _restore_stripped_input_order(
                candidate_data,
                circuit,
                active_indices,
                logical_qubit_count,
            )
        return Statevector(candidate_data, dims=logical_dims), " ".join(notes)
    except Exception as exc:
        notes.append(f"Fidelity skipped: {exc}")
        return None, " ".join(notes)


def logical_output_density_matrix(
    circuit,
    *,
    logical_qubit_count: int,
    max_qubits: int,
) -> tuple[DensityMatrix | None, str]:
    """Compatibility wrapper returning the logical state as a density matrix."""
    state, notes = logical_output_state(
        circuit,
        logical_qubit_count=logical_qubit_count,
        max_qubits=max_qubits,
    )
    if state is None or isinstance(state, DensityMatrix):
        return state, notes
    return DensityMatrix(state), notes
