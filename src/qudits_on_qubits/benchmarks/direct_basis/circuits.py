from __future__ import annotations

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import DiagonalGate, StatePreparation, UnitaryGate

from qudits_on_qubits.benchmarks.direct_basis.math_utils import (
    encoding_embedding,
    local_plus_state_in_direct_basis,
    physical_single_qutrit_gate_in_encoding,
    physical_two_qutrit_gate_in_encoding,
    qutrit_cz,
    qutrit_fourier,
)
from qudits_on_qubits.encoding_search.states import BenchmarkStateSpec, resolve_benchmark_state


def resolve_direct_state(state_name: str, n_qutrits: int | None = None) -> BenchmarkStateSpec:
    return resolve_benchmark_state(state_name, n_qutrits=n_qutrits)


def build_direct_basis_local_preparation(encoding: np.ndarray) -> StatePreparation:
    """Build a 2-qubit preparation of E_new @ |+> from |00>."""
    local_state = local_plus_state_in_direct_basis(encoding)
    return StatePreparation(local_state, label="plus_W")


def build_direct_basis_edge_gate(
    encoding: np.ndarray,
    *,
    power: int = 1,
) -> DiagonalGate | UnitaryGate:
    """Build the four-qubit encoded qutrit CZ gate."""
    if isinstance(power, bool) or not isinstance(power, int):
        raise ValueError("power must be an integer.")
    normalized_power = power % 3
    embedded = physical_two_qutrit_gate_in_encoding(
        np.linalg.matrix_power(qutrit_cz(), normalized_power),
        encoding,
    )
    diagonal = np.diag(embedded)
    if np.array_equal(embedded, np.diag(diagonal)):
        gate = DiagonalGate(diagonal)
        gate.label = "CZ_W"
        return gate
    gate = UnitaryGate(embedded, label="CZ_W")
    gate.name = "CZ_W"
    return gate


def build_direct_basis_fourier_gate(
    encoding: np.ndarray,
    *,
    leakage_phase: float = 0.0,
) -> UnitaryGate:
    """Build encoded F3 with a phase on the unused-state complement."""
    embedded = physical_single_qutrit_gate_in_encoding(
        qutrit_fourier(), encoding, leakage_phase=leakage_phase
    )
    gate = UnitaryGate(embedded, label="F3_W")
    gate.name = "F3_W"
    return gate


def gate_as_circuit(gate: UnitaryGate, num_qubits: int, name: str) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits, name=name)
    qc.append(gate, list(range(num_qubits)))
    return qc


def build_direct_basis_graph_state_circuit(
    state_name: str,
    basis_matrix: np.ndarray,
    *,
    n_qutrits: int | None = None,
) -> QuantumCircuit:
    """Build graph-state preparation directly in the requested qutrit encoding.

    This is intentionally not the legacy "append W as a physical
    basis-change gate" circuit.  The local target is E_new @ |+>, and every
    edge uses the physical embedding of logical qutrit CZ in that encoding.
    """
    state = resolve_direct_state(state_name, n_qutrits=n_qutrits)
    qubit_pairs = [[2 * idx, 2 * idx + 1] for idx in range(state.num_qutrits)]

    qc = QuantumCircuit(2 * state.num_qutrits, name=f"{state.state_id}_direct_basis")
    local_prep = build_direct_basis_local_preparation(basis_matrix)
    edge_gate = build_direct_basis_edge_gate(basis_matrix)

    for pair in qubit_pairs:
        qc.append(local_prep, pair)

    for left, right in state.edges:
        if left == right:
            continue
        # Logical statevectors use left-to-right tensor axes, while Qiskit
        # numbers physical qubit blocks from the little-endian end.
        left_pair = qubit_pairs[state.num_qutrits - 1 - left]
        right_pair = qubit_pairs[state.num_qutrits - 1 - right]
        qc.append(
            edge_gate,
            [
                left_pair[0],
                left_pair[1],
                right_pair[0],
                right_pair[1],
            ],
        )

    return qc


def build_direct_basis_fourier_graph_state_circuit(
    state_name: str,
    basis_matrix: np.ndarray,
    *,
    leakage_phase: float,
    n_qutrits: int | None = None,
) -> QuantumCircuit:
    """Prepare a graph state with explicit encoded F3 gates on every qutrit.

    Preparing E|0> first is essential: in a noncanonical monomial encoding,
    the physical |00> input need not represent logical zero (or even be coded).
    """
    state = resolve_direct_state(state_name, n_qutrits=n_qutrits)
    qubit_pairs = [[2 * index, 2 * index + 1] for index in range(state.num_qutrits)]
    qc = QuantumCircuit(
        2 * state.num_qutrits, name=f"{state.state_id}_direct_basis_f3"
    )
    encoded_zero = encoding_embedding(basis_matrix)[:, 0]
    zero_preparation = StatePreparation(encoded_zero, label="zero_W")
    fourier = build_direct_basis_fourier_gate(
        basis_matrix, leakage_phase=leakage_phase
    )
    edge_gate = build_direct_basis_edge_gate(basis_matrix)
    for pair in qubit_pairs:
        qc.append(zero_preparation, pair)
        qc.append(fourier, pair)
    for left, right in state.edges:
        if left == right:
            continue
        # Match the historical builder's logical/physical block ordering.
        left_pair = qubit_pairs[state.num_qutrits - 1 - left]
        right_pair = qubit_pairs[state.num_qutrits - 1 - right]
        qc.append(
            edge_gate,
            [left_pair[0], left_pair[1], right_pair[0], right_pair[1]],
        )
    return qc


def build_exact_optimized_direct_basis_graph_state_circuit(
    state_name: str,
    basis_matrix: np.ndarray,
    *,
    n_qutrits: int | None = None,
) -> QuantumCircuit:
    """Build an exactly equivalent graph state with weighted edge layers."""
    state = resolve_direct_state(state_name, n_qutrits=n_qutrits)
    qubit_pairs = [[2 * idx, 2 * idx + 1] for idx in range(state.num_qutrits)]

    qc = QuantumCircuit(
        2 * state.num_qutrits,
        name=f"{state.state_id}_direct_basis_exact_optimized",
    )
    local_prep = build_direct_basis_local_preparation(basis_matrix)
    for pair in qubit_pairs:
        qc.append(local_prep, pair)

    weighted_edges: dict[tuple[int, int], int] = {}
    for left, right in state.edges:
        if left == right:
            continue
        edge = tuple(sorted((left, right)))
        weighted_edges[edge] = (weighted_edges.get(edge, 0) + 1) % 3

    layers: list[list[tuple[int, int, int]]] = []
    for (left, right), power in weighted_edges.items():
        if power == 0:
            continue
        for layer in layers:
            if all(left not in edge[:2] and right not in edge[:2] for edge in layer):
                layer.append((left, right, power))
                break
        else:
            layers.append([(left, right, power)])

    edge_gates = {
        power: build_direct_basis_edge_gate(basis_matrix, power=power)
        for power in {power for layer in layers for _, _, power in layer}
    }
    for layer in layers:
        for left, right, power in layer:
            left_pair = qubit_pairs[state.num_qutrits - 1 - left]
            right_pair = qubit_pairs[state.num_qutrits - 1 - right]
            qc.append(
                edge_gates[power],
                [left_pair[0], left_pair[1], right_pair[0], right_pair[1]],
            )

    return qc
