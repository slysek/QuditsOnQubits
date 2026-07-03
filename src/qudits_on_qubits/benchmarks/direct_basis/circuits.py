from __future__ import annotations

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import StatePreparation, UnitaryGate

from qudits_on_qubits.benchmarks.direct_basis.math_utils import (
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


def build_direct_basis_edge_gate(encoding: np.ndarray) -> UnitaryGate:
    """Build the four-qubit encoded qutrit CZ gate."""
    embedded = physical_two_qutrit_gate_in_encoding(qutrit_cz(), encoding)
    gate = UnitaryGate(embedded, label="CZ_W")
    gate.name = "CZ_W"
    return gate


def build_direct_basis_fourier_gate(encoding: np.ndarray) -> UnitaryGate:
    """Build the two-qubit encoded qutrit F3 gate."""
    embedded = physical_single_qutrit_gate_in_encoding(qutrit_fourier(), encoding)
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
        qc.append(
            edge_gate,
            [
                qubit_pairs[left][0],
                qubit_pairs[left][1],
                qubit_pairs[right][0],
                qubit_pairs[right][1],
            ],
        )

    return qc
