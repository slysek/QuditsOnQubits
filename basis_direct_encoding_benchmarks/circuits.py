from __future__ import annotations

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import StatePreparation, UnitaryGate

from basis_direct_encoding_benchmarks.math_utils import (
    conjugated_qutrit_fourier,
    conjugated_qutrit_cz,
    embed_single_qutrit_gate_identity_leakage,
    embed_two_qutrit_gate_identity_leakage,
    local_plus_state_in_direct_basis,
    validate_unitary,
)
from encoding_search_v2.states import BenchmarkStateSpec, resolve_benchmark_state


def resolve_direct_state(state_name: str, n_qutrits: int | None = None) -> BenchmarkStateSpec:
    return resolve_benchmark_state(state_name, n_qutrits=n_qutrits)


def build_direct_basis_local_preparation(w: np.ndarray) -> StatePreparation:
    """Build a 2-qubit preparation of E_Z @ W @ |+> from |00>."""
    local_state = local_plus_state_in_direct_basis(w)
    return StatePreparation(local_state, label="plus_W")


def build_direct_basis_edge_gate(w: np.ndarray) -> UnitaryGate:
    """Build the four-qubit embedded CZ^(W) gate with identity leakage."""
    cz_w = conjugated_qutrit_cz(w)
    embedded = embed_two_qutrit_gate_identity_leakage(cz_w)
    gate = UnitaryGate(embedded, label="CZ_W")
    gate.name = "CZ_W"
    return gate


def build_direct_basis_fourier_gate(w: np.ndarray) -> UnitaryGate:
    """Build the two-qubit embedded F3^(W) gate with identity leakage."""
    f3_w = conjugated_qutrit_fourier(w)
    embedded = embed_single_qutrit_gate_identity_leakage(f3_w)
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
    """Build graph-state preparation directly in the W-defined qutrit basis.

    This is intentionally not the legacy "append W as a physical
    basis-change gate" circuit.  The local target is E_Z @ W @ |+>, and every
    edge uses the physical embedding of (W kron W) CZ (W^dag kron W^dag).
    """
    w = validate_unitary(basis_matrix, 3, name="basis_matrix")
    state = resolve_direct_state(state_name, n_qutrits=n_qutrits)
    qubit_pairs = [[2 * idx, 2 * idx + 1] for idx in range(state.num_qutrits)]

    qc = QuantumCircuit(2 * state.num_qutrits, name=f"{state.state_id}_direct_basis")
    local_prep = build_direct_basis_local_preparation(w)
    edge_gate = build_direct_basis_edge_gate(w)

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
