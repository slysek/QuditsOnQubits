"""Keep compiled circuit bit indices consistent for metrics and QPY."""

from copy import deepcopy
from itertools import chain

from qiskit import QuantumCircuit


def normalize_circuit_bit_indices(circuit):
    """Rebuild stale Qiskit bit-index caches without changing wire order.

    Some IQM/Qiskit passes reorder ``qubits`` while ``find_bit`` still reports
    the old indices. QPY uses both representations, corrupting layout metadata.
    Rebuild only affected circuits, using their actual ordered wire lists.
    """
    if not isinstance(circuit, QuantumCircuit):
        return circuit
    if all(
        circuit.find_bit(bit).index == index
        for bits in (circuit.qubits, circuit.clbits)
        for index, bit in enumerate(bits)
    ):
        return circuit

    normalized = QuantumCircuit(
        circuit.qubits, circuit.clbits,
        name=circuit.name, global_phase=circuit.global_phase,
        metadata=deepcopy(circuit.metadata),
        inputs=circuit.iter_input_vars(),
        captures=chain(circuit.iter_captured_vars(), circuit.iter_captured_stretches()),
    )
    for register in chain(circuit.qregs, circuit.cregs):
        normalized.add_register(register)
    for variable in circuit.iter_declared_vars():
        normalized.add_uninitialized_var(variable)
    for stretch in circuit.iter_declared_stretches():
        normalized.add_stretch(stretch)
    for instruction in circuit.data:
        normalized.append(instruction)
    normalized._layout = deepcopy(circuit.layout)
    return normalized
