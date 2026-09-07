import io

import numpy as np
from qiskit import qpy

from qudits_on_qubits.benchmarks.direct_basis.circuit_serialization import normalize_circuit_bit_indices
from qudits_on_qubits.benchmarks.direct_basis.circuits import build_direct_basis_graph_state_circuit
from qudits_on_qubits.benchmarks.direct_basis.iqm_backend import build_iqm_pass_manager


def test_normalization_preserves_source_and_qpy_wire_mapping():
    from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet

    source = build_direct_basis_graph_state_circuit('ghz3', np.eye(3), n_qutrits=3)
    circuit = build_iqm_pass_manager(IQMFakeGarnet(), seed_transpiler=18).run(source)
    circuit.measure_all()
    circuit.metadata = {'diagnostic': [18]}
    with circuit.if_test((circuit.clbits[0], True)):
        circuit.x(circuit.qubits[0])
    before = [circuit.find_bit(bit).index for bit in circuit.qubits]

    def instructions(qc):
        positions = {bit: i for i, bit in enumerate(qc.qubits)}
        return [(inst.operation.name, tuple(positions[b] for b in inst.qubits),
                 tuple(qc.clbits.index(b) for b in inst.clbits)) for inst in qc.data]

    fixed = normalize_circuit_bit_indices(circuit)
    assert instructions(fixed) == instructions(circuit)
    assert fixed.metadata == circuit.metadata
    assert fixed.cregs == circuit.cregs
    assert [fixed.find_bit(bit).index for bit in fixed.qubits] == list(range(circuit.num_qubits))
    assert [circuit.find_bit(bit).index for bit in circuit.qubits] == before
    assert normalize_circuit_bit_indices(fixed) is fixed
    stream = io.BytesIO()
    qpy.dump(fixed, stream)
    stream.seek(0)
    loaded = qpy.load(stream)[0]
    assert instructions(loaded) == instructions(fixed)
    assert loaded.layout.final_index_layout() == fixed.layout.final_index_layout()
