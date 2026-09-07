"""Spectator readout must use the same leakage convention as other parties."""
import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector

from qudits_on_qubits.bell_measurements.qiskit_measurements import append_measurement_for_global_setting


@pytest.mark.parametrize("unused", range(4))
def test_identity_spectator_decodes_every_codeword_and_leakage(unused):
    e = np.eye(4, dtype=complex)[:, [i for i in range(4) if i != unused]]
    qc, meta = append_measurement_for_global_setting(
        QuantumCircuit(2), [None], [(0,1)], e,
        lambda _: pytest.fail("spectator has no observable label"), add_measurements=False,
    )
    unitary = Operator(qc).data
    assert np.linalg.norm((unitary @ e)[3]) < 1e-12
    assert abs((unitary @ np.eye(4)[:,unused])[3]) == pytest.approx(1)
    assert (meta["local_basis_gates"][0] is None) == (unused == 3)
    assert all(len(inst.qubits) == 1 for inst in qc.data)


def test_identity_spectator_decodes_dense_isometry():
    rng = np.random.default_rng(31)
    u, _ = np.linalg.qr(rng.normal(size=(4,4))+1j*rng.normal(size=(4,4)))
    qc, _ = append_measurement_for_global_setting(QuantumCircuit(2), [None], [(0,1)], u[:,:3], lambda _: None, add_measurements=False)
    assert np.linalg.norm((Operator(qc).data @ u[:,:3])[3]) < 1e-12


def test_canonical_spectator_retains_zero_gate_cost():
    e = np.eye(4,3)[:,[1,2,0]] @ np.diag(np.exp(1j*np.array([.1,.7,-.2])))
    qc, _ = append_measurement_for_global_setting(QuantumCircuit(2), [None], [(0,1)], e, lambda _: None, add_measurements=False)
    assert qc.size() == 0


@pytest.mark.parametrize("e", [np.ones((4,3)), np.eye(3), np.zeros((4,3))])
def test_invalid_spectator_encoding_rejected_before_inplace_mutation(e):
    qc = QuantumCircuit(2)
    with pytest.raises(ValueError):
        append_measurement_for_global_setting(qc,[None],[(0,1)],e,lambda _: None,inplace=True)
    assert qc.size() == 0
