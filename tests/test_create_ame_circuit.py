import os
import unittest

import numpy as np
from qiskit import qpy
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Operator

from QuditsOnQubits.create_ame_circuit import (
    _append_encoded_czgate,
    _append_encoded_fgate,
    _build_encoding_change_circuits,
    create_ame_circuit,
)


def _repo_root():
    return os.path.dirname(os.path.dirname(__file__))


def _load_gate(relative_path):
    with open(os.path.join(_repo_root(), relative_path), "rb") as fd:
        return qpy.load(fd)[0]


def _sample_encoding():
    amp = 1.0 / np.sqrt(2.0)
    return np.array(
        [
            [amp, amp, 0],
            [amp, -amp, 0],
            [0, 0, 1],
            [0, 0, 0],
        ],
        dtype=complex,
    )


class EncodedAmeCircuitTests(unittest.TestCase):
    def test_encoded_blocks_match_previous_matrix_conjugation(self):
        e_new = _sample_encoding()
        w, w_qc, wdag_qc = _build_encoding_change_circuits(e_new)
        wdag = w.conj().T

        fgate = _load_gate("quantum_circuits/Fgate3.qpy")
        czgate = _load_gate("quantum_circuits/CZgate3.qpy")

        fgate_old = w @ Operator(fgate).data @ wdag
        czgate_old = np.kron(w, w) @ Operator(czgate).data @ np.kron(wdag, wdag)

        f_qc = QuantumCircuit(2)
        _append_encoded_fgate(f_qc, [0, 1], fgate, w_qc, wdag_qc)

        cz_qc = QuantumCircuit(4)
        _append_encoded_czgate(cz_qc, [0, 1, 2, 3], czgate, w_qc, wdag_qc)

        self.assertTrue(np.allclose(Operator(f_qc).data, fgate_old))
        self.assertTrue(np.allclose(Operator(cz_qc).data, czgate_old))

    def test_create_ame_circuit_with_encoding_change_builds_valid_circuit(self):
        qc, graph = create_ame_circuit(
            n=2,
            dim=3,
            graph_type="star",
            E_new=_sample_encoding(),
        )

        self.assertEqual(graph.vcount(), 2)
        self.assertEqual(qc.num_qubits, 4)
        self.assertGreater(qc.size(), 0)


if __name__ == "__main__":
    unittest.main()
