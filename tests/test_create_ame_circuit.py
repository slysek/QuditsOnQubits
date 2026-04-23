import unittest

import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Operator

from QuditsOnQubits.benchmark_encoding_bases import BASIS_GATES, COUPLING_MAP
from QuditsOnQubits.create_ame_circuit import (
    _build_encoding_change_circuits,
    create_ame_circuit,
)


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
    def test_build_encoding_change_circuits_return_w_and_wdag_blocks(self):
        e_new = _sample_encoding()
        w, w_qc, wdag_qc = _build_encoding_change_circuits(e_new)
        self.assertTrue(Operator(w_qc).equiv(Operator(w)))
        self.assertTrue(Operator(wdag_qc).equiv(Operator(w.conj().T)))

    def test_create_ame_circuit_with_encoding_change_matches_final_w_layer_model(self):
        e_new = _sample_encoding()
        qc, graph = create_ame_circuit(
            n=2,
            dim=3,
            graph_type="star",
            E_new=e_new,
        )
        baseline_qc, _ = create_ame_circuit(n=2, dim=3, graph_type="star", E_new=None)
        _, w_qc, _ = _build_encoding_change_circuits(e_new)

        expected_qc = baseline_qc.copy()
        expected_qc.append(w_qc, [0, 1])
        expected_qc.append(w_qc, [2, 3])

        self.assertEqual(graph.vcount(), 2)
        self.assertEqual(qc.num_qubits, 4)
        self.assertGreater(qc.size(), 0)
        self.assertTrue(np.allclose(Operator(qc).data, Operator(expected_qc).data))

    def test_create_ame_circuit_with_encoding_change_places_w_only_at_end(self):
        qc, _ = create_ame_circuit(
            n=2,
            dim=3,
            graph_type="star",
            E_new=_sample_encoding(),
        )
        op_names = [instruction.operation.name for instruction in qc.data]

        self.assertEqual(op_names[-2:], ["W", "W"])
        self.assertNotIn("W", op_names[:-2])

    def test_identity_encoding_change_transpiles_like_baseline(self):
        e_old = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 0],
            ],
            dtype=complex,
        )

        qc_identity, _ = create_ame_circuit(
            n=2,
            dim=3,
            graph_type="star",
            E_new=e_old,
        )
        baseline_qc, _ = create_ame_circuit(
            n=2,
            dim=3,
            graph_type="star",
            E_new=None,
        )
        baseline_t = transpile(
            baseline_qc,
            basis_gates=BASIS_GATES,
            coupling_map=COUPLING_MAP,
            optimization_level=3,
            seed_transpiler=0,
        )
        identity_t = transpile(
            qc_identity,
            basis_gates=BASIS_GATES,
            coupling_map=COUPLING_MAP,
            optimization_level=3,
            seed_transpiler=0,
        )

        self.assertEqual(
            dict(identity_t.count_ops()),
            dict(baseline_t.count_ops()),
        )
        self.assertEqual(identity_t.depth(), baseline_t.depth())
        self.assertEqual(identity_t.size(), baseline_t.size())


if __name__ == "__main__":
    unittest.main()
