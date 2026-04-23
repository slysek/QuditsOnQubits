import unittest

import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Operator, Statevector

from QuditsOnQubits.benchmark_encoding_bases import BASIS_GATES, COUPLING_MAP
from QuditsOnQubits.create_ame_circuit import (
    VALID_ENCODING_STRATEGIES,
    _build_encoding_change_circuits,
    _build_conjugated_cz_block,
    _build_local_w_plus_preparation,
    _load_gates_for_dim,
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


class TestAppendWStrategyBackwardCompat(unittest.TestCase):
    """append_w strategy must produce identical circuits to the old default."""

    def test_append_w_explicit_matches_default(self):
        e_new = _sample_encoding()
        qc_default, _ = create_ame_circuit(n=2, dim=3, graph_type="star", E_new=e_new)
        qc_explicit, _ = create_ame_circuit(
            n=2, dim=3, graph_type="star", E_new=e_new, encoding_strategy="append_w",
        )
        self.assertTrue(np.allclose(Operator(qc_default).data, Operator(qc_explicit).data))

    def test_append_w_without_encoding_matches_baseline(self):
        qc_baseline, _ = create_ame_circuit(n=2, dim=3, graph_type="star", E_new=None)
        qc_explicit, _ = create_ame_circuit(
            n=2, dim=3, graph_type="star", E_new=None, encoding_strategy="append_w",
        )
        self.assertTrue(np.allclose(Operator(qc_baseline).data, Operator(qc_explicit).data))


class TestPreparedWStrategy(unittest.TestCase):
    """Tests for the prepared_w_then_conjugated_entanglers strategy."""

    def test_invalid_encoding_strategy_raises(self):
        with self.assertRaises(ValueError):
            create_ame_circuit(n=2, dim=3, encoding_strategy="invalid_mode")

    def test_prepared_w_without_e_new_matches_baseline(self):
        qc_baseline, _ = create_ame_circuit(n=2, dim=3, E_new=None, encoding_strategy="append_w")
        qc_prepared, _ = create_ame_circuit(
            n=2, dim=3, E_new=None,
            encoding_strategy="prepared_w_then_conjugated_entanglers",
        )
        self.assertTrue(np.allclose(
            Operator(qc_baseline).data, Operator(qc_prepared).data,
        ))

    def test_prepared_w_builds_circuit_with_correct_qubit_count(self):
        e_new = _sample_encoding()
        qc, graph = create_ame_circuit(
            n=3, dim=3, graph_type="star", E_new=e_new,
            encoding_strategy="prepared_w_then_conjugated_entanglers",
        )
        self.assertEqual(qc.num_qubits, 6)
        self.assertEqual(graph.vcount(), 3)

    def test_prepared_w_strategy_requires_dim3(self):
        with self.assertRaises(ValueError):
            create_ame_circuit(
                n=2, dim=4,
                encoding_strategy="prepared_w_then_conjugated_entanglers",
            )


class TestConjugatedCzBlock(unittest.TestCase):
    """Verify the (W⊗W) CZ (W†⊗W†) block structure."""

    def test_conjugated_cz_block_is_unitary(self):
        e_new = _sample_encoding()
        W, W_qc, Wdag_qc = _build_encoding_change_circuits(e_new)
        Fgate, CZgate = _load_gates_for_dim(3)

        block = _build_conjugated_cz_block(W_qc, Wdag_qc, CZgate)
        op = Operator(block)
        self.assertTrue(op.is_unitary())

    def test_conjugated_cz_block_equals_explicit_matrix(self):
        e_new = _sample_encoding()
        W, W_qc, Wdag_qc = _build_encoding_change_circuits(e_new)
        Fgate, CZgate = _load_gates_for_dim(3)

        block = _build_conjugated_cz_block(W_qc, Wdag_qc, CZgate)
        block_op = Operator(block).data

        CZ_mat = Operator(CZgate).data
        W_mat = W
        Wdag_mat = W.conj().T

        WW = np.kron(W_mat, W_mat)
        WdWd = np.kron(Wdag_mat, Wdag_mat)
        expected = WW @ CZ_mat @ WdWd

        self.assertTrue(np.allclose(block_op, expected, atol=1e-10))


class TestLocalWPlusPreparation(unittest.TestCase):
    def test_local_prep_produces_w_plus_state(self):
        e_new = _sample_encoding()
        W, _, _ = _build_encoding_change_circuits(e_new)
        Fgate, _ = _load_gates_for_dim(3)

        prep = _build_local_w_plus_preparation(W, Fgate)
        sv = Statevector.from_label("00").evolve(prep)

        F_op = Operator(Fgate).data
        plus_state = F_op @ np.array([1, 0, 0, 0], dtype=complex)
        expected = W @ plus_state
        expected = expected / np.linalg.norm(expected)

        fidelity = abs(np.vdot(expected, sv.data)) ** 2
        self.assertAlmostEqual(fidelity, 1.0, places=10)


if __name__ == "__main__":
    unittest.main()
