from dataclasses import FrozenInstanceError
import unittest

import numpy as np

from qudits_on_qubits.reference_experiments import (
    EncodingSpec,
    LogicalStateSpec,
    get_encoding,
)


class ReferenceExperimentsTests(unittest.TestCase):
    def test_canonical_encoding_has_expected_code_and_leakage_geometry(self) -> None:
        encoding = get_encoding("canonical_ez")

        expected = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 0],
            ],
            dtype=complex,
        )
        matrix = encoding.as_array()
        leakage = encoding.leakage_array()

        self.assertEqual(matrix.shape, (4, 3))
        np.testing.assert_array_equal(matrix, expected)
        np.testing.assert_allclose(matrix.conj().T @ matrix, np.eye(3))
        self.assertEqual(leakage.shape, (4, 1))
        np.testing.assert_array_equal(
            leakage,
            np.array([[0], [0], [0], [1]], dtype=complex),
        )
        np.testing.assert_allclose(matrix.conj().T @ leakage, np.zeros((3, 1)))

    def test_encoding_arrays_are_defensive_copies(self) -> None:
        encoding = get_encoding("canonical_ez")
        matrix = encoding.as_array()
        leakage = encoding.leakage_array()

        matrix[0, 0] = 7
        leakage[3, 0] = 7

        self.assertEqual(encoding.as_array()[0, 0], 1)
        self.assertEqual(encoding.leakage_array()[3, 0], 1)

    def test_encoding_spec_is_frozen(self) -> None:
        encoding = get_encoding("canonical_ez")

        with self.assertRaises(FrozenInstanceError):
            encoding.encoding_id = "changed"

    def test_encoding_rejects_non_isometric_code_matrix(self) -> None:
        with self.assertRaisesRegex(ValueError, r"E\^dagger E"):
            EncodingSpec(
                encoding_id="bad",
                logical_dimension=3,
                physical_qubits_per_qutrit=2,
                isometry=(
                    (1, 0, 0),
                    (1, 0, 0),
                    (0, 0, 1),
                    (0, 0, 0),
                ),
                leakage_basis=((0,), (0,), (0,), (1,)),
            )

    def test_weighted_graph_state_has_expected_analytical_amplitudes(self) -> None:
        state_spec = LogicalStateSpec("test", 3, 2, (0, 1), ((0, 1, 1),))

        state = state_spec.statevector()
        omega = np.exp(2j * np.pi / 3)

        self.assertEqual(state.shape, (9,))
        np.testing.assert_allclose(np.linalg.norm(state), 1)
        np.testing.assert_allclose(state[0], 1 / 3)
        np.testing.assert_allclose(state[5], omega**2 / 3)

    def test_legacy_edges_repeat_weight_two_edge(self) -> None:
        state_spec = LogicalStateSpec("test", 3, 2, (0, 1), ((0, 1, 2),))

        self.assertEqual(state_spec.legacy_edges(), ((0, 1), (0, 1)))

    def test_logical_state_rejects_invalid_party_and_edge_metadata(self) -> None:
        with self.assertRaisesRegex(ValueError, "party_order"):
            LogicalStateSpec("bad-order", 3, 2, (0, 0), ((0, 1, 1),))

        with self.assertRaisesRegex(ValueError, "duplicate"):
            LogicalStateSpec(
                "duplicate-edge",
                3,
                2,
                (0, 1),
                ((0, 1, 1), (0, 1, 2)),
            )


if __name__ == "__main__":
    unittest.main()
