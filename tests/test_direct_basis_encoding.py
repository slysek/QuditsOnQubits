import os
import shutil
import unittest

import numpy as np
import pandas as pd
from qiskit import qpy
from qiskit.quantum_info import Statevector

from QuditsOnQubits.create_ame_circuit import create_ame_circuit
from basis_direct_encoding_benchmarks.benchmark import benchmark_direct_basis
from basis_direct_encoding_benchmarks.benchmark import benchmark_direct_basis_candidates
from basis_direct_encoding_benchmarks.candidates import (
    generate_sanity_basis_candidates,
)
from basis_direct_encoding_benchmarks.circuits import (
    build_direct_basis_graph_state_circuit,
)
from basis_direct_encoding_benchmarks.comparison import compare_old_vs_direct
from basis_direct_encoding_benchmarks.math_utils import (
    canonical_qutrit_embedding,
    code_subspace_indices,
    conjugated_qutrit_cz,
    direct_basis_embedding,
    embed_two_qutrit_gate_identity_leakage,
    is_unitary,
    qutrit_cz,
    qutrit_fourier,
    qutrit_plus_state,
)
from run_direct_basis_benchmarks import _load_candidates, build_parser


def _tmpdir():
    root = os.path.join(os.path.dirname(__file__), "_tmp_test_outputs")
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, "direct_basis_encoding_test")
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=False)
    return path


def _random_unitary_3(seed=123):
    rng = np.random.default_rng(seed)
    z = (rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))) / np.sqrt(2)
    q, r = np.linalg.qr(z)
    phases = np.diag(r)
    return q @ np.diag(phases / np.abs(phases))


class TestDirectBasisMath(unittest.TestCase):
    def test_direct_embedding_maps_w_basis_into_canonical_code_space(self):
        w = np.eye(3, dtype=complex)
        e_z = canonical_qutrit_embedding()
        e_w = direct_basis_embedding(w)
        plus_w = e_z @ w @ qutrit_plus_state()

        self.assertEqual(e_z.shape, (4, 3))
        self.assertTrue(np.allclose(e_w, e_z))
        self.assertTrue(np.allclose(plus_w, np.array([1, 1, 1, 0]) / np.sqrt(3)))

    def test_diagonal_w_commutes_with_qutrit_cz(self):
        omega = np.exp(2j * np.pi / 3)
        w = np.diag([1, omega, omega**2]).astype(complex)

        self.assertTrue(np.allclose(conjugated_qutrit_cz(w), qutrit_cz(), atol=1e-12))

    def test_permutation_w_keeps_cz_diagonal_with_permuted_phases(self):
        w = np.array(
            [
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
            ],
            dtype=complex,
        )

        cz_w = conjugated_qutrit_cz(w)
        off_diag = cz_w - np.diag(np.diag(cz_w))

        self.assertTrue(np.allclose(off_diag, 0.0, atol=1e-12))
        self.assertEqual(
            sorted(np.round(np.diag(cz_w), 12), key=lambda value: (value.real, value.imag)),
            sorted(np.round(np.diag(qutrit_cz()), 12), key=lambda value: (value.real, value.imag)),
        )

    def test_random_w_makes_dense_cz_and_identity_leakage_embedding(self):
        w = _random_unitary_3()
        cz_w = conjugated_qutrit_cz(w)
        embedded = embed_two_qutrit_gate_identity_leakage(cz_w)
        code = set(code_subspace_indices())
        leakage = [idx for idx in range(16) if idx not in code]

        self.assertTrue(is_unitary(cz_w))
        self.assertGreater(np.count_nonzero(np.abs(cz_w) > 1e-10), 20)
        self.assertTrue(is_unitary(embedded))

        for idx in leakage:
            expected_column = np.zeros(16, dtype=complex)
            expected_column[idx] = 1.0
            self.assertTrue(np.allclose(embedded[:, idx], expected_column, atol=1e-12))
            self.assertTrue(np.allclose(embedded[idx, :], expected_column, atol=1e-12))


class TestDirectBasisCircuit(unittest.TestCase):
    def test_identity_direct_basis_matches_standard_z_graph_state(self):
        direct_qc = build_direct_basis_graph_state_circuit("two_qutrit", np.eye(3))
        baseline_qc, _ = create_ame_circuit(
            n=2,
            dim=3,
            graph_type="star",
            E_new=None,
        )

        direct_state = Statevector.from_instruction(direct_qc).data
        baseline_state = Statevector.from_instruction(baseline_qc).data

        fidelity = abs(np.vdot(baseline_state, direct_state)) ** 2
        self.assertAlmostEqual(fidelity, 1.0, places=10)

    def test_sanity_candidate_set_contains_requested_small_bases(self):
        candidates = generate_sanity_basis_candidates(random_count=2, seed=7)
        by_name = {candidate.name: candidate for candidate in candidates}

        self.assertIn("I", by_name)
        self.assertIn("F3", by_name)
        self.assertIn("F3dg", by_name)
        self.assertIn("D001", by_name)
        self.assertIn("P021", by_name)
        self.assertEqual(len([c for c in candidates if c.candidate_type == "random_unitary"]), 2)
        self.assertTrue(np.allclose(by_name["F3"].matrix, qutrit_fourier()))


class TestDirectBasisBenchmarkAndComparison(unittest.TestCase):
    def test_benchmark_row_contains_compatibility_columns(self):
        row = benchmark_direct_basis(
            state_name="two_qutrit",
            basis_matrix=np.eye(3),
            basis_candidate_name="I",
            basis_candidate_type="identity",
            n_transpile_runs=1,
            coupling_map=[[0, 1], [1, 2], [2, 3]],
            basis_gates=["cz", "id", "rx", "rz", "rzz", "sx", "x"],
            compute_fidelity=True,
        )

        self.assertTrue(row["success"])
        self.assertEqual(row["method"], "direct_basis_encoding")
        self.assertEqual(row["state_name"], "two_qutrit")
        self.assertEqual(row["graph_name"], "star")
        self.assertEqual(row["basis_candidate_name"], "I")
        self.assertEqual(row["candidate_name"], "I")
        self.assertEqual(row["num_qutrits"], 2)
        self.assertEqual(row["num_physical_qubits"], 4)
        self.assertEqual(row["status"], "ok")
        self.assertGreaterEqual(row["compile_time_seconds"], 0.0)
        self.assertGreater(row["total_gate_count"], 0)
        self.assertAlmostEqual(row["fidelity"], 1.0, places=10)

    def test_comparison_joins_old_and_direct_metrics_and_writes_csv(self):
        tmpdir = _tmpdir()
        try:
            old_csv = os.path.join(tmpdir, "old.csv")
            direct_csv = os.path.join(tmpdir, "direct.csv")
            comparison_csv = os.path.join(tmpdir, "comparison.csv")
            pd.DataFrame(
                [
                    {
                        "state_name": "two_qutrit",
                        "class_name": "identity",
                        "candidate_name": "I",
                        "best_two_qubit_gate_count": 5,
                        "best_depth": 11,
                        "best_size": 30,
                        "status": "ok",
                    }
                ]
            ).to_csv(old_csv, index=False)
            pd.DataFrame(
                [
                    {
                        "state_name": "two_qutrit",
                        "graph_name": "star",
                        "basis_candidate_name": "I",
                        "two_qubit_gate_count": 4,
                        "circuit_depth": 9,
                        "total_gate_count": 25,
                        "success": True,
                    }
                ]
            ).to_csv(direct_csv, index=False)

            comparison, summary = compare_old_vs_direct(old_csv, direct_csv, comparison_csv)

            self.assertTrue(os.path.exists(comparison_csv))
            self.assertEqual(len(comparison), 1)
            self.assertEqual(comparison.loc[0, "delta_two_qubit_gate_count"], -1)
            self.assertEqual(comparison.loc[0, "delta_depth"], -2)
            self.assertEqual(comparison.loc[0, "delta_total_gate_count"], -5)
            self.assertEqual(summary["better"], 1)
            self.assertEqual(summary["worse"], 0)
            self.assertEqual(summary["tie"], 0)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_candidate_export_writes_basis_gates_and_full_circuit_qpy(self):
        tmpdir = _tmpdir()
        try:
            candidate = generate_sanity_basis_candidates(random_count=0)[0]
            df, _ = benchmark_direct_basis_candidates(
                state_name="two_qutrit",
                candidates=[candidate],
                n_transpile_runs=1,
                coupling_map=[[0, 1], [1, 2], [2, 3]],
                basis_gates=["cz", "id", "rx", "rz", "rzz", "sx", "x"],
                compute_fidelity=False,
                quantum_circuits_dir=tmpdir,
            )

            candidate_dir = os.path.join(tmpdir, "two_qutrit", "identity__I")
            f3_path = os.path.join(candidate_dir, "F3_W.qpy")
            cz_path = os.path.join(candidate_dir, "CZ3_W.qpy")
            circuit_path = os.path.join(candidate_dir, "graph_state_direct_basis.qpy")

            self.assertTrue(os.path.exists(f3_path))
            self.assertTrue(os.path.exists(cz_path))
            self.assertTrue(os.path.exists(circuit_path))
            self.assertEqual(df.loc[0, "quantum_circuit_dir"], candidate_dir)

            with open(f3_path, "rb") as handle:
                f3_qc = qpy.load(handle)[0]
            with open(cz_path, "rb") as handle:
                cz_qc = qpy.load(handle)[0]
            with open(circuit_path, "rb") as handle:
                full_qc = qpy.load(handle)[0]

            self.assertEqual(f3_qc.num_qubits, 2)
            self.assertEqual(cz_qc.num_qubits, 4)
            self.assertEqual(full_qc.num_qubits, 4)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestDirectBasisRunCli(unittest.TestCase):
    def test_all_qutrit_u3_candidate_set_does_not_require_old_csv(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "--state",
                "ame43",
                "--candidate-set",
                "all-qutrit-u3",
                "--limit-candidates",
                "12",
            ]
        )

        candidates = _load_candidates(args)

        self.assertEqual(args.state, "ame43")
        self.assertEqual(len(candidates), 12)
        self.assertIn(("baseline", "E_old"), {(c.class_name, c.candidate_name) for c in candidates})
        self.assertTrue(all(candidate.is_supported for candidate in candidates))


if __name__ == "__main__":
    unittest.main()
