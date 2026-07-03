from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from qiskit import qpy

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qudits_on_qubits.benchmarks.direct_basis.benchmark import (
    benchmark_direct_basis,
    benchmark_direct_basis_candidates,
    default_iqm_quantum_circuits_dir,
)
from qudits_on_qubits.benchmarks.direct_basis.candidates import DirectBasisCandidate
from qudits_on_qubits.benchmarks.direct_basis.iqm_backend import backend_metadata


def _fake_garnet():
    try:
        from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet
    except ImportError as exc:
        raise unittest.SkipTest(f"IQM fake backend is unavailable: {exc}") from exc
    return IQMFakeGarnet()


def _garnet_metadata(backend, *, optimization_level=3):
    return backend_metadata(
        backend,
        iqm_backend_name="garnet",
        iqm_use_metrics=False,
        optimization_level=optimization_level,
        layout_method=None,
        routing_method=None,
    )


class DirectBasisIqmBenchmarkTests(unittest.TestCase):
    def test_benchmark_direct_basis_accepts_iqm_backend(self):
        backend = _fake_garnet()
        metadata = _garnet_metadata(backend)

        row = benchmark_direct_basis(
            state_name="two_qutrit",
            n_qutrits=2,
            basis_matrix=np.eye(3, dtype=complex),
            basis_candidate_name="I",
            basis_candidate_type="identity",
            n_transpile_runs=1,
            compute_fidelity=False,
            transpiler_backend=backend,
            transpiler_metadata=metadata,
            layout_method=None,
            routing_method=None,
        )

        self.assertTrue(row["success"], row["error_message"])
        self.assertEqual(row["transpiler_backend"], "iqm")
        self.assertEqual(row["iqm_backend_name"], "garnet")
        self.assertEqual(row["optimization_level"], 3)
        self.assertIsInstance(row["best_one_qubit_gate_count"], int)
        self.assertIsInstance(row["mean_one_qubit_gate_count"], float)
        native_ops = json.loads(row["best_native_count_ops"])
        self.assertIn("r", native_ops)
        self.assertIn("cz", native_ops)
        self.assertNotIn("cx", row["best_count_ops"])

    def test_iqm_ranking_records_one_qubit_statistics(self):
        backend = _fake_garnet()
        metadata = _garnet_metadata(backend)

        row = benchmark_direct_basis(
            state_name="two_qutrit",
            n_qutrits=2,
            basis_matrix=np.eye(3, dtype=complex),
            basis_candidate_name="I",
            basis_candidate_type="identity",
            n_transpile_runs=2,
            compute_fidelity=False,
            transpiler_backend=backend,
            transpiler_metadata=metadata,
        )

        self.assertTrue(row["success"], row["error_message"])
        self.assertEqual(row["successful_trials"], 2)
        self.assertIsInstance(row["best_one_qubit_gate_count"], int)
        self.assertIsInstance(row["mean_one_qubit_gate_count"], float)

    def test_iqm_backend_without_metadata_marks_transpiler_backend(self):
        backend = _fake_garnet()

        row = benchmark_direct_basis(
            state_name="two_qutrit",
            n_qutrits=2,
            basis_matrix=np.eye(3, dtype=complex),
            basis_candidate_name="I",
            basis_candidate_type="identity",
            n_transpile_runs=1,
            compute_fidelity=False,
            transpiler_backend=backend,
        )

        self.assertTrue(row["success"], row["error_message"])
        self.assertEqual(row["transpiler_backend"], "iqm")

    def test_iqm_export_writes_full_transpiled_qpy(self):
        backend = _fake_garnet()
        metadata = _garnet_metadata(backend)

        with tempfile.TemporaryDirectory() as tmp:
            row = benchmark_direct_basis(
                state_name="two_qutrit",
                n_qutrits=2,
                basis_matrix=np.eye(3, dtype=complex),
                basis_candidate_name="I",
                basis_candidate_type="identity",
                n_transpile_runs=1,
                compute_fidelity=True,
                max_fidelity_qubits=1,
                quantum_circuits_dir=tmp,
                transpiler_backend=backend,
                transpiler_metadata=metadata,
            )

            self.assertTrue(row["success"], row["error_message"])
            qpy_path = row["graph_state_transpiled_qpy"]
            self.assertTrue(os.path.isfile(qpy_path))
            with open(qpy_path, "rb") as handle:
                circuits = list(qpy.load(handle))
            self.assertEqual(len(circuits), 1)
            self.assertEqual(circuits[0].num_qubits, row["num_qubits"])
            self.assertGreater(circuits[0].num_qubits, 1)

    def test_candidates_forward_iqm_backend_and_metadata(self):
        backend = _fake_garnet()
        metadata = _garnet_metadata(backend)
        candidates = [
            DirectBasisCandidate(
                name="I",
                candidate_type="identity",
                matrix=np.eye(3, dtype=complex),
            )
        ]

        df, _ = benchmark_direct_basis_candidates(
            state_name="two_qutrit",
            n_qutrits=2,
            candidates=candidates,
            n_transpile_runs=1,
            compute_fidelity=False,
            transpiler_backend=backend,
            transpiler_metadata=metadata,
        )

        self.assertEqual(list(df["transpiler_backend"]), ["iqm"])
        self.assertEqual(list(df["iqm_backend_name"]), ["garnet"])

    def test_unsupported_candidates_keep_iqm_metadata(self):
        backend = _fake_garnet()
        metadata = _garnet_metadata(backend)
        candidates = [
            DirectBasisCandidate(
                name="bad",
                candidate_type="unsupported",
                matrix=None,
                error_message="unsupported",
            )
        ]

        df, _ = benchmark_direct_basis_candidates(
            state_name="two_qutrit",
            n_qutrits=2,
            candidates=candidates,
            n_transpile_runs=1,
            compute_fidelity=False,
            transpiler_backend=backend,
            transpiler_metadata=metadata,
        )

        row = df.iloc[0]
        self.assertEqual(row["status"], "unsupported_direct_basis_candidate")
        self.assertEqual(row["transpiler_backend"], "iqm")
        self.assertEqual(row["iqm_backend_name"], "garnet")

    def test_default_iqm_quantum_circuits_dir_sanitizes_backend_name(self):
        path = default_iqm_quantum_circuits_dir("../bad name")

        self.assertTrue(path.endswith(os.path.join("quantum_circuits", "bad_name")))
        self.assertNotIn("..", Path(path).parts)


if __name__ == "__main__":
    unittest.main()
