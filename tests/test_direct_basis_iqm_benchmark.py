from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from qiskit import QuantumCircuit, qpy
from qiskit.quantum_info import Statevector

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qudits_on_qubits.benchmarks.direct_basis.benchmark import (
    _safe_fidelity,
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
    def test_safe_fidelity_uses_qiskit_state_fidelity(self):
        reference = QuantumCircuit(1)
        candidate = QuantumCircuit(1)

        with patch(
            "qudits_on_qubits.benchmarks.direct_basis.benchmark.state_fidelity",
            create=True,
        ) as mocked_state_fidelity:
            mocked_state_fidelity.return_value = 0.25

            fidelity, notes = _safe_fidelity(reference, candidate, max_qubits=10)

        self.assertEqual(fidelity, 0.25)
        self.assertEqual(notes, "")
        mocked_state_fidelity.assert_called_once()
        state1, state2 = mocked_state_fidelity.call_args.args
        self.assertIsInstance(state1, Statevector)
        self.assertIsInstance(state2, Statevector)

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

    def test_iqm_benchmark_uses_harness_strategies_and_exports_best(self):
        backend = object()
        calls = []

        def fake_strategy_runner(
            strategy_name,
            circuit,
            *,
            backend,
            seed_transpiler,
            optimization_level,
        ):
            calls.append((strategy_name, seed_transpiler, optimization_level, circuit.num_qubits))
            output = QuantumCircuit(2)
            if strategy_name == "worse_strategy":
                for index in range(4):
                    output.r(0.1 + index, 0.2, 0)
            else:
                output.r(0.1, 0.2, 0)
            output.r(0.3, 0.4, 1)
            output.cz(0, 1)
            return type(
                "StrategyResult",
                (),
                {
                    "strategy_name": strategy_name,
                    "seed_transpiler": seed_transpiler,
                    "success": True,
                    "circuit": output,
                    "compile_time_seconds": 0.01,
                    "error_type": "",
                    "error_message": "",
                },
            )()

        with tempfile.TemporaryDirectory() as tmp, patch(
            "qudits_on_qubits.benchmarks.direct_basis.benchmark.run_iqm_transpiler_strategy",
            side_effect=fake_strategy_runner,
        ):
            row = benchmark_direct_basis(
                state_name="two_qutrit",
                n_qutrits=2,
                basis_matrix=np.eye(3, dtype=complex),
                basis_candidate_name="I",
                basis_candidate_type="identity",
                n_transpile_runs=1,
                compute_fidelity=False,
                quantum_circuits_dir=tmp,
                transpiler_backend=backend,
                transpiler_metadata={
                    "transpiler_backend": "iqm",
                    "backend_operation_names": json.dumps(["r", "cz"]),
                },
                iqm_strategy_names=("worse_strategy", "better_strategy"),
            )

            self.assertTrue(row["success"], row["error_message"])
            self.assertEqual(
                calls,
                [
                    ("worse_strategy", 0, 3, 4),
                    ("better_strategy", 0, 3, 4),
                ],
            )
            self.assertEqual(row["successful_trials"], 2)
            self.assertEqual(row["iqm_transpiler_strategy"], "better_strategy")
            self.assertEqual(row["iqm_transpiler_seed"], 0)
            self.assertEqual(row["best_depth"], 2)
            self.assertEqual(row["best_two_qubit_gate_count"], 1)
            self.assertEqual(json.loads(row["best_native_count_ops"]), {"cz": 1, "r": 2})
            self.assertTrue(os.path.isfile(row["graph_state_transpiled_qpy"]))
            with open(row["graph_state_transpiled_qpy"], "rb") as handle:
                circuits = list(qpy.load(handle))
            self.assertEqual(circuits[0].depth(), 2)

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

    def test_iqm_candidate_jobs_run_serially_to_avoid_backend_thread_races(self):
        backend = object()
        candidates = [
            DirectBasisCandidate(
                name="first",
                candidate_type="identity",
                matrix=np.eye(3, dtype=complex),
            ),
            DirectBasisCandidate(
                name="second",
                candidate_type="identity",
                matrix=np.eye(3, dtype=complex),
            ),
        ]

        def fake_benchmark_direct_basis(**kwargs):
            return {
                "selection_label": kwargs["selection_label"],
                "state_name": kwargs["state_name"],
                "candidate_name": kwargs["basis_candidate_name"],
                "status": "ok",
                "success": True,
            }

        with (
            patch(
                "qudits_on_qubits.benchmarks.direct_basis.benchmark.ThreadPoolExecutor",
                side_effect=AssertionError("IQM backend must not use candidate threads"),
            ),
            patch(
                "qudits_on_qubits.benchmarks.direct_basis.benchmark.benchmark_direct_basis",
                side_effect=fake_benchmark_direct_basis,
            ) as mocked_benchmark,
        ):
            df, _ = benchmark_direct_basis_candidates(
                state_name="two_qutrit",
                n_qutrits=2,
                candidates=candidates,
                n_transpile_runs=1,
                compute_fidelity=False,
                jobs=4,
                transpiler_backend=backend,
                transpiler_metadata={"transpiler_backend": "iqm"},
            )

        self.assertEqual(mocked_benchmark.call_count, 2)
        self.assertEqual(list(df["candidate_name"]), ["first", "second"])

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
