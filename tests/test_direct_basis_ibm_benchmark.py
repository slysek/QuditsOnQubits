from __future__ import annotations

import pytest
pytestmark = pytest.mark.usefixtures("exact_cz3_synthesis")

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from qiskit.providers.fake_provider import GenericBackendV2

from qudits_on_qubits.benchmarks.direct_basis.benchmark import benchmark_direct_basis, benchmark_direct_basis_candidates
from qudits_on_qubits.benchmarks.direct_basis.candidates import DirectBasisCandidate


class IBMBenchmarkTests(unittest.TestCase):
    def test_routing_outside_requested_layout_is_reported(self):
        from qudits_on_qubits.benchmarks.direct_basis.benchmark import _transpile_one_trial
        backend = GenericBackendV2(5, basis_gates=["rz", "sx", "x", "cx"], seed=13)
        def rerouted(circuit, **options):
            return _transpile_one_trial(circuit, **{**options, "initial_layout": (1, 2, 3, 4)})
        with patch("qudits_on_qubits.benchmarks.direct_basis.benchmark._transpile_one_trial", side_effect=rerouted):
            row = benchmark_direct_basis(
                state_name="two_qutrit", basis_matrix=np.eye(3),
                basis_candidate_name="canonical", basis_candidate_type="baseline",
                transpiler_backend=backend, transpiler_provider="ibm",
                initial_layout=(0, 1, 2, 3), n_transpile_runs=1,
                ranking_workload="bell_measurements",
            )
        self.assertEqual(row["status"], "ok")
        metrics = json.loads(row["workload_metrics"])
        self.assertFalse(metrics["aggregate"]["uses_exact_physical_qubit_set"])

    def test_candidate_batch_preserves_ibm_target_and_failed_row_metadata(self):
        backend = GenericBackendV2(4, basis_gates=["rz", "sx", "x", "cx"], seed=13)
        with tempfile.TemporaryDirectory() as directory:
            frame, path = benchmark_direct_basis_candidates(
                state_name="two_qutrit",
                candidates=[DirectBasisCandidate("canonical", "baseline", np.eye(3)),
                            DirectBasisCandidate("unsupported", "other", None)],
                transpiler_backend=backend, transpiler_provider="ibm",
                initial_layout=(3, 2, 1, 0), n_transpile_runs=1,
                ranking_workload="bell_measurements",
                output_csv=str(Path(directory) / "benchmark.csv"), jobs=2,
            )
            self.assertEqual(frame.transpiler_backend.tolist(), ["ibm", "ibm"])
            self.assertEqual(frame.iloc[0]["status"], "ok")
            self.assertAlmostEqual(frame.iloc[0]["fidelity"], 1, places=8)
            workload = json.loads(frame.iloc[0]["workload_metrics"])
            self.assertTrue(all(record["physical_qubit_mapping"] == [3, 2, 1, 0] for record in workload["circuits"]))
            self.assertFalse(frame.iloc[1]["success"])
            self.assertTrue(Path(path).is_file())

    def test_ibm_compiles_complete_workload_and_preserves_logical_fidelity(self):
        backend = GenericBackendV2(7, basis_gates=["id", "rz", "sx", "x", "cx"], seed=13)
        with tempfile.TemporaryDirectory() as directory, patch(
            "qudits_on_qubits.benchmarks.direct_basis.iqm_backend.build_iqm_pass_manager",
            side_effect=AssertionError("IBM must not use IQM transpilation"),
        ):
            row = benchmark_direct_basis(
                state_name="two_qutrit", basis_matrix=np.eye(3),
                basis_candidate_name="canonical", basis_candidate_type="baseline",
                transpiler_backend=backend, transpiler_provider="ibm",
                initial_layout=(6, 4, 2, 0), n_transpile_runs=1,
                ranking_workload="bell_measurements", max_fidelity_qubits=6,
                quantum_circuits_dir=directory,
            )
            self.assertEqual(row["status"], "ok", row.get("error_message"))
            self.assertEqual(row["transpiler_backend"], "ibm")
            self.assertEqual(row["workload_circuit_count"], 9)
            self.assertAlmostEqual(row["fidelity"], 1, places=8)
            self.assertEqual(row["transpiler_seed"], 0)
            self.assertGreater(row["best_two_qubit_gate_count"], 0)
            self.assertGreater(row["best_two_qubit_depth"], 0)
            self.assertTrue(json.loads(row["workload_metrics"])["circuits"])
            self.assertTrue(Path(row["graph_state_transpiled_qpy"]).is_file())

    def test_ibm_rejects_iqm_strategies(self):
        with self.assertRaisesRegex(ValueError, "IQM"):
            benchmark_direct_basis(
                state_name="two_qutrit", basis_matrix=np.eye(3),
                basis_candidate_name="canonical", basis_candidate_type="baseline",
                transpiler_backend=object(), transpiler_provider="ibm",
                iqm_strategy_names=("default",),
            )


if __name__ == "__main__":
    unittest.main()


def test_ibm_bell_trial_ranking_prefers_parallel_two_qubit_depth(tmp_path):
    from qiskit import QuantumCircuit
    backend = GenericBackendV2(4, basis_gates=['rz','sx','x','cx'], seed=13)
    def compile_trial(circuit, *, trial, **kwargs):
        if not circuit.num_clbits:
            return circuit.copy()
        compiled = QuantumCircuit(4,4)
        compiled.cx(0,1)
        compiled.cx(1,2) if trial == 0 else compiled.cx(2,3)
        if trial == 1:
            for _ in range(3): compiled.x(0)
        compiled.measure(range(4), range(4))
        return compiled
    with patch('qudits_on_qubits.benchmarks.direct_basis.benchmark._transpile_one_trial', side_effect=compile_trial):
        row = benchmark_direct_basis(
            state_name='two_qutrit', basis_matrix=np.eye(3),
            basis_candidate_name='canonical', basis_candidate_type='baseline',
            transpiler_backend=backend, transpiler_provider='ibm',
            n_transpile_runs=2, ranking_workload='bell_measurements',
            quantum_circuits_dir=str(tmp_path),
        )
    assert row['status'] == 'ok'
    assert row['transpiler_seed'] == 1
    assert row['workload_max_two_qubit_depth'] == 1
    assert row['workload_total_two_qubit_depth'] == 9


def test_ibm_csv_selection_prefers_two_qubit_depth_without_changing_iqm():
    import pandas as pd
    from qudits_on_qubits.benchmarks.direct_basis.selection import select_top_k
    common = dict(success=True, status='ok', ranking_workload='bell_measurements',
                  workload_max_two_qubit_gate_count=2, workload_total_two_qubit_gate_count=18,
                  workload_max_size=8, workload_total_size=72)
    rows = [dict(common, candidate='serial', transpiler_backend='ibm', workload_max_depth=3,
                 workload_total_depth=27, workload_max_two_qubit_depth=2, workload_total_two_qubit_depth=18),
            dict(common, candidate='parallel', transpiler_backend='ibm', workload_max_depth=5,
                 workload_total_depth=45, workload_max_two_qubit_depth=1, workload_total_two_qubit_depth=9)]
    ranked = select_top_k(pd.DataFrame(rows), label='exact', top_k=1, fidelity_threshold=None)
    assert ranked.iloc[0]['candidate'] == 'parallel'
    for row in rows: row['transpiler_backend'] = 'iqm'
    ranked = select_top_k(pd.DataFrame(rows), label='exact', top_k=1, fidelity_threshold=None)
    assert ranked.iloc[0]['candidate'] == 'serial'
