from __future__ import annotations

import json
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, qpy

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qudits_on_qubits.benchmarks.direct_basis.candidates import DirectBasisCandidate
from qudits_on_qubits.benchmarks.direct_basis.piast_backend import (
    AQT_SCHEDULING_METHOD,
    AQT_TRANSLATION_METHOD,
)
from qudits_on_qubits.benchmarks.direct_basis.piast_transpiler_harness import (
    PiastTranspilerHarnessConfig,
    _best_trial_rows,
    _metric_row,
    _warning_flags,
    default_piast_transpiler_harness_output_dir,
    run_piast_transpiler_harness,
    write_piast_transpiler_harness_outputs,
)


def _native_aqt_circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(2)
    circuit.r(0.1, 0.2, 0)
    circuit.rz(0.3, 0)
    circuit.r(0.4, 0.5, 1)
    circuit.rxx(0.25, 0, 1)
    return circuit


class PanicException(BaseException):
    pass


PanicException.__module__ = "pyo3_runtime"


class PiastTranspilerHarnessTests(unittest.TestCase):
    def test_metric_row_counts_native_aqt_ops(self):
        row = _metric_row(_native_aqt_circuit())

        self.assertEqual(row["num_qubits"], 2)
        self.assertEqual(row["depth"], 3)
        self.assertEqual(row["size"], 4)
        self.assertEqual(row["rxx_count"], 1)
        self.assertEqual(row["r_count"], 2)
        self.assertEqual(row["rz_count"], 1)
        self.assertEqual(row["one_qubit_gate_count"], 3)
        self.assertEqual(row["two_qubit_gate_count"], 1)
        self.assertEqual(row["non_native_gate_count"], 0)
        self.assertTrue(row["native_compliant"])
        self.assertEqual(json.loads(row["count_ops_json"]), {"r": 2, "rxx": 1, "rz": 1})

    def test_metric_row_counts_non_native_ops_when_backend_ops_are_known(self):
        circuit = QuantumCircuit(1)
        circuit.h(0)

        row = _metric_row(circuit, native_operation_names=("r", "rz", "rxx"))

        self.assertEqual(row["non_native_gate_count"], 1)
        self.assertFalse(row["native_compliant"])
        self.assertEqual(json.loads(row["non_native_ops_json"]), {"h": 1})

    def test_warning_flags_reports_threshold_exceedances(self):
        flags = _warning_flags(
            {"depth": 101, "rxx_count": 51},
            max_depth_warning=100,
            max_rxx_warning=50,
        )

        self.assertEqual(flags, "depth_gt_100;rxx_gt_50")

    def test_best_trial_rows_selects_best_success_and_marks_all_failed(self):
        rows = [
            {
                "class_name": "class_a",
                "candidate_name": "candidate_a",
                "strategy_name": "worse",
                "success": True,
                "status": "ok",
                "depth": 5,
                "rxx_count": 3,
                "two_qubit_gate_count": 3,
                "r_count": 4,
                "rz_count": 1,
                "size": 12,
                "warning_flags": "",
            },
            {
                "class_name": "class_a",
                "candidate_name": "candidate_a",
                "strategy_name": "best",
                "success": True,
                "status": "ok",
                "depth": 5,
                "rxx_count": 2,
                "two_qubit_gate_count": 2,
                "r_count": 8,
                "rz_count": 4,
                "size": 14,
                "warning_flags": "",
            },
            {
                "class_name": "class_b",
                "candidate_name": "candidate_b",
                "strategy_name": "failed",
                "success": False,
                "status": "failed",
                "depth": None,
                "rxx_count": None,
                "two_qubit_gate_count": None,
                "r_count": None,
                "rz_count": None,
                "size": None,
                "warning_flags": "",
            },
        ]

        best_rows = _best_trial_rows(rows)

        by_candidate = {row["candidate_name"]: row for row in best_rows}
        self.assertEqual(by_candidate["candidate_a"]["strategy_name"], "best")
        self.assertEqual(by_candidate["candidate_b"]["status"], "failed_all_strategies")
        self.assertEqual(by_candidate["candidate_b"]["warning_flags"], "failed_all_strategies")

    def test_best_trial_rows_prefers_native_compliant_trials(self):
        rows = [
            {
                "class_name": "class_a",
                "candidate_name": "candidate_a",
                "strategy_name": "non_native_shallow",
                "success": True,
                "status": "ok",
                "native_compliant": False,
                "non_native_gate_count": 1,
                "depth": 1,
                "rxx_count": 0,
                "two_qubit_gate_count": 0,
                "r_count": 0,
                "rz_count": 0,
                "size": 1,
                "warning_flags": "non_native_gate_count_gt_0",
            },
            {
                "class_name": "class_a",
                "candidate_name": "candidate_a",
                "strategy_name": "native_deeper",
                "success": True,
                "status": "ok",
                "native_compliant": True,
                "non_native_gate_count": 0,
                "depth": 5,
                "rxx_count": 1,
                "two_qubit_gate_count": 1,
                "r_count": 2,
                "rz_count": 1,
                "size": 4,
                "warning_flags": "",
            },
        ]

        best_rows = _best_trial_rows(rows)

        self.assertEqual(best_rows[0]["strategy_name"], "native_deeper")

    def test_run_piast_transpiler_harness_records_success_failure_and_summary(self):
        candidate = DirectBasisCandidate(
            name="I",
            candidate_type="identity",
            matrix=np.eye(3, dtype=complex),
            source_class_name="baseline",
            source_candidate_name="I",
        )

        def fake_runner(strategy_name, circuit, *, backend, seed_transpiler, optimization_level):
            self.assertEqual(circuit.num_qubits, 4)
            self.assertEqual(seed_transpiler, 0)
            self.assertEqual(optimization_level, 2)
            if strategy_name == "ok_strategy":
                return SimpleNamespace(
                    strategy_name=strategy_name,
                    seed_transpiler=seed_transpiler,
                    success=True,
                    circuit=_native_aqt_circuit(),
                    compile_time_seconds=0.25,
                    error_type="",
                    error_message="",
                )
            return SimpleNamespace(
                strategy_name=strategy_name,
                seed_transpiler=seed_transpiler,
                success=False,
                circuit=None,
                compile_time_seconds=0.5,
                error_type="RuntimeError",
                error_message="boom",
            )

        config = PiastTranspilerHarnessConfig(
            state_name="two_qutrit",
            n_qutrits=2,
            backend=object(),
            candidates=[candidate],
            strategy_names=("ok_strategy", "bad_strategy"),
            n_transpile_runs=1,
            optimization_level=2,
        )

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            all_trials, best_by_candidate, summary = run_piast_transpiler_harness(
                config,
                strategy_runner=fake_runner,
            )

        self.assertEqual(len(all_trials), 2)
        self.assertEqual(set(all_trials["status"]), {"ok", "failed"})
        success_trial = all_trials[all_trials["status"] == "ok"].iloc[0]
        failed_trial = all_trials[all_trials["status"] == "failed"].iloc[0]
        self.assertEqual(success_trial["num_qubits"], 2)
        self.assertTrue(pd.isna(failed_trial["num_qubits"]))
        self.assertEqual(len(best_by_candidate), 1)
        self.assertEqual(best_by_candidate.iloc[0]["strategy_name"], "ok_strategy")
        self.assertEqual(summary["candidate_count"], 1)
        self.assertEqual(summary["trial_count"], 2)
        self.assertEqual(summary["successful_trial_count"], 1)
        self.assertEqual(summary["failed_trial_count"], 1)
        self.assertEqual(summary["unsupported_candidate_count"], 0)
        self.assertEqual(summary["failed_all_strategy_count"], 0)
        progress_output = stdout.getvalue()
        self.assertIn(
            "candidate 1/1 remaining_candidates=0 trials_per_candidate=2 baseline/I",
            progress_output,
        )
        self.assertIn(
            "[trial 1/2 remaining_trials=1 candidate_trial=1/2] strategy=ok_strategy seed=0",
            progress_output,
        )
        self.assertIn("[trial 1/2] ok depth=3 rxx=1 native=True time=0.250s", progress_output)
        self.assertIn(
            "[trial 2/2 remaining_trials=0 candidate_trial=2/2] strategy=bad_strategy seed=0",
            progress_output,
        )
        self.assertIn("[trial 2/2] failed RuntimeError: boom", progress_output)

    def test_run_piast_transpiler_harness_exports_trial_circuit_and_encoding_matrices(self):
        candidate = DirectBasisCandidate(
            name="I",
            candidate_type="identity",
            matrix=np.eye(3, dtype=complex),
            source_class_name="baseline",
            source_candidate_name="I",
        )

        def fake_runner(strategy_name, circuit, *, backend, seed_transpiler, optimization_level):
            return SimpleNamespace(
                strategy_name=strategy_name,
                seed_transpiler=seed_transpiler,
                success=True,
                circuit=_native_aqt_circuit(),
                compile_time_seconds=0.1,
                error_type="",
                error_message="",
            )

        with tempfile.TemporaryDirectory() as tmp:
            config = PiastTranspilerHarnessConfig(
                state_name="two_qutrit",
                n_qutrits=2,
                backend=object(),
                candidates=[candidate],
                strategy_names=("ok_strategy",),
                n_transpile_runs=1,
                quantum_circuits_dir=tmp,
            )

            all_trials, best_by_candidate, _ = run_piast_transpiler_harness(
                config,
                strategy_runner=fake_runner,
            )

            trial = all_trials.iloc[0]
            best = best_by_candidate.iloc[0]
            self.assertEqual(trial["graph_state_transpiled_qpy"], best["graph_state_transpiled_qpy"])
            self.assertEqual(Path(best["graph_state_transpiled_qpy"]).name, "t_ok_strategy_s0.qpy")
            for column in (
                "f3_w_qpy",
                "cz3_w_qpy",
                "graph_state_qpy",
                "graph_state_transpiled_qpy",
                "basis_change_matrix_npy",
                "E_npy",
                "W_npy",
            ):
                self.assertTrue(Path(best[column]).is_file(), column)

            self.assertEqual(best["basis_change_matrix_npy"], best["W_npy"])
            np.testing.assert_allclose(np.load(best["W_npy"]), np.eye(3, dtype=complex))

            with Path(best["graph_state_transpiled_qpy"]).open("rb") as handle:
                circuits = list(qpy.load(handle))
            self.assertEqual(len(circuits), 1)
            self.assertEqual(circuits[0].num_qubits, 2)

    def test_run_piast_transpiler_harness_exports_4x3_encoding_without_w_alias(self):
        candidate = DirectBasisCandidate(
            name="E",
            candidate_type="isometry",
            matrix=np.eye(4, 3, dtype=complex),
            source_class_name="isometry",
            source_candidate_name="E",
        )

        def fake_runner(strategy_name, circuit, *, backend, seed_transpiler, optimization_level):
            return SimpleNamespace(
                strategy_name=strategy_name,
                seed_transpiler=seed_transpiler,
                success=True,
                circuit=_native_aqt_circuit(),
                compile_time_seconds=0.1,
                error_type="",
                error_message="",
            )

        with tempfile.TemporaryDirectory() as tmp:
            config = PiastTranspilerHarnessConfig(
                state_name="two_qutrit",
                n_qutrits=2,
                backend=object(),
                candidates=[candidate],
                strategy_names=("ok_strategy",),
                n_transpile_runs=1,
                quantum_circuits_dir=Path(tmp) / "circuits",
            )

            all_trials, best_by_candidate, summary = run_piast_transpiler_harness(
                config,
                strategy_runner=fake_runner,
            )
            output_paths = write_piast_transpiler_harness_outputs(
                Path(tmp) / "out",
                all_trials=all_trials,
                best_by_candidate=best_by_candidate,
                summary=summary,
            )

            best = best_by_candidate.iloc[0]
            self.assertTrue(Path(best["E_npy"]).is_file())
            self.assertEqual(best["basis_change_matrix_npy"], best["E_npy"])
            self.assertEqual(best["W_npy"], "")
            self.assertEqual(np.load(best["E_npy"]).shape, (4, 3))

            round_tripped = pd.read_csv(output_paths["best_by_candidate_csv"])
            self.assertTrue(Path(round_tripped.loc[0, "E_npy"]).is_file())
            self.assertTrue(pd.isna(round_tripped.loc[0, "W_npy"]))

    def test_run_piast_transpiler_harness_records_native_panic_failure(self):
        candidate = DirectBasisCandidate(
            name="I",
            candidate_type="identity",
            matrix=np.eye(3, dtype=complex),
            source_class_name="baseline",
            source_candidate_name="I",
        )

        def fake_runner(*args, **kwargs):
            raise PanicException("native failure")

        config = PiastTranspilerHarnessConfig(
            state_name="two_qutrit",
            n_qutrits=2,
            backend=object(),
            candidates=[candidate],
            strategy_names=("bad_strategy",),
        )

        all_trials, best_by_candidate, summary = run_piast_transpiler_harness(
            config,
            strategy_runner=fake_runner,
        )

        self.assertEqual(len(all_trials), 1)
        trial = all_trials.iloc[0]
        self.assertEqual(trial["status"], "failed")
        self.assertEqual(trial["error_type"], "PanicException")
        self.assertIn("native failure", trial["error_message"])
        self.assertEqual(best_by_candidate.iloc[0]["status"], "failed_all_strategies")
        self.assertEqual(summary["failed_trial_count"], 1)
        self.assertEqual(summary["failed_all_strategy_count"], 1)

    def test_run_piast_transpiler_harness_re_raises_process_level_failures(self):
        candidate = DirectBasisCandidate(
            name="I",
            candidate_type="identity",
            matrix=np.eye(3, dtype=complex),
            source_class_name="baseline",
            source_candidate_name="I",
        )

        def fake_runner(*args, **kwargs):
            raise MemoryError("fatal")

        config = PiastTranspilerHarnessConfig(
            state_name="two_qutrit",
            n_qutrits=2,
            backend=object(),
            candidates=[candidate],
            strategy_names=("bad_strategy",),
        )

        with self.assertRaises(MemoryError):
            run_piast_transpiler_harness(config, strategy_runner=fake_runner)

    def test_run_piast_transpiler_harness_records_builtin_strategy_metadata(self):
        candidate = DirectBasisCandidate(
            name="I",
            candidate_type="identity",
            matrix=np.eye(3, dtype=complex),
            source_class_name="baseline",
            source_candidate_name="I",
        )

        def fake_runner(strategy_name, circuit, *, backend, seed_transpiler, optimization_level):
            return SimpleNamespace(
                strategy_name=strategy_name,
                seed_transpiler=seed_transpiler,
                success=True,
                circuit=_native_aqt_circuit(),
                compile_time_seconds=0.1,
                error_type="",
                error_message="",
            )

        config = PiastTranspilerHarnessConfig(
            state_name="two_qutrit",
            n_qutrits=2,
            backend=object(),
            candidates=[candidate],
            strategy_names=("transpile_aqt_plugin", "preset_aqt_plugin"),
        )

        all_trials, _, _ = run_piast_transpiler_harness(
            config,
            strategy_runner=fake_runner,
        )

        by_strategy = {row["strategy_name"]: row for row in all_trials.to_dict(orient="records")}
        self.assertEqual(by_strategy["transpile_aqt_plugin"]["strategy_kind"], "transpile")
        self.assertEqual(
            by_strategy["transpile_aqt_plugin"]["strategy_translation_method"],
            AQT_TRANSLATION_METHOD,
        )
        self.assertEqual(
            by_strategy["preset_aqt_plugin"]["strategy_scheduling_method"],
            AQT_SCHEDULING_METHOD,
        )

    def test_run_piast_transpiler_harness_records_unsupported_candidate(self):
        candidate = DirectBasisCandidate(
            name="missing",
            candidate_type="unknown",
            matrix=None,
            source_class_name="legacy",
            source_candidate_name="missing",
            error_message="not found",
        )

        def fake_runner(*args, **kwargs):
            raise AssertionError("strategy runner should not be called")

        config = PiastTranspilerHarnessConfig(
            state_name="two_qutrit",
            n_qutrits=2,
            backend=object(),
            candidates=[candidate],
            strategy_names=("ok_strategy",),
        )

        all_trials, best_by_candidate, summary = run_piast_transpiler_harness(
            config,
            strategy_runner=fake_runner,
        )

        self.assertEqual(len(all_trials), 1)
        self.assertEqual(len(best_by_candidate), 1)
        self.assertEqual(all_trials.iloc[0]["status"], "unsupported_candidate")
        self.assertEqual(best_by_candidate.iloc[0]["status"], "unsupported_candidate")
        self.assertEqual(all_trials.iloc[0]["error_message"], "not found")
        self.assertTrue(pd.isna(all_trials.iloc[0]["num_qubits"]))
        self.assertEqual(summary["unsupported_candidate_count"], 1)

    def test_write_piast_transpiler_harness_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_piast_transpiler_harness_outputs(
                tmp,
                all_trials=pd.DataFrame([{"status": "ok", "depth": 2}]),
                best_by_candidate=pd.DataFrame([{"status": "ok", "depth": 2}]),
                summary={"candidate_count": 1},
            )

            self.assertTrue(Path(paths["all_trials_csv"]).is_file())
            self.assertTrue(Path(paths["best_by_candidate_csv"]).is_file())
            self.assertTrue(Path(paths["summary_json"]).is_file())

    def test_default_piast_transpiler_harness_output_dir_uses_run_id(self):
        output_dir = default_piast_transpiler_harness_output_dir("run123")

        self.assertTrue(
            output_dir.endswith(
                str(
                    Path("artifacts")
                    / "piast_runs"
                    / "processed"
                    / "transpiler_harness"
                    / "run123"
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
