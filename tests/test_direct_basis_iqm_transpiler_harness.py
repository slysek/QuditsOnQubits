from __future__ import annotations

import json
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
from qudits_on_qubits.benchmarks.direct_basis.iqm_backend import (
    EXACT_RZ_SCHEDULING_METHOD,
)
from qudits_on_qubits.benchmarks.direct_basis.iqm_transpiler_harness import (
    IqmTranspilerHarnessConfig,
    _best_trial_rows,
    _metric_row,
    _warning_flags,
    default_iqm_transpiler_harness_output_dir,
    run_iqm_transpiler_harness,
    write_iqm_transpiler_harness_outputs,
)


def _native_iqm_circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(2)
    circuit.r(0.1, 0.2, 0)
    circuit.r(0.3, 0.4, 1)
    circuit.cz(0, 1)
    return circuit


class NonExceptionHarnessFailure(BaseException):
    pass


class IqmTranspilerHarnessTests(unittest.TestCase):
    def test_metric_row_counts_native_ops(self):
        row = _metric_row(_native_iqm_circuit())

        self.assertEqual(row["num_qubits"], 2)
        self.assertEqual(row["depth"], 2)
        self.assertEqual(row["size"], 3)
        self.assertEqual(row["cz_count"], 1)
        self.assertEqual(row["r_count"], 2)
        self.assertEqual(row["one_qubit_gate_count"], 2)
        self.assertEqual(row["two_qubit_gate_count"], 1)
        self.assertEqual(json.loads(row["count_ops_json"]), {"cz": 1, "r": 2})

    def test_metric_row_counts_one_qubit_ops_by_instruction_arity(self):
        circuit = QuantumCircuit(3)
        circuit.x(0)
        circuit.cx(0, 1)
        circuit.ccx(0, 1, 2)

        row = _metric_row(circuit)

        self.assertEqual(row["size"], 3)
        self.assertEqual(row["one_qubit_gate_count"], 1)
        self.assertEqual(row["two_qubit_gate_count"], 1)
        self.assertEqual(
            json.loads(row["count_ops_json"]),
            {"ccx": 1, "cx": 1, "x": 1},
        )

    def test_warning_flags_reports_threshold_exceedances(self):
        flags = _warning_flags(
            {"depth": 101, "cz_count": 51},
            max_depth_warning=100,
            max_cz_warning=50,
        )

        self.assertEqual(flags, "depth_gt_100;cz_gt_50")

    def test_warning_flags_returns_empty_string_below_thresholds(self):
        flags = _warning_flags(
            {"depth": 35, "cz_count": 18},
            max_depth_warning=100,
            max_cz_warning=50,
        )

        self.assertEqual(flags, "")

    def test_best_trial_rows_selects_best_success_and_marks_all_failed(self):
        rows = [
            {
                "class_name": "class_a",
                "candidate_name": "candidate_a",
                "strategy_name": "failed",
                "success": False,
                "status": "failed",
                "depth": None,
                "cz_count": None,
                "r_count": None,
                "size": None,
                "warning_flags": "",
            },
            {
                "class_name": "class_a",
                "candidate_name": "candidate_a",
                "strategy_name": "worse",
                "success": True,
                "status": "ok",
                "depth": 5,
                "cz_count": 3,
                "r_count": 4,
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
                "cz_count": 2,
                "r_count": 8,
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
                "cz_count": None,
                "r_count": None,
                "size": None,
                "warning_flags": "",
            },
        ]

        best_rows = _best_trial_rows(rows)

        by_candidate = {
            row["candidate_name"]: row
            for row in best_rows
        }
        self.assertEqual(by_candidate["candidate_a"]["strategy_name"], "best")
        self.assertEqual(by_candidate["candidate_b"]["status"], "failed_all_strategies")
        self.assertEqual(
            by_candidate["candidate_b"]["warning_flags"],
            "failed_all_strategies",
        )

    def test_run_iqm_transpiler_harness_records_success_failure_and_summary(self):
        candidate = DirectBasisCandidate(
            name="I",
            candidate_type="identity",
            matrix=np.eye(3, dtype=complex),
            source_class_name="baseline",
            source_candidate_name="I",
        )
        backend = object()

        def fake_runner(
            strategy_name,
            circuit,
            *,
            backend: object,
            seed_transpiler: int,
            optimization_level: int,
        ):
            self.assertEqual(circuit.num_qubits, 4)
            self.assertEqual(seed_transpiler, 0)
            self.assertEqual(optimization_level, 2)
            if strategy_name == "ok_strategy":
                return SimpleNamespace(
                    strategy_name=strategy_name,
                    seed_transpiler=seed_transpiler,
                    success=True,
                    circuit=_native_iqm_circuit(),
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

        config = IqmTranspilerHarnessConfig(
            state_name="two_qutrit",
            n_qutrits=2,
            backend=backend,
            iqm_backend_name="fake_backend",
            iqm_use_metrics=False,
            candidates=[candidate],
            strategy_names=("ok_strategy", "bad_strategy"),
            n_transpile_runs=1,
            optimization_level=2,
        )

        all_trials, best_by_candidate, summary = run_iqm_transpiler_harness(
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
        self.assertEqual(best_by_candidate.iloc[0]["status"], "ok")
        self.assertEqual(best_by_candidate.iloc[0]["num_qubits"], 2)
        self.assertEqual(summary["candidate_count"], 1)
        self.assertEqual(summary["trial_count"], 2)
        self.assertEqual(summary["successful_trial_count"], 1)
        self.assertEqual(summary["failed_trial_count"], 1)
        self.assertEqual(summary["unsupported_candidate_count"], 0)
        self.assertEqual(summary["failed_all_strategy_count"], 0)

    def test_run_iqm_transpiler_harness_exports_trial_circuit_and_encoding_matrices(self):
        candidate = DirectBasisCandidate(
            name="I",
            candidate_type="identity",
            matrix=np.eye(3, dtype=complex),
            source_class_name="baseline",
            source_candidate_name="I",
        )

        def fake_runner(
            strategy_name,
            circuit,
            *,
            backend: object,
            seed_transpiler: int,
            optimization_level: int,
        ):
            return SimpleNamespace(
                strategy_name=strategy_name,
                seed_transpiler=seed_transpiler,
                success=True,
                circuit=_native_iqm_circuit(),
                compile_time_seconds=0.1,
                error_type="",
                error_message="",
            )

        with tempfile.TemporaryDirectory() as tmp:
            config = IqmTranspilerHarnessConfig(
                state_name="two_qutrit",
                n_qutrits=2,
                backend=object(),
                iqm_backend_name="fake_backend",
                iqm_use_metrics=False,
                candidates=[candidate],
                strategy_names=("ok_strategy",),
                n_transpile_runs=1,
                quantum_circuits_dir=tmp,
            )

            all_trials, best_by_candidate, _ = run_iqm_transpiler_harness(
                config,
                strategy_runner=fake_runner,
            )

            trial = all_trials.iloc[0]
            best = best_by_candidate.iloc[0]
            self.assertEqual(trial["graph_state_transpiled_qpy"], best["graph_state_transpiled_qpy"])
            self.assertIn("ok_strategy_seed0", Path(best["graph_state_transpiled_qpy"]).name)
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
            E = np.load(best["E_npy"])
            self.assertEqual(E.shape, (4, 3))
            np.testing.assert_allclose(E[:3, :], np.eye(3, dtype=complex))
            np.testing.assert_allclose(E[3, :], np.zeros(3, dtype=complex))

            with Path(best["graph_state_transpiled_qpy"]).open("rb") as handle:
                circuits = list(qpy.load(handle))
            self.assertEqual(len(circuits), 1)
            self.assertEqual(circuits[0].num_qubits, 2)

    def test_run_iqm_transpiler_harness_records_base_exception_failure(self):
        candidate = DirectBasisCandidate(
            name="I",
            candidate_type="identity",
            matrix=np.eye(3, dtype=complex),
            source_class_name="baseline",
            source_candidate_name="I",
        )

        def fake_runner(*args, **kwargs):
            raise NonExceptionHarnessFailure("native failure")

        config = IqmTranspilerHarnessConfig(
            state_name="two_qutrit",
            n_qutrits=2,
            backend=object(),
            iqm_backend_name="fake_backend",
            iqm_use_metrics=False,
            candidates=[candidate],
            strategy_names=("bad_strategy",),
        )

        all_trials, best_by_candidate, summary = run_iqm_transpiler_harness(
            config,
            strategy_runner=fake_runner,
        )

        self.assertEqual(len(all_trials), 1)
        trial = all_trials.iloc[0]
        self.assertEqual(trial["status"], "failed")
        self.assertEqual(trial["error_type"], "NonExceptionHarnessFailure")
        self.assertIn("native failure", trial["error_message"])
        self.assertEqual(best_by_candidate.iloc[0]["status"], "failed_all_strategies")
        self.assertEqual(summary["failed_trial_count"], 1)
        self.assertEqual(summary["failed_all_strategy_count"], 1)

    def test_run_iqm_transpiler_harness_rejects_zero_transpile_runs(self):
        config = IqmTranspilerHarnessConfig(
            state_name="two_qutrit",
            n_qutrits=2,
            backend=object(),
            iqm_backend_name="fake_backend",
            iqm_use_metrics=False,
            candidates=[],
            n_transpile_runs=0,
        )

        with self.assertRaisesRegex(ValueError, "n_transpile_runs must be >= 1"):
            run_iqm_transpiler_harness(config)

    def test_run_iqm_transpiler_harness_records_builtin_strategy_metadata(self):
        candidate = DirectBasisCandidate(
            name="I",
            candidate_type="identity",
            matrix=np.eye(3, dtype=complex),
            source_class_name="baseline",
            source_candidate_name="I",
        )

        def fake_runner(
            strategy_name,
            circuit,
            *,
            backend: object,
            seed_transpiler: int,
            optimization_level: int,
        ):
            return SimpleNamespace(
                strategy_name=strategy_name,
                seed_transpiler=seed_transpiler,
                success=True,
                circuit=_native_iqm_circuit(),
                compile_time_seconds=0.1,
                error_type="",
                error_message="",
            )

        config = IqmTranspilerHarnessConfig(
            state_name="two_qutrit",
            n_qutrits=2,
            backend=object(),
            iqm_backend_name="fake_backend",
            iqm_use_metrics=False,
            candidates=[candidate],
            strategy_names=("preset_default", "preset_exact"),
        )

        all_trials, _, _ = run_iqm_transpiler_harness(
            config,
            strategy_runner=fake_runner,
        )

        by_strategy = {
            row["strategy_name"]: row
            for row in all_trials.to_dict(orient="records")
        }
        self.assertTrue(pd.isna(all_trials.iloc[0]["scheduling_method"]))
        self.assertEqual(by_strategy["preset_default"]["strategy_kind"], "preset")
        self.assertEqual(
            by_strategy["preset_default"]["strategy_scheduling_method"],
            "",
        )
        self.assertEqual(
            by_strategy["preset_default"]["strategy_remove_final_rzs"],
            True,
        )
        self.assertEqual(by_strategy["preset_exact"]["strategy_kind"], "preset")
        self.assertEqual(
            by_strategy["preset_exact"]["strategy_scheduling_method"],
            EXACT_RZ_SCHEDULING_METHOD,
        )
        self.assertEqual(
            by_strategy["preset_exact"]["strategy_remove_final_rzs"],
            False,
        )

    def test_run_iqm_transpiler_harness_passes_distinct_circuit_per_trial(self):
        candidate = DirectBasisCandidate(
            name="I",
            candidate_type="identity",
            matrix=np.eye(3, dtype=complex),
            source_class_name="baseline",
            source_candidate_name="I",
        )
        seen_circuits = []

        def fake_runner(
            strategy_name,
            circuit,
            *,
            backend: object,
            seed_transpiler: int,
            optimization_level: int,
        ):
            seen_circuits.append(circuit)
            return SimpleNamespace(
                strategy_name=strategy_name,
                seed_transpiler=seed_transpiler,
                success=False,
                circuit=None,
                compile_time_seconds=0.1,
                error_type="RuntimeError",
                error_message="boom",
            )

        config = IqmTranspilerHarnessConfig(
            state_name="two_qutrit",
            n_qutrits=2,
            backend=object(),
            iqm_backend_name="fake_backend",
            iqm_use_metrics=False,
            candidates=[candidate],
            strategy_names=("strategy_a", "strategy_b"),
            n_transpile_runs=2,
        )

        run_iqm_transpiler_harness(config, strategy_runner=fake_runner)

        self.assertEqual(len(seen_circuits), 4)
        self.assertEqual(len({id(circuit) for circuit in seen_circuits}), 4)

    def test_run_iqm_transpiler_harness_records_unsupported_candidate(self):
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

        config = IqmTranspilerHarnessConfig(
            state_name="two_qutrit",
            n_qutrits=2,
            backend=object(),
            iqm_backend_name="fake_backend",
            iqm_use_metrics=False,
            candidates=[candidate],
            strategy_names=("ok_strategy",),
        )

        all_trials, best_by_candidate, summary = run_iqm_transpiler_harness(
            config,
            strategy_runner=fake_runner,
        )

        self.assertEqual(len(all_trials), 1)
        self.assertEqual(len(best_by_candidate), 1)
        self.assertEqual(all_trials.iloc[0]["status"], "unsupported_candidate")
        self.assertEqual(best_by_candidate.iloc[0]["status"], "unsupported_candidate")
        self.assertEqual(all_trials.iloc[0]["error_message"], "not found")
        self.assertEqual(all_trials.iloc[0]["warning_flags"], "")
        self.assertEqual(best_by_candidate.iloc[0]["warning_flags"], "")
        self.assertTrue(pd.isna(all_trials.iloc[0]["num_qubits"]))
        self.assertTrue(pd.isna(best_by_candidate.iloc[0]["num_qubits"]))
        self.assertEqual(summary["unsupported_candidate_count"], 1)

    def test_write_iqm_transpiler_harness_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = write_iqm_transpiler_harness_outputs(
                tmp,
                all_trials=pd.DataFrame([{"status": "ok", "depth": 2}]),
                best_by_candidate=pd.DataFrame([{"status": "ok", "depth": 2}]),
                summary={"candidate_count": 1},
            )

            self.assertTrue(Path(paths["all_trials_csv"]).is_file())
            self.assertTrue(Path(paths["best_by_candidate_csv"]).is_file())
            self.assertTrue(Path(paths["summary_json"]).is_file())
            self.assertEqual(
                json.loads(Path(paths["summary_json"]).read_text(encoding="utf-8")),
                {"candidate_count": 1},
            )

    def test_default_iqm_transpiler_harness_output_dir_uses_run_id(self):
        output_dir = default_iqm_transpiler_harness_output_dir("run123")

        self.assertTrue(
            output_dir.endswith(
                str(
                    Path("artifacts")
                    / "iqm_runs"
                    / "processed"
                    / "transpiler_harness"
                    / "run123"
                )
            )
        )


if __name__ == "__main__":
    unittest.main()
