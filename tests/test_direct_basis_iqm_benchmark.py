from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from qiskit import QuantumCircuit, qpy, transpile
from qiskit.quantum_info import DensityMatrix, Statevector, state_fidelity
from qiskit.transpiler import CouplingMap

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qudits_on_qubits.benchmarks.direct_basis.benchmark import (
    _safe_fidelity,
    benchmark_direct_basis,
    benchmark_direct_basis_candidates,
    default_iqm_quantum_circuits_dir,
    logical_output_density_matrix,
)
from qudits_on_qubits.benchmarks.direct_basis.candidates import DirectBasisCandidate
from qudits_on_qubits.benchmarks.direct_basis.iqm_backend import backend_metadata
from qudits_on_qubits.bell_measurements import build_sampler_circuits_for_candidate


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
    @staticmethod
    def _with_final_layout(circuit, input_to_output):
        circuit._layout = SimpleNamespace(
            final_layout=object(),
            final_index_layout=lambda *, filter_ancillas: list(input_to_output),
        )
        return circuit

    @staticmethod
    def _workload_ranking_transpiler(calls):
        def fake_transpile(circuit, *, trial, iqm_strategy_name, **kwargs):
            measured = circuit.num_clbits > 0
            calls.append(
                {
                    "trial": trial,
                    "strategy": iqm_strategy_name,
                    "measured": measured,
                    "options": kwargs,
                }
            )
            compiled = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
            if measured:
                compiled.cx(0, 1)
                if trial == 0:
                    compiled.cx(1, 2)
                compiled.measure(
                    range(circuit.num_clbits),
                    range(circuit.num_clbits),
                )
            elif trial == 1:
                compiled.x(0)
            return compiled

        return fake_transpile

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
        self.assertIsInstance(state2, DensityMatrix)

    def test_logical_output_density_matrix_restores_layout_order(self):
        reference = QuantumCircuit(2)
        reference.x(0)
        candidate = QuantumCircuit(2)
        candidate.x(1)
        self._with_final_layout(candidate, [1, 0])

        logical, notes = logical_output_density_matrix(
            candidate,
            logical_qubit_count=2,
            max_qubits=2,
        )

        self.assertIsNotNone(logical, notes)
        self.assertGreater(state_fidelity(Statevector.from_instruction(reference), logical), 1 - 1e-10)

    def test_initial_only_transpiler_layout_restores_logical_order(self):
        reference = QuantumCircuit(2)
        reference.x(0)
        candidate = transpile(
            reference,
            basis_gates=["u", "cx"],
            coupling_map=CouplingMap.from_line(2),
            initial_layout=[1, 0],
            optimization_level=0,
        )
        self.assertIsNone(candidate.layout.final_layout)
        self.assertEqual(candidate.layout.final_index_layout(filter_ancillas=True), [1, 0])

        logical, notes = logical_output_density_matrix(
            candidate,
            logical_qubit_count=2,
            max_qubits=2,
        )
        fidelity, fidelity_notes = _safe_fidelity(
            reference,
            candidate,
            max_qubits=2,
        )

        self.assertIsNotNone(logical, notes)
        self.assertGreater(
            state_fidelity(Statevector.from_instruction(reference), logical),
            1 - 1e-10,
        )
        self.assertIsNotNone(fidelity, fidelity_notes)
        self.assertGreater(fidelity, 1 - 1e-10)

    def test_initial_only_layout_restores_logical_order_with_extra_physical_width(self):
        reference = QuantumCircuit(1)
        reference.x(0)
        candidate = transpile(
            reference,
            basis_gates=["u", "cx"],
            coupling_map=CouplingMap.from_line(2),
            initial_layout=[1],
            optimization_level=0,
        )
        self.assertEqual(candidate.num_qubits, 2)
        self.assertIsNone(candidate.layout.final_layout)
        self.assertEqual(candidate.layout.final_index_layout(filter_ancillas=True), [1])

        logical, notes = logical_output_density_matrix(
            candidate,
            logical_qubit_count=1,
            max_qubits=2,
        )
        fidelity, fidelity_notes = _safe_fidelity(
            reference,
            candidate,
            max_qubits=2,
        )

        self.assertIsNotNone(logical, notes)
        self.assertGreater(
            state_fidelity(Statevector.from_instruction(reference), logical),
            1 - 1e-10,
        )
        self.assertIsNotNone(fidelity, fidelity_notes)
        self.assertGreater(fidelity, 1 - 1e-10)

    def test_logical_output_density_matrix_strips_idle_qubits(self):
        reference = QuantumCircuit(1)
        reference.x(0)
        candidate = QuantumCircuit(3)
        candidate.x(2)
        self._with_final_layout(candidate, [2])

        logical, notes = logical_output_density_matrix(
            candidate,
            logical_qubit_count=1,
            max_qubits=1,
        )

        self.assertIsNotNone(logical, notes)
        self.assertIn("Idle qubits stripped", notes)
        self.assertGreater(state_fidelity(Statevector.from_instruction(reference), logical), 1 - 1e-10)

    def test_logical_output_density_matrix_traces_extra_active_qubits(self):
        reference = QuantumCircuit(1)
        reference.h(0)
        candidate = QuantumCircuit(2)
        candidate.h(0)
        candidate.x(1)
        self._with_final_layout(candidate, [0])

        logical, notes = logical_output_density_matrix(
            candidate,
            logical_qubit_count=1,
            max_qubits=2,
        )

        self.assertIsNotNone(logical, notes)
        self.assertIn("Extra active qubits traced", notes)
        self.assertGreater(state_fidelity(Statevector.from_instruction(reference), logical), 1 - 1e-10)

    def test_logical_output_density_matrix_rejects_measurements_before_simulation(self):
        candidate = QuantumCircuit(1, 1)
        candidate.h(0)
        candidate.measure(0, 0)

        logical, notes = logical_output_density_matrix(
            candidate,
            logical_qubit_count=1,
            max_qubits=1,
        )

        self.assertIsNone(logical)
        self.assertIn("measurement", notes.casefold())

    def test_logical_output_density_matrix_reports_unsafe_widths(self):
        too_wide = QuantumCircuit(2)
        too_wide.x(0)
        too_wide.x(1)
        below_logical = QuantumCircuit(1)

        wide_state, wide_notes = logical_output_density_matrix(
            too_wide,
            logical_qubit_count=1,
            max_qubits=1,
        )
        narrow_state, narrow_notes = logical_output_density_matrix(
            below_logical,
            logical_qubit_count=2,
            max_qubits=2,
        )

        self.assertIsNone(wide_state)
        self.assertIn("2 qubits", wide_notes)
        self.assertIsNone(narrow_state)
        self.assertIn("active qubit count 1", narrow_notes)

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

    def test_bell_workload_ranking_selects_cheaper_measured_trial(self):
        calls = []
        with (
            patch(
                "qudits_on_qubits.benchmarks.direct_basis.benchmark."
                "build_sampler_circuits_for_candidate",
                wraps=build_sampler_circuits_for_candidate,
                create=True,
            ) as build_workload,
            patch(
                "qudits_on_qubits.benchmarks.direct_basis.benchmark._transpile_one_trial",
                side_effect=self._workload_ranking_transpiler(calls),
            ),
        ):
            row = benchmark_direct_basis(
                state_name="ghz3",
                basis_matrix=np.eye(4, 3, dtype=complex),
                basis_candidate_name="candidate",
                basis_candidate_type="test",
                n_transpile_runs=2,
                transpiler_backend=object(),
                iqm_strategy_names=("strategy",),
                ranking_workload="bell_measurements",
                compute_fidelity=False,
            )

        self.assertTrue(row["success"], row["error_message"])
        build_workload.assert_called_once()
        self.assertEqual(row["iqm_transpiler_seed"], 1)
        self.assertEqual(row["iqm_transpiler_strategy"], "strategy")
        self.assertEqual(row["ranking_workload"], "bell_measurements")
        self.assertEqual(row["successful_trials"], 2)
        self.assertEqual(
            {
                key: row[key]
                for key in (
                    "workload_circuit_count",
                    "workload_max_depth",
                    "workload_total_depth",
                    "workload_max_two_qubit_gate_count",
                    "workload_total_two_qubit_gate_count",
                    "workload_max_size",
                    "workload_total_size",
                )
            },
            {
                "workload_circuit_count": 12,
                "workload_max_depth": 2,
                "workload_total_depth": 24,
                "workload_max_two_qubit_gate_count": 1,
                "workload_total_two_qubit_gate_count": 12,
                "workload_max_size": 7,
                "workload_total_size": 84,
            },
        )
        self.assertEqual(
            Counter((call["trial"], call["measured"]) for call in calls),
            Counter({(0, False): 1, (0, True): 12, (1, False): 1, (1, True): 12}),
        )
        self.assertEqual({call["strategy"] for call in calls}, {"strategy"})
        self.assertTrue(
            all(call["options"] == calls[0]["options"] for call in calls[1:])
        )

    def test_bell_workload_ranking_uses_size_before_seed_tie_breaker(self):
        def fake_transpile(circuit, *, trial, **_kwargs):
            compiled = QuantumCircuit(circuit.num_qubits, circuit.num_clbits)
            if circuit.num_clbits > 0:
                compiled.cx(0, 1)
                if trial == 0:
                    compiled.x(2)
                compiled.measure(
                    range(circuit.num_clbits),
                    range(circuit.num_clbits),
                )
            return compiled

        with patch(
            "qudits_on_qubits.benchmarks.direct_basis.benchmark._transpile_one_trial",
            side_effect=fake_transpile,
        ):
            row = benchmark_direct_basis(
                state_name="ghz3",
                basis_matrix=np.eye(4, 3, dtype=complex),
                basis_candidate_name="candidate",
                basis_candidate_type="test",
                n_transpile_runs=2,
                transpiler_backend=object(),
                iqm_strategy_names=("strategy",),
                ranking_workload="bell_measurements",
                compute_fidelity=False,
            )

        self.assertTrue(row["success"], row["error_message"])
        self.assertEqual(row["iqm_transpiler_seed"], 1)
        self.assertEqual(row["workload_max_two_qubit_gate_count"], 1)
        self.assertEqual(row["workload_total_two_qubit_gate_count"], 12)
        self.assertEqual(row["workload_max_depth"], 2)
        self.assertEqual(row["workload_total_depth"], 24)
        self.assertEqual(row["workload_max_size"], 7)
        self.assertEqual(row["workload_total_size"], 84)

    def test_default_state_ranking_preserves_state_only_selection(self):
        calls = []
        with (
            patch(
                "qudits_on_qubits.benchmarks.direct_basis.benchmark."
                "build_sampler_circuits_for_candidate",
                side_effect=AssertionError("state ranking must not build Bell workload"),
                create=True,
            ),
            patch(
                "qudits_on_qubits.benchmarks.direct_basis.benchmark._transpile_one_trial",
                side_effect=self._workload_ranking_transpiler(calls),
            ),
        ):
            row = benchmark_direct_basis(
                state_name="ghz3",
                basis_matrix=np.eye(4, 3, dtype=complex),
                basis_candidate_name="candidate",
                basis_candidate_type="test",
                n_transpile_runs=2,
                transpiler_backend=object(),
                iqm_strategy_names=("strategy",),
                compute_fidelity=False,
            )

        self.assertTrue(row["success"], row["error_message"])
        self.assertEqual(row["iqm_transpiler_seed"], 0)
        self.assertEqual(row["ranking_workload"], "state_preparation")
        self.assertEqual(len(calls), 2)
        self.assertNotIn("workload_circuit_count", row)

    def test_bell_workload_ranking_embeds_three_by_three_basis(self):
        calls = []
        with patch(
            "qudits_on_qubits.benchmarks.direct_basis.benchmark._transpile_one_trial",
            side_effect=self._workload_ranking_transpiler(calls),
        ):
            row = benchmark_direct_basis(
                state_name="ghz3",
                basis_matrix=np.eye(3, dtype=complex),
                basis_candidate_name="I",
                basis_candidate_type="identity",
                n_transpile_runs=1,
                transpiler_backend=object(),
                iqm_strategy_names=("strategy",),
                ranking_workload="bell_measurements",
                compute_fidelity=False,
            )

        self.assertTrue(row["success"], row["error_message"])
        self.assertEqual(row["workload_circuit_count"], 12)

    def test_bell_workload_compile_failure_rejects_only_affected_trial(self):
        calls = []
        fake_transpile = self._workload_ranking_transpiler(calls)

        def fail_first_measured_circuit(circuit, *, trial, **kwargs):
            if trial == 0 and circuit.num_clbits > 0:
                raise RuntimeError("measured compile failed")
            return fake_transpile(circuit, trial=trial, **kwargs)

        with (
            patch(
                "qudits_on_qubits.benchmarks.direct_basis.benchmark."
                "build_sampler_circuits_for_candidate",
                wraps=build_sampler_circuits_for_candidate,
                create=True,
            ),
            patch(
                "qudits_on_qubits.benchmarks.direct_basis.benchmark._transpile_one_trial",
                side_effect=fail_first_measured_circuit,
            ),
        ):
            row = benchmark_direct_basis(
                state_name="ghz3",
                basis_matrix=np.eye(4, 3, dtype=complex),
                basis_candidate_name="candidate",
                basis_candidate_type="test",
                n_transpile_runs=2,
                transpiler_backend=object(),
                iqm_strategy_names=("strategy",),
                ranking_workload="bell_measurements",
                compute_fidelity=False,
            )

        self.assertTrue(row["success"], row["error_message"])
        self.assertEqual(row["iqm_transpiler_seed"], 1)
        self.assertEqual(row["successful_trials"], 1)
        self.assertEqual(row["failed_trials"], 1)

    def test_bell_workload_compile_propagates_memory_error(self):
        with patch(
            "qudits_on_qubits.benchmarks.direct_basis.benchmark._transpile_one_trial",
            side_effect=MemoryError("direct-basis transpiler exhausted"),
        ):
            with self.assertRaisesRegex(
                MemoryError,
                "direct-basis transpiler exhausted",
            ):
                benchmark_direct_basis(
                    state_name="ghz3",
                    basis_matrix=np.eye(4, 3, dtype=complex),
                    basis_candidate_name="candidate",
                    basis_candidate_type="test",
                    n_transpile_runs=2,
                    transpiler_backend=object(),
                    iqm_strategy_names=("strategy",),
                    ranking_workload="bell_measurements",
                    compute_fidelity=False,
                )

    def test_benchmark_rejects_unknown_ranking_workload(self):
        with self.assertRaisesRegex(ValueError, "ranking_workload"):
            benchmark_direct_basis(
                state_name="ghz3",
                basis_matrix=np.eye(4, 3, dtype=complex),
                basis_candidate_name="candidate",
                basis_candidate_type="test",
                ranking_workload="unknown",
                compute_fidelity=False,
            )

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

    def test_candidates_forward_ranking_workload_and_iqm_strategies_to_benchmark(self):
        candidate = DirectBasisCandidate(
            name="I",
            candidate_type="identity",
            matrix=np.eye(3, dtype=complex),
        )
        with patch(
            "qudits_on_qubits.benchmarks.direct_basis.benchmark.benchmark_direct_basis",
            return_value={
                "selection_label": "exact",
                "state_name": "two_qutrit",
                "candidate_name": "I",
                "status": "ok",
                "success": True,
            },
        ) as mocked_benchmark:
            benchmark_direct_basis_candidates(
                state_name="two_qutrit",
                candidates=[candidate],
                n_transpile_runs=1,
                compute_fidelity=False,
                transpiler_backend=object(),
                iqm_strategy_names=("custom_strategy",),
                ranking_workload="bell_measurements",
            )

        self.assertEqual(
            mocked_benchmark.call_args.kwargs["ranking_workload"],
            "bell_measurements",
        )
        self.assertEqual(
            mocked_benchmark.call_args.kwargs["iqm_strategy_names"],
            ("custom_strategy",),
        )

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
