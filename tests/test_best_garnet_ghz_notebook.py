import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from qiskit import QuantumCircuit


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = (
    REPO_ROOT / "notebooks" / "working" / "iqm" / "best_garnet_ghz.ipynb"
)


def load_cell_namespace(marker, **injected):
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code" and marker in "".join(cell["source"])
    )
    namespace = {
        "np": np,
        "QuantumCircuit": QuantumCircuit,
        "repo_root": REPO_ROOT,
        **injected,
    }
    exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    return namespace


class ReadoutCalibrationCacheTests(unittest.TestCase):
    @staticmethod
    def _backend(name="garnet", counts=None):
        backend = Mock()
        backend.num_qubits = 4
        backend.name = name
        if counts is not None:
            backend.run.return_value.result.return_value.get_counts.return_value = (
                counts
            )
        return backend

    @staticmethod
    def _cache(matrix, backend_name="garnet", qubit=2):
        return {
            "version": 1,
            "backends": {
                backend_name: {
                    str(qubit): {
                        "matrix": np.asarray(matrix).tolist(),
                        "shots": 10_000,
                        "created_at": "2026-07-13T10:00:00+00:00",
                    }
                }
            },
        }

    def test_cache_hit_does_not_run_backend(self):
        expected = np.array([[0.9, 0.2], [0.1, 0.8]], dtype=np.float32)
        backend = self._backend()

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "readout_matrices.json"
            cache_path.write_text(
                json.dumps(self._cache(expected)),
                encoding="utf-8",
            )

            namespace = load_cell_namespace(
                "def build_readout_calibration_matrices"
            )
            matrices = namespace["build_readout_calibration_matrices"](
                backend,
                [2],
                cache_path=cache_path,
                verbose=False,
            )

        backend.run.assert_not_called()
        np.testing.assert_allclose(matrices[2], expected)

    def test_cache_miss_runs_backend_and_persists_matrix(self):
        backend = self._backend(
            counts=[{"0": 90, "1": 10}, {"0": 20, "1": 80}]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "readout_matrices.json"
            function = load_cell_namespace(
                "def build_readout_calibration_matrices"
            )["build_readout_calibration_matrices"]

            matrices = function(
                backend, [2], shots=100, cache_path=cache_path, verbose=False
            )
            payload = json.loads(cache_path.read_text(encoding="utf-8"))

        self.assertEqual(backend.run.call_count, 1)
        self.assertEqual(backend.run.call_args.kwargs["shots"], 100)
        np.testing.assert_allclose(matrices[2], [[0.9, 0.2], [0.1, 0.8]])
        np.testing.assert_allclose(
            payload["backends"]["garnet"]["2"]["matrix"],
            [[0.9, 0.2], [0.1, 0.8]],
        )
        self.assertEqual(payload["backends"]["garnet"]["2"]["shots"], 100)
        self.assertIn("created_at", payload["backends"]["garnet"]["2"])

    def test_partial_hit_calibrates_only_missing_qubit(self):
        cached = np.array([[0.95, 0.1], [0.05, 0.9]], dtype=np.float32)
        backend = self._backend(
            counts=[{"0": 90, "1": 10}, {"0": 20, "1": 80}]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "readout_matrices.json"
            cache_path.write_text(
                json.dumps(self._cache(cached, qubit=1)), encoding="utf-8"
            )
            function = load_cell_namespace(
                "def build_readout_calibration_matrices"
            )["build_readout_calibration_matrices"]
            matrices = function(
                backend, [1, 2], shots=100, cache_path=cache_path, verbose=False
            )

        backend.run.assert_called_once()
        self.assertEqual(len(backend.run.call_args.args[0]), 2)
        np.testing.assert_allclose(matrices[1], cached)
        np.testing.assert_allclose(matrices[2], [[0.9, 0.2], [0.1, 0.8]])

    def test_cache_entries_are_isolated_by_backend_name(self):
        backend = self._backend(
            counts=[{"0": 90, "1": 10}, {"0": 20, "1": 80}]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "readout_matrices.json"
            cache_path.write_text(
                json.dumps(self._cache(np.eye(2), backend_name="other")),
                encoding="utf-8",
            )
            function = load_cell_namespace(
                "def build_readout_calibration_matrices"
            )["build_readout_calibration_matrices"]
            function(
                backend, [2], shots=100, cache_path=cache_path, verbose=False
            )

        backend.run.assert_called_once()

    def test_iqm_devices_with_default_backend_name_use_separate_entries(self):
        backend = self._backend(
            name="IQMBackend",
            counts=[{"0": 90, "1": 10}, {"0": 20, "1": 80}],
        )
        backend.client._iqm_server_client.root_url = "https://resonance.iqm.tech"
        backend.client._iqm_server_client._quantum_computer = "crystal"
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "readout_matrices.json"
            cache_path.write_text(
                json.dumps(
                    self._cache(np.eye(2), backend_name="IQMBackend")
                ),
                encoding="utf-8",
            )
            function = load_cell_namespace(
                "def build_readout_calibration_matrices"
            )["build_readout_calibration_matrices"]
            function(
                backend, [2], shots=100, cache_path=cache_path, verbose=False
            )

        backend.run.assert_called_once()

    def test_force_recalibration_replaces_cached_entry(self):
        backend = self._backend(
            counts=[{"0": 90, "1": 10}, {"0": 20, "1": 80}]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "readout_matrices.json"
            cache_path.write_text(
                json.dumps(self._cache(np.eye(2))), encoding="utf-8"
            )
            function = load_cell_namespace(
                "def build_readout_calibration_matrices"
            )["build_readout_calibration_matrices"]
            function(
                backend,
                [2],
                shots=100,
                cache_path=cache_path,
                force_recalibration=True,
                verbose=False,
            )
            payload = json.loads(cache_path.read_text(encoding="utf-8"))

        backend.run.assert_called_once()
        np.testing.assert_allclose(
            payload["backends"]["garnet"]["2"]["matrix"],
            [[0.9, 0.2], [0.1, 0.8]],
        )

    def test_malformed_json_fails_before_backend_execution(self):
        backend = self._backend()

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "readout_matrices.json"
            cache_path.write_text("{not-json", encoding="utf-8")
            function = load_cell_namespace(
                "def build_readout_calibration_matrices"
            )["build_readout_calibration_matrices"]
            with self.assertRaisesRegex(ValueError, "Cannot read"):
                function(backend, [2], cache_path=cache_path, verbose=False)

        backend.run.assert_not_called()

    def test_wrong_matrix_shape_fails_before_backend_execution(self):
        backend = self._backend()

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "readout_matrices.json"
            cache_path.write_text(
                json.dumps(self._cache([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])),
                encoding="utf-8",
            )
            function = load_cell_namespace(
                "def build_readout_calibration_matrices"
            )["build_readout_calibration_matrices"]
            with self.assertRaisesRegex(ValueError, "Invalid cached matrix"):
                function(backend, [2], cache_path=cache_path, verbose=False)

        backend.run.assert_not_called()


class QubitSelectorTests(unittest.TestCase):
    def test_selects_best_layout_and_transpiles_complete_sampler_batch(self):
        logical_state_circuit = object()
        sampler_circuits = [object(), object()]
        transpiled_circuits = [object(), object()]
        best_layout = [2, 5, 7, 8, 11, 13]
        reduced_coupling_map = object()

        backend = Mock()
        backend.coupling_map.reduce.return_value = reduced_coupling_map
        evaluator = Mock()
        evaluator.get_top_layouts.return_value = ([best_layout], [0.031])
        cost_evaluator = Mock(return_value=evaluator)
        perform_backend_transpilation = Mock(return_value=transpiled_circuits)

        namespace = load_cell_namespace(
            "def select_and_transpile_candidate",
            CostEvaluator=cost_evaluator,
            perform_backend_transpilation=perform_backend_transpilation,
        )
        transpiled, layout, cost = namespace[
            "select_and_transpile_candidate"
        ](
            backend,
            logical_state_circuit,
            sampler_circuits,
            candidate="candidate-a",
        )

        cost_evaluator.assert_called_once_with(
            backend=backend,
            quantum_circuit=logical_state_circuit,
        )
        evaluator.get_top_layouts.assert_called_once_with(num_layouts=1)
        backend.coupling_map.reduce.assert_called_once_with(mapping=best_layout)
        perform_backend_transpilation.assert_called_once_with(
            sampler_circuits,
            backend,
            best_layout,
            reduced_coupling_map,
            qiskit_optim_level=3,
        )
        self.assertIs(transpiled, transpiled_circuits)
        self.assertEqual(layout, best_layout)
        self.assertEqual(cost, 0.031)

    def test_no_layout_names_candidate_and_skips_transpilation(self):
        backend = Mock()
        evaluator = Mock()
        evaluator.get_top_layouts.return_value = ([], [])
        cost_evaluator = Mock(return_value=evaluator)
        perform_backend_transpilation = Mock()

        namespace = load_cell_namespace(
            "def select_and_transpile_candidate",
            CostEvaluator=cost_evaluator,
            perform_backend_transpilation=perform_backend_transpilation,
        )

        with self.assertRaisesRegex(RuntimeError, "candidate-b"):
            namespace["select_and_transpile_candidate"](
                backend,
                object(),
                [object()],
                candidate="candidate-b",
            )

        perform_backend_transpilation.assert_not_called()


class PretranspiledStateTests(unittest.TestCase):
    def test_compacts_physical_wires_back_to_logical_order(self):
        source = QuantumCircuit(4)
        source.x(3)
        source.cx(3, 1)
        source._layout = Mock()
        source._layout.final_index_layout.return_value = [3, 1]

        namespace = load_cell_namespace("def compact_pretranspiled_state")
        compact = namespace["compact_pretranspiled_state"](source)

        self.assertEqual(compact.num_qubits, 2)
        self.assertEqual(compact.count_ops(), {"x": 1, "cx": 1})
        self.assertEqual(
            [
                tuple(compact.find_bit(qubit).index for qubit in item.qubits)
                for item in compact.data
            ],
            [(0,), (0, 1)],
        )

    def test_requires_final_layout_metadata(self):
        namespace = load_cell_namespace("def compact_pretranspiled_state")

        with self.assertRaisesRegex(ValueError, "final layout"):
            namespace["compact_pretranspiled_state"](QuantumCircuit(2))


class NotebookPipelineContractTests(unittest.TestCase):
    def test_pipeline_uses_compiled_state_and_explicit_execution_backend(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        source = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        provider_line = next(
            line
            for line in source.splitlines()
            if "provider_garnet = IQMProvider" in line
        )

        self.assertNotIn("token=", provider_line)
        self.assertIn(
            "select_and_transpile_candidate(\n"
            "        transpile_backend, logical_state_circuit, sampler_circuits, candidate",
            source,
        )
        self.assertIn("compact_pretranspiled_state(qcsuptrans)", source)
        self.assertIn("backend=execution_backend", source)
        self.assertNotIn("backend=garnet_noisy_backend,", source)
        self.assertIn('"selected_layout": tuple(selected_layout)', source)
        self.assertIn('"selector_cost": selector_cost', source)
        self.assertIn('"transpiled_depth": isa_sampler_qc_1[0].depth()', source)
        self.assertIn('"transpiled_cz_count":', source)
        self.assertIn(
            "full_pipeline(\n"
            "        backend_garnet, garnet_noisy_backend, compact_qc, Esup",
            source,
        )


class DependencyContractTests(unittest.TestCase):
    def test_iqm_qubit_selector_is_declared(self):
        for relative_path in ("pyproject.toml", "requirements.txt"):
            with self.subTest(path=relative_path):
                content = (REPO_ROOT / relative_path).read_text(encoding="utf-8")
                self.assertIn("iqm-qubit-selector>=1,<2", content)


if __name__ == "__main__":
    unittest.main()
