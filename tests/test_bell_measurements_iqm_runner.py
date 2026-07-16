from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from qiskit import QuantumCircuit


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qudits_on_qubits.bell_measurements import (
    run_iqm_sampler_circuits_to_counts_by_setting,
    run_sampler_circuits_to_counts_by_setting,
)


class _CountsResult:
    def __init__(self, counts):
        self._counts = list(counts)

    def get_counts(self, index=0):
        return self._counts[index]


class _Job:
    def __init__(self, counts):
        self._result = _CountsResult(counts)

    def result(self):
        return self._result


class _RecordingBackend:
    def __init__(self, counts):
        self.counts = counts
        self.circuits = None
        self.options = None

    def run(self, circuits, **options):
        self.circuits = list(circuits)
        self.options = dict(options)
        return _Job(self.counts)


class _RecordingSampler(_RecordingBackend):
    pass


def _measured_circuit() -> QuantumCircuit:
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)
    return circuit


class BellMeasurementIqmRunnerTests(unittest.TestCase):
    def test_backend_run_is_default_without_transpilation(self):
        circuit = _measured_circuit()
        backend = _RecordingBackend([{"0": 7}])
        metadata = {"setting_by_circuit_index": [("A0",)]}

        with patch(
            "qudits_on_qubits.bell_measurements.sampler_circuits._transpile_circuits",
            side_effect=AssertionError("transpile should not be called"),
        ):
            counts_by_setting, execution = run_sampler_circuits_to_counts_by_setting(
                [circuit],
                metadata,
                shots=7,
                backend=backend,
            )

        self.assertEqual(counts_by_setting, {("A0",): {"0": 7}})
        self.assertIs(backend.circuits[0], circuit)
        self.assertEqual(backend.options, {"shots": 7})
        self.assertEqual(execution["execution_target"], "backend")
        self.assertFalse(execution["transpile_circuits"])

    def test_sampler_default_path_is_also_without_transpilation(self):
        circuit = _measured_circuit()
        sampler = _RecordingSampler([{"1": 3}])
        metadata = {"setting_by_circuit_index": [("B1",)]}

        with patch(
            "qudits_on_qubits.bell_measurements.sampler_circuits._transpile_circuits",
            side_effect=AssertionError("transpile should not be called"),
        ):
            counts_by_setting, execution = run_sampler_circuits_to_counts_by_setting(
                [circuit],
                metadata,
                shots=3,
                sampler=sampler,
            )

        self.assertEqual(counts_by_setting, {("B1",): {"1": 3}})
        self.assertIs(sampler.circuits[0], circuit)
        self.assertEqual(sampler.options, {"shots": 3})
        self.assertEqual(execution["execution_target"], "sampler")
        self.assertFalse(execution["transpile_circuits"])

    def test_iqm_helper_runs_backend_without_transpilation(self):
        circuit = _measured_circuit()
        backend = _RecordingBackend([{"0": 4, "1": 2}])
        metadata = {"setting_by_circuit_index": [("C0",)]}

        counts_by_setting, execution = run_iqm_sampler_circuits_to_counts_by_setting(
            [circuit],
            metadata,
            shots=6,
            backend=backend,
            run_options={"use_timeslot": True},
        )

        self.assertEqual(counts_by_setting, {("C0",): {"0": 4, "1": 2}})
        self.assertIs(backend.circuits[0], circuit)
        self.assertEqual(backend.options, {"shots": 6, "use_timeslot": True})
        self.assertEqual(execution["backend"], backend)
        self.assertFalse(execution["transpile_circuits"])


if __name__ == "__main__":
    unittest.main()
