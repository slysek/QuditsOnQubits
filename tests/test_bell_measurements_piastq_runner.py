from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import qudits_on_qubits.bell_measurements as bell_measurements


class PiastQBellRunnerTests(unittest.TestCase):
    def test_public_package_exports_aqt_runner(self):
        self.assertTrue(
            hasattr(bell_measurements, "compute_bell_value_from_counts_aqt")
        )

    def test_runs_all_ordered_circuits_in_one_piastq_job(self):
        for circuit_count in (1, 3, 5):
            with self.subTest(circuit_count=circuit_count):
                circuits = [object() for _ in range(circuit_count)]
                settings = [(f"A{index}",) for index in range(circuit_count)]
                counts = [
                    {format(index, "02b"): index + 1}
                    for index in range(circuit_count)
                ]
                terms = [object()]
                qutrit_bits = {setting: [(0, 1)] for setting in settings}
                outcome_map = {0: 0, 1: 1, 2: 2, 3: None}
                metadata = {
                    "setting_by_circuit_index": [list(setting) for setting in settings],
                    "terms": terms,
                    "qutrit_bit_indices_by_setting": qutrit_bits,
                    "physical_to_logical_outcome_map": outcome_map,
                    "d": 3,
                }
                backend = object()
                sampler_options = {"seed": 17}
                run_options = {"memory": True, "tag": "bell-test"}
                sampler_type = MagicMock(name="PiastQSampler")
                sampler = sampler_type.return_value
                job = sampler.run.return_value
                result = object()
                job.result.return_value = result
                job.counts.return_value = counts
                expected_counts_by_setting = dict(
                    zip(settings, counts, strict=True)
                )

                with (
                    patch(
                        "qudits_on_qubits.bell_measurements.piastq_runner."
                        "_load_piastq_sampler",
                        return_value=sampler_type,
                    ) as load_sampler,
                    patch(
                        "qudits_on_qubits.bell_measurements.piastq_runner."
                        "compute_bell_value_from_counts",
                        return_value=3 + 4j,
                    ) as compute_bell,
                ):
                    bell_value, execution = (
                        bell_measurements.compute_bell_value_from_counts_aqt(
                            tuple(circuits),
                            metadata,
                            backend=backend,
                            shots=211,
                            sampler_options=sampler_options,
                            run_options=run_options,
                            timeout=12.5,
                            poll_interval=0.25,
                        )
                    )

                load_sampler.assert_called_once_with()
                sampler_type.assert_called_once_with(
                    backend,
                    options=sampler_options,
                )
                sampler.run.assert_called_once_with(
                    circuits,
                    shots=211,
                    **run_options,
                )
                job.result.assert_called_once_with(
                    timeout=12.5,
                    poll_interval=0.25,
                )
                job.counts.assert_called_once_with()
                compute_bell.assert_called_once_with(
                    expected_counts_by_setting,
                    terms,
                    qutrit_bits,
                    outcome_map=outcome_map,
                    d=3,
                )
                self.assertEqual(bell_value, 3 + 4j)
                self.assertEqual(
                    execution,
                    {
                        "backend": backend,
                        "sampler": sampler,
                        "job": job,
                        "result": result,
                        "counts_by_setting": expected_counts_by_setting,
                        "circuits": circuits,
                        "shots": 211,
                    },
                )


if __name__ == "__main__":
    unittest.main()
