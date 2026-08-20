from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from qiskit import QuantumCircuit


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import qudits_on_qubits.bell_measurements as bell_measurements
from qudits_on_qubits.bell_measurements import piastq_runner


CFT_PIASTQ_AVAILABLE = importlib.util.find_spec("cft_piastq") is not None
QISKIT_AER_AVAILABLE = importlib.util.find_spec("qiskit_aer") is not None


def _metadata_for(settings: list[tuple[str, ...]]) -> dict[str, object]:
    return {
        "setting_by_circuit_index": [list(setting) for setting in settings],
        "terms": [object()],
        "qutrit_bit_indices_by_setting": {
            setting: [(0, 1)] for setting in settings
        },
        "physical_to_logical_outcome_map": {0: 0, 1: 1, 2: 2, 3: None},
        "d": 3,
    }


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

    def test_rejects_empty_circuit_list_before_loading_piastq(self):
        with (
            patch.object(
                piastq_runner,
                "_load_piastq_sampler",
                side_effect=AssertionError("PiastQ must not load"),
            ),
            self.assertRaisesRegex(ValueError, "at least one circuit"),
        ):
            piastq_runner.compute_bell_value_from_counts_aqt(
                [],
                _metadata_for([]),
                backend=object(),
            )

    def test_rejects_invalid_shots_before_loading_piastq(self):
        circuits = [object()]
        metadata = _metadata_for([("A0",)])
        for shots in (True, 0, -1, 1.5, "100"):
            with self.subTest(shots=shots):
                with (
                    patch.object(
                        piastq_runner,
                        "_load_piastq_sampler",
                        side_effect=AssertionError("PiastQ must not load"),
                    ),
                    self.assertRaisesRegex(
                        ValueError,
                        "shots must be a positive integer",
                    ),
                ):
                    piastq_runner.compute_bell_value_from_counts_aqt(
                        circuits,
                        metadata,
                        backend=object(),
                        shots=shots,  # type: ignore[arg-type]
                    )

    def test_rejects_nonpositive_poll_interval_before_loading_piastq(self):
        circuits = [object()]
        metadata = _metadata_for([("A0",)])
        for poll_interval in (True, 0, -0.5):
            with self.subTest(poll_interval=poll_interval):
                with (
                    patch.object(
                        piastq_runner,
                        "_load_piastq_sampler",
                        side_effect=AssertionError("PiastQ must not load"),
                    ),
                    self.assertRaisesRegex(
                        ValueError,
                        "poll_interval must be a positive number",
                    ),
                ):
                    piastq_runner.compute_bell_value_from_counts_aqt(
                        circuits,
                        metadata,
                        backend=object(),
                        poll_interval=poll_interval,
                    )

    def test_rejects_shots_inside_run_options(self):
        with (
            patch.object(
                piastq_runner,
                "_load_piastq_sampler",
                side_effect=AssertionError("PiastQ must not load"),
            ),
            self.assertRaisesRegex(
                ValueError,
                "pass shots via the shots argument",
            ),
        ):
            piastq_runner.compute_bell_value_from_counts_aqt(
                [object()],
                _metadata_for([("A0",)]),
                backend=object(),
                run_options={"shots": 200},
            )

    def test_rejects_circuit_setting_length_mismatch(self):
        with (
            patch.object(
                piastq_runner,
                "_load_piastq_sampler",
                side_effect=AssertionError("PiastQ must not load"),
            ),
            self.assertRaisesRegex(
                ValueError,
                "number of sampler_circuits must match metadata settings",
            ),
        ):
            piastq_runner.compute_bell_value_from_counts_aqt(
                [object()],
                _metadata_for([("A0",), ("A1",)]),
                backend=object(),
            )

    def test_rejects_duplicate_settings(self):
        duplicate = ("A0", "B0")
        with (
            patch.object(
                piastq_runner,
                "_load_piastq_sampler",
                side_effect=AssertionError("PiastQ must not load"),
            ),
            self.assertRaisesRegex(ValueError, "metadata settings must be unique"),
        ):
            piastq_runner.compute_bell_value_from_counts_aqt(
                [object(), object()],
                _metadata_for([duplicate, duplicate]),
                backend=object(),
            )

    def test_rejects_result_count_length_mismatch(self):
        settings = [("A0",), ("A1",)]
        job = MagicMock()
        job.result.return_value = object()
        job.counts.return_value = [{"00": 100}]
        sampler = MagicMock()
        sampler.run.return_value = job
        sampler_type = MagicMock(return_value=sampler)

        with (
            patch.object(
                piastq_runner,
                "_load_piastq_sampler",
                return_value=sampler_type,
            ),
            self.assertRaisesRegex(
                ValueError,
                "expected 2 count dictionaries, received 1",
            ),
        ):
            piastq_runner.compute_bell_value_from_counts_aqt(
                [object(), object()],
                _metadata_for(settings),
                backend=object(),
            )

    def test_missing_cft_piastq_reports_separate_install(self):
        with patch.dict(sys.modules, {"cft_piastq": None}):
            with self.assertRaisesRegex(
                ImportError,
                "install cft-piastq separately in this environment",
            ):
                piastq_runner._load_piastq_sampler()

    def test_backend_mode_is_opaque_to_the_runner(self):
        metadata = _metadata_for([("A0",)])
        for mode in ("auto", "managed", "direct"):
            with self.subTest(mode=mode):
                backend = SimpleNamespace(mode=mode)
                job = MagicMock()
                job.result.return_value = object()
                job.counts.return_value = [{"00": 32}]
                sampler = MagicMock()
                sampler.run.return_value = job
                sampler_type = MagicMock(return_value=sampler)

                with (
                    patch.object(
                        piastq_runner,
                        "_load_piastq_sampler",
                        return_value=sampler_type,
                    ),
                    patch.object(
                        piastq_runner,
                        "compute_bell_value_from_counts",
                        return_value=1.0 + 0.0j,
                    ),
                ):
                    piastq_runner.compute_bell_value_from_counts_aqt(
                        [object()],
                        metadata,
                        backend=backend,
                        shots=32,
                    )

                sampler_type.assert_called_once_with(backend, options={})

    @unittest.skipUnless(
        CFT_PIASTQ_AVAILABLE and QISKIT_AER_AVAILABLE,
        "requires cft-piastq and qiskit-aer",
    )
    def test_real_cft_piastq_fake_job_counts_feed_bell_postprocessing(self):
        from cft_piastq import PiastQClient

        circuit = QuantumCircuit(2, 2, name="piastq-fake-zero")
        circuit.measure([0, 1], [0, 1])
        setting = ("A0",)
        metadata = {
            "setting_by_circuit_index": [setting],
            "terms": [
                {
                    "settings": setting,
                    "powers": (1,),
                    "coeff": 1.0,
                }
            ],
            "qutrit_bit_indices_by_setting": {setting: [(0, 1)]},
            "physical_to_logical_outcome_map": {0: 0, 1: 1, 2: 2, 3: None},
            "d": 3,
        }
        client = PiastQClient(mode="fake", owner="bell-pipeline-test")
        backend = client.fake_backend(use_backend_noise=False)

        value, execution = piastq_runner.compute_bell_value_from_counts_aqt(
            [circuit],
            metadata,
            backend=backend,
            shots=32,
            sampler_options={"cft_job_name": "fake-bell-contract"},
        )

        counts = execution["counts_by_setting"][setting]
        self.assertEqual(sum(counts.values()), 32)
        self.assertAlmostEqual(value.real, 1.0)
        self.assertAlmostEqual(value.imag, 0.0)


if __name__ == "__main__":
    unittest.main()
