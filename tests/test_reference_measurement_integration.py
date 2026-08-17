from __future__ import annotations

import math
import unittest

import numpy as np

from qudits_on_qubits.bell_measurements import (
    ReferenceBellEvaluation,
    build_general_graph_bell_settings,
    build_sampler_circuits_for_candidate,
    evaluate_reference_bell_values_from_counts,
)
from qudits_on_qubits.bell_measurements.sampler_circuits import (
    _candidate_bell_settings_data,
)
from qudits_on_qubits.reference_experiments import get_reference_experiment


_OUTCOME_MAP = {0: 0, 1: 1, 2: 2, 3: None}
_QUTRIT_BITS = ((0, 1), (2, 3))


class _Edge:
    def __init__(self, u: int, v: int, weight: complex) -> None:
        self.tuple = (u, v)
        self._weight = weight

    def __getitem__(self, key: str) -> complex:
        if key != "weight":
            raise KeyError(key)
        return self._weight


class _Edges(list[_Edge]):
    def attributes(self) -> list[str]:
        return ["weight"]


class _Graph:
    def __init__(self, weight: complex = 1) -> None:
        self.es = _Edges([_Edge(0, 1, weight)])

    def vcount(self) -> int:
        return 2


def _two_qutrit_counts(
    counts: dict[str, int],
) -> tuple[
    dict[tuple[str | None, ...], dict[str, int]],
    dict[tuple[str | None, ...], tuple[tuple[int, int], ...]],
]:
    settings = get_reference_experiment("two_qutrit").measurement_settings()
    return (
        {setting: dict(counts) for setting in settings},
        {setting: _QUTRIT_BITS for setting in settings},
    )


class ReferenceMeasurementRegistryIntegrationTests(unittest.TestCase):
    def test_candidate_settings_are_converted_from_canonical_registry_specs(self) -> None:
        for candidate, expected_settings in (
            ("two_qutrit", 9),
            ("ghz3", 12),
            ("ame43", 13),
        ):
            with self.subTest(candidate=candidate):
                spec = get_reference_experiment(candidate)
                data = _candidate_bell_settings_data(
                    candidate,
                    d=3,
                    drop_conjugate_half=False,
                )

                self.assertEqual(data["candidate"], candidate)
                self.assertEqual(data["spec_hash"], spec.stable_hash())
                self.assertEqual(data["party_order"], spec.state.party_order)
                self.assertEqual(len(data["measurement_settings"]), expected_settings)
                self.assertEqual(
                    data["physical_to_logical_outcome_map"],
                    _OUTCOME_MAP,
                )
                self.assertEqual(
                    data["terms"][0]["coeff"],
                    spec.bell_functional.terms[0].sampling_coefficient(),
                )
                self.assertIsInstance(
                    data["observables_by_label"][spec.observables[0].label],
                    np.ndarray,
                )

    def test_candidate_alias_canonicalizes_all_registry_derived_metadata(self) -> None:
        canonical = _candidate_bell_settings_data(
            "two_qutrit",
            d=3,
            drop_conjugate_half=False,
        )
        alias = _candidate_bell_settings_data(
            "2qutrit",
            d=3,
            drop_conjugate_half=False,
        )

        self.assertEqual(alias["candidate"], "two_qutrit")
        for key in (
            "spec_hash",
            "party_order",
            "measurement_settings",
            "terms",
            "physical_to_logical_outcome_map",
        ):
            self.assertEqual(alias[key], canonical[key])
        for label, observable in canonical["observables_by_label"].items():
            np.testing.assert_array_equal(alias["observables_by_label"][label], observable)
            self.assertIsNot(alias["observables_by_label"][label], observable)

    def test_candidate_conversion_filters_conjugates_and_rejects_non_qutrit_dimension(
        self,
    ) -> None:
        data = _candidate_bell_settings_data(
            "two_qutrit",
            d=3,
            drop_conjugate_half=True,
        )

        self.assertTrue(data["terms"])
        self.assertTrue(
            all(term["graph_power"] == 1 for term in data["terms"])
        )
        self.assertEqual(
            [term["source"] for term in data["terms"]],
            [f"two_qutrit:{index}" for index in range(9)],
        )
        with self.assertRaisesRegex(ValueError, "only for d=3"):
            _candidate_bell_settings_data(
                "two_qutrit",
                d=2,
                drop_conjugate_half=False,
            )

    def test_two_qutrit_graph_settings_are_registry_derived(self) -> None:
        weight = 2.5 - 0.25j
        spec = get_reference_experiment("two_qutrit")
        data = build_general_graph_bell_settings(n=3, graph=_Graph(weight))

        self.assertEqual(data["candidate"], "two_qutrit")
        self.assertEqual(data["spec_hash"], spec.stable_hash())
        self.assertEqual(data["party_order"], spec.state.party_order)
        self.assertEqual(data["measurement_settings"], list(spec.measurement_settings()))
        self.assertEqual(data["physical_to_logical_outcome_map"], _OUTCOME_MAP)
        self.assertEqual(data["construction"], "two_qutrit_pdf")
        self.assertEqual(
            data["terms"][0]["coeff"],
            weight * spec.bell_functional.terms[0].sampling_coefficient(),
        )
        self.assertEqual(data["terms"][0]["source"], "two_qutrit_pdf:0-1")

    def test_sampler_metadata_uses_registry_map_for_non_monomial_isometry(self) -> None:
        from qiskit import QuantumCircuit

        qutrit_fourier = np.fft.fft(np.eye(4), norm="ortho")[:, :3]
        circuits, metadata = build_sampler_circuits_for_candidate(
            "2qutrit",
            QuantumCircuit(4),
            E=qutrit_fourier,
            add_measurements=False,
        )
        spec = get_reference_experiment("two_qutrit")

        self.assertEqual(len(circuits), 9)
        self.assertEqual(metadata["candidate"], "two_qutrit")
        self.assertEqual(metadata["spec_hash"], spec.stable_hash())
        self.assertEqual(metadata["physical_to_logical_outcome_map"], _OUTCOME_MAP)


class ReferenceBellEvaluationIntegrationTests(unittest.TestCase):
    def test_dual_leakage_reports_conditional_and_unconditional_values(self) -> None:
        counts, indices = _two_qutrit_counts({"0000": 8, "0011": 2})

        result = evaluate_reference_bell_values_from_counts(
            "two_qutrit",
            counts,
            indices,
            bit_order="qiskit",
        )

        self.assertIsInstance(result, ReferenceBellEvaluation)
        self.assertEqual(result.total_shots, 90)
        self.assertEqual(result.accepted_shots, 72)
        self.assertAlmostEqual(result.leakage_rate, 0.2, places=12)
        self.assertIsInstance(result.unconditional, complex)
        self.assertIsInstance(result.conditional, complex)
        self.assertTrue(math.isfinite(result.unconditional.real))
        self.assertTrue(math.isfinite(result.conditional.real))
        self.assertGreater(result.conditional.real, 3.0)
        self.assertAlmostEqual(result.conditional.imag, 0.0, places=12)
        self.assertAlmostEqual(
            result.unconditional.real,
            0.8 * result.conditional.real,
            places=10,
        )

    def test_all_leakage_has_defined_zero_values(self) -> None:
        counts, indices = _two_qutrit_counts({"0011": 5})

        result = evaluate_reference_bell_values_from_counts(
            "two_qutrit",
            counts,
            indices,
        )

        self.assertEqual(result.total_shots, 45)
        self.assertEqual(result.accepted_shots, 0)
        self.assertEqual(result.leakage_rate, 1.0)
        self.assertEqual(result.unconditional, 0j)
        self.assertEqual(result.conditional, 0j)

    def test_zero_total_counts_has_defined_zero_rate(self) -> None:
        counts, indices = _two_qutrit_counts({"0000": 0})

        result = evaluate_reference_bell_values_from_counts(
            "two_qutrit",
            counts,
            indices,
        )

        self.assertEqual(
            result,
            ReferenceBellEvaluation(
                unconditional=0j,
                conditional=0j,
                leakage_rate=0.0,
                total_shots=0,
                accepted_shots=0,
            ),
        )

    def test_missing_settings_and_invalid_counts_preserve_compute_errors(self) -> None:
        counts, indices = _two_qutrit_counts({"0000": 1})
        missing = get_reference_experiment("two_qutrit").measurement_settings()[0]
        del counts[missing]

        with self.assertRaises(KeyError):
            evaluate_reference_bell_values_from_counts(
                "two_qutrit",
                counts,
                indices,
            )

        counts, indices = _two_qutrit_counts({"0000": -1})
        with self.assertRaisesRegex(ValueError, "non-negative integers"):
            evaluate_reference_bell_values_from_counts(
                "two_qutrit",
                counts,
                indices,
            )


if __name__ == "__main__":
    unittest.main()
