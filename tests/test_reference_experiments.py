from dataclasses import FrozenInstanceError
from dataclasses import replace
from itertools import product
import json
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from qiskit.quantum_info import Statevector

from qudits_on_qubits.bell_functionals.bell_builders import (
    bell_terms,
    build_bell_operator,
    candidate_statevector,
    num_qutrits_for_candidate,
)
from qudits_on_qubits.bell_functionals.classical_bounds import (
    bound_for_candidate,
    brute_force_classical_bound,
)
from qudits_on_qubits.bell_functionals.encoding import (
    default_qutrit_encoding,
    encode_qutrit_state,
)
from qudits_on_qubits.bell_functionals.operators import (
    make_XZ_qutrit,
    make_measurement_observables_qutrit_d3,
    qutrit_lambda,
)
from qudits_on_qubits.benchmarks.direct_basis.circuits import (
    build_direct_basis_graph_state_circuit,
)
from qudits_on_qubits.core.graph_states import resolve_graph_state_or_raise
from qudits_on_qubits.reference_experiments import (
    REFERENCE_EXPERIMENTS,
    BellFactorSpec,
    BellTermSpec,
    EncodingSpec,
    ExpectedValueSpec,
    LocalObservableSpec,
    LogicalStateSpec,
    OutcomeConventionSpec,
    ReferenceExperimentSpec,
    _canonical_value,
    _lambda,
    _make_xz,
    _measurement_observables,
    _omega,
    _ordered_eigenbasis,
    _root_expectation_scale,
    get_encoding,
    get_reference_experiment,
    list_reference_experiments,
)


def _probability_bell_value(spec: ReferenceExperimentSpec) -> complex:
    state = spec.state.statevector()
    omega = np.exp(2j * np.pi / 3)
    bell_value = 0j

    for term in spec.bell_functional.terms:
        correlator = 0j
        for outcomes in product(range(3), repeat=len(term.factors)):
            local_projectors = {
                party: np.eye(3, dtype=complex)
                for party in spec.state.party_order
            }
            phase = 1 + 0j
            for factor, outcome in zip(term.factors, outcomes):
                basis, _ = spec.observable(factor.setting_label).ordered_eigenbasis()
                vector = basis[:, outcome]
                local_projectors[factor.party] = np.outer(vector, vector.conj())
                phase *= omega ** ((factor.outcome_power * outcome) % 3)

            projector = np.array([[1]], dtype=complex)
            for party in spec.state.party_order:
                projector = np.kron(projector, local_projectors[party])
            probability = np.vdot(state, projector @ state).real
            correlator += probability * phase

        bell_value += term.sampling_coefficient() * correlator

    return complex(bell_value)


def _brute_force_bound(spec: ReferenceExperimentSpec) -> float:
    keys = sorted(
        {
            (factor.party, factor.setting_label)
            for term in spec.bell_functional.terms
            for factor in term.factors
        }
    )
    key_indices = {key: index for index, key in enumerate(keys)}
    omega = np.exp(2j * np.pi / 3)
    best = -math.inf

    for assignment in product(range(3), repeat=len(keys)):
        value = 0j
        for term in spec.bell_functional.terms:
            phase = 1 + 0j
            for factor in term.factors:
                outcome = assignment[
                    key_indices[(factor.party, factor.setting_label)]
                ]
                phase *= omega ** (factor.outcome_power * outcome)
            value += term.sampling_coefficient() * phase
        best = max(best, value.real)

    return float(best)


class PublicReferenceApiTests(unittest.TestCase):
    def test_top_level_reference_exports_resolve_registry_objects(self) -> None:
        from qudits_on_qubits import (
            EncodingSpec as PublicEncodingSpec,
            ReferenceExperimentSpec as PublicReferenceExperimentSpec,
            get_encoding as public_get_encoding,
            get_reference_experiment as public_get_reference_experiment,
            list_reference_experiments as public_list_reference_experiments,
        )

        encoding = public_get_encoding("canonical_ez")
        reference = public_get_reference_experiment("ghz3")

        self.assertIs(PublicEncodingSpec, EncodingSpec)
        self.assertIs(PublicReferenceExperimentSpec, ReferenceExperimentSpec)
        self.assertIs(encoding, get_encoding("canonical_ez"))
        self.assertIs(reference, get_reference_experiment("ghz3"))
        self.assertEqual(
            public_list_reference_experiments(),
            ("two_qutrit", "ghz3", "ame43"),
        )


class ReferenceExperimentsTests(unittest.TestCase):
    def test_canonical_encoding_has_expected_code_and_leakage_geometry(self) -> None:
        encoding = get_encoding("canonical_ez")

        expected = np.array(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 0],
            ],
            dtype=complex,
        )
        matrix = encoding.as_array()
        leakage = encoding.leakage_array()

        self.assertEqual(matrix.shape, (4, 3))
        np.testing.assert_array_equal(matrix, expected)
        np.testing.assert_allclose(matrix.conj().T @ matrix, np.eye(3))
        self.assertEqual(leakage.shape, (4, 1))
        np.testing.assert_array_equal(
            leakage,
            np.array([[0], [0], [0], [1]], dtype=complex),
        )
        np.testing.assert_allclose(matrix.conj().T @ leakage, np.zeros((3, 1)))

    def test_encoding_arrays_are_defensive_copies(self) -> None:
        encoding = get_encoding("canonical_ez")
        matrix = encoding.as_array()
        leakage = encoding.leakage_array()

        matrix[0, 0] = 7
        leakage[3, 0] = 7

        self.assertEqual(encoding.as_array()[0, 0], 1)
        self.assertEqual(encoding.leakage_array()[3, 0], 1)

    def test_encoding_spec_is_frozen(self) -> None:
        encoding = get_encoding("canonical_ez")

        with self.assertRaises(FrozenInstanceError):
            encoding.encoding_id = "changed"

    def test_encoding_rejects_non_isometric_code_matrix(self) -> None:
        with self.assertRaisesRegex(ValueError, r"E\^dagger E"):
            EncodingSpec(
                encoding_id="bad",
                logical_dimension=3,
                physical_qubits_per_qutrit=2,
                isometry=(
                    (1, 0, 0),
                    (1, 0, 0),
                    (0, 0, 1),
                    (0, 0, 0),
                ),
                leakage_basis=((0,), (0,), (0,), (1,)),
            )

    def test_weighted_graph_state_has_expected_analytical_amplitudes(self) -> None:
        state_spec = LogicalStateSpec("test", 3, 2, (0, 1), ((0, 1, 1),))

        state = state_spec.statevector()
        omega = np.exp(2j * np.pi / 3)

        self.assertEqual(state.shape, (9,))
        np.testing.assert_allclose(np.linalg.norm(state), 1)
        np.testing.assert_allclose(state[0], 1 / 3)
        np.testing.assert_allclose(state[5], omega**2 / 3)

    def test_legacy_edges_repeat_weight_two_edge(self) -> None:
        state_spec = LogicalStateSpec("test", 3, 2, (0, 1), ((0, 1, 2),))

        self.assertEqual(state_spec.legacy_edges(), ((0, 1), (0, 1)))

    def test_logical_state_rejects_invalid_party_and_edge_metadata(self) -> None:
        with self.assertRaisesRegex(ValueError, "party_order"):
            LogicalStateSpec("bad-order", 3, 2, (0, 0), ((0, 1, 1),))

        with self.assertRaisesRegex(ValueError, "duplicate"):
            LogicalStateSpec(
                "duplicate-edge",
                3,
                2,
                (0, 1),
                ((0, 1, 1), (0, 1, 2)),
            )

    def test_logical_state_freezes_mutable_metadata_inputs(self) -> None:
        party_order = [0, 1]
        weighted_edges = [[0, 1, 1]]

        state_spec = LogicalStateSpec(
            "mutable-inputs",
            3,
            2,
            party_order,
            weighted_edges,
        )
        party_order[0] = 1
        weighted_edges[0][0] = 1

        self.assertEqual(state_spec.party_order, (0, 1))
        self.assertEqual(state_spec.weighted_edges, ((0, 1, 1),))

    def test_logical_state_rejects_noninteger_party_values(self) -> None:
        for party_order in ((0.0, 1), ("0", 1), (False, 1)):
            with self.subTest(party_order=party_order):
                with self.assertRaisesRegex(ValueError, "party_order.*integers"):
                    LogicalStateSpec(
                        "bad-party-type",
                        3,
                        2,
                        party_order,
                        ((0, 1, 1),),
                    )

    def test_logical_state_rejects_noninteger_weighted_edge_values(self) -> None:
        for weighted_edges in (
            ((0, "1", 1),),
            ((0, 1.0, 1),),
            ((0, 1, 1.0),),
            ((0, 1, True),),
        ):
            with self.subTest(weighted_edges=weighted_edges):
                with self.assertRaisesRegex(ValueError, "weighted edge.*integers"):
                    LogicalStateSpec(
                        "bad-edge-type",
                        3,
                        2,
                        (0, 1),
                        weighted_edges,
                    )

    def test_logical_state_rejects_empty_or_nonstring_state_id(self) -> None:
        for state_id in ("", " \t", None, 7):
            with self.subTest(state_id=state_id):
                with self.assertRaisesRegex(ValueError, "state_id.*nonempty string"):
                    LogicalStateSpec(
                        state_id,
                        3,
                        2,
                        (0, 1),
                        ((0, 1, 1),),
                    )


class ReferenceExperimentRegistryTests(unittest.TestCase):
    def test_registry_has_stable_order_and_legacy_alias(self) -> None:
        self.assertEqual(
            list_reference_experiments(),
            ("two_qutrit", "ghz3", "ame43"),
        )
        self.assertIs(
            get_reference_experiment("2qutrit"),
            get_reference_experiment("two_qutrit"),
        )
        self.assertIs(
            get_reference_experiment("  ghz3  "),
            REFERENCE_EXPERIMENTS["ghz3"],
        )

    def test_unknown_reference_id_lists_canonical_ids(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "two_qutrit, ghz3, ame43",
        ):
            get_reference_experiment("missing")

    def test_registry_metadata_matches_reference_contract(self) -> None:
        expected = {
            "two_qutrit": (2, 18, 9, 6.0),
            "ghz3": (3, 24, 12, 6.0),
            "ame43": (4, 26, 13, 8.0),
        }

        for experiment_id, contract in expected.items():
            with self.subTest(experiment_id=experiment_id):
                spec = get_reference_experiment(experiment_id)
                self.assertEqual(
                    (
                        spec.state.num_parties,
                        len(spec.bell_functional.terms),
                        len(spec.measurement_settings()),
                        spec.expected.ideal_bell_value,
                    ),
                    contract,
                )
                self.assertEqual(
                    len(spec.measurement_settings()),
                    spec.expected_unique_measurement_settings,
                )

    def test_ame43_graph_outcomes_and_leakage_policy_match_contract(self) -> None:
        spec = get_reference_experiment("ame43")

        self.assertEqual(
            spec.state.weighted_edges,
            ((0, 1, 1), (0, 3, 1), (1, 2, 1), (2, 3, 2)),
        )
        self.assertEqual(
            dict(spec.outcome_convention.measurement_basis_index_map),
            {0: 0, 1: 1, 2: 2, 3: None},
        )
        self.assertTrue(spec.leakage_policy.report_rate)
        self.assertTrue(spec.leakage_policy.compute_unconditional)
        self.assertTrue(spec.leakage_policy.compute_conditional)
        self.assertEqual(spec.leakage_policy.leakage_contribution, 0)

    def test_fixed_graph_resolver_preserves_registry_legacy_edges(self) -> None:
        expected_edges = {
            "two_qutrit": ((0, 1),),
            "ghz3": ((0, 1), (0, 2)),
            "ame43": ((0, 1), (0, 3), (1, 2), (2, 3), (2, 3)),
        }

        for state_name, edges in expected_edges.items():
            with self.subTest(state_name=state_name):
                self.assertEqual(
                    resolve_graph_state_or_raise(state_name).edges,
                    edges,
                )

    def test_direct_basis_fixed_graphs_match_reference_states(self) -> None:
        encoding = get_encoding("canonical_ez").as_array()

        for state_name, num_parties in (
            ("two_qutrit", 2),
            ("ghz3", 3),
            ("ame43", 4),
        ):
            with self.subTest(state_name=state_name):
                spec = get_reference_experiment(state_name)
                self.assertEqual(spec.state.num_parties, num_parties)
                circuit = build_direct_basis_graph_state_circuit(
                    state_name,
                    encoding,
                )
                actual = Statevector.from_instruction(circuit).data
                expected = encode_qutrit_state(
                    spec.state.statevector(),
                    encoding,
                    num_parties,
                )

                fidelity = abs(np.vdot(expected, actual)) ** 2
                self.assertAlmostEqual(fidelity, 1.0, places=10)

    def test_registry_and_nested_values_are_immutable(self) -> None:
        spec = get_reference_experiment("two_qutrit")

        with self.assertRaises(TypeError):
            REFERENCE_EXPERIMENTS["new"] = spec
        with self.assertRaises(FrozenInstanceError):
            spec.experiment_id = "changed"
        with self.assertRaises(FrozenInstanceError):
            spec.observables[0].label = "changed"

    def test_reference_operators_reproduce_ideal_values(self) -> None:
        for experiment_id in list_reference_experiments():
            with self.subTest(experiment_id=experiment_id):
                spec = get_reference_experiment(experiment_id)
                operator = spec.logical_bell_operator()
                state = spec.state.statevector()
                value = np.vdot(state, operator @ state)

                np.testing.assert_allclose(operator, operator.conj().T, atol=1e-9)
                np.testing.assert_allclose(
                    value,
                    spec.expected.ideal_bell_value,
                    atol=spec.expected.absolute_tolerance,
                )

    def test_term_helpers_expose_party_ordered_settings_and_powers(self) -> None:
        spec = get_reference_experiment("two_qutrit")
        term = spec.bell_functional.terms[0]

        self.assertEqual(spec.setting_for_term(term), ("A0", "B0"))
        self.assertEqual(spec.powers_for_term(term), (1, 1))
        self.assertEqual(
            spec.measurement_settings(),
            tuple((f"A{a}", f"B{b}") for a in range(3) for b in range(3)),
        )

    def test_serialization_is_canonical_and_hash_is_stable(self) -> None:
        spec = get_reference_experiment("ghz3")
        payload = spec.to_dict()

        self.assertEqual(payload["schema_version"], "reference-experiment-v1")
        serialized_bound = payload["bell_functional"]["classical_bound"]
        self.assertEqual(float(serialized_bound), spec.bell_functional.classical_bound)
        self.assertEqual(
            serialized_bound,
            format(spec.bell_functional.classical_bound, ".17g"),
        )
        self.assertIsInstance(payload["bell_functional"]["terms"][0]["coefficient"], list)
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        self.assertEqual(spec.stable_hash(), spec.stable_hash())
        self.assertEqual(len(spec.stable_hash()), 64)

    def test_adjacent_float_values_serialize_and_hash_distinctly(self) -> None:
        spec = get_reference_experiment("two_qutrit")
        value = 1.234567890123456
        adjacent = math.nextafter(value, math.inf)
        first = replace(spec, expected=ExpectedValueSpec(value, 1e-10))
        second = replace(spec, expected=ExpectedValueSpec(adjacent, 1e-10))

        first_value = first.to_dict()["expected"]["ideal_bell_value"]
        second_value = second.to_dict()["expected"]["ideal_bell_value"]

        self.assertNotEqual(first_value, second_value)
        self.assertEqual(float(first_value), value)
        self.assertEqual(float(second_value), adjacent)
        self.assertNotEqual(first.stable_hash(), second.stable_hash())


class ScientificReferenceTests(unittest.TestCase):
    def test_operator_probability_and_analytical_values_agree(self) -> None:
        expected_ideal_values = {
            "two_qutrit": 6.0,
            "ghz3": 6.0,
            "ame43": 8.0,
        }

        for experiment_id in list_reference_experiments():
            with self.subTest(experiment_id=experiment_id):
                spec = get_reference_experiment(experiment_id)
                operator = spec.logical_bell_operator()
                state = spec.state.statevector()
                matrix_value = np.vdot(state, operator @ state)
                probability_value = _probability_bell_value(spec)

                np.testing.assert_allclose(
                    operator,
                    operator.conj().T,
                    rtol=0,
                    atol=1e-10,
                )
                self.assertAlmostEqual(
                    matrix_value.real,
                    expected_ideal_values[experiment_id],
                    places=10,
                )
                self.assertAlmostEqual(matrix_value.imag, 0.0, places=10)
                self.assertAlmostEqual(
                    probability_value.real,
                    matrix_value.real,
                    places=10,
                )
                self.assertAlmostEqual(probability_value.imag, 0.0, places=10)

    def test_exhaustive_classical_bounds_match_frozen_values(self) -> None:
        decimal_places = {
            "two_qutrit": 10,
            "ghz3": 10,
            "ame43": 5,
        }

        for experiment_id in list_reference_experiments():
            with self.subTest(experiment_id=experiment_id):
                spec = get_reference_experiment(experiment_id)
                self.assertAlmostEqual(
                    _brute_force_bound(spec),
                    spec.bell_functional.classical_bound,
                    places=decimal_places[experiment_id],
                )

    def test_serialization_and_hash_are_deterministic_for_every_reference(self) -> None:
        for experiment_id in list_reference_experiments():
            with self.subTest(experiment_id=experiment_id):
                spec = get_reference_experiment(experiment_id)

                self.assertEqual(spec.to_dict(), spec.to_dict())
                self.assertEqual(spec.to_dict()["experiment_id"], experiment_id)
                self.assertEqual(spec.stable_hash(), spec.stable_hash())
                self.assertEqual(len(spec.stable_hash()), 64)


class LegacyBellFunctionalCompatibilityTests(unittest.TestCase):
    def test_num_qutrits_delegates_to_reference_registry(self) -> None:
        fake_spec = SimpleNamespace(state=SimpleNamespace(num_parties=7))

        with patch(
            "qudits_on_qubits.bell_functionals.bell_builders.get_reference_experiment",
            return_value=fake_spec,
        ) as getter:
            parties = num_qutrits_for_candidate("custom")

        self.assertEqual(parties, 7)
        getter.assert_called_once_with("custom")

    def test_legacy_terms_preserve_registry_factor_contract(self) -> None:
        for experiment_id in list_reference_experiments():
            with self.subTest(experiment_id=experiment_id):
                spec = get_reference_experiment(experiment_id)
                legacy_terms = bell_terms(experiment_id)

                self.assertEqual(len(legacy_terms), len(spec.bell_functional.terms))
                for legacy_term, registry_term in zip(
                    legacy_terms,
                    spec.bell_functional.terms,
                ):
                    self.assertEqual(legacy_term.coefficient, registry_term.coefficient)
                    self.assertEqual(len(legacy_term.factors), len(registry_term.factors))
                    for legacy_factor, registry_factor in zip(
                        legacy_term.factors,
                        registry_term.factors,
                    ):
                        observable = spec.observable(registry_factor.setting_label)
                        numeric_suffix = registry_factor.setting_label.lstrip(
                            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                        )
                        self.assertEqual(
                            (
                                legacy_factor.party,
                                legacy_factor.setting,
                                legacy_factor.power,
                                legacy_factor.label,
                            ),
                            (
                                registry_factor.party,
                                int(numeric_suffix),
                                registry_factor.outcome_power,
                                registry_factor.setting_label,
                            ),
                        )
                        np.testing.assert_allclose(
                            legacy_factor.matrix,
                            registry_factor.logical_operator(observable),
                            rtol=0,
                            atol=1e-10,
                        )

    def test_legacy_term_base_observables_are_defensive_arrays(self) -> None:
        spec = get_reference_experiment("two_qutrit")
        registry_observable = spec.observable("A0").as_array()
        first_factor = next(
            factor
            for term in bell_terms("two_qutrit")
            for factor in term.factors
            if factor.label == "A0" and factor.power == 2
        )
        expected = first_factor.base_observable.copy()
        first = first_factor.base_observable
        first[0, 0] = 123 + 456j

        np.testing.assert_allclose(
            spec.observable("A0").as_array(),
            registry_observable,
        )
        fresh_factor = next(
            factor
            for term in bell_terms("two_qutrit")
            for factor in term.factors
            if factor.label == "A0" and factor.power == 2
        )
        np.testing.assert_allclose(
            fresh_factor.base_observable,
            expected,
        )

    def test_legacy_factor_fields_store_separate_effective_operator_arrays(self) -> None:
        for experiment_id in list_reference_experiments():
            with self.subTest(experiment_id=experiment_id):
                factors = [
                    factor
                    for term in bell_terms(experiment_id)
                    for factor in term.factors
                ]
                power_two = [factor for factor in factors if factor.power == 2]
                other_powers = [factor for factor in factors if factor.power != 2]
                self.assertTrue(power_two)

                for factor in power_two + other_powers:
                    with self.subTest(
                        label=factor.label,
                        party=factor.party,
                        power=factor.power,
                    ):
                        self.assertIsNot(
                            factor.base_observable,
                            factor.matrix,
                        )
                        np.testing.assert_allclose(
                            factor.base_observable,
                            factor.matrix,
                            rtol=0,
                            atol=0,
                        )

    def test_legacy_states_and_operators_match_registry_ideals(self) -> None:
        encoding = default_qutrit_encoding()

        for experiment_id in list_reference_experiments():
            with self.subTest(experiment_id=experiment_id):
                spec = get_reference_experiment(experiment_id)
                state = candidate_statevector(experiment_id, encoding)
                expected_state = encode_qutrit_state(
                    spec.state.statevector(),
                    encoding,
                    spec.state.num_parties,
                )

                self.assertEqual(
                    num_qutrits_for_candidate(experiment_id),
                    spec.state.num_parties,
                )
                np.testing.assert_allclose(
                    state.data,
                    expected_state,
                    rtol=0,
                    atol=1e-10,
                )
                value = state.expectation_value(
                    build_bell_operator(experiment_id, encoding)
                )
                self.assertAlmostEqual(
                    value.real,
                    spec.expected.ideal_bell_value,
                    places=10,
                )
                self.assertAlmostEqual(value.imag, 0.0, places=10)

        self.assertEqual(num_qutrits_for_candidate("2qutrit"), 2)

    def test_legacy_operator_helpers_delegate_to_canonical_primitives(self) -> None:
        x, z, omega = make_XZ_qutrit()
        canonical_x, canonical_z = _make_xz()

        np.testing.assert_allclose(x, canonical_x, rtol=0, atol=0)
        np.testing.assert_allclose(z, canonical_z, rtol=0, atol=0)
        self.assertEqual(omega, np.exp(2j * np.pi / 3))
        for power in (1, 2):
            with self.subTest(power=power):
                self.assertEqual(qutrit_lambda(power), _lambda(power))
                actual = make_measurement_observables_qutrit_d3(power)
                canonical = _measurement_observables(power)
                self.assertIsInstance(actual, list)
                for actual_matrix, canonical_matrix in zip(actual, canonical):
                    np.testing.assert_allclose(
                        actual_matrix,
                        canonical_matrix,
                        rtol=0,
                        atol=0,
                    )

        first = make_measurement_observables_qutrit_d3()
        expected = _measurement_observables(1)[0]
        first[0][0, 0] = 123 + 456j
        np.testing.assert_allclose(
            make_measurement_observables_qutrit_d3()[0],
            expected,
            rtol=0,
            atol=0,
        )

    def test_legacy_bounds_match_registry_values(self) -> None:
        places = {"two_qutrit": 10, "ghz3": 10, "ame43": 5}

        for experiment_id in list_reference_experiments():
            with self.subTest(experiment_id=experiment_id):
                spec = get_reference_experiment(experiment_id)
                frozen = bound_for_candidate(experiment_id)
                brute_force = brute_force_classical_bound(experiment_id)

                self.assertEqual(frozen.candidate, spec.experiment_id)
                self.assertEqual(frozen.quantum, spec.expected.ideal_bell_value)
                self.assertEqual(
                    frozen.classical,
                    spec.bell_functional.classical_bound,
                )
                self.assertEqual(
                    frozen.classical_source,
                    spec.bell_functional.classical_bound_source,
                )
                self.assertEqual(brute_force.candidate, spec.experiment_id)
                self.assertEqual(
                    brute_force.quantum,
                    spec.expected.ideal_bell_value,
                )
                self.assertEqual(
                    brute_force.classical_source,
                    "numeric_bruteforce",
                )
                self.assertAlmostEqual(
                    brute_force.classical,
                    spec.bell_functional.classical_bound,
                    places=places[experiment_id],
                )

        alias = bound_for_candidate("2qutrit")
        self.assertEqual(alias.candidate, "two_qutrit")


class ReferenceExperimentModelTests(unittest.TestCase):
    def test_numpy_integer_metadata_is_normalized_to_python_ints(self) -> None:
        base_encoding = get_encoding("canonical_ez")
        encoding = EncodingSpec(
            encoding_id="numpy-integers",
            logical_dimension=np.int64(3),
            physical_qubits_per_qutrit=np.int64(2),
            isometry=base_encoding.isometry,
            leakage_basis=base_encoding.leakage_basis,
        )
        state = LogicalStateSpec(
            "numpy-integers",
            np.int64(3),
            np.int64(2),
            (np.int64(0), np.int64(1)),
            ((np.int64(0), np.int64(1), np.int64(1)),),
        )
        factor = BellFactorSpec(
            np.int64(0),
            "A0",
            np.int64(1),
        )
        outcomes = OutcomeConventionSpec(
            local_dimension=np.int64(3),
            logical_outcomes=(np.int64(0), np.int64(1), np.int64(2)),
            leakage_outcome=np.int64(9),
            measurement_basis_index_map=(
                (np.int64(0), np.int64(0)),
                (np.int64(1), np.int64(1)),
                (np.int64(2), np.int64(2)),
                (np.int64(3), np.int64(9)),
            ),
            root_phase_sign=np.int64(1),
        )
        spec = replace(
            get_reference_experiment("two_qutrit"),
            state=state,
            outcome_convention=outcomes,
            expected_unique_measurement_settings=np.int64(9),
        )

        integer_values = (
            encoding.logical_dimension,
            encoding.physical_qubits_per_qutrit,
            state.local_dimension,
            state.num_parties,
            *state.party_order,
            *state.weighted_edges[0],
            factor.party,
            factor.outcome_power,
            outcomes.local_dimension,
            *outcomes.logical_outcomes,
            outcomes.leakage_outcome,
            *(value for entry in outcomes.measurement_basis_index_map for value in entry),
            outcomes.root_phase_sign,
            spec.expected_unique_measurement_settings,
        )
        self.assertTrue(all(type(value) is int for value in integer_values))
        self.assertEqual(_canonical_value(np.int64(7)), 7)

    def test_integer_metadata_rejects_float_lookalikes(self) -> None:
        with self.assertRaisesRegex(ValueError, "logical_dimension"):
            EncodingSpec(
                "float-dimension",
                3.0,
                2,
                get_encoding("canonical_ez").isometry,
                get_encoding("canonical_ez").leakage_basis,
            )
        with self.assertRaisesRegex(ValueError, "local_dimension"):
            LogicalStateSpec(
                "float-dimension",
                3.0,
                2,
                (0, 1),
                ((0, 1, 1),),
            )
        with self.assertRaisesRegex(ValueError, "outcomes"):
            OutcomeConventionSpec(
                3,
                (0, 1, 2),
                None,
                ((0, 0.0), (1, 1), (2, 2), (3, None)),
                1,
            )
        with self.assertRaisesRegex(ValueError, "root_phase_sign"):
            OutcomeConventionSpec(
                3,
                (0, 1, 2),
                None,
                ((0, 0), (1, 1), (2, 2), (3, None)),
                1.0,
            )

    def test_lambda_phase_matches_reference_and_rejects_other_powers(self) -> None:
        np.testing.assert_allclose(_lambda(1), np.exp(1j * np.pi / 18))
        np.testing.assert_allclose(_lambda(2), np.exp(-1j * np.pi / 18))
        with self.assertRaisesRegex(ValueError, "1 or 2"):
            _lambda(0)

    def test_local_observable_defensively_freezes_a_unitary(self) -> None:
        _, z = _make_xz()
        matrix = z.copy()
        observable = LocalObservableSpec("Z", matrix)
        matrix[0, 0] = 9

        basis, gamma = observable.ordered_eigenbasis()

        self.assertEqual(observable.matrix[0][0], 1)
        np.testing.assert_allclose(basis.conj().T @ basis, np.eye(3), atol=1e-10)
        np.testing.assert_allclose(gamma, 1, atol=1e-10)

    def test_local_observable_rejects_bad_label_shape_unitarity_and_spectrum(self) -> None:
        for label, matrix, message in (
            ("", np.eye(3), "label"),
            ("bad", np.eye(2), "3x3"),
            ("bad", np.diag([1, 1, 1]), "spectrum"),
            ("bad", np.diag([1, 1, 2]), "unitary"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    LocalObservableSpec(label, matrix)

    def test_ordered_eigenbasis_is_deterministic_and_phase_fixed(self) -> None:
        x, _ = _make_xz()

        first_basis, first_gamma = _ordered_eigenbasis(x)
        second_basis, second_gamma = _ordered_eigenbasis(x)

        np.testing.assert_allclose(first_basis, second_basis, atol=1e-12)
        np.testing.assert_allclose(first_gamma, second_gamma, atol=1e-12)
        for column in range(3):
            vector = first_basis[:, column]
            pivot = int(np.argmax(np.abs(vector)))
            self.assertAlmostEqual(vector[pivot].imag, 0.0, places=12)
            self.assertGreater(vector[pivot].real, 0)

    def test_measurement_helpers_have_root_spectrum_and_consistent_scale(self) -> None:
        first = _measurement_observables(1)
        second = _measurement_observables(2)

        self.assertEqual(len(first), 3)
        for setting in range(3):
            np.testing.assert_allclose(
                first[setting].conj().T @ first[setting],
                np.eye(3),
                atol=1e-10,
            )
            scale = _root_expectation_scale(
                LocalObservableSpec(f"A{setting}", first[setting]),
                second[setting],
                2,
            )
            self.assertAlmostEqual(abs(scale), 1.0, places=10)

    def test_bell_factor_and_term_apply_operator_scale_once(self) -> None:
        _, z = _make_xz()
        observable = LocalObservableSpec("Z", z)
        factor = BellFactorSpec(0, "Z", 2, 2j)
        term = BellTermSpec(3 - 1j, (factor,))

        np.testing.assert_allclose(
            factor.logical_operator(observable),
            2j * np.diag([1, _omega() ** 2, _omega()]),
            atol=1e-10,
        )
        self.assertEqual(term.sampling_coefficient(), (3 - 1j) * 2j)

    def test_reference_validation_rejects_broken_term_contracts(self) -> None:
        spec = get_reference_experiment("two_qutrit")
        term = spec.bell_functional.terms[0]
        invalid_factors = (
            (replace(term.factors[0], party=8), "party"),
            (replace(term.factors[0], setting_label="unknown"), "unknown observable"),
            (replace(term.factors[0], outcome_power=3), "outcome_power"),
        )

        for factor, message in invalid_factors:
            with self.subTest(message=message):
                bad_term = replace(term, factors=(factor, *term.factors[1:]))
                bad_functional = replace(
                    spec.bell_functional,
                    terms=(bad_term, *spec.bell_functional.terms[1:]),
                )
                with self.assertRaisesRegex(ValueError, message):
                    replace(spec, bell_functional=bad_functional)

        duplicate_party_term = replace(term, factors=(term.factors[0], term.factors[0]))
        with self.assertRaisesRegex(ValueError, "at most one factor"):
            replace(
                spec,
                bell_functional=replace(
                    spec.bell_functional,
                    terms=(duplicate_party_term, *spec.bell_functional.terms[1:]),
                ),
            )

    def test_reference_validation_rejects_schema_duplicates_and_setting_count(self) -> None:
        spec = get_reference_experiment("two_qutrit")

        with self.assertRaisesRegex(ValueError, "schema_version"):
            replace(spec, schema_version="reference-experiment-v2")
        with self.assertRaisesRegex(ValueError, "unique observable"):
            replace(spec, observables=(*spec.observables, spec.observables[0]))
        with self.assertRaisesRegex(ValueError, "measurement setting"):
            replace(spec, expected_unique_measurement_settings=8)


if __name__ == "__main__":
    unittest.main()
