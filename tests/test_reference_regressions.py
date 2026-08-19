from __future__ import annotations

from functools import reduce
import math

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from qudits_on_qubits.bell_functionals.bell_builders import (
    build_bell_operator,
    candidate_statevector,
)
from qudits_on_qubits.bell_functionals.encoding import default_qutrit_encoding
from qudits_on_qubits.bell_functionals.estimator_backend import bell_value_estimator
from qudits_on_qubits.bell_functionals.sampler_backend import bell_value_sampler
from qudits_on_qubits.bell_measurements import bit_pair_to_qutrit_outcome
from qudits_on_qubits.reference_experiments import (
    BellTermSpec,
    ReferenceExperimentSpec,
    get_reference_experiment,
)


REFERENCE_CASES = (
    pytest.param("two_qutrit", 6.0, id="two-qutrit"),
    pytest.param("ghz3", 6.0, id="ghz3"),
    pytest.param("ame43", 8.0, id="ame43"),
)
FINITE_SHOTS = 4_096
SAMPLER_SEED = 20_260_818


def _logical_term_contribution(
    spec: ReferenceExperimentSpec,
    term: BellTermSpec,
) -> complex:
    local_operators = [
        np.eye(spec.state.local_dimension, dtype=complex)
        for _ in spec.state.party_order
    ]
    for factor in term.factors:
        local_operators[factor.party] = factor.logical_operator(
            spec.observable(factor.setting_label)
        )
    operator = reduce(np.kron, local_operators)
    state = spec.state.statevector()
    return complex(term.coefficient * np.vdot(state, operator @ state))


def _six_sigma_sampling_tolerance(
    spec: ReferenceExperimentSpec,
    shots: int,
) -> float:
    coefficient_square_sum = sum(
        abs(term.sampling_coefficient()) ** 2
        for term in spec.bell_functional.terms
    )
    return 6 * math.sqrt(coefficient_square_sum / shots)


@pytest.mark.parametrize(("candidate", "expected"), REFERENCE_CASES)
def test_reference_statevector_regression_matches_ideal_value(
    candidate: str,
    expected: float,
) -> None:
    spec = get_reference_experiment(candidate)
    encoding = default_qutrit_encoding()
    state = candidate_statevector(candidate, encoding)

    result = bell_value_estimator(
        state,
        build_bell_operator(candidate, encoding),
        E=encoding,
    )

    assert result.backend == "StatevectorEstimator"
    assert result.value.real == pytest.approx(
        expected,
        abs=spec.expected.absolute_tolerance,
    )
    assert result.value.imag == pytest.approx(
        0.0,
        abs=spec.expected.absolute_tolerance,
    )
    assert result.leakage_probability == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(("candidate", "expected"), REFERENCE_CASES)
def test_reference_sampler_regression_matches_exact_and_finite_shot_values(
    candidate: str,
    expected: float,
) -> None:
    spec = get_reference_experiment(candidate)
    encoding = default_qutrit_encoding()
    state = candidate_statevector(candidate, encoding)

    exact = bell_value_sampler(state, candidate, E=encoding, shots=None)
    sampled = bell_value_sampler(
        state,
        candidate,
        E=encoding,
        shots=FINITE_SHOTS,
        seed=SAMPLER_SEED,
    )
    tolerance = _six_sigma_sampling_tolerance(spec, FINITE_SHOTS)

    assert exact.backend == "ProjectorSampler"
    assert exact.value.real == pytest.approx(
        expected,
        abs=spec.expected.absolute_tolerance,
    )
    assert exact.value.imag == pytest.approx(
        0.0,
        abs=spec.expected.absolute_tolerance,
    )
    assert exact.leakage_probability == pytest.approx(0.0, abs=1e-12)
    assert abs(sampled.value - expected) <= tolerance
    assert abs(sampled.value.imag) <= tolerance
    assert sampled.shots == FINITE_SHOTS


@pytest.mark.parametrize(("candidate", "expected"), REFERENCE_CASES)
def test_reference_conjugate_term_symmetry_cancels_imaginary_part(
    candidate: str,
    expected: float,
) -> None:
    spec = get_reference_experiment(candidate)
    terms = spec.bell_functional.terms
    midpoint = len(terms) // 2
    first_power = tuple(
        _logical_term_contribution(spec, term) for term in terms[:midpoint]
    )
    conjugate_power = tuple(
        _logical_term_contribution(spec, term) for term in terms[midpoint:]
    )

    assert len(first_power) == len(conjugate_power)
    assert conjugate_power == pytest.approx(
        tuple(value.conjugate() for value in first_power),
        abs=1e-10,
    )
    total = sum(first_power + conjugate_power)
    assert total.real == pytest.approx(expected, abs=1e-10)
    assert total.imag == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize(("candidate", "_expected"), REFERENCE_CASES)
def test_reference_outcome_mapping_marks_11_as_leakage(
    candidate: str,
    _expected: float,
) -> None:
    spec = get_reference_experiment(candidate)
    outcome_map = dict(spec.outcome_convention.measurement_basis_index_map)

    assert {
        bits: bit_pair_to_qutrit_outcome(*bits, outcome_map=outcome_map)
        for bits in ((0, 0), (1, 0), (0, 1), (1, 1))
    } == {
        (0, 0): 0,
        (1, 0): 1,
        (0, 1): 2,
        (1, 1): None,
    }


@pytest.mark.parametrize(("candidate", "expected"), REFERENCE_CASES)
def test_reference_sampler_reports_and_postselects_known_leakage(
    candidate: str,
    expected: float,
) -> None:
    encoding = default_qutrit_encoding()
    ideal = np.asarray(candidate_statevector(candidate, encoding).data)
    leakage = np.zeros_like(ideal)
    leakage[-1] = 1.0
    leakage_probability = 0.125
    state = Statevector(
        math.sqrt(1 - leakage_probability) * ideal
        + math.sqrt(leakage_probability) * leakage
    )

    conditional = bell_value_sampler(
        state,
        candidate,
        E=encoding,
        shots=None,
        postselect=True,
    )
    unconditional = bell_value_sampler(
        state,
        candidate,
        E=encoding,
        shots=None,
        postselect=False,
    )

    assert conditional.leakage_probability == pytest.approx(
        leakage_probability,
        abs=1e-12,
    )
    assert conditional.value == pytest.approx(expected, abs=1e-10)
    assert unconditional.value == pytest.approx(
        (1 - leakage_probability) * expected,
        abs=1e-10,
    )
