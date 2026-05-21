import math
import json
import subprocess
import sys

import numpy as np

from bell_functionals_qutrit.bell_builders import (
    build_bell_operator_ame43,
    build_bell_operator_ghz_graph,
    build_bell_operator_two_qutrit,
    candidate_statevector,
)
from bell_functionals_qutrit.classical_bounds import (
    bound_for_candidate,
    brute_force_classical_bound,
)
from bell_functionals_qutrit.encoding import (
    default_qutrit_encoding,
    embed_operator_E,
    embed_projector_E,
    leakage_probability,
    projector_E,
    validate_isometry,
)
from bell_functionals_qutrit.estimator_backend import bell_value_estimator
from bell_functionals_qutrit.operators import (
    make_A_tilde_qutrit_d3,
    make_measurement_observables_qutrit_d3,
    make_XZ_qutrit,
    split_nonhermitian,
)
from bell_functionals_qutrit.sampler_backend import bell_value_sampler


def test_default_encoding_is_an_isometry_and_projector():
    E = default_qutrit_encoding()

    assert validate_isometry(E)
    P = projector_E(E)
    np.testing.assert_allclose(P @ P, P, atol=1e-12)
    np.testing.assert_allclose(P.conj().T, P, atol=1e-12)


def test_embedding_supports_general_isometry_with_11_support():
    E = np.array(
        [
            [1 / math.sqrt(2), 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1 / math.sqrt(2), 0, 0],
        ],
        dtype=complex,
    )
    X, _, _ = make_XZ_qutrit()

    assert validate_isometry(E)
    embedded = embed_operator_E(E, X)
    np.testing.assert_allclose(
        embedded.conj().T @ embedded,
        np.eye(4),
        atol=1e-12,
    )

    P0 = np.diag([1, 0, 0])
    embedded_projector = embed_projector_E(E, P0)
    np.testing.assert_allclose(
        embedded_projector @ embedded_projector,
        embedded_projector,
        atol=1e-12,
    )


def test_a_tilde_recovers_qutrit_xz_stabilizer_observables():
    X, Z, _ = make_XZ_qutrit()

    for n in (1, 2):
        observables = make_measurement_observables_qutrit_d3(n)
        for k in range(3):
            expected = np.linalg.matrix_power(X @ np.linalg.matrix_power(Z, k), n)
            actual = make_A_tilde_qutrit_d3(observables, k, n)
            np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_split_nonhermitian_returns_hermitian_parts():
    X, Z, _ = make_XZ_qutrit()
    O = X @ Z

    real, imag = split_nonhermitian(O)

    np.testing.assert_allclose(real.conj().T, real, atol=1e-12)
    np.testing.assert_allclose(imag.conj().T, imag, atol=1e-12)
    np.testing.assert_allclose(real + 1j * imag, O, atol=1e-12)


def test_bell_operators_are_hermitian_and_match_pdf_quantum_values():
    E = default_qutrit_encoding()
    cases = [
        ("two_qutrit", build_bell_operator_two_qutrit, 6.0),
        ("ghz3", build_bell_operator_ghz_graph, 6.0),
        ("ame43", build_bell_operator_ame43, 8.0),
    ]

    for candidate, builder, expected in cases:
        bell_operator = builder(E)
        state = candidate_statevector(candidate, E)

        np.testing.assert_allclose(
            bell_operator.conj().T,
            bell_operator,
            atol=1e-10,
        )
        result = bell_value_estimator(state, bell_operator, E=E)
        assert result.leakage_probability == pytest_approx_zero()
        assert result.value.real == pytest_approx(expected)
        assert abs(result.value.imag) < 1e-10


def test_sampler_exact_path_matches_estimator_for_two_qutrit():
    E = default_qutrit_encoding()
    state = candidate_statevector("two_qutrit", E)
    bell_operator = build_bell_operator_two_qutrit(E)

    estimator_result = bell_value_estimator(state, bell_operator, E=E)
    sampler_result = bell_value_sampler(state, "two_qutrit", E=E, shots=None)

    assert sampler_result.value.real == pytest_approx(estimator_result.value.real)
    assert abs(sampler_result.value.imag) < 1e-10
    assert sampler_result.leakage_probability == pytest_approx_zero()


def test_classical_bounds_distinguish_pdf_and_numeric_results():
    two = bound_for_candidate("two_qutrit")
    ame = bound_for_candidate("ame43")
    ghz = bound_for_candidate("ghz3")

    assert two.quantum == pytest_approx(6.0)
    assert two.classical == pytest_approx(6 * math.cos(math.pi / 9))
    assert two.classical_source == "pdf"

    assert ame.quantum == pytest_approx(8.0)
    assert ame.classical == pytest_approx(7.63816, abs=5e-5)
    assert ame.classical_source == "pdf"

    assert ghz.quantum == pytest_approx(6.0)
    assert ghz.classical_source == "numeric_bruteforce"
    assert ghz.classical == pytest_approx(brute_force_classical_bound("ghz3").classical)


def test_cli_estimator_outputs_json_value_for_two_qutrit():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "bell_functionals_qutrit.cli",
            "--candidate",
            "two_qutrit",
            "--backend",
            "estimator",
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)
    assert payload["candidate"] == "two_qutrit"
    assert payload["backend"] == "estimator"
    assert payload["value_real"] == pytest_approx(6.0)
    assert payload["leakage_probability"] == pytest_approx_zero()


def pytest_approx(value, abs=1e-10):
    import pytest

    return pytest.approx(value, abs=abs)


def pytest_approx_zero(abs=1e-10):
    return pytest_approx(0.0, abs=abs)
