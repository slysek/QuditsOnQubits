"""Independent matrix, projector, and local deterministic strategy oracles."""

from itertools import product

import numpy as np
import pytest

from qudits_on_qubits.reference_experiments import get_reference_experiment


def tensor(matrices):
    result = np.ones((1, 1), dtype=complex)
    for matrix in matrices:
        result = np.kron(result, matrix)
    return result


def alphabets(reference):
    # Keep this independent of the production schedule implementation.
    return tuple(tuple(sorted({factor.setting_label for term in reference.bell_functional.terms for factor in term.factors if factor.party == party})) for party in reference.state.party_order)


def matches(setting, pattern):
    return all(label is None or label == actual for actual, label in zip(setting, pattern))


def sampled_operator(reference):
    root = np.exp(2j * np.pi / 3)
    result = np.zeros((3**reference.state.num_parties,) * 2, dtype=complex)
    for term in reference.bell_functional.terms:
        active = {factor.party: factor for factor in term.factors}
        matrices = []
        for party in reference.state.party_order:
            factor = active.get(party)
            if factor is None:
                matrices.append(np.eye(3))
            else:
                basis, _ = reference.observable(factor.setting_label).ordered_eigenbasis()
                diagonal = np.diag([root ** ((factor.outcome_power * outcome) % 3) for outcome in range(3)])
                matrices.append(basis @ diagonal @ basis.conj().T)
        result += term.sampling_coefficient() * tensor(matrices)
    return result


@pytest.mark.parametrize("state,total,patterns,useful", [
    ("two_qutrit", 9, 9, 9), ("ghz3", 18, 12, 12), ("ame43", 36, 13, 15),
])
def test_reference_input_spaces_and_outcome_operator(state, total, patterns, useful):
    reference = get_reference_experiment(state)
    settings = list(product(*alphabets(reference)))
    required = reference.measurement_settings()
    assert len(settings) == total
    assert len(required) == patterns
    assert sum(any(matches(setting, pattern) for pattern in required) for setting in settings) == useful
    np.testing.assert_allclose(sampled_operator(reference), reference.logical_bell_operator(), atol=1e-10, rtol=0)


@pytest.mark.parametrize("state", ["two_qutrit", "ghz3", "ame43"])
@pytest.mark.parametrize("state_kind", ["reference", "arbitrary", "product"])
def test_full_context_projector_probabilities_recover_reference_expectation(state, state_kind):
    reference = get_reference_experiment(state)
    parties = reference.state.num_parties
    rng = np.random.default_rng(804)
    if state_kind == "reference":
        psi = reference.state.statevector()
    elif state_kind == "product":
        vectors = [rng.normal(size=3) + 1j * rng.normal(size=3) for _ in range(parties)]
        psi = tensor([vector[:, None] / np.linalg.norm(vector) for vector in vectors]).ravel()
    else:
        psi = rng.normal(size=3**parties) + 1j * rng.normal(size=3**parties)
        psi /= np.linalg.norm(psi)
    outcomes = np.array(list(product(range(3), repeat=parties)))
    root = np.exp(2j * np.pi / 3)
    probabilities = {}
    for setting in product(*alphabets(reference)):
        basis = tensor([reference.observable(label).ordered_eigenbasis()[0] for label in setting])
        probabilities[setting] = np.abs(basis.conj().T @ psi) ** 2
    total = 0j
    for term in reference.bell_functional.terms:
        pattern = reference.setting_for_term(term)
        weights = root ** (sum(factor.outcome_power * outcomes[:, factor.party] for factor in term.factors) % 3)
        context_values = [complex(probs @ weights) for setting, probs in probabilities.items() if matches(setting, pattern)]
        # In particular, AME's identity marginals must agree between full contexts.
        np.testing.assert_allclose(context_values, context_values[0], atol=1e-10, rtol=0)
        total += term.sampling_coefficient() * np.mean(context_values)
    expected = np.vdot(psi, reference.logical_bell_operator() @ psi)
    assert abs(total - expected) < 1e-10


@pytest.mark.parametrize("state,expected", [
    ("two_qutrit", 5.638155724715452),
    ("ghz3", 5.638155724715452),
    ("ame43", 7.638155724715452),
])
def test_uniform_context_local_bound_with_global_leakage_acceptance(state, expected):
    reference = get_reference_experiment(state)
    local = alphabets(reference)
    widths = tuple(len(labels) for labels in local)
    offsets = np.cumsum((0,) + widths)
    root = np.exp(2j * np.pi / 3)
    # 3 denotes leakage. Enumerate in bounded chunks instead of retaining a
    # global table of all strategies or any operator-valued strategy payload.
    powers = 4 ** np.arange(sum(widths), dtype=np.int64)
    best = -np.inf
    for start in range(0, 4 ** sum(widths), 65536):
        strategy_ids = np.arange(start, min(start + 65536, 4 ** sum(widths)), dtype=np.int64)
        responses = (strategy_ids[:, None] // powers[None, :]) % 4
        accepted_fractions = [np.mean(responses[:, offsets[party]:offsets[party + 1]] != 3, axis=1) for party in range(len(local))]
        scores = np.zeros(len(responses), dtype=complex)
        for term in reference.bell_functional.terms:
            active = {factor.party: factor for factor in term.factors}
            value = np.full(len(responses), term.sampling_coefficient(), dtype=complex)
            for party, labels in enumerate(local):
                factor = active.get(party)
                if factor is None:
                    value *= accepted_fractions[party]
                else:
                    response = responses[:, offsets[party] + labels.index(factor.setting_label)]
                    value *= (response != 3) * root ** ((factor.outcome_power * response) % 3)
            scores += value
        assert np.max(np.abs(scores.imag)) < 1e-10
        best = max(best, float(np.max(scores.real)))
    assert best == pytest.approx(expected, abs=1e-10)
