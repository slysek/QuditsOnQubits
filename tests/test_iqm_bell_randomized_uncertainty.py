"""Regression tests for uncertainty with unequal randomized setting frequencies."""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.iqm_bell_randomized_uncertainty import (
    block_statistics, bootstrap, summarize, load_run, write_outputs,
)


def example():
    # Two settings with two and three blocks: average within setting, then sum.
    counts = np.array([[[8, 2], [6, 4], [9, 1], [5, 5], [4, 6]]] * 3)
    return counts, np.array([0, 0, 1, 1, 1]), np.array([[0., 2.], [-1., 1.]])


def test_estimator_and_variance_match_expanded_shots():
    counts, settings, weights = example()
    stats = block_statistics(counts, settings, weights)
    values, variances = [], []
    for b, setting in enumerate(settings):
        shots = np.repeat(weights[setting], counts[0, b])
        values.append(shots.mean())
        variances.append(shots.var(ddof=1) / len(shots))
    denominators = np.array([2, 2, 3, 3, 3])
    expected = np.sum(np.array(values) / denominators)
    np.testing.assert_allclose(stats['bell'], expected)
    np.testing.assert_allclose(stats['shot_variance'], np.sum(variances / denominators**2))
    assert expected != pytest.approx(np.mean(values) * 2)
    np.testing.assert_allclose(stats['block_variance_components'][0], [np.var(values[:2], ddof=1)/2, np.var(values[2:], ddof=1)/3])


def test_zero_weight_leakage_stays_in_denominator():
    counts = np.array([[[5, 0, 5], [5, 0, 5]]] * 3)
    stats = block_statistics(counts, np.zeros(2, dtype=int), np.array([[1., -1., 0.]]))
    np.testing.assert_allclose(stats['bell'], .5)
    np.testing.assert_allclose(stats['shot_variance'], .25 / 9 / 2)


@pytest.mark.parametrize('case', ['negative', 'fractional', 'one_shot', 'nan_weights', 'float_settings', 'missing_setting', 'one_block', 'shape'])
def test_invalid_inputs_fail(case):
    counts, settings, weights = example()
    if case == 'negative': counts[0, 0, 0] = -1
    elif case == 'fractional': counts = counts.astype(float); counts[0, 0, 0] = 1.5
    elif case == 'one_shot': counts[0, 0] = [1, 0]
    elif case == 'nan_weights': weights[0, 0] = np.nan
    elif case == 'float_settings': settings = settings.astype(float)
    elif case == 'missing_setting': weights = np.vstack([weights, weights[0]])
    elif case == 'one_block': settings[0] = 1
    elif case == 'shape': counts = counts[:, :, :1]
    with pytest.raises(ValueError): block_statistics(counts, settings, weights)


def test_bootstrap_reproducible_and_matches_multinomial_variance():
    counts, settings, weights = example()
    stats = block_statistics(counts, settings, weights)
    draws = bootstrap(counts, settings, weights, draws=12000, seed=31)
    again = bootstrap(counts, settings, weights, draws=12000, seed=31)
    for key in draws:
        np.testing.assert_array_equal(draws[key], again[key])
    # Plug-in multinomial and empirical block bootstraps have finite-sample factors.
    np.testing.assert_allclose(draws['shot'].var(axis=0, ddof=1), stats['shot_variance'] * .9, rtol=.045)
    expected = (stats['block_variance_components'] * np.array([.5, 2/3])).sum(axis=1)
    np.testing.assert_allclose(draws['block'].var(axis=0, ddof=1), expected, rtol=.045)
    np.testing.assert_allclose(draws['shot'].mean(axis=0), stats['bell'], atol=.01)
    with pytest.raises(ValueError): bootstrap(counts, settings, weights, draws=1, seed=0)


def test_three_scale_weights_and_curvature():
    counts, settings, weights = example()
    stats = block_statistics(counts, settings, weights)
    stats['bell'] = 6 - .1 * np.array([1., 3., 5.]) - .01 * np.array([1., 3., 5.])**2
    result = summarize(stats, None, bound=5.6, ideal=6.)
    rows = {r['model']: r for r in result['models']}
    assert rows['quadratic']['value'] == pytest.approx(6.)
    assert rows['quadratic']['residual_dof'] == 0
    assert rows['linear']['residual_dof'] == 1
    np.testing.assert_allclose(rows['linear']['weights'], [13/12, 1/3, -5/12])
    np.testing.assert_allclose(rows['quadratic']['weights'], [15/8, -5/4, 3/8])
    assert result['linearity']['contrast']['value'] == pytest.approx(-.08)
    assert result['linearity']['quadratic_coefficient'] == pytest.approx(-.01)
    assert result['model_sensitivity']['linear_minus_quadratic']['value'] == pytest.approx(.0633333333333)
    assert rows['linear']['shot']['se']**2 == pytest.approx(np.dot(np.array(rows['linear']['weights'])**2, stats['shot_variance']))
    json.dumps(result, allow_nan=False)


def test_degenerate_variance_does_not_produce_invalid_json():
    stats = block_statistics(np.array([[[10, 0], [10, 0]]] * 3), np.zeros(2, dtype=int), np.array([[1., 0.]]))
    result = summarize(stats, None, bound=1., ideal=1.)
    assert result['linearity']['shot_chi_square_p'] is None
    json.dumps(result, allow_nan=False)


RUN = Path(__file__).resolve().parents[1] / 'artifacts/bell_optimized/iqm_emerald_dd_zne_50x1000_20260914'
SOURCE = RUN.parent / 'iqm_emerald_randomized_50x1000_20260914'


@pytest.mark.skipif(not (RUN/'completion_audit.json').is_file(), reason='Hardware artifacts unavailable')
def test_saved_run_replay_and_outputs(tmp_path):
    loaded = load_run(RUN, SOURCE)
    stats = block_statistics(loaded['counts'], loaded['settings'], loaded['weights'])
    draws = bootstrap(loaded['counts'], loaded['settings'], loaded['weights'], draws=300, seed=17)
    result = summarize(stats, draws, bound=loaded['analysis']['classical_bound'], ideal=loaded['ideal'])
    np.testing.assert_allclose(stats['bell'], [5.541927650303156, 5.242990503374911, 4.736356815163742], atol=1e-12)
    assert result['models'][0]['value'] == pytest.approx(5.777936449301835)
    assert result['models'][0]['shot']['se'] == pytest.approx(.04120699690699726)
    assert result['models'][1]['value'] == pytest.approx(5.613510020786179)
    assert result['linearity']['shot_chi_square_p'] == pytest.approx(.013742113209908645)
    result['provenance'] = loaded['provenance']
    write_outputs(tmp_path, result, stats, draws)
    assert (tmp_path/'report.md').stat().st_size > 2000
    assert (tmp_path/'zne_uncertainty.png').stat().st_size > 10000
    assert json.loads((tmp_path/'analysis.json').read_text())['models'][0]['value'] == result['models'][0]['value']


@pytest.mark.skipif(not (RUN/'completion_audit.json').is_file(), reason='Hardware artifacts unavailable')
def test_corrupt_evidence_rejected(tmp_path):
    import shutil
    for name in ['completion_audit.json', 'protocol.json', 'analysis.json']:
        shutil.copyfile(RUN/name, tmp_path/name)
    (tmp_path/'analysis.json').write_text('{}')
    with pytest.raises(ValueError, match='SHA256'):
        load_run(tmp_path, SOURCE)
