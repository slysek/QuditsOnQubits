import importlib
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def api():
    return importlib.import_module('scripts.iqm_bell_zne_models')


def test_model_comparison_module_exists():
    assert (Path(__file__).resolve().parents[1] / 'scripts/iqm_bell_zne_models.py').is_file()


@pytest.mark.parametrize('model,parameters', [
    ('linear', [5.7, -.3]),
    ('quadratic', [5.8, -.4, .04]),
    ('exponential', [5.9, .07]),
    ('exponential_offset', [2., 3.8, .2]),
])
def test_recovers_known_curves(api, model, parameters):
    x = np.array([1., 2., 3.])
    y = api.predict(model, parameters, x)
    fitted = api.fit_model(model, y)
    np.testing.assert_allclose(api.predict(model, fitted, x), y, atol=1e-9)
    assert api.predict(model, fitted, 0.) == pytest.approx(api.predict(model, parameters, 0.), abs=1e-8)


def test_offset_exponential_does_not_invent_a_fit_at_singularity(api):
    with pytest.raises(ValueError):
        api.fit_model('exponential_offset', [5.4, 5.1, 4.8])
    with pytest.raises(ValueError):
        api.fit_model('exponential_offset', [5.4, 5.1, 5.2])
    with pytest.raises(ValueError):
        api.fit_model('exponential', [0., -1., -2.])
    with pytest.raises(ValueError):
        api.fit_model('cubic', [5., 4., 3.])


def test_covariance_floor_preserves_correlated_uncertainty(api):
    empirical = np.array([[4., 3., 0.], [3., 4., 0.], [0., 0., .01]])
    floor = np.diag([1., 1., 1.])
    result = api.covariance_floor(empirical, floor)
    assert np.linalg.eigvalsh(result - empirical).min() > -1e-12
    assert np.linalg.eigvalsh(result - floor).min() > -1e-12
    assert result[0, 1] > 2


def test_comparison_propagates_uncertainty_and_identifies_saturated_models(api):
    blocks = np.tile([5.4, 5.1, 4.8], (8, 1))
    statistics = {'DD': dict(blocks=blocks, shot_covariance=np.eye(3) * .0025)}
    result = api.compare_statistics(statistics, draws=300, seed=42)
    rows = {row['model']: row for row in result['rows']}
    assert rows['linear']['bell_zero'] == pytest.approx(5.7)
    assert rows['quadratic']['bell_zero'] == pytest.approx(5.7)
    assert rows['linear']['ci95_low'] < 5.7 < rows['linear']['ci95_high']
    assert rows['quadratic']['ci95_high'] - rows['quadratic']['ci95_low'] > 2 * (rows['linear']['ci95_high'] - rows['linear']['ci95_low'])
    assert rows['quadratic']['residual_dof'] == 0
    assert rows['exponential_offset']['status'] == 'not_identifiable'
    assert result['unidentified_polynomials']['minimum_degree'] == 3
    json.dumps(result, allow_nan=False)


def test_failed_resamples_are_not_hidden_in_a_confidence_interval(api):
    statistics = {'DD': dict(blocks=np.tile([5.4, 5.1, 4.81], (8, 1)), shot_covariance=np.eye(3) * .05)}
    result = api.compare_statistics(statistics, draws=300, seed=7)
    row = next(r for r in result['rows'] if r['model'] == 'exponential_offset')
    assert row['failed_draws'] > 0
    assert row['ci95_low'] is None and row['ci95_high'] is None


def test_cubic_family_has_identical_data_and_arbitrary_zero_intercept():
    x = np.array([1., 2., 3.])
    assert np.all((x - 1) * (x - 2) * (x - 3) == 0)
    assert (0 - 1) * (0 - 2) * (0 - 3) == -6


def test_real_campaign_statistics_and_plot(api, tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    from scripts.iqm_bell_short import read_samples
    root = Path(__file__).resolve().parents[1]
    directory = root / 'artifacts/iqm_bell_short/emerald_20260910'
    if not (directory / 'bell_weights.npy').exists():
        pytest.skip('Local hardware evidence is unavailable')
    samples = read_samples(directory)
    weights = np.load(directory / 'bell_weights.npy', allow_pickle=False)
    result = api.compare_models(samples, weights, draws=100, seed=4)
    rows = {(r['arm'], r['model']): r for r in result['rows']}
    assert rows['DD', 'linear']['bell_zero'] == pytest.approx(5.689542413225696)
    assert rows['DD', 'quadratic']['bell_zero'] == pytest.approx(5.824128864708198)
    assert rows['TWIRLING', 'exponential_offset']['decaying_to_finite_offset'] is False
    figure = api.plot_comparison(result)
    assert len(figure.axes) == 2
    figure.savefig(tmp_path / 'comparison.png')
