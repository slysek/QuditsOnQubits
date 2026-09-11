import json
from pathlib import Path

import numpy as np
import pytest
from qiskit import QuantumCircuit

from scripts import iqm_bell_short as short
from scripts import iqm_bell_zne_models as models


SCALES = (1., 1.5, 2., 2.5, 3.)


def synthetic_samples():
    weights = np.zeros((9, 16))
    weights[:, 0] = 1
    samples = []
    for repeat in range(32):
        for setting in range(9):
            for arm in ('RAW', 'DD', 'TWIRLING'):
                for scale in ((1.,) if arm == 'RAW' else SCALES):
                    counts = int(600 - 50 * (scale - 1) + 2 * (repeat // 4 - 3))
                    samples.append(dict(arm=arm, scale=scale, actual_scale=scale, repeat=repeat,
                        setting=setting, counts={'0000': counts, '1111': 1000-counts}, shots=1000))
    return samples, weights


def test_fractional_folding_balances_every_block_and_preserves_probabilities():
    circuits = []
    for ncz in (6, 7, 8, 9, 6, 7, 8, 9, 6):
        circuit = QuantumCircuit(4, 4)
        circuit.r(np.pi / 2, np.pi / 2, range(4))
        for _ in range(ncz):
            circuit.cz(0, 1)
            circuit.r(.23, np.pi / 2, 0)
        circuit.measure(range(4), range(4))
        circuits.append(circuit)
    plan = short.make_plan(circuits, repeats=8, shots=128, seed=3, scales=SCALES)
    assert max(len(job['circuits']) for job in plan) <= 100
    assert len(plan) == 10
    assert sum(len(j['circuits']) * j['shots'] for j in plan) == 9 * 8 * 128 * 11
    observed = {}
    for job in plan:
        assert job['dd'] == (job['arm'] == 'DD')
        for circuit, record in zip(job['circuits'], job['records'], strict=True):
            original = circuits[record['setting']]
            ratio = circuit.count_ops()['cz'] / original.count_ops()['cz']
            assert ratio == record['actual_scale']
            np.testing.assert_allclose(short.probabilities(circuit), short.probabilities(original), atol=1e-10)
            key = job['arm'], record['setting'], record['scale'], record['repeat'] // 4
            observed.setdefault(key, []).append(ratio)
    for (_, _, scale, _), actual in observed.items():
        assert len(actual) == 4
        assert np.mean(actual) == pytest.approx(scale)
    with pytest.raises(ValueError):
        short.make_plan(circuits, repeats=6, scales=SCALES)
    with pytest.raises(ValueError):
        short.make_plan(circuits, scales=(1., 1.6, 3.))


def test_five_scale_analysis_validates_all_scales_and_balancing():
    samples, weights = synthetic_samples()
    result = short.analyze(samples, weights, scales=SCALES)
    rows = {r['variant']: r for r in result['rows']}
    assert rows['DD + ZNE']['bell'] == pytest.approx(9 * .651)
    assert rows['DD + ZNE']['shots'] == 5 * 32 * 9 * 1000
    assert result['fits']['DD']['scales'] == list(SCALES)
    assert result['fits']['DD']['curvature'] == pytest.approx(0., abs=1e-10)
    stats = models.campaign_statistics(samples, weights, scales=SCALES)
    assert stats['DD']['blocks'].shape == (8, 5)
    with pytest.raises(ValueError, match='Incomplete'):
        short.analyze([s for s in samples if s['scale'] != 1.5], weights, scales=SCALES)
    damaged = [dict(s) for s in samples]
    next(s for s in damaged if s['arm'] == 'DD' and s['scale'] == 1.5)['actual_scale'] += .1
    with pytest.raises(ValueError, match='Unbalanced'):
        short.analyze(damaged, weights, scales=SCALES)


@pytest.mark.parametrize('model,parameters', [
    ('linear', [5.7, -.3]), ('quadratic', [5.8, -.4, .04]),
    ('exponential', [5.9, .07]), ('exponential_offset', [2., 3.8, .2]),
    ('exponential_offset', [8., -2., -.2]),
])
def test_models_fit_five_points(model, parameters):
    y = models.predict(model, parameters, SCALES)
    fit = models.fit_model(model, y, scales=SCALES)
    np.testing.assert_allclose(models.predict(model, fit, SCALES), y, atol=1e-7)
    assert models.predict(model, fit, 0) == pytest.approx(models.predict(model, parameters, 0), abs=1e-6)


def test_offset_rate_is_not_identifiable_for_constant_data():
    with pytest.raises(ValueError, match='identifiable'):
        models.fit_model('exponential_offset', [5.] * 5, scales=SCALES)


def test_five_scales_constrain_quadratic_without_interpolation():
    samples, weights = synthetic_samples()
    result = models.compare_models(samples, weights, draws=100, scales=SCALES)
    quad = next(r for r in result['rows'] if r['model'] == 'quadratic')
    assert quad['residual_dof'] == 2
    assert quad['status'] == 'ok'
    assert quad['ci95_low'] < quad['bell_zero'] < quad['ci95_high']
    assert result['unidentified_polynomials']['minimum_degree'] == 5
    assert result['data']['DD']['repeat_blocks'] == 8
    json.dumps(result, allow_nan=False)
    figure = models.plot_comparison(result)
    assert len(figure.axes) == 2


def test_five_scale_notebook_is_offline_by_default_and_code_only():
    path = Path(__file__).resolve().parents[1] / 'notebooks/working/iqm/bell_short_5scales.ipynb'
    notebook = json.loads(path.read_text(encoding='utf-8'))
    sources = [c['source'] if isinstance(c['source'], str) else ''.join(c['source']) for c in notebook['cells']]
    assert all(c['cell_type'] == 'code' for c in notebook['cells'])
    assert any('RUN_HARDWARE = False' in s for s in sources)
    assert any('REPEATS = 32' in s for s in sources)
    assert any('SHOTS = 128' in s for s in sources)
    for source in sources:
        assert not any(line.lstrip().startswith('#') for line in source.splitlines())
        compile(source, str(path), 'exec')
