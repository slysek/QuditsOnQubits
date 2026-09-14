"""Execution, ordering and block-estimator checks for the local benchmark."""
import json
from pathlib import Path

import numpy as np
import pytest
from qiskit import qpy

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'experiment_inputs/bell_optimized/raw_reference.json'


@pytest.fixture(scope='module')
def api():
    from scripts import bell_optimized_benchmark
    return bell_optimized_benchmark


@pytest.fixture(scope='module')
def catalog():
    from qudits_on_qubits.bell_measurements.bell_optimized import build_catalog
    return build_catalog(SOURCE)


def test_blocks_keep_repeats_separate_and_counterbalance_order(api, catalog):
    plan = api.make_plan(catalog, blocks=8, shots=32, seed=42)
    assert len(plan) == 8
    first = np.zeros((9, 2), dtype=int)
    for block, job in enumerate(plan):
        assert job['block'] == block
        assert job['shots'] == 32
        assert len(job['circuits']) == len(job['records']) == 18
        assert {(r['variant'], r['setting_index']) for r in job['records']} == {
            (v, s) for v in ('baseline', 'optimized') for s in range(9)}
        for position in range(0, 18, 2):
            left, right = job['records'][position:position + 2]
            assert left['setting_index'] == right['setting_index']
            first[left['setting_index'], int(left['variant'] == 'optimized')] += 1
    np.testing.assert_array_equal(first, np.full((9, 2), 4))
    again = api.make_plan(catalog, blocks=8, shots=32, seed=42)
    assert [j['records'] for j in plan] == [j['records'] for j in again]


def samples():
    rows = []
    for block in range(4):
        for variant in ('baseline', 'optimized'):
            for setting in range(9):
                n = 40 + block + 10 * (variant == 'optimized')
                rows.append(dict(block=block, variant=variant, setting_index=setting,
                                 shots=100, counts={'0000': n, '1111': 100-n}))
    return rows


def test_estimator_uses_blocks_pairs_and_keeps_invalid_words(api):
    weights = np.zeros((9, 16))
    weights[:, 0] = 1
    result = api.analyze(samples(), weights)
    assert result['variants']['baseline']['bell'] == pytest.approx(9 * .415)
    assert result['variants']['baseline']['mean_invalid_code_probability'] == pytest.approx(.585)
    assert result['paired_difference']['bell'] == pytest.approx(.9)
    assert result['paired_difference']['standard_error'] == pytest.approx(0, abs=1e-14)
    expected_se = np.std(np.arange(4) * .09, ddof=1) / 2
    assert result['variants']['baseline']['standard_error'] == pytest.approx(expected_se)


@pytest.mark.parametrize('change', ['missing', 'duplicate', 'bad_total', 'negative', 'bad_key', 'unequal_shots'])
def test_analysis_rejects_corrupt_results(api, change):
    rows = samples()
    if change == 'missing': rows.pop()
    if change == 'duplicate': rows.append(rows[0])
    if change == 'bad_total': rows[0]['shots'] = 101
    if change == 'negative': rows[0]['counts'] = {'0000': -1, '1111': 101}
    if change == 'bad_key': rows[0]['counts'] = {'0': 100}
    if change == 'unequal_shots':
        rows[0]['shots'] = 200
        rows[0]['counts'] = {'0000': 100, '1111': 100}
    with pytest.raises(ValueError):
        api.analyze(rows, np.zeros((9, 16)))


@pytest.mark.parametrize('blocks,shots', [(1, 32), (3, 32), (True, 32), (4, 0)])
def test_plan_rejects_invalid_budget(api, catalog, blocks, shots):
    with pytest.raises(ValueError):
        api.make_plan(catalog, blocks=blocks, shots=shots)


def test_local_pipeline_exports_replayable_counts_and_compilation(api, tmp_path):
    output = tmp_path / 'benchmark'
    result = api.run_local(SOURCE, output, blocks=4, shots=256, seed=42)
    assert result['status'] == 'completed_local'
    assert result['hardware_submitted'] is False
    assert result['total_shots'] == 4 * 18 * 256
    assert result['circuit_evidence']['max_probability_error'] < 1e-10
    assert set(result['compilation']) == {'iqm_rcz', 'ibm_cz', 'ibm_ecr'}
    for profile in result['compilation'].values():
        assert profile['max_probability_error'] < 1e-10
        assert profile['physical_backend_validated'] is False
    rows = json.loads((output / 'samples.json').read_text(encoding='utf-8'))
    assert len(rows) == 4 * 18
    replay = api.analyze(rows, np.load(output / 'bell_weights.npy'))
    assert replay == result['statistics']
    for variant in ('baseline', 'optimized'):
        assert abs(replay['variants'][variant]['bell'] - 6) < .25
        with (output / f'{variant}.qpy').open('rb') as handle:
            assert len(qpy.load(handle)) == 9
    assert (output / 'report.md').is_file()
    with pytest.raises(FileExistsError):
        api.run_local(SOURCE, output, blocks=4, shots=256, seed=42)
