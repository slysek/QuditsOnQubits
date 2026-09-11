"""Replay committed hardware evidence without contacting a provider."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from qiskit import qpy

from scripts import iqm_bell_short as short
from scripts import iqm_bell_zne_models as models
from scripts import iqm_randomized_bell_campaign as campaign
from qudits_on_qubits.experiments.preparation import prepare_measurements
from qudits_on_qubits.experiments.setting_schedule import SettingSchedule


ROOT = Path(__file__).resolve().parents[1]
FIVE = ROOT / 'artifacts/iqm_bell_short/emerald_20260910_5scales'


@pytest.mark.parametrize('name', ['bell_short', 'bell_short_5scales'])
def test_saved_short_notebooks_have_consistent_offline_outputs(name):
    notebook = json.loads((ROOT / f'notebooks/working/iqm/{name}.ipynb').read_text(encoding='utf-8'))
    cells = notebook['cells']
    assert any('RUN_HARDWARE = False' in ''.join(cell['source']) for cell in cells)
    assert [cell['execution_count'] for cell in cells] == list(range(1, len(cells) + 1))
    guarded = [cell for cell in cells if ''.join(cell['source']).startswith('if RUN_HARDWARE:')]
    assert len(guarded) == 2
    assert all(cell['outputs'] == [] for cell in guarded)
    assert all(output['output_type'] != 'error' for cell in cells for output in cell['outputs'])


@pytest.mark.parametrize('name,scales,jobs,total_shots', [
    ('emerald_20260910', (1, 2, 3), 24, 258048),
    ('emerald_20260910_5scales', (1, 1.5, 2, 2.5, 3), 40, 405504),
])
def test_short_archives_reproduce_estimates_and_submitted_circuits(name, scales, jobs, total_shots):
    directory = ROOT / 'artifacts/iqm_bell_short' / name
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert len(manifest['jobs']) == jobs
    samples = short.read_samples(directory)
    assert sum(sample['shots'] for sample in samples) == total_shots
    weights = np.load(directory / 'bell_weights.npy', allow_pickle=False)
    _, expected_weights, evidence = short.build_catalog(ROOT)
    np.testing.assert_allclose(weights, expected_weights, atol=1e-12)
    assert evidence['ideal_bell'] == pytest.approx(6, abs=1e-10)
    actual = short.analyze(samples, weights, scales=scales)
    expected = json.loads((directory / 'analysis.json').read_text())
    for got, saved in zip(actual['rows'], expected['rows'], strict=True):
        assert got['variant'] == saved['variant']
        for key in ('bell', 'standard_error', 'ci95_low', 'ci95_high'):
            assert got[key] == pytest.approx(saved[key], abs=1e-11)
    stats = models.campaign_statistics(samples, weights, scales=scales)
    comparison = json.loads((directory / 'model_comparison.json').read_text())
    for saved in comparison['rows']:
        parameters = models.fit_model(saved['model'], stats[saved['arm']]['blocks'].mean(axis=0), scales=scales)
        assert models.predict(saved['model'], parameters, 0) == pytest.approx(saved['bell_zero'], abs=1e-6)
    for index, job in enumerate(manifest['jobs']):
        with (directory / f'job_{index:03d}.qpy').open('rb') as handle:
            circuits = qpy.load(handle)
        assert [short._fingerprint(c) for c in circuits] == job['circuit_hashes']
        assert len(circuits) == len(job['records']) <= 100


def test_diagnostics_input_hashes_and_block_statistics_match_counts():
    diagnostics = json.loads((FIVE / 'statistical_diagnostics.json').read_text())
    hashes = diagnostics['provenance']['input_sha256']
    assert len(hashes) == 42
    for name, expected in hashes.items():
        assert Path(name).name == name
        assert hashlib.sha256((FIVE / name).read_bytes()).hexdigest() == expected
    actual = models.campaign_statistics(short.read_samples(FIVE),
        np.load(FIVE / 'bell_weights.npy', allow_pickle=False), scales=(1, 1.5, 2, 2.5, 3))
    for arm, saved in diagnostics['source_statistics'].items():
        for key in ('blocks', 'shot_covariance'):
            np.testing.assert_allclose(actual[arm][key], saved[key], atol=1e-12)


def test_original_archive_reproduces_all_point_estimates(monkeypatch):
    directory = ROOT / 'artifacts/iqm_randomized_bell/campaign_1ee8e73b3efb45ee921e8be945833fad'
    schedule = SettingSchedule.from_safe_dict(json.loads((directory / 'schedule.json').read_text()))
    artifacts, _ = campaign.load_state(ROOT / 'experiment_inputs/iqm_randomized_bell/canonical_optimized_20260909')
    metadata = prepare_measurements(artifacts, schedule).metadata
    monkeypatch.setattr(campaign, 'write_json', lambda *_: None)
    actual = campaign.analyze_campaign(directory, schedule, metadata)
    expected = json.loads((directory / 'analysis.json').read_text())
    for got, saved in zip(actual['rows'], expected['rows'], strict=True):
        assert (got['variant'], got['factor']) == (saved['variant'], saved['factor'])
        for key in ('real', 'imag'):
            assert got[key] == pytest.approx(saved[key], abs=1e-11)
