"""Offline regressions for optimized parents in the existing randomized pipeline."""
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import iqm_randomized_bell_campaign as campaign
from scripts import iqm_bell_short as short
from scripts import iqm_bell_optimized_randomized as run
from qudits_on_qubits.experiments import RandomizedBlocks
from qudits_on_qubits.experiments.setting_schedule import generate_schedule
from qudits_on_qubits.reference_experiments import get_reference_experiment


@pytest.fixture(scope='module')
def schedule():
    rng = random.Random(19)
    result = generate_schedule(get_reference_experiment('two_qutrit'),
        RandomizedBlocks(50, 1000, 50), _randbelow=rng.randrange,
        _source='test_seeded', _seed=19)
    assert result.complete
    return result


@pytest.fixture(scope='module')
def native():
    from qudits_on_qubits.bell_measurements.bell_optimized import build_catalog
    return build_catalog(run.sequential.SOURCE)


@pytest.fixture(scope='module')
def planned(schedule, native):
    return run.prepare_batches(native.optimized, native.weights, schedule, instances=8, seed=19)


def test_native_schedule_order_budget_and_ideal(planned, schedule, native):
    _, batches, calibrations, evidence = planned
    assert not calibrations
    assert evidence['max_ideal_probability_error'] < 1e-6
    assert [(b['arm'], b['factor'], b['shots'], len(b['circuits'])) for b in batches] == [
        ('RAW', 1, 1000, 50), ('TWIRL_DD', 1, 125, 400),
        ('TWIRL_DD', 3, 125, 400), ('TWIRL_DD', 5, 125, 400)]
    assert sum(len(b['circuits'])*b['shots'] for b in batches) == 200000
    for batch in batches:
        for index in (0, len(batch['circuits'])//2, len(batch['circuits'])-1):
            record = batch['records'][index]
            setting = schedule.blocks[record['block_id']].settings
            original = native.optimized[run.SETTINGS.index(setting)]
            circuit = batch['circuits'][index]
            assert circuit.count_ops()['cz'] == original.count_ops()['cz'] * batch['factor']
            assert campaign.measurement_mapping(circuit) == (2, 3, 0, 1)
            np.testing.assert_allclose(short.probabilities(circuit), short.probabilities(original), atol=1e-12)
    assert len(set(b.settings for b in schedule.blocks)) == 9


def test_wrong_parent_order_and_weights_rejected(schedule, native):
    reordered = list(native.optimized)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with pytest.raises(ValueError, match='setting|histogram'):
        run.prepare_batches(reordered, native.weights, schedule, instances=8, seed=19)
    with pytest.raises(ValueError, match='weights'):
        run.prepare_batches(native.optimized, native.weights + .1, schedule, instances=8, seed=19)


class Backend:
    name = 'fake'
    calibration_set_id = 'pinned'
    architecture = SimpleNamespace(calibration_set_id='pinned')

    def __init__(self, wrong_calibration=False):
        self.calls = []
        self.wrong_calibration = wrong_calibration

    def run(self, circuits, **options):
        self.calls.append(options)
        counts = []
        for circuit in circuits:
            # Largest remainder sampling in actual classical-bit order.
            expected = short.probabilities(circuit) * options['shots']
            values = np.floor(expected).astype(int)
            remaining = options['shots'] - int(values.sum())
            values[np.argsort(expected-values)[::-1][:remaining]] += 1
            counts.append({format(i, '04b'): int(v) for i, v in enumerate(values) if v})
        result = SimpleNamespace(get_counts=lambda index: counts[index],
            results=[SimpleNamespace(calibration_set_id='wrong' if self.wrong_calibration else 'pinned')]
            * len(circuits))
        return SimpleNamespace(job_id=lambda: str(len(self.calls)), result=lambda: result)


def test_pipeline_execution_and_analysis_without_readout(planned, schedule, tmp_path):
    measurements, batches, calibrations, evidence = planned
    backend = Backend()
    target = tmp_path/'hardware'
    campaign.execute_campaign(target, schedule, evidence, batches, calibrations, backend,
        allow_hardware=True, expected_calibration_set_id='pinned')
    result = campaign.analyze_campaign(target, schedule, measurements.metadata)
    assert len(backend.calls) == 13
    assert result['rows'][0]['real'] == pytest.approx(6, abs=.02)
    assert result['rows'][-1]['real'] == pytest.approx(6, abs=.12)
    assert result['rows'][-1]['variant'] == 'TWIRL_DD_ZNE'
    assert result['rows'][0]['real'] == pytest.approx(result['raw_block_analysis']['raw']['real'])
    assert result['mitigated_uncertainty'] is None
    assert all(o['circuit_compilation_options'].dd_mode.value == 'enabled' for o in backend.calls[1:])
    assert backend.calls[0]['circuit_compilation_options'].dd_mode.value == 'disabled'
    assert json.loads((target/'backend.json').read_text())['calibration_set_id'] == 'pinned'
    with pytest.raises(FileExistsError):
        campaign.execute_campaign(target, schedule, evidence, batches, calibrations, backend,
            allow_hardware=True, expected_calibration_set_id='pinned')
    (target/'TWIRL_DD_5_300.counts.json').unlink()
    with pytest.raises(FileNotFoundError):
        campaign.analyze_campaign(target, schedule, measurements.metadata)


def test_calibration_mismatch_stops_submission_and_results(planned, schedule, tmp_path):
    _, batches, calibrations, evidence = planned
    backend = Backend()
    with pytest.raises(ValueError, match='calibration'):
        campaign.execute_campaign(tmp_path/'wrong', schedule, evidence, batches, calibrations, backend,
            allow_hardware=True, expected_calibration_set_id='different')
    assert not backend.calls
    bad_result = Backend(wrong_calibration=True)
    with pytest.raises(ValueError, match='calibration'):
        campaign.execute_campaign(tmp_path/'wrong_result', schedule, evidence, batches, calibrations, bad_result,
            allow_hardware=True, expected_calibration_set_id='pinned')
    assert len(bad_result.calls) == 1
    assert not (tmp_path/'wrong_result'/'RAW_1_0.counts.json').exists()


@pytest.fixture(scope='module')
def frozen(tmp_path_factory, native, schedule):
    from dataclasses import replace
    from unittest.mock import patch
    output = tmp_path_factory.mktemp('randomized')/'prepared'
    selection = dict(backend='fake', calibration_set_id='pinned', layout=[0,1,2,3],
        qubit_names_logical_order=['QB1','QB2','QB3','QB4'])
    parent = dict(selection=selection, versions={})
    production = replace(schedule, source='system_secrets', seed=None)
    with patch.object(run.sequential, 'load_prepared', return_value=(native.optimized, native.weights, parent)), \
         patch.object(run, 'generate_schedule', return_value=production):
        run.prepare(output, output.parent/'parent', instances=1)
    return output


def test_frozen_inputs_pipeline_finish_and_guards(frozen, tmp_path):
    import shutil
    output = tmp_path/'prepared'
    shutil.copytree(frozen, output)
    backend = Backend()
    backend.index_to_qubit_name = lambda i: f'QB{i+1}'
    with pytest.raises(PermissionError):
        run.execute(output, backend)
    assert not backend.calls
    backend.name = 'wrong'
    with pytest.raises(ValueError, match='Backend'):
        run.execute(output, backend, allow_hardware=True)
    backend.name = 'fake'
    run.execute(output, backend, allow_hardware=True)
    result = run.finish(output)
    assert result['protocol']['total_shots'] == 200000
    assert result['rows'][0]['real'] == pytest.approx(6, abs=.02)
    assert result['classical_bound'] == pytest.approx(5.638155724715451)
    assert (output/'report.md').exists()
    assert run.finish(output) == result
    with pytest.raises(FileExistsError):
        run.execute(output, backend, allow_hardware=True)
    assert len(backend.calls) == 4
    request_path = output/'hardware'/'RAW_1_0.request.json'
    request = run.read_json(request_path)
    request['records'][0]['block_id'] = 999
    campaign.write_json(request_path, request)
    with pytest.raises(ValueError, match='request'):
        run.finish(output)


@pytest.mark.parametrize('name', ['schedule.json', 'batch_00.qpy', 'batches.json', 'bell_weights.npy'])
def test_frozen_hash_mismatch(frozen, tmp_path, name):
    import shutil
    output = tmp_path/'prepared'
    shutil.copytree(frozen, output)
    with (output/name).open('ab') as handle:
        handle.write(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        run.load_prepared(output)


def test_no_redraw_and_invalid_budget(frozen, tmp_path):
    with pytest.raises(FileExistsError):
        run.prepare(frozen)
    with pytest.raises(ValueError, match='divisible'):
        run.prepare(tmp_path/'bad', shots=1001, instances=8)


def test_incomplete_schedule_persisted_before_rejection(tmp_path, native, monkeypatch):
    from dataclasses import replace
    rng = random.Random(1)
    incomplete = generate_schedule(get_reference_experiment('two_qutrit'),
        RandomizedBlocks(1, 1000, 1), _randbelow=rng.randrange,
        _source='test_seeded', _seed=1)
    incomplete = replace(incomplete, source='system_secrets', seed=None)
    monkeypatch.setattr(run, 'generate_schedule', lambda *a: incomplete)
    monkeypatch.setattr(run.sequential, 'load_prepared', lambda *a: (native.optimized, native.weights, {}))
    output = tmp_path/'incomplete'
    with pytest.raises(ValueError, match='incomplete coverage'):
        run.prepare(output, draws=1)
    assert run.read_json(output/'schedule.json') == incomplete.to_safe_dict()
    assert not (output/'hardware').exists()
