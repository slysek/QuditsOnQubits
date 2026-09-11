import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import CZGate, RGate
from qiskit.transpiler import InstructionProperties, Target


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks/working/iqm/bell_short.ipynb'


@pytest.fixture(scope='module')
def api():
    return importlib.import_module('scripts.iqm_bell_short')


@pytest.fixture(scope='module')
def catalog(api):
    return api.build_catalog(ROOT)


def test_notebook_is_code_only():
    assert NOTEBOOK.exists()
    notebook = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    assert all(c['cell_type'] == 'code' for c in notebook['cells'])
    for cell in notebook['cells']:
        source = ''.join(cell['source'])
        assert not any(line.lstrip().startswith('#') for line in source.splitlines())
        compile(source, str(NOTEBOOK), 'exec')


def test_short_catalog_preserves_bell_probabilities(api, catalog):
    circuits, weights, evidence = catalog
    assert len(circuits) == 9
    assert evidence['state_cx'] == 5
    assert evidence['state_fidelity'] > 1 - 1e-10
    assert evidence['max_probability_error'] < 1e-6
    assert [min(c.count_ops()['cz'] for c in circuits), max(c.count_ops()['cz'] for c in circuits)] == [6, 9]
    assert sum(w @ api.probabilities(c) for w, c in zip(weights, circuits)) == pytest.approx(6, abs=1e-10)


def calibrated_backend():
    target = Target(num_qubits=6)
    edges = [(0, 1), (0, 2), (1, 3), (2, 3), (2, 4), (3, 5), (4, 5)]
    target.add_instruction(CZGate(), {edge: InstructionProperties(error=.002 if min(edge) >= 2 else .04) for edge in edges})
    target.add_instruction(RGate(.2, .3), {(q,): InstructionProperties(error=.001) for q in range(6)})
    from qiskit.circuit import Measure
    target.add_instruction(Measure(), {(q,): InstructionProperties(error=.01) for q in range(6)})
    return SimpleNamespace(target=target, num_qubits=6, name='test', architecture=SimpleNamespace(calibration_set_id='test-calibration'))


def test_layout_uses_calibrations_and_keeps_classical_mapping(api, catalog):
    backend = calibrated_backend()
    circuits, weights, _ = catalog
    mapped, selection = api.select_layout(circuits, backend)
    assert set(selection['layout']) == {2, 3, 4, 5}
    assert selection['calibration_set_id'] == 'test-calibration'
    assert selection['candidates'] > 1
    for before, after in zip(circuits, mapped):
        np.testing.assert_allclose(api.probabilities(before), api.probabilities(after), atol=1e-12)
        assert after.layout is None


def test_missing_metrics_fail_closed(api, catalog):
    backend = calibrated_backend()
    for props in backend.target['cz'].values():
        props.error = None
    with pytest.raises(ValueError, match='calibrat'):
        api.select_layout(catalog[0], backend)


def test_plan_budget_folding_and_twirling_are_matched(api, catalog):
    circuits, _, _ = catalog
    plan = api.make_plan(circuits, repeats=4, shots=32, seed=42)
    assert sum(len(job['circuits']) * job['shots'] for job in plan) == 9 * 4 * 32 * 7
    observed = {}
    for job in plan:
        assert job['dd'] == (job['arm'] == 'DD')
        assert job['shots'] == 32
        for circuit, record in zip(job['circuits'], job['records']):
            original = circuits[record['setting']]
            np.testing.assert_allclose(api.probabilities(circuit), api.probabilities(original), atol=1e-11)
            ratio = circuit.count_ops()['cz'] / original.count_ops()['cz']
            assert ratio == record['actual_scale']
            observed.setdefault((job['arm'], record['setting'], record['scale']), []).append(ratio)
    for (_, _, scale), ratios in observed.items():
        assert np.mean(ratios) == pytest.approx(scale)


def test_analysis_uses_paired_repeat_uncertainty_and_all_shots(api):
    weights = np.zeros((9, 16))
    weights[:, 0] = 1
    samples = []
    for repeat in range(8):
        for setting in range(9):
            for arm, scale in [('RAW', 1), ('DD', 1), ('DD', 2), ('DD', 3), ('TWIRLING', 1), ('TWIRLING', 2), ('TWIRLING', 3)]:
                n = 600 - 50 * (scale - 1) + 2 * (repeat - 3)
                samples.append(dict(arm=arm, scale=scale, actual_scale=float(scale), repeat=repeat, setting=setting, counts={'0000': n, '1111': 1000-n}, shots=1000))
    result = api.analyze(samples, weights)
    rows = {r['variant']: r for r in result['rows']}
    assert set(rows) == {'RAW', 'DD', 'TWIRLING', 'DD + ZNE', 'TWIRLING + ZNE'}
    assert rows['RAW']['bell'] == pytest.approx(9 * .601)
    assert rows['DD + ZNE']['bell'] == pytest.approx(9 * .651)
    assert rows['RAW']['ci95_high'] > rows['RAW']['bell'] > rows['RAW']['ci95_low']
    assert rows['RAW']['leakage'] == pytest.approx(.399)
    assert result['fits']['DD']['curvature'] == pytest.approx(0, abs=1e-12)
    with pytest.raises(ValueError, match='Incomplete'):
        api.analyze(samples[:-1], weights)


def test_checkpoint_resume_never_resubmits_completed_or_uncertain_job(api, tmp_path):
    c = QuantumCircuit(4, 4)
    c.measure(range(4), range(4))
    record = dict(setting=0, scale=1, actual_scale=1., repeat=0)
    plan = [dict(arm='RAW', dd=False, shots=8, records=[record], circuits=[c])]
    class Backend:
        def __init__(self):
            self.calls = 0
        def run(self, circuits, **options):
            self.calls += 1
            result = SimpleNamespace(get_counts=lambda i: {'0000': 8}, results=[SimpleNamespace(calibration_set_id='test')])
            return SimpleNamespace(job_id=lambda: 'id-1', result=lambda: result)
    backend = Backend()
    path = tmp_path / 'campaign'
    api.execute(path, plan, backend, {'calibration_set_id': 'test'}, allow_hardware=True)
    api.execute(path, plan, backend, {'calibration_set_id': 'test'}, allow_hardware=True)
    assert backend.calls == 1
    other = tmp_path / 'uncertain'
    other.mkdir()
    (other / 'job_000.submission.json').write_text('{"status": "submission_unknown"}')
    with pytest.raises(RuntimeError, match='submission_unknown'):
        api.execute(other, plan, backend, {'calibration_set_id': 'test'}, allow_hardware=True)
    assert backend.calls == 1


def test_execution_requires_explicit_opt_in(api, tmp_path):
    with pytest.raises(PermissionError):
        api.execute(tmp_path / 'campaign', [], None, {})
