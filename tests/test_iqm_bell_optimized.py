"""Safety and statistics for nine ordered IQM Bell settings."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from qiskit.circuit import Measure
from qiskit.circuit.library import CZGate, RGate
from qiskit.transpiler import Target, InstructionProperties

SOURCE = Path(__file__).resolve().parents[1] / 'experiment_inputs/bell_optimized/raw_reference.json'


@pytest.fixture(scope='module')
def api():
    from scripts import iqm_bell_optimized
    return iqm_bell_optimized


def backend():
    target = Target(num_qubits=4)
    target.add_instruction(CZGate(), {edge: InstructionProperties(error=.002) for edge in [(0,1),(0,2),(1,3),(2,3)]})
    target.add_instruction(RGate(.2,.3), {(q,): InstructionProperties(error=.001) for q in range(4)})
    target.add_instruction(Measure(), {(q,): InstructionProperties(error=.01) for q in range(4)})
    class Backend:
        name = 'emerald'
        num_qubits = 4
        architecture = SimpleNamespace(calibration_set_id='test-calibration')
        def __init__(self):
            self.target = target
            self.calls = []
        def index_to_qubit_name(self, q):
            return f'QB{q+1}'
        def run(self, circuits, **options):
            self.calls.append(circuits[0].metadata['setting_index'])
            assert len(circuits) == 1
            assert options['shots'] == 32
            result = SimpleNamespace(get_counts=lambda i: {'0000':32},
                results=[SimpleNamespace(calibration_set_id='test-calibration')])
            return SimpleNamespace(job_id=lambda: f'job-{len(self.calls)}', result=lambda: result)
    return Backend()


@pytest.fixture
def prepared(api, tmp_path):
    output = tmp_path/'campaign'
    device = backend()
    api.prepare(SOURCE, output, device, shots=32)
    return output, device


def test_preflight_order_budget_and_cost(api, prepared):
    output, device = prepared
    circuits, weights, protocol = api.load_prepared(output)
    assert device.calls == []
    assert protocol['shots_per_setting'] == 32
    assert protocol['total_shots'] == 9*32
    assert protocol['randomized'] is False
    assert [c.metadata['setting_index'] for c in circuits] == list(range(9))
    assert [c.count_ops()['cz'] for c in circuits] == [5,7,7]*3
    assert weights.shape == (9,16)


def test_requires_opt_in_then_executes_in_order_without_resubmitting(api, prepared):
    output, device = prepared
    with pytest.raises(PermissionError):
        api.execute(output, device)
    assert device.calls == []
    samples = api.execute(output, device, allow_hardware=True)
    assert device.calls == list(range(9))
    assert [s['setting'] for s in samples] == list(range(9))
    api.execute(output, device, allow_hardware=True)
    assert device.calls == list(range(9))


def test_unknown_submission_never_retried(api, prepared):
    output, device = prepared
    (output/'job_000.submission.json').write_text('{"status":"submission_unknown"}')
    with pytest.raises(ValueError, match='manifest'):
        api.execute(output, device, allow_hardware=True)
    assert device.calls == []


def test_modified_frozen_inputs_rejected(api, prepared):
    output, device = prepared
    np.save(output/'bell_weights.npy', np.zeros((9,16)))
    with pytest.raises(ValueError, match='hash'):
        api.execute(output, device, allow_hardware=True)
    assert device.calls == []


def test_calibration_change_rejected_before_submission(api, prepared):
    output, device = prepared
    device.architecture = SimpleNamespace(calibration_set_id='different')
    with pytest.raises(ValueError, match='calibration'):
        api.execute(output, device, allow_hardware=True)
    assert device.calls == []


def test_corrupt_completed_counts_rejected(api, prepared):
    output, device = prepared
    api.execute(output, device, allow_hardware=True)
    path=output/'job_000.counts.json'
    doc=json.loads(path.read_text());doc['counts']=[{'0000':31}]
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='shots'):
        api.execute(output, device, allow_hardware=True)
    assert device.calls == list(range(9))


def test_raw_estimate_retains_code11_and_counting_uncertainty(api):
    weights=np.zeros((9,16));weights[:,0]=1
    samples=[dict(setting=s,arm='RAW',shots=100,counts={'0000':60,'1111':40}) for s in range(9)]
    result=api.analyze(samples,weights,classical_bound=5.638155724715451)
    assert result['bell'] == pytest.approx(5.4)
    assert result['standard_error'] == pytest.approx(np.sqrt(9*.24/99))
    assert result['mean_invalid_code_probability'] == pytest.approx(.4)
    assert result['point_estimate_above_classical'] is False
    assert result['ci95_entirely_above_classical'] is False
    with pytest.raises(ValueError,match='nine'):
        api.analyze(samples[:-1],weights,classical_bound=5.6)
    with pytest.raises(ValueError,match='nine'):
        api.analyze(samples+[samples[0]],weights,classical_bound=5.6)


@pytest.mark.parametrize('shots',[0,1,True,3.2])
def test_invalid_shots_rejected(api,tmp_path,shots):
    with pytest.raises(ValueError,match='shots'):
        api.prepare(SOURCE,tmp_path/'bad',backend(),shots=shots)


def test_finish_replays_results_and_writes_report(api,prepared,tmp_path,monkeypatch):
    output,device=prepared
    api.execute(output,device,allow_hardware=True)
    monkeypatch.setattr(api,'ROOT',tmp_path/'no-history')
    result=api.finish(output)
    assert result['total_shots']==9*32
    assert result['historical_raw']==[]
    assert (output/'report.md').is_file()
    assert api.finish(output)==result
    assert api.main(['analyze','--output',str(output)])==0


def test_concurrent_executor_is_blocked(api,prepared):
    from filelock import FileLock, Timeout
    output,device=prepared
    with FileLock(str(output/'.execution.lock'),timeout=0):
        with pytest.raises(Timeout):
            api.execute(output,device,allow_hardware=True)
    assert device.calls==[]



def test_historical_comparison_reads_raw_counts(api,prepared,tmp_path,monkeypatch):
    import shutil
    output,device=prepared
    api.execute(output,device,allow_hardware=True)
    root=tmp_path/'historical'
    destination=root/'artifacts/iqm_bell_by_setting/high_shots_20260911'
    shutil.copytree(output,destination)
    monkeypatch.setattr(api,'ROOT',root)
    result=api.finish(output)
    assert len(result['historical_raw'])==1
    assert result['historical_raw'][0]['delta_new_minus_old']==0
    assert len(result['historical_raw'][0]['counts_sha256'])==9
