"""Checks for balanced fractional CZ folding and recoverable DD execution."""
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest
from qiskit import QuantumCircuit
from iqm.iqm_client import STANDARD_DD_STRATEGY

from scripts import iqm_bell_interleaved_zne as run
from scripts import iqm_bell_short as short


def parents():
    result = []
    for j in range(9):
        c = QuantumCircuit(4, 4)
        c.r(.7, .2, 0)
        for k in range(5 if j % 3 == 0 else 7):
            c.cz(0, 1); c.r(.11, .3, 1)
        c.measure([2, 3, 0, 1], range(4))
        c.metadata = dict(setting_index=j)
        result.append(c)
    return result


def schedule():
    # Same uneven setting multiplicities as the frozen hardware run.
    labels = [j for j, n in enumerate([4,4,6,7,2,9,5,9,4]) for _ in range(n)]
    return dict(blocks=[dict(block_id=i, settings=[f'A{j//3}',f'B{j%3}']) for i,j in enumerate(labels)])


def test_fractional_folding_balances_every_gate_and_preserves_state():
    rng = np.random.default_rng(12)
    for c in parents()[:3]:
        n = c.count_ops()['cz']
        for scale in run.SCALES:
            variants, repetitions = run.fold_variants(c, scale, rng)
            assert repetitions.shape == (4,n)
            np.testing.assert_allclose(repetitions.mean(axis=0), scale)
            assert np.max(repetitions.sum(axis=1))-np.min(repetitions.sum(axis=1)) <= 2
            for v, rep in zip(variants, repetitions):
                assert v.count_ops()['cz'] == sum(rep)
                assert v.count_ops()['r'] == c.count_ops()['r']
                np.testing.assert_allclose(short.probabilities(v),short.probabilities(c),atol=1e-12)
    with pytest.raises(ValueError): run.fold_variants(c,1.2,rng)
    with pytest.raises(ValueError): run.fold_variants(QuantumCircuit(4,4),1.5,rng)


@pytest.fixture(scope='module')
def plan():
    return run.make_plan(parents(),schedule(),seed=17)


def test_interleaved_plan_budget_and_reproducibility(plan):
    assert len(plan)==10
    assert all(len(j['circuits'])==100 and j['shots']==250 for j in plan)
    assert sum(len(j['circuits'])*j['shots'] for j in plan)==250000
    for job in plan:
        keys={(r['block_id'],r['scale'],r['instance']) for r in job['records']}
        assert len(keys)==100
        for block in {r['block_id'] for r in job['records']}:
            for scale in run.SCALES:
                records=[r for r in job['records'] if r['block_id']==block and r['scale']==scale]
                assert len(records)==4
                assert np.mean([r['actual_scale'] for r in records])==pytest.approx(scale)
        assert len(set(r['scale'] for r in job['records'][:10]))>1
    again=run.make_plan(parents(),schedule(),seed=17)
    assert [j['records'] for j in again]==[j['records'] for j in plan]


class FakeBackend:
    name='emerald'
    architecture=SimpleNamespace(calibration_set_id='pinned')
    def __init__(self, fail_result=False, unknown=False, wrong_calibration=False):
        self.calls=[];self.retrievals=[];self.fail_result=fail_result;self.unknown=unknown;self.wrong=wrong_calibration;self.jobs={}
    def index_to_qubit_name(self,q): return f'QB{q+1}'
    def run(self,circuits,**options):
        self.calls.append(options)
        if self.unknown: raise RuntimeError('transport interruption')
        identifier=str(len(self.calls));n=len(circuits);shots=options['shots']
        def result():
            if self.fail_result: raise RuntimeError('pending read interrupted')
            return SimpleNamespace(results=[SimpleNamespace(calibration_set_id='wrong' if self.wrong else 'pinned')]*n,get_counts=lambda i:{'0000':shots})
        job=SimpleNamespace(job_id=lambda:identifier,result=result)
        self.jobs[identifier]=job
        return job
    def retrieve_job(self,identifier):
        self.retrievals.append(identifier)
        return self.jobs[identifier]


@pytest.fixture
def frozen(tmp_path,plan):
    selection=dict(backend='emerald',calibration_set_id='pinned',layout=[0,1,2,3],qubit_names_logical_order=['QB1','QB2','QB3','QB4'],measurement_map_c0_to_c3=[2,3,0,1])
    run.freeze(tmp_path/'run',parents(),np.zeros((9,16)),schedule(),selection,plan,source_evidence={},seed=17)
    return tmp_path/'run'


def test_execute_dd_and_resume_without_duplicates(frozen):
    backend=FakeBackend()
    with pytest.raises(PermissionError):run.execute(frozen,backend)
    run.execute(frozen,backend,allow_hardware=True)
    assert len(backend.calls)==10
    for call in backend.calls:
        options=call['circuit_compilation_options']
        assert options.dd_mode.value=='enabled'
        assert options.dd_strategy==STANDARD_DD_STRATEGY
    run.execute(frozen,backend,allow_hardware=True)
    assert len(backend.calls)==10
    samples=run.read_samples(frozen)
    assert len(samples)==1000 and sum(s['shots'] for s in samples)==250000
    from scripts.iqm_bell_interleaved_analysis import finish
    result=finish(frozen,draws=200,seed=4)
    assert result['total_shots']==250000
    assert len(run.read_samples(frozen))==1000
    counts_path=frozen/'hardware/job_000.counts.json'
    saved=json.loads(counts_path.read_text());saved['counts'][0]={'1111':250}
    counts_path.write_text(json.dumps(saved))
    with pytest.raises(ValueError,match='hash'):run.read_samples(frozen)


def test_interrupted_result_is_retrieved(frozen):
    backend=FakeBackend(fail_result=True)
    with pytest.raises(RuntimeError,match='interrupted'):run.execute(frozen,backend,allow_hardware=True)
    assert len(backend.calls)==1
    backend.fail_result=False
    run.execute(frozen,backend,allow_hardware=True)
    assert len(backend.calls)==10 and backend.retrievals==['1']


def test_unknown_submission_never_retried(frozen):
    backend=FakeBackend(unknown=True)
    with pytest.raises(RuntimeError,match='transport'):run.execute(frozen,backend,allow_hardware=True)
    with pytest.raises(RuntimeError,match='submission_unknown'):run.execute(frozen,backend,allow_hardware=True)
    assert len(backend.calls)==1


def test_corruption_and_calibration_rejected(frozen):
    backend=FakeBackend();backend.architecture=SimpleNamespace(calibration_set_id='different')
    with pytest.raises(ValueError,match='calibration'):run.execute(frozen,backend,allow_hardware=True)
    assert not backend.calls
    (frozen/'batch_00.qpy').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='hash'):run.load_prepared(frozen)


def test_wrong_result_calibration_not_saved(frozen):
    backend=FakeBackend(wrong_calibration=True)
    with pytest.raises(ValueError,match='calibration'):run.execute(frozen,backend,allow_hardware=True)
    assert not (frozen/'hardware/job_000.counts.json').exists()


@pytest.mark.skipif(not (run.SOURCE/'protocol.json').exists(),reason='Local frozen source unavailable')
def test_prepare_against_architecture_without_submission(tmp_path):
    class PreparationBackend(FakeBackend):
        def create_run_request(self,circuits,**kwargs):
            assert kwargs['shots']==250
            assert kwargs['circuit_compilation_options'].dd_strategy==STANDARD_DD_STRATEGY
            return SimpleNamespace(circuits=[SimpleNamespace(instructions=[SimpleNamespace(name='prx' if item.operation.name=='r' else item.operation.name) for item in c.data]) for c in circuits])
    backend=PreparationBackend()
    protocol=run.prepare(tmp_path/'prepared',backend=backend,seed=5)
    assert protocol['source_evidence']['ideal_bell']==pytest.approx(6)
    assert protocol['source_evidence']['serialized_circuits']==1000
    assert not backend.calls
