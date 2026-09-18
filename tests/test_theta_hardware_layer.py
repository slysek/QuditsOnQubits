"""Additive, immutable and resumable hardware layer contracts."""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
from qiskit import QuantumCircuit

from qudits_on_qubits.benchmarks.theta_continuation.artifacts import RunStore, fingerprint


def module():
    return importlib.import_module('qudits_on_qubits.benchmarks.theta_continuation.hardware_layer')


@pytest.fixture
def snapshot():
    # Actual connected ladder around the archived four-qubit baseline.
    return dict(schema_version=1, backend='emerald', calibration_set_id='test-calibration',
        qubit_names=[f'QB{i+1}' for i in range(54)], native_gates=['r','cz','measure'],
        edges=[[27,28],[27,19],[28,20],[19,20],[27,35],[28,36],[35,36],
               [19,11],[20,12],[11,12]], hardware_submitted=False)


@pytest.fixture
def source(tmp_path):
    payload=dict(benchmark='theta_threshold_reassessment_v1', config={'theta_points':2})
    store=RunStore.create(tmp_path/'source', {**payload,'fingerprint':fingerprint(payload)})
    for index,theta in enumerate((0.,float(np.pi/4))):
        e=np.array([[np.cos(theta),0,0],[0,1,0],[0,0,1],[-np.sin(theta),0,0]])
        store.write_bundle(f'points/{index:05d}',metadata=dict(index=index,theta=theta,correct=True),
            circuits={'f3_optimal.qpy':QuantumCircuit(2),'cz3_selected.qpy':QuantumCircuit(4)}, arrays={'E.npy':e})
        store.write_bundle(f'circuits/two_qutrit/{index:05d}',
            metadata=dict(index=index,theta=theta,state_name='two_qutrit',success=True,fidelity=1.),
            circuits={'graph_state_transpiled.qpy':QuantumCircuit(4)})
    return store.root


def inventory(path):
    return {str(p.relative_to(path)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in path.rglob('*') if p.is_file()}


@pytest.fixture
def fake_compilers(monkeypatch):
    m=module()
    cost=dict(n_cz=0,n_1q=0,depth=0,cz_depth=0,size=0)
    calls=[]
    def gate(circuit,edges,*,seeds):
        calls.append('gate')
        return dict(metadata=dict(original=cost,native=cost,routed=cost,full_operator_error=0.,
                                  routing={'seed':seeds[0],'final_permutation':list(range(circuit.num_qubits))}),
                    circuits={'native.qpy':circuit,'routed.qpy':circuit},arrays={})
    def case(state_name,circuit,encoding,edges,*,seeds):
        calls.append(state_name)
        measured=QuantumCircuit(4,4);measured.measure([2,3,0,1],range(4))
        return dict(metadata=dict(preparation_original=cost,preparation_native=cost,preparation_routed=cost,
            bell_prefix_native=cost,bell_prefix_routed=cost,
            settings=[['A0','B0']],per_setting=[dict(index=0,labels=['A0','B0'],native=cost,routed=cost,
            max_probability_error=0.,ideal_contribution=6.)],ideal_bell=6.,source_fidelity=1.,max_probability_error=0.,
            routing={'preparation':{'seed':seeds[0],'final_permutation':[0,1,2,3]},
                     'bell':{'seed':seeds[0],'final_permutation':[0,1,2,3],'repair_swaps':[],'repair_count':0}}),
            circuits={'preparation_native.qpy':circuit,'preparation_routed.qpy':circuit,
                      'bell_native_000.qpy':measured,'bell_routed_000.qpy':measured},
            arrays={'weights.npy':np.zeros((1,16))})
    monkeypatch.setattr(m,'compile_gate',gate)
    monkeypatch.setattr(m,'compile_case',case)
    return calls


def test_additive_run_and_resume_preserve_source(source,snapshot,tmp_path,fake_compilers):
    before=inventory(source)
    output=tmp_path/'hardware'
    messages=[]
    result=module().run_hardware_layer(source,output,snapshot,states=['two_qutrit'],progress=messages.append)
    assert len(messages)==6
    assert result['cases']==2 and result['gate_cases']==4 and result['hardware_submitted'] is False
    assert inventory(source)==before
    assert len(fake_compilers)==6
    module().run_hardware_layer(source,output,snapshot,states=['two_qutrit'],resume=True)
    assert len(fake_compilers)==6
    report=(output/'report.md').read_text(encoding='utf-8')
    assert 'baseline' in report.lower() and 'routing' in report.lower()
    assert (output/'bell_settings.csv').is_file()
    assert (output/'summary.csv').is_file()
    assert (output/'gates.csv').is_file()


def test_corrupt_output_is_not_silently_recomputed(source,snapshot,tmp_path,fake_compilers):
    output=tmp_path/'hardware'
    module().run_hardware_layer(source,output,snapshot,states=['two_qutrit'])
    (output/'circuits/two_qutrit/00000/preparation_routed.qpy').write_bytes(b'corrupt')
    with pytest.raises(ValueError):
        module().run_hardware_layer(source,output,snapshot,states=['two_qutrit'],resume=True)


def test_changed_configuration_and_source_rejected(source,snapshot,tmp_path,fake_compilers):
    output=tmp_path/'hardware'
    module().run_hardware_layer(source,output,snapshot,states=['two_qutrit'])
    with pytest.raises(ValueError):
        module().run_hardware_layer(source,output,snapshot,states=['two_qutrit'],seeds=(7,),resume=True)
    (source/'added.txt').write_text('new source evidence')
    with pytest.raises(ValueError):
        module().run_hardware_layer(source,output,snapshot,states=['two_qutrit'],resume=True)


@pytest.mark.parametrize('change',[{'seeds':()}, {'seeds':(True,)}, {'indices':[1]}, {'states':[]}, {'states':['unknown']}])
def test_invalid_selection_rejected_before_output(source,snapshot,tmp_path,change):
    output=tmp_path/'hardware'
    with pytest.raises(ValueError):module().run_hardware_layer(source,output,snapshot,**change)
    assert not output.exists()


def test_output_must_not_be_inside_source(source,snapshot):
    with pytest.raises(ValueError):
        module().run_hardware_layer(source,source/'hardware',snapshot,states=['two_qutrit'])


@pytest.mark.parametrize('edges', [[[0,True]], [[27,27]], [[27,100]], [[27,28],[28,27]], []])
def test_invalid_snapshot_edges_rejected(snapshot,edges):
    snapshot['edges']=edges
    with pytest.raises(ValueError):module().validate_snapshot(snapshot)


def test_profile_edges_come_from_snapshot(snapshot):
    validated=module().validate_snapshot(snapshot)
    profile=module().hardware_profile(validated,'two_qutrit')
    assert profile['physical_qubits']==[27,28,19,20]
    assert set(map(tuple,profile['edges']))=={(0,1),(0,2),(1,3),(2,3)}


def test_cli_requires_explicit_local_inputs():
    import importlib.util
    script=Path(__file__).resolve().parents[1]/'scripts/run_theta_hardware_layer.py'
    spec=importlib.util.spec_from_file_location('theta_hardware_cli',script)
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    with pytest.raises(SystemExit) as error:cli.main([])
    assert error.value.code==2
