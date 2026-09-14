"""Frozen randomized Bell benchmark: interleaved half-integer CZ folding, DD only."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import sys
from uuid import UUID

import numpy as np
from qiskit import qpy
from qiskit.circuit.library import CZGate

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT/'src'):
    if str(path) not in sys.path:
        sys.path.insert(0,str(path))

from scripts import iqm_bell_short as short
from scripts import iqm_bell_optimized_randomized as prior
from scripts.iqm_randomized_bell_campaign import measurement_mapping, write_json

SCALES = (1.,1.5,2.,3.,5.)
SOURCE = ROOT/'artifacts/bell_optimized/iqm_emerald_randomized_50x1000_20260914'
DEFAULT_OUTPUT = ROOT/'artifacts/bell_optimized/iqm_emerald_interleaved_dd_zne_50x1000_20260915'


def now():
    return datetime.now(timezone.utc).isoformat()


def options():
    from iqm.iqm_client import CircuitCompilationOptions, DDMode, STANDARD_DD_STRATEGY
    return CircuitCompilationOptions(dd_mode=DDMode.ENABLED,dd_strategy=STANDARD_DD_STRATEGY)


def _repeat(circuit, repetitions):
    result=circuit.copy_empty_like();index=0
    for item in circuit.data:
        times=int(repetitions[index]) if item.operation.name=='cz' else 1
        if item.operation.name=='cz':index+=1
        for _ in range(times):result.append(item.operation,item.qubits,item.clbits)
    return result


def fold_variants(circuit, scale, rng):
    if scale not in SCALES:
        raise ValueError('Use approved scales 1,1.5,2,3,5')
    gates=[item.operation for item in circuit.data if item.operation.name=='cz']
    if not gates or any(not isinstance(g,CZGate) or getattr(g,'condition',None) is not None for g in gates):
        raise ValueError('Expected unconditioned native CZ gates')
    if any(i.operation.name not in ('r','cz','measure','barrier') for i in circuit.data):
        raise ValueError('Only frozen native R/CZ/measurement circuits are supported')
    n=len(gates);base=int((scale-1)//2);extra=int(round(2*(scale-(1+2*base))))
    repetitions=np.full((4,n),1+2*base,dtype=int)
    # Each original gate receives exactly extra additional CZ pairs across the
    # four variants, with balanced row totals. Randomness changes folding sites,
    # not measurement settings or the frozen overall shot budget.
    ranks=rng.permutation(n);offset=int(rng.integers(4))
    for rank,gate in enumerate(ranks):
        for step in range(extra):repetitions[(rank+offset+step)%4,gate]+=2
    return [_repeat(circuit,row) for row in repetitions],repetitions


def make_plan(parents, schedule, *, seed=20260915, check_ideal=True):
    if len(parents)!=9 or [c.metadata.get('setting_index') for c in parents]!=list(range(9)):
        raise ValueError('Expected nine optimized parents in setting order')
    blocks=schedule['blocks']
    if len(blocks)!=50 or [b['block_id'] for b in blocks]!=list(range(50)):
        raise ValueError('Expected the frozen 50-block schedule')
    labels=[(f'A{a}',f'B{b}') for a in range(3) for b in range(3)]
    rng=np.random.default_rng(seed);plan=[];checked=set()
    ideal=[short.probabilities(c) for c in parents] if check_ideal else None
    for start in range(0,50,5):
        circuits=[];records=[]
        for block in blocks[start:start+5]:
            j=labels.index(tuple(block['settings']));parent=parents[j]
            for scale in SCALES:
                variants,repetitions=fold_variants(parent,scale,rng)
                for instance,(c,rep) in enumerate(zip(variants,repetitions,strict=True)):
                    key=(j,tuple(rep))
                    if check_ideal and key not in checked:
                        if not np.allclose(short.probabilities(c),ideal[j],atol=1e-12,rtol=0):
                            raise ValueError('Folding changed ideal probabilities')
                        checked.add(key)
                    record=dict(block_id=block['block_id'],setting=j,scale=scale,instance=instance,
                        actual_scale=float(np.mean(rep)),cz_repetitions=rep.tolist(),
                        native_cz=int(np.sum(rep)),parent_cz=len(rep),mapping=list(measurement_mapping(c)))
                    circuits.append(c);records.append(record)
        order=rng.permutation(len(circuits))
        plan.append(dict(shots=250,circuits=[circuits[i] for i in order],records=[records[i] for i in order]))
    return plan


def freeze(output, parents, weights, schedule, selection, plan, *, source_evidence, seed):
    output=Path(output)
    if output.exists():raise FileExistsError('Choose a new preparation directory')
    weights=np.asarray(weights)
    if weights.shape!=(9,16) or not np.isfinite(weights).all():raise ValueError('Invalid Bell weights')
    output.mkdir(parents=True,exist_ok=False)
    with (output/'parents.qpy').open('wb') as f:qpy.dump(parents,f)
    np.save(output/'bell_weights.npy',weights,allow_pickle=False)
    write_json(output/'schedule.json',schedule)
    write_json(output/'source_evidence.json',source_evidence)
    batches=[]
    for i,job in enumerate(plan):
        name=f'batch_{i:02d}.qpy'
        with (output/name).open('wb') as f:qpy.dump(job['circuits'],f)
        batches.append(dict(file=name,shots=job['shots'],records=job['records'],
            circuit_hashes=[short._fingerprint(c) for c in job['circuits']]))
    write_json(output/'batches.json',batches)
    protocol=dict(schema_version=1,variant='DD_ZNE_INTERLEAVED',selection=selection,
        scales=list(SCALES),draws=50,shots_per_draw_per_scale=1000,instances=4,shots_per_instance=250,
        total_shots=250000,jobs=10,max_circuits_per_job=100,dd=dict(mode=options().dd_mode.value,strategy=options().dd_strategy.model_dump(mode='json')),
        twirling=False,readout_mitigation=False,seed=seed,created_at=now(),
        interleaving='all five scales for five saved blocks in each job; shuffled circuits',
        fractional_folding='each native CZ has exact requested mean repetition count across four equally weighted variants',
        estimator='unconditional sum of nine within-setting block means; zero leakage weight',
        model_protocol='primary unweighted linear across all five scales; quadratic sensitivity uses actual scale moments per setting',
        source_evidence=source_evidence,versions={n:version(n) for n in ('numpy','scipy','qiskit','iqm-client')},
        input_sha256={p.name:prior.sha256(p) for p in output.iterdir() if p.is_file()})
    write_json(output/'protocol.json',protocol)
    (output/'protocol.sha256').write_text(prior.sha256(output/'protocol.json'),encoding='ascii')
    load_prepared(output)
    return protocol


def load_prepared(output):
    output=Path(output)
    if prior.sha256(output/'protocol.json')!=(output/'protocol.sha256').read_text(encoding='ascii'):
        raise ValueError('Frozen protocol hash mismatch')
    p=prior.read_json(output/'protocol.json')
    expected={'parents.qpy','bell_weights.npy','schedule.json','source_evidence.json','batches.json',
        *[f'batch_{i:02d}.qpy' for i in range(10)]}
    if (p.get('schema_version')!=1 or p.get('scales')!=list(SCALES) or p.get('total_shots')!=250000
        or p.get('dd')!=dict(mode=options().dd_mode.value,strategy=options().dd_strategy.model_dump(mode='json')) or p.get('twirling') is not False
        or p.get('readout_mitigation') is not False or set(p.get('input_sha256',{}))!=expected):
        raise ValueError('Invalid frozen DD-only protocol')
    for name,digest in p['input_sha256'].items():
        if prior.sha256(output/name)!=digest:raise ValueError(f'Frozen input hash mismatch: {name}')
    with (output/'parents.qpy').open('rb') as f:parents=qpy.load(f)
    schedule=prior.read_json(output/'schedule.json')
    expected_plan=make_plan(parents,schedule,seed=p['seed'],check_ideal=False)
    batches=prior.read_json(output/'batches.json');jobs=[]
    if len(batches)!=10:raise ValueError('Expected ten interleaved batches')
    for i,(batch,expected_job) in enumerate(zip(batches,expected_plan,strict=True)):
        if (batch['file']!=f'batch_{i:02d}.qpy' or batch['shots']!=250
            or batch['records']!=expected_job['records']):raise ValueError('Saved records differ from interleaved design')
        with (output/batch['file']).open('rb') as f:circuits=qpy.load(f)
        hashes=[short._fingerprint(c) for c in circuits]
        if hashes!=batch['circuit_hashes'] or hashes!=[short._fingerprint(c) for c in expected_job['circuits']]:
            raise ValueError('Frozen circuit hash mismatch')
        jobs.append(dict(**batch,circuits=circuits))
    weights=np.load(output/'bell_weights.npy',allow_pickle=False)
    if weights.shape!=(9,16) or not np.isfinite(weights).all():raise ValueError('Invalid frozen weights')
    return p,schedule,jobs,weights


def backend_for(output=None):
    from qudits_on_qubits.benchmarks.direct_basis.iqm_backend import load_iqm_environment
    from iqm.qiskit_iqm import IQMProvider
    env=load_iqm_environment()
    provider=IQMProvider(env.server_url,quantum_computer='emerald')
    kwargs={} if output is None else dict(calibration_set_id=UUID(prior.read_json(Path(output)/'protocol.json')['selection']['calibration_set_id']))
    return provider.get_backend(use_metrics=False,**kwargs)


def prepare(output, source=SOURCE, *, backend, seed=20260915):
    source=Path(source)
    original,schedule,_=prior.load_prepared(source)
    with (source/'parents.qpy').open('rb') as f:parents=qpy.load(f)
    weights=np.load(source/'bell_weights.npy',allow_pickle=False)
    selection=dict(original['selection']);previous_calibration=selection['calibration_set_id']
    selection['calibration_set_id']=str(backend.architecture.calibration_set_id)
    if backend.name!=selection['backend'] or [backend.index_to_qubit_name(q) for q in selection['layout']]!=selection['qubit_names_logical_order']:
        raise ValueError('Current backend differs from frozen physical layout')
    # Previous error metrics describe the earlier calibration, not a fresh ranking.
    selection['layout_selection_calibration_set_id']=previous_calibration
    plan=make_plan(parents,schedule.to_safe_dict(),seed=seed)
    serialized=0
    for job in plan:
        request=backend.create_run_request(job['circuits'],shots=job['shots'],circuit_compilation_options=options())
        if len(request.circuits)!=len(job['circuits']):raise ValueError('Serialization lost circuits')
        serialized+=len(request.circuits)
        for native,c in zip(request.circuits,job['circuits'],strict=True):
            counts=Counter(i.name for i in native.instructions)
            if counts['cz']!=c.count_ops()['cz'] or counts['prx']!=c.count_ops().get('r',0):
                raise ValueError('Serialization changed native CZ/R budget')
    evidence=dict(source_directory=str(source.resolve()),source_protocol_sha256=prior.sha256(source/'protocol.json'),
        source_input_sha256=original['input_sha256'],ideal_bell=float(sum(w@short.probabilities(c) for w,c in zip(weights,parents,strict=True))),
        ideal_probabilities_preserved=True,serialized_circuits=serialized,previous_calibration_set_id=previous_calibration)
    if abs(evidence['ideal_bell']-6)>1e-10:raise ValueError('Ideal Bell reference is not six')
    return freeze(output,parents,weights,schedule.to_safe_dict(),selection,plan,source_evidence=evidence,seed=seed)


def _counts_payload(output,index,job,p):
    hw=Path(output)/'hardware';stem=f'job_{index:03d}'
    marker=prior.read_json(hw/f'{stem}.submission.json')
    if marker.get('status')!='submitted' or not marker.get('job_id'):
        raise ValueError('Completed counts lack submitted job identity')
    expected=dict(index=index,shots=job['shots'],dd=p['dd'],records=job['records'],qpy_sha256=p['input_sha256'][job['file']])
    if prior.read_json(hw/f'{stem}.request.json')!=expected:raise ValueError('Submitted request differs from frozen plan')
    saved=prior.read_json(hw/f'{stem}.counts.json')
    if (saved.get('job_id')!=marker['job_id'] or saved.get('calibration_set_id')!=p['selection']['calibration_set_id']
        or saved.get('qpy_sha256')!=expected['qpy_sha256'] or len(saved.get('counts',[]))!=len(job['records'])):
        raise ValueError('Saved counts identity, calibration or length mismatch')
    for counts in saved['counts']:short.count_vector(counts,job['shots'])
    return saved


def execute(output,backend,*,allow_hardware=False):
    if allow_hardware is not True:raise PermissionError('Hardware execution requires explicit opt-in')
    from filelock import FileLock
    output=Path(output)
    with FileLock(str(output/'.execution.lock'),timeout=0):
        p,_,jobs,_=load_prepared(output);selection=p['selection']
        if (backend.name!=selection['backend'] or str(backend.architecture.calibration_set_id)!=selection['calibration_set_id']
            or [backend.index_to_qubit_name(q) for q in selection['layout']]!=selection['qubit_names_logical_order']):
            raise ValueError('Backend or calibration differs from frozen selection')
        hw=output/'hardware';hw.mkdir(exist_ok=True)
        # Freeze every request before the first submission, also when resuming.
        for i,job in enumerate(jobs):
            request=dict(index=i,shots=job['shots'],dd=p['dd'],records=job['records'],qpy_sha256=p['input_sha256'][job['file']])
            path=hw/f'job_{i:03d}.request.json'
            if path.exists() and prior.read_json(path)!=request:raise ValueError('Saved request differs from frozen plan')
            if not path.exists():write_json(path,request)
        for i,job in enumerate(jobs):
            stem=f'job_{i:03d}';marker=hw/f'{stem}.submission.json';counts_file=hw/f'{stem}.counts.json'
            if counts_file.exists():
                _counts_payload(output,i,job,p)
                print(f'{i+1}/10 already complete',flush=True);continue
            if marker.exists():
                saved=prior.read_json(marker)
                if saved.get('status')!='submitted' or not saved.get('job_id'):
                    raise RuntimeError(f'{stem}: submission_unknown; verify remote job before retry')
                pending=backend.retrieve_job(saved['job_id'])
            else:
                write_json(marker,dict(status='submission_unknown',started_at=now()))
                pending=backend.run(job['circuits'],shots=job['shots'],circuit_compilation_options=options())
                write_json(marker,dict(status='submitted',job_id=str(pending.job_id()),submitted_at=now()))
            print(f'{i+1}/10 DD, five interleaved scales, job_id={pending.job_id()}',flush=True)
            result=pending.result()
            if len(result.results)!=len(job['circuits']) or {str(getattr(r,'calibration_set_id','')) for r in result.results}!={selection['calibration_set_id']}:
                raise ValueError('Result calibration or circuit count differs from frozen plan')
            counts=[result.get_counts(k) for k in range(len(job['circuits']))]
            for count in counts:short.count_vector(count,job['shots'])
            write_json(counts_file,dict(job_id=str(pending.job_id()),calibration_set_id=selection['calibration_set_id'],
                qpy_sha256=p['input_sha256'][job['file']],counts=counts,finished_at=now()))
            _counts_payload(output,i,job,p)
            print(f'{i+1}/10 complete; {(i+1)*25000}/250000 shots saved',flush=True)
    return output


def read_samples(output):
    output=Path(output)
    audit_path=output/'completion_audit.json'
    if audit_path.exists():
        audit=prior.read_json(audit_path)
        for name,digest in audit['input_sha256'].items():
            path=(output/name).resolve()
            if not path.is_relative_to(output.resolve()) or prior.sha256(path)!=digest:
                raise ValueError('Completed evidence hash mismatch')
    p,schedule,jobs,weights=load_prepared(output);samples=[]
    for i,job in enumerate(jobs):
        saved=_counts_payload(output,i,job,p)
        for record,counts in zip(job['records'],saved['counts'],strict=True):
            samples.append(dict(**record,job_index=i,shots=job['shots'],counts=counts))
    return samples


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','execute','analyze'])
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--allow-hardware',action='store_true')
    args=parser.parse_args()
    if args.action=='prepare':
        result=prepare(args.output,args.source,backend=backend_for())
        print(json.dumps({k:result[k] for k in ('scales','total_shots','selection')},indent=2))
    elif args.action=='execute':
        execute(args.output,backend_for(args.output),allow_hardware=args.allow_hardware)
    else:
        from scripts.iqm_bell_interleaved_analysis import finish
        result=finish(args.output)
        print(json.dumps(result['models'],indent=2))


if __name__=='__main__':main()
