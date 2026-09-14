"""Nine optimized Bell settings executed sequentially on a pinned IQM calibration."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
from uuid import UUID

import numpy as np
from qiskit import qpy
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT/'src'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import iqm_bell_short as short
from qudits_on_qubits.bell_measurements.bell_optimized import SETTINGS, build_catalog

SOURCE = ROOT/'experiment_inputs/bell_optimized/raw_reference.json'
INPUT_FILES = ('parents.qpy', 'bell_weights.npy', 'ideal_probabilities.npy', 'circuit_evidence.json')


def _plan(circuits, shots):
    if type(shots) is not int or shots < 2:
        raise ValueError('Expected integer shots >=2')
    if len(circuits) != 9 or [c.metadata.get('setting_index') for c in circuits] != list(range(9)):
        raise ValueError('Expected nine settings ordered A0/B0 through A2/B2')
    return [dict(arm='RAW', dd=False, shots=shots, circuits=[c],
                 records=[dict(setting=i, repeat=0, scale=1, actual_scale=1.)])
            for i, c in enumerate(circuits)]


def prepare(source, output, backend, *, shots=8192):
    """Read-only hardware preflight; freeze all input before any submission."""
    if type(shots) is not int or shots < 2:
        raise ValueError('Expected integer shots >=2')
    output = Path(output)
    if output.exists():
        raise FileExistsError('Choose a new campaign directory')
    catalog = build_catalog(source)
    mapped, selection = short.select_layout(catalog.optimized, backend)
    for index, (logical, physical) in enumerate(zip(catalog.optimized, mapped, strict=True)):
        physical.metadata = dict(logical.metadata)
        physical.name = f'optimized_{SETTINGS[index][0]}_{SETTINGS[index][1]}'
        if physical.count_ops() != logical.count_ops():
            raise ValueError('Mapping changed the native gate budget')
    _plan(mapped, shots)
    selection['qubit_names_logical_order'] = [backend.index_to_qubit_name(q) for q in selection['layout']]
    selection['measurement_map_c0_to_c3'] = [selection['layout'][q] for q in (2,3,0,1)]
    ideal = np.array([short.probabilities(c) for c in mapped])
    output.mkdir(parents=True, exist_ok=False)
    with (output/'parents.qpy').open('wb') as handle:
        qpy.dump(mapped, handle)
    np.save(output/'bell_weights.npy', catalog.weights, allow_pickle=False)
    np.save(output/'ideal_probabilities.npy', ideal, allow_pickle=False)
    short.write_json(output/'circuit_evidence.json', catalog.evidence)
    protocol = dict(schema_version=1, settings=[list(s) for s in SETTINGS], randomized=False,
        mode='nine sequential single-circuit jobs, each completed before the next submission',
        variant='optimized RAW', shots_per_setting=shots, total_shots=9*shots,
        dynamical_decoupling=False, twirling=False, zne=False, selection=selection,
        circuit_hashes=[short._fingerprint(c) for c in mapped],
        input_sha256={name: hashlib.sha256((output/name).read_bytes()).hexdigest() for name in INPUT_FILES},
        versions={name:version(name) for name in ('qiskit','iqm-client','numpy','scipy')},
        created_at=datetime.now(timezone.utc).isoformat())
    short.write_json(output/'protocol.json', protocol)
    return protocol


def load_prepared(output):
    output = Path(output)
    protocol = json.loads((output/'protocol.json').read_text(encoding='utf-8'))
    if (protocol.get('schema_version') != 1 or protocol.get('randomized') is not False
            or protocol.get('settings') != [list(s) for s in SETTINGS]
            or set(protocol.get('input_sha256', {})) != set(INPUT_FILES)):
        raise ValueError('Invalid frozen protocol')
    for name in INPUT_FILES:
        if hashlib.sha256((output/name).read_bytes()).hexdigest() != protocol['input_sha256'][name]:
            raise ValueError(f'Frozen input hash mismatch: {name}')
    with (output/'parents.qpy').open('rb') as handle:
        circuits = qpy.load(handle)
    if [short._fingerprint(c) for c in circuits] != protocol['circuit_hashes']:
        raise ValueError('Frozen circuit hash mismatch')
    _plan(circuits, protocol['shots_per_setting'])
    if protocol['total_shots'] != 9 * protocol['shots_per_setting']:
        raise ValueError('Frozen shot budget differs')
    weights = np.load(output/'bell_weights.npy', allow_pickle=False)
    if weights.shape != (9,16) or not np.isfinite(weights).all():
        raise ValueError('Invalid frozen Bell weights')
    return circuits, weights, protocol


def _verify_jobs(output, plan, selection, *, complete=False):
    output = Path(output)
    path = output/'manifest.json'
    expected = dict(selection=selection, jobs=[dict(arm='RAW', dd=False, shots=j['shots'],
        records=j['records'], circuit_hashes=[short._fingerprint(j['circuits'][0])]) for j in plan])
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8')) != expected:
            raise ValueError('Saved manifest differs from frozen sequential plan')
    elif complete or any(output.glob('job_*')):
        raise ValueError('Missing manifest for existing jobs; recover provenance')
    for index, job in enumerate(expected['jobs']):
        stem = output/f'job_{index:03d}'
        circuit_path = stem.with_suffix('.qpy')
        marker = stem.with_suffix('.submission.json')
        counts_path = stem.with_suffix('.counts.json')
        if complete or marker.exists() or counts_path.exists():
            if not circuit_path.exists():
                raise ValueError('Missing submitted circuit evidence')
        if circuit_path.exists():
            with circuit_path.open('rb') as handle:
                saved = qpy.load(handle)
            if [short._fingerprint(c) for c in saved] != job['circuit_hashes']:
                raise ValueError('Submitted circuit hash mismatch')
        if counts_path.exists():
            counts = json.loads(counts_path.read_text(encoding='utf-8'))
            if (counts['calibration_set_id'] != selection['calibration_set_id']
                    or len(counts['counts']) != 1):
                raise ValueError('Saved counts differ from selected calibration or circuit count')
            short.count_vector(counts['counts'][0], job['shots'])
        elif complete:
            raise ValueError('Incomplete nine-setting campaign')


def execute(output, backend, *, allow_hardware=False):
    """Resume known jobs only; execute settings 0..8 with no randomization."""
    if allow_hardware is not True:
        raise PermissionError('Hardware execution requires explicit opt-in')
    from filelock import FileLock
    with FileLock(str(Path(output)/'.execution.lock'), timeout=0):
        return _execute_locked(output, backend)


def _execute_locked(output, backend):
    circuits, _, protocol = load_prepared(output)
    selection = protocol['selection']
    if (str(backend.architecture.calibration_set_id) != selection['calibration_set_id']
            or backend.name != selection['backend']):
        raise ValueError('Backend or calibration differs from frozen selection')
    names = [backend.index_to_qubit_name(q) for q in selection['layout']]
    if names != selection['qubit_names_logical_order']:
        raise ValueError('Physical qubit names differ from frozen layout')
    plan = _plan(circuits, protocol['shots_per_setting'])
    _verify_jobs(output, plan, selection)
    samples = short.execute(output, plan, backend, selection, allow_hardware=True)
    _verify_jobs(output, plan, selection, complete=True)
    return samples


def analyze(samples, weights, *, classical_bound):
    """Unconditional RAW Bell sum; approximate CI accounts for shot noise only."""
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (9,16) or not np.isfinite(weights).all() or not np.isfinite(classical_bound):
        raise ValueError('Expected finite weights and bound')
    if len(samples) != 9 or sorted(s['setting'] for s in samples) != list(range(9)):
        raise ValueError('Expected exactly nine distinct RAW settings')
    rows = []
    for sample in sorted(samples, key=lambda s:s['setting']):
        setting, shots = sample['setting'], sample['shots']
        if (type(setting) is not int or type(shots) is not int or shots < 2
                or sample['arm'] != 'RAW' or sample.get('scale',1) != 1):
            raise ValueError('Invalid RAW sample')
        p = short.count_vector(sample['counts'], shots)
        w = weights[setting]
        mean = float(w @ p)
        variance = max(0., float(w*w @ p - mean*mean)) / (shots-1)
        invalid = float(sum(p[k] for k in range(16) if k & 3 == 3 or k >> 2 == 3))
        rows.append(dict(setting=list(SETTINGS[setting]), setting_index=setting, shots=shots,
                         bell=mean, variance=variance, standard_error=float(np.sqrt(variance)),
                         invalid_code_probability=invalid))
    bell = float(sum(r['bell'] for r in rows))
    se = float(np.sqrt(sum(r['variance'] for r in rows)))
    half = float(norm.ppf(.975)*se)
    bound = float(classical_bound)
    return dict(bell=bell, standard_error=se, ci95_low=bell-half, ci95_high=bell+half,
        classical_bound=bound, delta_from_classical=bell-bound,
        point_estimate_above_classical=bool(bell>bound), ci95_entirely_above_classical=bool(bell-half>bound),
        mean_invalid_code_probability=float(np.mean([r['invalid_code_probability'] for r in rows])),
        total_shots=sum(r['shots'] for r in rows), settings=rows,
        uncertainty='Approximate normal 95% interval: independent multinomial counting noise only; excludes drift and calibration/systematic errors. No repeated blocks.')


def finish(output):
    output = Path(output)
    circuits, weights, protocol = load_prepared(output)
    _verify_jobs(output, _plan(circuits, protocol['shots_per_setting']), protocol['selection'], complete=True)
    evidence = json.loads((output/'circuit_evidence.json').read_text(encoding='utf-8'))
    result = analyze(short.read_samples(output), weights, classical_bound=evidence['classical_bound'])
    result['selection'] = protocol['selection']
    result['historical_raw'] = []
    for name, directory in [('high_20260911', ROOT/'artifacts/iqm_bell_by_setting/high_shots_20260911'),
                             ('readout_20260911', ROOT/'artifacts/iqm_bell_readout/emerald_20260911')]:
        if not (directory/'manifest.json').exists():
            continue
        raw = [s for s in short.read_samples(directory) if s['arm']=='RAW' and s.get('scale',1)==1]
        old_weights = np.load(directory/'bell_weights.npy', allow_pickle=False)
        if not np.allclose(weights, old_weights, atol=1e-12, rtol=0):
            raise ValueError('Historical Bell weights differ')
        old = analyze(raw, weights, classical_bound=evidence['classical_bound'])
        result['historical_raw'].append(dict(series=name, bell=old['bell'], standard_error=old['standard_error'],
            delta_new_minus_old=result['bell']-old['bell'], source=str(directory),
            counts_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.glob('job_*.counts.json')}))
    result['input_sha256'] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in output.iterdir() if p.is_file() and p.suffix in ('.json','.npy','.qpy') and p.name!='analysis.json'}
    short.write_json(output/'analysis.json', result)
    lines = ['# IQM: dziewięć zoptymalizowanych ustawień Bella', '',
        f"Backend: {protocol['selection']['backend']}. Layout: {protocol['selection']['layout']}.",
        f"Kalibracja: {protocol['selection']['calibration_set_id']}.", '',
        f"RAW: **{result['bell']:.8f}**, SE {result['standard_error']:.8f}; 95% CI [{result['ci95_low']:.8f}, {result['ci95_high']:.8f}].",
        f"Granica klasyczna: **{result['classical_bound']:.12f}**. Różnica: {result['delta_from_classical']:+.8f}.",
        f"Estymata powyżej granicy: {result['point_estimate_above_classical']}. Cały 95% CI powyżej: {result['ci95_entirely_above_classical']}.", '',
        f"Budżet: 9 × {protocol['shots_per_setting']} = {result['total_shots']} shotów. Kolejność A0/B0 do A2/B2, osobne zadania wykonywane kolejno.",
        'Bez losowania ustawień, DD, twirlingu, ZNE i postselekcji. Nie wykonywano osobnego testu POVM.', '',
        '| Ustawienie | CZ | RAW | SE |', '|---|---:|---:|---:|']
    for row in result['settings']:
        lines.append(f"| {' / '.join(row['setting'])} | {5 if row['setting_index']%3==0 else 7} | {row['bell']:.8f} | {row['standard_error']:.8f} |")
    lines += ['', '## Porównanie historyczne', '']
    for old in result['historical_raw']:
        lines.append(f"- {old['series']}: {old['bell']:.8f}; zmiana nowy − stary: {old['delta_new_minus_old']:+.8f}.")
    lines += ['', 'Przedział obejmuje tylko niepewność zliczeń przy założeniu niezależnych prób. Nie obejmuje dryfu ani błędów systematycznych.',
              'Porównanie historyczne obejmuje inny czas wykonania i może obejmować inne kubity oraz kalibrację. Nie izoluje wpływu samej optymalizacji.', '']
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('prepare','execute','analyze'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shots', type=int, default=8192)
    args = parser.parse_args(argv)
    if args.mode=='analyze':
        result=finish(args.output)
        print(json.dumps({k:result[k] for k in ('bell','standard_error','ci95_low','ci95_high','classical_bound','ci95_entirely_above_classical')},indent=2),flush=True)
        return 0
    from iqm.qiskit_iqm import IQMProvider
    from qudits_on_qubits.benchmarks.direct_basis.iqm_backend import load_iqm_environment
    env=load_iqm_environment()
    provider=IQMProvider(env.server_url,quantum_computer='emerald')
    if args.mode=='prepare':
        backend=provider.get_backend(use_metrics=True)
        protocol=prepare(SOURCE,args.output,backend,shots=args.shots)
        print(json.dumps(protocol,indent=2),flush=True)
    else:
        _,_,protocol=load_prepared(args.output)
        backend=provider.get_backend(use_metrics=False,calibration_set_id=UUID(protocol['selection']['calibration_set_id']))
        execute(args.output,backend,allow_hardware=True)
        result=finish(args.output)
        print(json.dumps({k:result[k] for k in ('bell','standard_error','ci95_low','ci95_high','classical_bound','ci95_entirely_above_classical')},indent=2),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
