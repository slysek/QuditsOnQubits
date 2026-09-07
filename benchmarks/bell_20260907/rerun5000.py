"""Exact 5000-shot repeat, factor 1 only, with the original global budget ledger."""
from contextlib import contextmanager
from datetime import datetime, timezone
import argparse
import json
import math
from pathlib import Path
import shutil
import time

import execute
import hardware
import study

LEDGER = study.ROOT
OUTPUT = LEDGER / 'rerun5000'
SHOTS = 5000
SETTING_COUNTS = dict(zip(study.STATES, (9, 12, 13)))


def arms():
    selection = json.loads((LEDGER / 'selection.json').read_text())
    return [(s, n, v) for s in study.STATES
            for n in ['canonical_ez'] + selection[s]['selected'] for v in study.VARIANTS]


def select_arm(circuits, mapping, arm):
    if len(circuits) != len(mapping):
        raise ValueError('circuit mapping mismatch')
    selected = [(c, m) for c, m in zip(circuits, mapping)
                if 'calibration' in m or tuple(m.get(k) for k in ('state', 'name', 'variant')) == arm]
    settings = [m['setting_index'] for c, m in selected if 'calibration' not in m]
    calibrations = [m['calibration'] for c, m in selected if 'calibration' in m]
    if sorted(settings) != list(range(SETTING_COUNTS[arm[0]])) or sorted(calibrations) != [0, 1]:
        raise ValueError('requires complete measurement arm and two calibrations')
    return [c for c, m in selected], [m for c, m in selected]


@contextmanager
def configured(provider, arm=None):
    """Only circuit loading uses the stage root; accounting always uses LEDGER."""
    original_backend, original_batch, original_root = execute.backend_for, execute.make_batch, execute.ROOT
    backend, service = hardware.backend_for(provider)
    if provider == 'ibm':
        backend = service.backend('ibm_fez')
    if backend.name != ('ibm_fez' if provider == 'ibm' else 'garnet'):
        raise ValueError('unexpected real backend')
    source = LEDGER / 'fez' if provider == 'ibm' else LEDGER

    def batch(p, b, factor, seed, states, pilot=False):
        saved = execute.ROOT
        try:
            execute.ROOT = source
            circuits, mapping = original_batch(p, b, factor, seed, states, pilot)
        finally:
            execute.ROOT = saved
        return select_arm(circuits, mapping, arm) if arm is not None else (circuits, mapping)

    def get_backend(p):
        if p != provider:
            raise ValueError('provider context mismatch')
        return backend, service

    try:
        execute.ROOT = LEDGER
        execute.backend_for = get_backend
        execute.make_batch = batch
        yield backend, service
    finally:
        execute.backend_for, execute.make_batch, execute.ROOT = original_backend, original_batch, original_root


def initialize():
    OUTPUT.mkdir(exist_ok=True)
    for name in ('selection.json', 'shortlist.json', 'coding_bases.json'):
        shutil.copy2(LEDGER / name, OUTPUT / name)
    for provider in ('ibm', 'iqm'):
        source = LEDGER / 'fez' if provider == 'ibm' else LEDGER
        (OUTPUT / provider).mkdir(exist_ok=True)
        for name in ('final_compile.json', 'verification.json', 'calibration.json'):
            shutil.copy2(source / provider / name, OUTPUT / provider / name)
    study.save_json(OUTPUT / 'protocol.json', {
        'shots_per_setting': SHOTS, 'factors': [1], 'zne_omission': 'User explicitly approved omission when full 5000-shot ZNE does not fit.',
        'backends': ['ibm_fez', 'garnet'], 'arms_per_provider': arms(),
        'circuit_sources': {'ibm': str(LEDGER / 'fez/ibm/final'), 'iqm': str(LEDGER / 'iqm/final')},
        'limits_seconds': execute.LIMITS, 'historical_usage_seconds': {'ibm': 311, 'iqm': 218.937842},
        'ibm_rep_delay': .000075, 'ibm_randomizations': 8, 'ibm_shots_per_randomization': 625,
        'iqm_partition': 'One complete state/basis/F3 arm per job, with two readout calibration circuits. No duplicate baseline arms.',
        'analysis': 'Separate 5000-shot series; no pooling with historical counts; 2000 bootstrap samples.',
        'created_at': datetime.now(timezone.utc).isoformat(),
    })


def run_ibm():
    label = 'raw5000-fez'
    with configured('ibm'):
        path = execute.job_directory('ibm', label) / 'receipt.json'
        if not path.exists():
            execute.prepare('ibm', label, SHOTS, 1, 220, list(study.STATES), rep_delay=.000075)
        receipt = json.loads(path.read_text())
        if receipt['shots'] != SHOTS or receipt['backend'] != 'ibm_fez' or receipt['factor'] != 1:
            raise ValueError('existing rerun receipt mismatch')
        if receipt['status'] == 'prepared':
            execute.submit('ibm', label)
        else:
            execute.retrieve('ibm', label)


def run_iqm():
    for index, arm in enumerate(arms()):
        label = f'raw5000-iqm-{index:02d}'
        path = execute.job_directory('iqm', label) / 'receipt.json'
        with configured('iqm', arm) as (backend, _):
            if not path.exists():
                circuits, mapping = execute.make_batch('iqm', backend, 1, 5908, [arm[0]])
                required, model = execute.iqm_reservation(circuits, SHOTS, backend)
                reservation = math.ceil(required) + 2
                print(json.dumps({'label': label, 'arm': arm, 'reservation': reservation,
                                  'used': execute.used_budget('iqm')}), flush=True)
                execute.prepare('iqm', label, SHOTS, 1, reservation, [arm[0]])
            receipt = json.loads(path.read_text())
            if receipt['shots'] != SHOTS or receipt['backend'] != 'garnet' or receipt['factor'] != 1:
                raise ValueError('existing rerun receipt mismatch')
            if receipt['status'] == 'prepared':
                execute.submit('iqm', label)
            while True:
                receipt = json.loads(path.read_text())
                if receipt['status'] == 'completed':
                    break
                if receipt['status'] not in ('submitted',):
                    raise ValueError('uncertain or failed submission; manual retrieval required, never resubmit')
                execute.retrieve('iqm', label)
                receipt = json.loads(path.read_text())
                if any(s in receipt.get('remote_status', '') for s in ('error', 'cancel', 'failed')):
                    raise RuntimeError('IQM job did not complete; preserve budget reservation and inspect')
                if receipt['status'] != 'completed':
                    time.sleep(5)


def analyze_completed():
    import analyze
    for provider in ('ibm', 'iqm'):
        for path in (LEDGER / provider / 'jobs').glob('raw5000-*/receipt.json'):
            receipt = json.loads(path.read_text())
            if receipt['status'] != 'completed':
                continue
            dest = OUTPUT / provider / 'jobs' / path.parent.name
            dest.mkdir(parents=True, exist_ok=True)
            for name in ('receipt.json', 'counts.json', 'metrics.json', 'timeline.json'):
                source = path.parent / name
                if source.exists():
                    shutil.copy2(source, dest / name)
            if not (dest / 'analysis.json').exists():
                analyze.analyze_job(dest / 'receipt.json', 2000)
                print(json.dumps({'analyzed': path.parent.name}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['initialize', 'ibm', 'iqm', 'analyze'])
    args = parser.parse_args()
    {'initialize': initialize, 'ibm': run_ibm, 'iqm': run_iqm, 'analyze': analyze_completed}[args.action]()
