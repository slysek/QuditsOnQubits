"""Run the optimized two-qutrit Bell experiment locally; no hardware submission."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys

import numpy as np
from qiskit import qpy, transpile
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / 'src') not in sys.path:
    sys.path.insert(0, str(ROOT / 'src'))

from qudits_on_qubits.bell_measurements.bell_optimized import (
    ATOL, SETTINGS, BellCatalog, build_catalog, probabilities,
)

VARIANTS = ('baseline', 'optimized')
DEFAULT_SOURCE = ROOT / 'experiment_inputs/bell_optimized/raw_reference.json'


def make_plan(catalog: BellCatalog, *, blocks=16, shots=512, seed=20260914):
    """Keep every repetition; pair variants and counterbalance which runs first."""
    if (type(blocks) is not int or blocks < 2 or blocks % 2
            or type(shots) is not int or shots < 1
            or type(seed) is not int or not 0 <= seed < 2**32 - blocks):
        raise ValueError('Use even blocks >=2, positive shots, and a valid nonnegative seed')
    rng = np.random.default_rng(seed)
    offsets = rng.integers(0, 2, size=9)
    plan = []
    for block in range(blocks):
        circuits, records = [], []
        for setting in rng.permutation(9):
            first = int((block + offsets[setting]) % 2)
            for variant_index in (first, 1 - first):
                variant = VARIANTS[variant_index]
                circuit = getattr(catalog, variant)[setting].copy()
                position = len(records)
                circuit.name = f'block_{block:03d}_{position:02d}_{variant}_{setting}'
                circuits.append(circuit)
                records.append(dict(block=block, variant=variant, setting_index=int(setting),
                                    position_in_block=position, execution_index=18 * block + position))
        plan.append(dict(block=block, shots=shots, circuits=circuits, records=records))
    return plan


def _summary(values):
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    half = float(t.ppf(.975, len(values) - 1) * se)
    return dict(bell=mean, standard_error=se, ci95_low=mean-half, ci95_high=mean+half)


def analyze(samples, weights):
    """Estimate over complete independent blocks, retaining all code words."""
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (9, 16) or not np.isfinite(weights).all():
        raise ValueError('Expected finite weights with shape (9,16)')
    rows, shots_seen = {}, set()
    for sample in samples:
        block, variant, setting = (sample[k] for k in ('block', 'variant', 'setting_index'))
        if (type(block) is not int or block < 0 or variant not in VARIANTS
                or type(setting) is not int or not 0 <= setting < 9):
            raise ValueError('Invalid block, variant or setting')
        key = block, variant, setting
        if key in rows:
            raise ValueError('Duplicate block/variant/setting result')
        shots, counts = sample['shots'], sample['counts']
        if type(shots) is not int or shots < 1 or not isinstance(counts, dict):
            raise ValueError('Invalid shots or counts')
        vector = np.zeros(16)
        for bits, count in counts.items():
            if (not isinstance(bits, str) or len(bits) != 4 or set(bits) - {'0', '1'}
                    or type(count) is not int or count < 0):
                raise ValueError('Expected four-bit keys and nonnegative integer counts')
            vector[int(bits, 2)] = count
        if vector.sum() != shots:
            raise ValueError('Counts do not sum to declared shots')
        rows[key] = vector / shots
        shots_seen.add(shots)
    blocks = sorted({key[0] for key in rows})
    expected = {(b, v, s) for b in blocks for v in VARIANTS for s in range(9)}
    if (len(blocks) < 2 or blocks != list(range(len(blocks))) or set(rows) != expected
            or len(shots_seen) != 1):
        raise ValueError('Require complete consecutive blocks with equal shots; no merging')
    invalid = [k for k in range(16) if (k & 3) == 3 or (k >> 2) == 3]
    estimates, variants, block_rows = {}, {}, []
    for variant in VARIANTS:
        contributions = np.array([[weights[s] @ rows[b, variant, s] for s in range(9)] for b in blocks])
        values = contributions.sum(axis=1)
        estimates[variant] = values
        variants[variant] = dict(**_summary(values),
            mean_invalid_code_probability=float(np.mean([rows[b, variant, s][invalid].sum()
                                                         for b in blocks for s in range(9)])),
            settings=[dict(setting=list(SETTINGS[s]), **_summary(contributions[:, s])) for s in range(9)])
        block_rows.extend(dict(block=b, variant=variant, bell=float(values[b])) for b in blocks)
    return dict(blocks=len(blocks), shots_per_setting_per_block=next(iter(shots_seen)),
                interval_method='Student-t over independent block sums; paired difference by block',
                variants=variants, block_results=block_rows,
                paired_difference=_summary(estimates['optimized'] - estimates['baseline']))


def compile_locally(catalog, *, seed):
    """Check basis translation; do not claim physical-device readiness."""
    from iqm.qiskit_iqm.iqm_transpilation import IQMOptimizeSingleQubitGates
    from qiskit.transpiler import PassManager

    summaries, exports = {}, {}
    for profile, basis in (('iqm_rcz', ['r', 'cz']),
                           ('ibm_cz', ['rz', 'sx', 'x', 'cz']),
                           ('ibm_ecr', ['rz', 'sx', 'x', 'ecr'])):
        costs, error, profile_circuits = {}, 0., {}
        for variant in VARIANTS:
            originals = getattr(catalog, variant)
            compiled = transpile(list(originals), basis_gates=basis, optimization_level=1,
                                 seed_transpiler=seed, num_processes=1)
            if profile == 'iqm_rcz':
                compiled = PassManager([IQMOptimizeSingleQubitGates()]).run(compiled, num_processes=1)
            for old, new in zip(originals, compiled, strict=True):
                error = max(error, float(np.max(abs(probabilities(old) - probabilities(new)))))
                actual = sum(item.operation.num_qubits == 2 for item in new.data)
                if actual > old.count_ops()['cz'] or new.count_ops().get('swap', 0):
                    raise ValueError(f'{profile}: increased entangling-gate cost')
            costs[variant] = [dict(c.count_ops()) for c in compiled]
            profile_circuits[variant] = compiled
        if error > ATOL:
            raise ValueError(f'{profile}: changed Bell histogram')
        summaries[profile] = dict(basis_gates=basis, max_probability_error=error, costs=costs,
                                  physical_backend_validated=False,
                                  scope='Offline basis translation; no physical routing, timing or calibration')
        exports[profile] = profile_circuits
    return summaries, exports


def _write_json(path, document):
    path.write_text(json.dumps(document, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def _write_report(output, result):
    evidence, stats = result['circuit_evidence'], result['statistics']
    lines = ['# Lokalny benchmark eksperymentu Bella', '',
             'Wykonanie: idealny Aer, bez modelu szumu i bez zadań sprzętowych.', '',
             f"Budżet: {stats['blocks']} bloków × 18 obwodów × {stats['shots_per_setting_per_block']} shotów = {result['total_shots']} shotów.", '',
             'Każdy blok zawiera dziewięć ustawień obu wariantów. Kolejność ustawień jest losowana; ',
             'kolejność starego i nowego obwodu w parze jest zbalansowana między blokami. ',
             'Powtórzenia pozostają oddzielne. Kod 11 zachowuje masę w normalizacji.', '',
             '| Wariant | CZ (B0 / B1,B2) | R (B0 / B1,B2) | Idealna suma | Aer | 95% CI po blokach |',
             '|---|---|---|---:|---:|---|']
    for variant, label, cz, r in [('baseline', 'Referencja', '6 / 9', '14 / 22'),
                                  ('optimized', 'Optymalizacja', '5 / 7', '12 / 18')]:
        row = stats['variants'][variant]
        lines.append(f"| {label} | {cz} | {r} | {evidence['ideal_bell_' + variant]:.12f} | {row['bell']:.8f} | [{row['ci95_low']:.8f}, {row['ci95_high']:.8f}] |")
    difference = stats['paired_difference']
    lines += ['', f"Największa różnica idealnych histogramów: {evidence['max_probability_error']:.3g}.", '',
              f"Granica lokalna: {evidence['classical_bound']:.12f}.", '',
              f"Różnica sparowana (nowy − stary): {difference['bell']:.8f}; 95% CI [{difference['ci95_low']:.8f}, {difference['ci95_high']:.8f}].", '',
              'W idealnym symulatorze oba obwody dają ten sam rozkład. Różnica estymat wynika ze skończonej liczby shotów. ',
              'Ten test potwierdza obwody, histogramy i lokalny pipeline; nie przewiduje wyniku IBM ani IQM.', '',
              'Przejście na sprzęt wymaga aktualnej kalibracji, wyboru backendu, weryfikacji fizycznego layoutu, ',
              'mapy bitów i kosztu po routingu. Sprawdzono lokalną translację do baz IQM R/CZ oraz IBM CZ i ECR.', '',
              '## Artefakty', '',
              '- `baseline.qpy`, `optimized.qpy`: po dziewięć kompletnych obwodów z pomiarami.',
              '- `*_baseline.qpy`, `*_optimized.qpy`: lokalne translacje baz dla obu wariantów.',
              '- `execution_plan.json`: kolejność każdego wykonania, seed i budżet.',
              '- `samples.json`, `bell_weights.npy`: surowe zliczenia i wagi do ponownej analizy.',
              '- `result.json`: parametry syntezy, koszty, wersje bibliotek i wyniki bloków.',
              '- `files.sha256.json`: hashe artefaktów.', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')


def run_local(source=DEFAULT_SOURCE, output=None, *, blocks=16, shots=512, seed=20260914):
    """Execute one Aer job per block and export reproducible evidence."""
    from qiskit_aer import AerSimulator

    if output is None:
        raise ValueError('Choose a new output directory')
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Output already exists: {output}')
    catalog = build_catalog(source)
    plan = make_plan(catalog, blocks=blocks, shots=shots, seed=seed)
    compilation, exports = compile_locally(catalog, seed=seed)
    output.mkdir(parents=True, exist_ok=False)
    for name in VARIANTS:
        with (output / f'{name}.qpy').open('wb') as handle:
            qpy.dump(list(getattr(catalog, name)), handle)
    for profile, variants in exports.items():
        for name, circuits in variants.items():
            with (output / f'{profile}_{name}.qpy').open('wb') as handle:
                qpy.dump(circuits, handle)
    np.save(output / 'bell_weights.npy', catalog.weights, allow_pickle=False)
    _write_json(output / 'execution_plan.json', dict(seed=seed, blocks=blocks, shots=shots,
                simulator='AerSimulator', noise_model=None, deduplication=False,
                reference_digest=catalog.evidence['reference_digest'],
                jobs=[dict(block=j['block'], shots=j['shots'], seed_simulator=seed+j['block'],
                           records=j['records'], circuit_names=[c.name for c in j['circuits']]) for j in plan]))
    simulator = AerSimulator(max_parallel_threads=1)
    samples = []
    for job in plan:
        started = datetime.now(timezone.utc).isoformat()
        result = simulator.run(job['circuits'], shots=shots, seed_simulator=seed+job['block']).result()
        if not result.success or len(result.results) != len(job['records']):
            raise RuntimeError('Incomplete or failed Aer block')
        finished = datetime.now(timezone.utc).isoformat()
        for i, record in enumerate(job['records']):
            samples.append(dict(**record, shots=shots, counts=result.get_counts(i),
                                block_started_at=started, block_completed_at=finished,
                                simulator_seed=seed+job['block']))
        _write_json(output / 'samples.json', samples)
    result = dict(status='completed_local', hardware_submitted=False, noise_model=None,
                  total_shots=blocks * 18 * shots,
                  versions={name: version(name) for name in ('qiskit', 'qiskit-aer', 'iqm-client', 'numpy', 'scipy')},
                  python=platform.python_version(), seed=seed, circuit_evidence=catalog.evidence,
                  compilation=compilation, statistics=analyze(samples, catalog.weights))
    _write_json(output / 'result.json', result)
    _write_report(output, result)
    _write_json(output / 'files.sha256.json', {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                              for p in sorted(output.iterdir()) if p.is_file()})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--blocks', type=int, default=16)
    parser.add_argument('--shots', type=int, default=512)
    parser.add_argument('--seed', type=int, default=20260914)
    args = parser.parse_args(argv)
    result = run_local(args.source, args.output, blocks=args.blocks, shots=args.shots, seed=args.seed)
    print(json.dumps(dict(status=result['status'], total_shots=result['total_shots'],
                          report=str(args.output.resolve() / 'report.md'),
                          statistics=result['statistics']), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
