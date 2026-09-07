"""Offline, checksummed replay of the published Bell hardware data. No QPU calls."""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import redirect_stdout
import copy
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import zipfile

HERE = Path(__file__).resolve().parent
KEYS = ('raw_unconditional', 'raw_conditional', 'readout_unconditional', 'readout_conditional')


def verify_archive(path, manifest):
    for name in manifest['files']:
        parts = PurePosixPath(name)
        if parts.is_absolute() or '..' in parts.parts or '\\' in name or ':' in name:
            raise ValueError('unsafe archive path')
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['archive_sha256']:
        raise ValueError('archive checksum mismatch')
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(manifest['files']):
            raise ValueError('archive member set mismatch')
        if sum(x.file_size for x in archive.infolist()) > 512_000_000:
            raise ValueError('archive exceeds size limit')
        for name in names:
            parts = PurePosixPath(name)
            if parts.is_absolute() or '..' in parts.parts or '\\' in name or ':' in name:
                raise ValueError('unsafe archive path')
            data = archive.read(name)
            entry = manifest['files'][name]
            if len(data) != entry['bytes'] or hashlib.sha256(data).hexdigest() != entry['sha256']:
                raise ValueError('member checksum mismatch: ' + name)


def validate_counts(receipt, counts):
    if len(counts) != receipt['circuit_count'] or len(counts) != len(receipt['mapping']):
        raise ValueError('counts and circuit mapping mismatch')
    for circuit in counts:
        if not circuit or any(type(n) is not int or n < 0 for n in circuit.values()):
            raise ValueError('invalid counts')
        if sum(circuit.values()) != receipt['shots']:
            raise ValueError('wrong shots in hardware result')


def compare_estimates(actual, expected, keys=KEYS, bootstrap=False):
    def keyed(rows):
        result = {(r['state'], r['name'], r['variant']): r for r in rows}
        if len(result) != len(rows):
            raise ValueError('duplicate analysis groups')
        return result
    left, right = keyed(actual), keyed(expected)
    if set(left) != set(right):
        raise ValueError('missing or additional analysis groups')
    for group in left:
        for field in ('invalid_fraction', 'total_shots', 'shots_per_setting'):
            if field not in right[group]:
                continue
            a, b = left[group].get(field), right[group][field]
            matches = (a is not None and math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)) if field == 'invalid_fraction' else a == b
            if not matches:
                raise ValueError(f'analysis metadata mismatch: {group} {field}')
        for key in keys:
            fields = ('value', 'se') if bootstrap else ('value',)
            for field in fields:
                if not math.isclose(left[group][key][field], right[group][key][field], rel_tol=1e-9, abs_tol=1e-9):
                    raise ValueError(f'estimate mismatch: {group} {key} {field}')
            if bootstrap:
                for a, b in zip(left[group][key]['ci95'], right[group][key]['ci95'], strict=True):
                    if not math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9):
                        raise ValueError(f'estimate CI mismatch: {group} {key}')


def point_estimates(receipt, counts, expected):
    """Recalculate all four estimates directly from counts, without resampling."""
    import numpy as np
    import analyze
    from study import workload, outcome_scores
    model, _ = analyze.calibration_models(receipt, counts, 1, np.random.default_rng(0))
    grouped = defaultdict(dict)
    for i, item in enumerate(receipt['mapping']):
        if 'calibration' not in item:
            key = (item['state'], item['name'], item['variant'])
            if item['setting_index'] in grouped[key]:
                raise ValueError('duplicate setting')
            grouped[key][item['setting_index']] = i
    rows = copy.deepcopy(expected)
    if set(grouped) != {(x['state'], x['name'], x['variant']) for x in rows}:
        raise ValueError('unexpected measurement groups')
    for row in rows:
        key = (row['state'], row['name'], row['variant'])
        _, meta = workload(*key)
        width = 2 * len(meta['qutrit_qubits'])
        scores, valid = outcome_scores(meta, width)
        settings = grouped[key]
        if sorted(settings) != list(range(len(scores))):
            raise ValueError('incomplete settings')
        values = np.zeros(4)
        shots_per_setting = []
        accepted_shots = 0
        for setting, index in sorted(settings.items()):
            count = analyze.count_array(counts[index], width)
            physical = receipt['mapping'][index]['physical_by_classical']
            score, accepted = scores[setting], valid[setting]
            corrected = analyze.inverse_readout_vectors(score, model, physical)
            norm = analyze.inverse_readout_vectors(accepted.astype(float), model, physical)
            denominators = (count.sum(), count @ accepted, count.sum(), count @ norm)
            shots_per_setting.append(int(count.sum()))
            accepted_shots += float(count @ accepted)
            if min(denominators) <= 0:
                raise ValueError('nonpositive normalization')
            values += np.array([count @ score, count @ score, count @ corrected, count @ corrected]) / denominators
        for key, value in zip(KEYS, values):
            row[key]['value'] = float(value)
        row['shots_per_setting'] = shots_per_setting
        row['total_shots'] = sum(shots_per_setting)
        row['invalid_fraction'] = 1 - accepted_shots / row['total_shots']
    return rows


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(items, output, bootstrap):
    import numpy as np
    groups = defaultdict(list)
    for job, row, draws in items:
        groups[(job['series'], job['backend'], row['state'], row['name'], row['variant'], row['factor'])].append((row, draws))
    data = {}
    results = []
    for key, rows in sorted(groups.items()):
        weights = np.array([row['total_shots'] for row, _ in rows], float)
        weights /= weights.sum()
        draws = sum(weight * sample for weight, (_, sample) in zip(weights, rows))
        values = [sum(weight * row[name]['value'] for weight, (row, _) in zip(weights, rows)) for name in KEYS]
        data[key] = (values, draws)
        flat = dict(zip(('series', 'backend', 'state', 'name', 'variant', 'factor'), key))
        flat.update(total_shots=sum(row['total_shots'] for row, _ in rows),
                    invalid_fraction=float(sum(weight * row['invalid_fraction'] for weight, (row, _) in zip(weights, rows))),
                    job_ids='|'.join(row['job_id'] for row, _ in rows), bootstrap_recomputed=bootstrap)
        for i, name in enumerate(KEYS):
            flat.update({name: values[i], name+'_se': float(np.std(draws[:, i], ddof=1)),
                         name+'_ci95_low': float(np.quantile(draws[:, i], .025)),
                         name+'_ci95_high': float(np.quantile(draws[:, i], .975))})
        results.append(flat)
    write_csv(output/'results.csv', results)
    differences = []
    for key, (values, draws) in sorted(data.items()):
        series, backend, state, name, variant, factor = key
        if factor != 1:
            continue
        comparisons = [('candidate_minus_baseline_same_f3', (series, backend, state, 'canonical_ez', variant, 1))]
        if variant == 'optimal':
            comparisons.append(('f3_opt_minus_standard', (series, backend, state, name, 'standard', 1)))
            comparisons.append(('candidate_opt_minus_baseline_standard', (series, backend, state, 'canonical_ez', 'standard', 1)))
        if series == 'rerun5000':
            comparisons.append(('new_minus_original', ('original', backend, state, name, variant, 1)))
        for comparison, other in comparisons:
            if other not in data:
                continue
            old_values, old_draws = data[other]
            for i, estimator in enumerate(KEYS):
                delta = values[i] - old_values[i]
                samples = draws[:, i] - old_draws[:, i]
                differences.append({'series': series, 'backend': backend, 'state': state, 'name': name,
                    'variant': variant, 'comparison': comparison, 'estimator': estimator, 'delta': delta,
                    'relative_percent': 100 * delta / old_values[i] if abs(old_values[i]) > 1e-12 else None,
                    'se': float(np.std(samples, ddof=1)), 'ci95_low': float(np.quantile(samples, .025)),
                    'ci95_high': float(np.quantile(samples, .975))})
    write_csv(output/'differences.csv', differences)
    zne = []
    for key in sorted(data):
        if key[-1] != 1 or not all((*key[:-1], f) in data for f in (3, 5)):
            continue
        points = [np.array(data[(*key[:-1], f)][0]) for f in (1, 3, 5)]
        samples = [data[(*key[:-1], f)][1] for f in (1, 3, 5)]
        intercept = sum(w*p for w, p in zip((13/12, 1/3, -5/12), points))
        intercept_samples = sum(w*p for w, p in zip((13/12, 1/3, -5/12), samples))
        curvature = points[0] - 2*points[1] + points[2]
        curvature_samples = samples[0] - 2*samples[1] + samples[2]
        for i, estimator in enumerate(KEYS):
            lo, hi = np.quantile(curvature_samples[:, i], [.025, .975])
            zne.append({'series': key[0], 'backend': key[1], 'state': key[2], 'name': key[3],
                'variant': key[4], 'estimator': estimator, 'value': float(intercept[i]),
                'se': float(np.std(intercept_samples[:, i], ddof=1)),
                'ci95_low': float(np.quantile(intercept_samples[:, i], .025)),
                'ci95_high': float(np.quantile(intercept_samples[:, i], .975)),
                'curvature': float(curvature[i]), 'curvature_ci95_low': float(lo), 'curvature_ci95_high': float(hi),
                'linearity_rejected': bool(lo > 0 or hi < 0)})
    write_csv(output/'zne.csv', zne)
    return results


def replay(output, bootstrap=False, series='all'):
    import numpy as np
    import analyze
    manifest = json.loads((HERE/'manifest.json').read_text())
    verify_archive(HERE/'data.zip', manifest)
    output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(HERE/'data.zip') as archive:
        archive.extractall(output/'data')  # Every member verified above, before any extraction.
    items = []
    checked = 0
    for job in manifest['jobs']:
        if series != 'all' and job['series'] != series:
            continue
        directory = output/'data'/job['directory']
        receipt = json.loads((directory/'receipt.json').read_text())
        if job['status'] != 'completed':
            continue
        counts = json.loads((directory/'counts.json').read_text())
        validate_counts(receipt, counts)
        if hashlib.sha256((directory/'submitted.qpy').read_bytes()).hexdigest() != receipt['payload_sha256']:
            raise ValueError('submitted QPY checksum mismatch')
        checked += 1
        if not job['has_analysis']:
            continue  # The completed low-shot pilot remains evidence, not a benchmark arm.
        expected = json.loads((directory/'expected_analysis.json').read_text())
        if bootstrap:
            with redirect_stdout(io.StringIO()):
                rows = analyze.analyze_job(directory/'receipt.json', 2000)
            bootstrap_path = directory/'bootstrap.npz'
        else:
            rows = point_estimates(receipt, counts, expected)
            bootstrap_path = directory/'expected_bootstrap.npz'
        compare_estimates(rows, expected, bootstrap=bootstrap)
        with np.load(bootstrap_path, allow_pickle=False) as draws:
            items.extend((job, row, draws[row['bootstrap_key']].copy()) for row in rows)
        print('Verified '+job['backend']+'/'+job['label'], flush=True)
    results = aggregate(items, output, bootstrap)
    # Compare all published primary point estimates, in addition to per-job checks.
    lookup = {(r['series'], r['backend'], r['state'], r['name'], r['variant'], r['factor']): r for r in results}
    for selected, path in [('original', HERE/'reports/all_results.csv'), ('rerun5000', HERE/'reports/rerun5000/results.csv')]:
        if series != 'all' and series != selected:
            continue
        with path.open(encoding='utf-8-sig', newline='') as stream:
            for row in csv.DictReader(stream):
                backend = row['backend'].replace('iqm_garnet', 'garnet')
                actual = lookup[(selected, backend, row['state'], row['name'], row['variant'], int(row['factor']))]
                for name in (*KEYS, 'invalid_fraction', 'total_shots'):
                    if name not in row:
                        continue
                    if not math.isclose(actual[name], float(row[name]), rel_tol=1e-9, abs_tol=1e-9):
                        raise ValueError('published primary result mismatch: ' + name)
    summary = {'verified_completed_jobs': checked, 'aggregate_rows': len(results), 'series': series,
               'bootstrap_recomputed': bootstrap, 'samples': 2000,
               'point_estimates_recomputed_from_counts': True, 'published_primary_values_match': True,
               'postselection_and_shot_metadata_verified': True,
               'no_network_or_qpu_calls': True,
               'uncertainty_note': 'Recomputed from counts and calibration bootstrap.' if bootstrap else 'Uses archived bootstrap draws; use --bootstrap to recompute SE/CI.'}
    (output/'verification.json').write_text(json.dumps(summary, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true', help='Verify archive SHA-256 without extraction or dependencies.')
    parser.add_argument('--bootstrap', action='store_true', help='Recompute all 2000-sample bootstraps and compare SE/CI to publication.')
    parser.add_argument('--series', choices=('all', 'original', 'rerun5000'), default='all')
    parser.add_argument('--output', type=Path, default=Path('artifacts/reproduced-bell-20260907'))
    args = parser.parse_args()
    if args.verify_only:
        verify_archive(HERE/'data.zip', json.loads((HERE/'manifest.json').read_text()))
        print('Archive and all member SHA-256 checks passed.')
    else:
        replay(args.output.resolve(), args.bootstrap, args.series)


if __name__ == '__main__':
    main()
