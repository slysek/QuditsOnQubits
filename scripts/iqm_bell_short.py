from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math

import numpy as np
from qiskit import QuantumCircuit, qpy, transpile
from qiskit.quantum_info import Statevector
from qiskit.transpiler import CouplingMap, PassManager
from scipy.stats import t

from scripts.iqm_randomized_bell_campaign import count_vector, load_state, twirl_cz, write_json
from qudits_on_qubits.bell_measurements.postprocessing import compute_bell_value_from_counts
from qudits_on_qubits.bell_measurements.sampler_circuits import decoding_kwargs_from_metadata
from qudits_on_qubits.experiments.artifacts import BasisArtifacts
from qudits_on_qubits.experiments.preparation import prepare_measurements
from qudits_on_qubits.reference_experiments import get_reference_experiment


def probabilities(circuit):
    active = sorted({circuit.find_bit(q).index for item in circuit.data for q in item.qubits})
    positions = {q: i for i, q in enumerate(active)}
    small = QuantumCircuit(len(active))
    measurements = {}
    for item in circuit.data:
        qubits = [positions[circuit.find_bit(q).index] for q in item.qubits]
        if item.operation.name == 'measure':
            measurements[circuit.find_bit(item.clbits[0]).index] = qubits[0]
        elif item.operation.name != 'barrier':
            small.append(item.operation, qubits)
    if set(measurements) != set(range(4)):
        raise ValueError('Expected four measured classical bits')
    result = np.zeros(16)
    for basis, probability in enumerate(Statevector(small).probabilities()):
        result[sum(((basis >> measurements[bit]) & 1) << bit for bit in range(4))] += probability
    return result


def optimize_rotations(circuit):
    from iqm.qiskit_iqm.iqm_transpilation import IQMOptimizeSingleQubitGates
    return PassManager([IQMOptimizeSingleQubitGates()]).run(circuit)


def build_catalog(root):
    directory = Path(root) / 'experiment_inputs/iqm_randomized_bell/canonical_optimized_20260909'
    original, _ = load_state(directory)
    left, singular, right = np.linalg.svd(np.array([[1, 1], [1, 0]]) / np.sqrt(3))
    state = QuantumCircuit(4)
    state.ry(2 * np.arctan2(singular[1], singular[0]), 0)
    state.cx(0, 1)
    state.unitary(left, [1])
    state.unitary(right.T, [0])
    state.cx(0, 2)
    state.cx(1, 3)
    with (directory / 'F3_W.qpy').open('rb') as handle:
        state.compose(qpy.load(handle)[0], [2, 3], inplace=True)
    state = transpile(state, basis_gates=['u', 'cx'], optimization_level=0)
    reference = get_reference_experiment('two_qutrit')
    expected = np.kron(original.encoding, original.encoding) @ reference.state.statevector()
    fidelity = float(abs(np.vdot(expected, Statevector(state).data)) ** 2)
    if fidelity < 1 - 1e-10:
        raise ValueError('Short preparation differs from the reference state')
    artifacts = BasisArtifacts(directory, original.state, state, original.encoding, {}, {})
    prepared = prepare_measurements(artifacts)
    metadata = prepared.metadata
    edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
    coupling = CouplingMap(edges + [(b, a) for a, b in edges])
    native = transpile(list(prepared.circuits), basis_gates=['r', 'cz'], coupling_map=coupling,
                       initial_layout=[0, 1, 2, 3], optimization_level=3, seed_transpiler=9)
    native = [optimize_rotations(c) for c in native]
    settings = metadata['setting_by_circuit_index']
    weights = []
    for setting in settings:
        terms = [term for term in metadata['terms'] if tuple(term['settings']) == tuple(setting)]
        weights.append([compute_bell_value_from_counts(
            {tuple(setting): {format(i, '04b'): 1}}, terms,
            metadata['qutrit_bit_indices_by_setting'], renormalize_after_discard=False,
            **decoding_kwargs_from_metadata(metadata)).real for i in range(16)])
    error = max(float(np.max(abs(probabilities(c) - probabilities(p))))
                for c, p in zip(native, prepared.circuits))
    if error > 1e-10 or any(c.count_ops().get('cz', 0) > 9 for c in native):
        raise ValueError('Native compilation failed the short-circuit check')
    return native, np.asarray(weights), dict(state_cx=state.count_ops()['cx'], state_fidelity=fidelity,
        max_probability_error=error, settings=[list(s) for s in settings],
        ideal_bell=float(sum(w @ probabilities(c) for w, c in zip(weights, native))),
        classical_bound=reference.bell_functional.classical_bound)


def select_layout(circuits, backend):
    target = backend.target
    required = Counter((item.operation.name, tuple(c.find_bit(q).index for q in item.qubits))
                       for c in circuits for item in c.data if item.operation.name != 'barrier')
    links = {tuple(sorted(wires)) for (gate, wires) in required if gate == 'cz'}
    adjacent = {q: set() for q in range(backend.num_qubits)}
    for a, b in target['cz']:
        adjacent[a].add(b)
        adjacent[b].add(a)

    def properties(gate, wires):
        value = target[gate].get(wires)
        if value is None and gate == 'cz':
            value = target[gate].get(wires[::-1])
        if value is None or value.error is None or not math.isfinite(value.error) or not 0 <= value.error < 1:
            raise ValueError('Missing valid calibration metrics')
        return value

    candidates = []
    def visit(layout):
        if len(layout) == 4:
            try:
                score = sum(n * -math.log1p(-properties(gate, tuple(layout[q] for q in wires)).error)
                            for (gate, wires), n in required.items()) / len(circuits)
            except ValueError:
                return
            candidates.append((score, tuple(layout)))
            return
        logical = len(layout)
        choices = set(range(backend.num_qubits)) - set(layout)
        for previous in range(logical):
            if tuple(sorted((previous, logical))) in links:
                choices &= adjacent[layout[previous]]
        for physical in sorted(choices):
            visit([*layout, physical])
    visit([])
    if not candidates:
        raise ValueError('No connected four-qubit layout with complete calibration metrics')
    score, layout = min(candidates)
    mapped = []
    for c in circuits:
        output = QuantumCircuit(backend.num_qubits, 4)
        for item in c.data:
            wires = tuple(layout[c.find_bit(q).index] for q in item.qubits)
            if item.operation.name == 'cz' and wires not in target['cz']:
                wires = wires[::-1]
            output.append(item.operation, wires, [c.find_bit(b).index for b in item.clbits])
        np.testing.assert_allclose(probabilities(output), probabilities(c), atol=1e-12, rtol=0)
        mapped.append(output)
    calibration = str(getattr(getattr(backend, 'architecture', None), 'calibration_set_id', ''))
    if not calibration:
        raise ValueError('Missing calibration_set_id')
    metrics = []
    for gate, wires in required:
        physical = tuple(layout[q] for q in wires)
        prop = properties(gate, physical)
        metrics.append(dict(gate=gate, qubits=list(physical), error=float(prop.error),
                            duration_seconds=None if prop.duration is None else float(prop.duration)))
    return mapped, dict(layout=list(layout), score=score, candidates=len(candidates),
                        calibration_set_id=calibration, backend=backend.name, metrics=metrics)


def folding_block_size(scales):
    if (len(scales) < 3 or tuple(sorted(set(scales))) != tuple(scales)
        or scales[0] != 1 or scales[-1] != 3
        or any(s not in (1, 1.5, 2, 2.5, 3) for s in scales)):
        raise ValueError('Use increasing half-integer scales from 1 to 3')
    return 4 if any(s % 1 for s in scales) else 2


def make_plan(circuits, repeats=16, shots=256, seed=20260910, *, scales=(1, 2, 3)):
    block_size = folding_block_size(scales)
    if type(repeats) is not int or repeats < 2 * block_size or repeats % block_size or type(shots) is not int or shots < 1:
        raise ValueError('Use complete repeat blocks and positive integer shots')
    rng = np.random.default_rng(seed)
    plan = []
    ideal = [probabilities(c) for c in circuits]
    for round_index in range(repeats // block_size):
        arms = rng.permutation(['RAW', 'DD', 'TWIRLING'])
        for arm in arms:
            batch, records = [], []
            for repeat in range(block_size * round_index, block_size * (round_index + 1)):
                for setting, circuit in enumerate(circuits):
                    ncz = circuit.count_ops()['cz']
                    for scale in ([1] if arm == 'RAW' else scales):
                        target = (scale - 1) * ncz / 2
                        position = repeat % block_size
                        extra = math.floor((position + 1) * target) - math.floor(position * target)
                        folded = set(rng.choice(ncz, size=extra, replace=False).tolist())
                        output = circuit.copy_empty_like()
                        index = 0
                        for item in circuit.data:
                            times = 3 if item.operation.name == 'cz' and index in folded else 1
                            for _ in range(times):
                                output.append(item.operation, item.qubits, item.clbits)
                            index += item.operation.name == 'cz'
                        if arm == 'TWIRLING':
                            output = optimize_rotations(twirl_cz(output, rng))
                        actual = output.count_ops()['cz'] / ncz
                        if output.count_ops()['cz'] != ncz + 2 * extra:
                            raise ValueError('Optimization changed CZ folding')
                        np.testing.assert_allclose(probabilities(output), ideal[setting], atol=1e-10, rtol=0)
                        batch.append(output)
                        records.append(dict(repeat=repeat, setting=setting, scale=scale, actual_scale=actual))
            order = rng.permutation(len(batch))
            for start in range(0, len(order), 100):
                indices = order[start:start + 100]
                plan.append(dict(arm=str(arm), dd=arm == 'DD', shots=shots,
                                 circuits=[batch[i] for i in indices], records=[records[i] for i in indices]))
    return plan


def _fingerprint(circuit):
    data = [(i.operation.name, [float(p) for p in i.operation.params],
             [circuit.find_bit(q).index for q in i.qubits], [circuit.find_bit(c).index for c in i.clbits])
            for i in circuit.data]
    return hashlib.sha256(json.dumps(data).encode()).hexdigest()


def execute(directory, plan, backend, selection, *, allow_hardware=False):
    if allow_hardware is not True:
        raise PermissionError('Hardware execution requires explicit opt-in')
    from iqm.iqm_client import CircuitCompilationOptions, DDMode, DDStrategy
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = dict(selection=selection, jobs=[dict(arm=j['arm'], dd=bool(j['dd']), shots=j['shots'],
        records=j['records'], circuit_hashes=[_fingerprint(c) for c in j['circuits']]) for j in plan])
    path = directory / 'manifest.json'
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError('Saved campaign differs from this plan; select a new directory')
    write_json(path, manifest)
    for index, job in enumerate(plan):
        name = f'job_{index:03d}'
        output = directory / f'{name}.counts.json'
        if output.exists():
            continue
        marker = directory / f'{name}.submission.json'
        if marker.exists():
            saved = json.loads(marker.read_text())
            if not saved.get('job_id'):
                raise RuntimeError(f'{name}: submission_unknown; inspect the IQM dashboard')
            pending = backend.retrieve_job(saved['job_id'])
        else:
            with (directory / f'{name}.qpy').open('wb') as handle:
                qpy.dump(job['circuits'], handle)
            options = CircuitCompilationOptions(dd_mode=DDMode.ENABLED if job['dd'] else DDMode.DISABLED,
                dd_strategy=DDStrategy(gate_sequences=[(2, 'XX', 'center')]) if job['dd'] else None)
            write_json(marker, dict(status='submission_unknown', started_at=datetime.now(timezone.utc).isoformat()))
            pending = backend.run(job['circuits'], shots=job['shots'], circuit_compilation_options=options)
            write_json(marker, dict(status='submitted', job_id=str(pending.job_id())))
        print(f'{index + 1}/{len(plan)} {job["arm"]} {len(job["circuits"])} circuits', flush=True)
        result = pending.result()
        counts = [result.get_counts(i) for i in range(len(job['circuits']))]
        for count in counts:
            count_vector(count, job['shots'])
        calibrations = {str(getattr(r, 'calibration_set_id', '')) for r in result.results}
        if calibrations != {selection['calibration_set_id']}:
            raise ValueError(f'Execution calibration differs from selection: {calibrations}')
        write_json(output, dict(counts=counts, calibration_set_id=selection['calibration_set_id'],
                                finished_at=datetime.now(timezone.utc).isoformat()))
    return read_samples(directory)


def read_samples(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    result = []
    for index, job in enumerate(manifest['jobs']):
        output = directory / f'job_{index:03d}.counts.json'
        if not output.exists():
            raise ValueError(f'Incomplete campaign: {output.name}')
        data = json.loads(output.read_text())
        if data['calibration_set_id'] != manifest['selection']['calibration_set_id']:
            raise ValueError('Mixed calibration sets')
        for record, counts in zip(job['records'], data['counts'], strict=True):
            result.append(dict(**record, counts=counts, arm=job['arm'], shots=job['shots']))
    return result


def analyze(samples, weights, *, scales=(1, 2, 3)):
    block_size = folding_block_size(scales)
    keys = [('RAW', 1)] + [(arm, scale) for arm in ('DD', 'TWIRLING') for scale in scales]
    count = len(scales)
    if not samples:
        raise ValueError('Incomplete experiment samples')
    repeats = max(s['repeat'] for s in samples) + 1
    if repeats < 2 * block_size or repeats % block_size:
        raise ValueError('Incomplete repeat blocks')
    values = np.full((len(keys), repeats, 9), np.nan)
    variances, leakage, actual_scales = (np.zeros_like(values) for _ in range(3))
    shots = {s['shots'] for s in samples}
    if len(shots) != 1:
        raise ValueError('Unequal shot counts')
    for sample in samples:
        arm = keys.index((sample['arm'], sample['scale']))
        repeat, setting = sample['repeat'], sample['setting']
        if not np.isnan(values[arm, repeat, setting]):
            raise ValueError('Duplicate sample')
        p = count_vector(sample['counts'], sample['shots'])
        w = weights[setting]
        mean = float(w @ p)
        values[arm, repeat, setting] = mean
        variances[arm, repeat, setting] = max(0., float(w * w @ p - mean * mean)) / sample['shots']
        leakage[arm, repeat, setting] = sum(p[i] for i in range(16) if i & 3 == 3 or (i >> 2) & 3 == 3)
        actual_scales[arm, repeat, setting] = sample['actual_scale']
    if not np.isfinite(values).all():
        raise ValueError('Incomplete experiment samples')
    for index, (_, scale) in enumerate(keys):
        if not np.allclose(actual_scales[index].reshape(repeats // block_size, block_size, 9).mean(axis=1), scale):
            raise ValueError('Unbalanced effective CZ scales')
    observations = values.sum(axis=2)
    def estimate(coefficients):
        per_repeat = np.asarray(coefficients) @ observations
        paired = per_repeat.reshape(-1, block_size).mean(axis=1)
        iid_variance = sum(c * c * v.sum() for c, v in zip(coefficients, variances)) / repeats ** 2
        error = math.sqrt(max(float(np.var(paired, ddof=1) / len(paired)), iid_variance))
        mean = float(np.mean(paired))
        radius = float(t.ppf(.975, len(paired) - 1)) * error
        return dict(bell=mean, standard_error=error, ci95_low=mean-radius, ci95_high=mean+radius)
    rows = []
    for label, base in [('RAW', 0), ('DD', 1), ('TWIRLING', 1 + count)]:
        coefficients = np.zeros(len(keys))
        coefficients[base] = 1
        rows.append(dict(variant=label, **estimate(coefficients), leakage=float(leakage[base].mean()),
                         shots=repeats * 9 * next(iter(shots))))
    fits = {}
    for arm, base in [('DD', 1), ('TWIRLING', 1 + count)]:
        coefficients = np.zeros(len(keys))
        coefficients[base:base+count] = np.linalg.pinv(np.polynomial.polynomial.polyvander(scales, 1))[0]
        rows.append(dict(variant=f'{arm} + ZNE', **estimate(coefficients), leakage=None,
                         shots=count * repeats * 9 * next(iter(shots))))
        coefficients[base:base+count] = 2 * np.linalg.pinv(np.polynomial.polynomial.polyvander(scales, 2))[2]
        curvature = estimate(coefficients)
        fits[arm] = dict(scales=list(scales), values=observations[base:base+count].mean(axis=1).tolist(),
            curvature=curvature['bell'], curvature_standard_error=curvature['standard_error'],
            linear_model_warning=not curvature['ci95_low'] <= 0 <= curvature['ci95_high'])
    return dict(rows=rows, fits=fits, uncertainty=f'95% Student-t intervals across interleaved blocks of {block_size} repeats with a shot-noise floor; excludes ZNE model bias and arbitrary drift',
                readout_mitigation=False, estimator='unconditional_zero_for_invalid_codewords')
