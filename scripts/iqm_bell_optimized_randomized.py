"""Use the existing randomized IQM campaign with frozen optimized Bell parents."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from uuid import UUID

import numpy as np
from qiskit import qpy

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT/'src'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import iqm_bell_optimized as sequential
from scripts import iqm_bell_short as short
from scripts import iqm_randomized_bell_campaign as campaign
from qudits_on_qubits.bell_measurements.bell_optimized import SETTINGS
from qudits_on_qubits.experiments import RandomizedBlocks
from qudits_on_qubits.experiments.preparation import prepare_measurements
from qudits_on_qubits.experiments.setting_schedule import SettingSchedule, generate_schedule
from qudits_on_qubits.reference_experiments import get_reference_experiment

GATES = ROOT/'experiment_inputs/iqm_randomized_bell/canonical_optimized_20260909'
PARENT = ROOT/'artifacts/bell_optimized/iqm_emerald_sequential_20260914'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_batches(parents, weights, schedule, *, instances=8, seed=20260914):
    """Map by explicit labels; prove agreement with the existing Bell decoder."""
    if len(parents) != 9 or [c.metadata.get('setting_index') for c in parents] != list(range(9)):
        raise ValueError('Expected nine optimized parents in setting order')
    artifacts, gate_evidence = campaign.load_state(GATES)
    prepared = prepare_measurements(artifacts, schedule)
    settings = prepared.metadata['setting_by_circuit_index']
    ordered = tuple(parents[SETTINGS.index(tuple(s))] for s in settings)
    errors = [float(np.max(np.abs(short.probabilities(a)-short.probabilities(b))))
              for a, b in zip(ordered, prepared.circuits, strict=True)]
    if max(errors) > 1e-6:
        raise ValueError('Optimized parent histogram differs from randomized pipeline')
    reference_weights = np.empty((9, 16))
    for index, setting in enumerate(SETTINGS):
        for word in range(16):
            reference_weights[index, word] = campaign.compute_bell_value_from_counts(
                {s: {format(word, '04b') if s == setting else '1111': 1} for s in SETTINGS},
                prepared.metadata['terms'],
                prepared.metadata['qutrit_bit_indices_by_setting'], discard_leakage=True,
                renormalize_after_discard=False,
                **campaign.decoding_kwargs_from_metadata(prepared.metadata)).real
    if np.shape(weights) != (9, 16) or not np.allclose(weights, reference_weights, atol=1e-12, rtol=0):
        raise ValueError('Bell weights differ from the randomized decoder')
    prepared, batches, calibrations = campaign.prepare_campaign(
        artifacts, schedule, None, None, instances, seed,
        native_circuits=ordered, readout_mitigation=False)
    evidence = dict(metadata_gates=gate_evidence, max_ideal_probability_error=max(errors),
                    ideal_bell=float(sum(w @ short.probabilities(c) for w, c in zip(weights, parents))))
    return prepared, batches, calibrations, evidence


def prepare(output, parent=PARENT, *, draws=50, shots=1000, instances=8, seed=20260914):
    """One production schedule, saved even if coverage is incomplete. No provider call."""
    output, parent = Path(output), Path(parent)
    if output.exists():
        raise FileExistsError('Choose a new preparation directory; never redraw an existing schedule')
    config = RandomizedBlocks(draws, shots, draws)
    if type(instances) is not int or instances < 1 or shots % instances:
        raise ValueError('shots_per_draw must be divisible by twirling instances')
    parents, weights, original = sequential.load_prepared(parent)
    schedule = generate_schedule(get_reference_experiment('two_qutrit'), config)
    output.mkdir(parents=True, exist_ok=False)
    campaign.write_json(output/'schedule.json', schedule.to_safe_dict())
    if not schedule.complete:
        raise ValueError('Saved schedule has incomplete coverage; no hardware submitted. Extend the budget explicitly.')
    prepared, batches, calibrations, evidence = prepare_batches(
        parents, weights, schedule, instances=instances, seed=seed)
    max_error = 0.
    for batch in batches:
        for circuit, record in zip(batch['circuits'], batch['records'], strict=True):
            original_circuit = parents[SETTINGS.index(schedule.blocks[record['block_id']].settings)]
            max_error = max(max_error, float(np.max(np.abs(
                short.probabilities(circuit) - short.probabilities(original_circuit)))))
    if max_error > 1e-10:
        raise ValueError('Folding/twirling changed ideal probabilities')
    evidence['max_transformed_probability_error'] = max_error
    campaign.write_json(output/'evidence.json', evidence)
    with (output/'parents.qpy').open('wb') as handle:
        qpy.dump(parents, handle)
    np.save(output/'bell_weights.npy', weights, allow_pickle=False)
    campaign.write_json(output/'parent_protocol.json', original)
    batch_documents = []
    for index, batch in enumerate(batches):
        name = f'batch_{index:02d}.qpy'
        with (output/name).open('wb') as handle:
            qpy.dump(batch['circuits'], handle)
        batch_documents.append({**{k:v for k,v in batch.items() if k != 'circuits'}, 'file':name})
    campaign.write_json(output/'batches.json', batch_documents)
    protocol = dict(schema_version=1, parent_directory=str(parent.resolve()),
        selection=original['selection'], draws=len(schedule.blocks), shots_per_draw=shots,
        instances=instances, twirling_seed=seed, zne_factors=[1,3,5], readout_mitigation=False,
        total_shots=sum(len(b['circuits'])*b['shots'] for b in batches),
        setting_frequencies={','.join(s):n for s,n in Counter(b.settings for b in schedule.blocks).items()},
        versions=original['versions'],
        input_sha256={p.name:sha256(p) for p in output.iterdir() if p.is_file()})
    campaign.write_json(output/'protocol.json', protocol)
    return protocol


def load_prepared(output):
    output = Path(output)
    protocol = read_json(output/'protocol.json')
    expected_files = {'schedule.json', 'evidence.json', 'parents.qpy', 'bell_weights.npy',
                      'parent_protocol.json', 'batches.json', *[f'batch_{i:02d}.qpy' for i in range(4)]}
    if protocol.get('schema_version') != 1 or set(protocol.get('input_sha256', {})) != expected_files:
        raise ValueError('Invalid frozen input manifest')
    for name, digest in protocol['input_sha256'].items():
        if sha256(output/name) != digest:
            raise ValueError(f'Frozen input hash mismatch: {name}')
    schedule = SettingSchedule.from_safe_dict(read_json(output/'schedule.json'))
    if not schedule.complete or schedule.source != 'system_secrets':
        raise ValueError('Expected a complete production schedule')
    original = read_json(output/'parent_protocol.json')
    if protocol['selection'] != original['selection']:
        raise ValueError('Frozen hardware selection differs from parent')
    batches = []
    for index, document in enumerate(read_json(output/'batches.json')):
        if document['file'] != f'batch_{index:02d}.qpy':
            raise ValueError('Unexpected frozen batch filename')
        with (output/document['file']).open('rb') as handle:
            circuits = qpy.load(handle)
        batches.append({**{k:v for k,v in document.items() if k != 'file'}, 'circuits':circuits})
    if (len(batches) != 4 or [(b['arm'], b['factor']) for b in batches] !=
            [('RAW',1), ('TWIRL_DD',1), ('TWIRL_DD',3), ('TWIRL_DD',5)]
            or protocol['total_shots'] != 4*len(schedule.blocks)*schedule.config.shots_per_draw
            or protocol['total_shots'] != sum(len(b['circuits'])*b['shots'] for b in batches)):
        raise ValueError('Frozen budget or campaign arms differ')
    return protocol, schedule, batches


def verify_hardware(output, protocol, schedule, batches):
    """Tie every analyzed request and result to this frozen preparation."""
    output = Path(output)
    hardware = output/'hardware'
    backend = read_json(hardware/'backend.json')
    selection = protocol['selection']
    if backend != dict(name=selection['backend'], calibration_set_id=selection['calibration_set_id']):
        raise ValueError('Saved backend or calibration differs from preparation')
    expected_names = []
    for batch in batches:
        for start in range(0, len(batch['circuits']), 100):
            name = f"{batch['arm']}_{batch['factor']}_{start}"
            expected_names.append(name)
            request = read_json(hardware/f'{name}.request.json')
            expected = dict(name=name, shots=batch['shots'], dd=batch['arm'] != 'RAW',
                arm=batch['arm'], factor=batch['factor'], records=batch['records'][start:start+100],
                qpy_sha256=sha256(hardware/f'{name}.qpy'))
            if request != expected:
                raise ValueError('Hardware request differs from frozen batch')
            with (hardware/f'{name}.qpy').open('rb') as handle:
                circuits = qpy.load(handle)
            if [short._fingerprint(c) for c in circuits] != [
                    short._fingerprint(c) for c in batch['circuits'][start:start+100]]:
                raise ValueError('Hardware circuit differs from frozen batch')
            submission = read_json(hardware/f'{name}.submission.json')
            if submission.get('status') != 'submitted' or not submission.get('job_id'):
                raise ValueError('Missing known submission identity')
            if read_json(hardware/f'{name}.result.json') != dict(
                    job_id=submission['job_id'], calibration_set_id=selection['calibration_set_id'],
                    circuit_count=len(circuits)):
                raise ValueError('Result identity or calibration differs from preparation')
    if read_json(hardware/'index.json') != expected_names:
        raise ValueError('Hardware job index differs from preparation')
    if read_json(hardware/'schedule.json') != schedule.to_safe_dict():
        raise ValueError('Hardware schedule differs from preparation')


def execute(output, backend, *, allow_hardware=False):
    if allow_hardware is not True:
        raise PermissionError('Hardware execution requires explicit opt-in')
    output = Path(output)
    protocol, schedule, batches = load_prepared(output)
    selection = protocol['selection']
    if (backend.name != selection['backend'] or
            [backend.index_to_qubit_name(q) for q in selection['layout']] != selection['qubit_names_logical_order']):
        raise ValueError('Backend or physical qubits differ from frozen selection')
    return campaign.execute_campaign(output/'hardware', schedule, read_json(output/'evidence.json'),
        batches, {}, backend, allow_hardware=True,
        expected_calibration_set_id=selection['calibration_set_id'])


def finish(output):
    output = Path(output)
    protocol, schedule, batches = load_prepared(output)
    verify_hardware(output, protocol, schedule, batches)
    artifacts, evidence = campaign.load_state(GATES)
    if evidence != read_json(output/'evidence.json')['metadata_gates']:
        raise ValueError('Decoder source gates changed since preparation')
    metadata = prepare_measurements(artifacts, schedule).metadata
    result = campaign.analyze_campaign(output/'hardware', schedule, metadata)
    bound = float(get_reference_experiment('two_qutrit').bell_functional.classical_bound)
    result['classical_bound'] = bound
    for row in result['rows']:
        row['delta_from_classical'] = row['real'] - bound
        row['point_above_classical'] = row['real'] > bound
    result['protocol'] = protocol
    campaign.write_json(output/'analysis.json', result)
    lines = ['# IQM: losowany eksperyment Bella, obwody zoptymalizowane', '',
        f"Backend: {protocol['selection']['backend']}; layout: {protocol['selection']['layout']}.",
        f"Kalibracja: {protocol['selection']['calibration_set_id']}.",
        f"{protocol['draws']} losowań × {protocol['shots_per_draw']} shotów w każdym wariancie; razem {protocol['total_shots']} shotów.",
        f"Twirling: {protocol['instances']} realizacji na blok. ZNE: skale 1, 3, 5, liniowy fit.",
        f'Granica klasyczna: {bound:.9f}. Wynik idealny: 6.', '',
        '| Wariant | Skala | Bell | Różnica od granicy klasycznej |',
        '|---|---:|---:|---:|']
    for row in result['rows']:
        lines.append(f"| {row['variant']} | {row['factor']} | {row['real']:.9f} | {row['delta_from_classical']:+.9f} |")
    lines += ['', 'Losowanie: istniejący generate_schedule, niezależny uniform A/B, system_secrets. Ten sam zapisany harmonogram we wszystkich wariantach.',
        'Estymator: suma korelatorów uśrednionych w pasujących blokach. Kod 11 ma wagę zero i pozostaje w mianowniku. Bez postselekcji ani korekcji odczytu.',
        'Przedziały Hoeffdinga dla RAW: analysis.json → raw_block_analysis → uncertainty. ZNE: estymata po ekstrapolacji, bez wyznaczonej niepewności; samo przekroczenie granicy nie stanowi potwierdzenia naruszenia.',
        'DD zamówione u IQM w każdej skali twirlingu, STANDARD_DD_STRATEGY. Warianty wykonywane seriami, więc porównanie może obejmować dryf.', '',
        '## Liczba losowań każdej kombinacji', '']
    lines += [f'- {setting}: {count}' for setting, count in sorted(protocol['setting_frequencies'].items())]
    (output/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare','execute','analyze'])
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--parent', type=Path, default=PARENT)
    parser.add_argument('--draws', type=int, default=50)
    parser.add_argument('--shots', type=int, default=1000)
    parser.add_argument('--instances', type=int, default=8)
    args = parser.parse_args(argv)
    if args.action == 'prepare':
        result = prepare(args.output, args.parent, draws=args.draws, shots=args.shots, instances=args.instances)
    elif args.action == 'execute':
        protocol, _, _ = load_prepared(args.output)
        from qudits_on_qubits.benchmarks.direct_basis.iqm_backend import load_iqm_environment
        from iqm.qiskit_iqm import IQMProvider
        env = load_iqm_environment()
        provider = IQMProvider(env.server_url, quantum_computer=protocol['selection']['backend'])
        backend = provider.get_backend(use_metrics=False,
            calibration_set_id=UUID(protocol['selection']['calibration_set_id']))
        execute(args.output, backend, allow_hardware=True)
        result = finish(args.output)
    else:
        result = finish(args.output)
    print(json.dumps(result.get('rows', result), indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
