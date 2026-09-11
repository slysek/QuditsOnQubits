"""Notebook support for an opt-in, two-qutrit IQM mitigation comparison.

Uses the audited setting sampler and raw block estimator. This is a separate
experimental campaign, not an extension of the raw-only RandomizedBlocks runner.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import math

import numpy as np
from qiskit import QuantumCircuit, qpy
from qiskit.circuit.library import RGate
from qiskit.quantum_info import Statevector

from qudits_on_qubits.benchmarks.direct_basis.math_utils import qutrit_cz, qutrit_fourier
from qudits_on_qubits.benchmarks.direct_basis.circuit_serialization import normalize_circuit_bit_indices
from qudits_on_qubits.benchmarks.direct_basis.optimized_gates import validate_code_space_gate
from qudits_on_qubits.bell_measurements.postprocessing import compute_bell_value_from_counts
from qudits_on_qubits.bell_measurements.sampler_circuits import decoding_kwargs_from_metadata
from qudits_on_qubits.experiments.artifacts import BasisArtifacts
from qudits_on_qubits.experiments.block_estimation import evaluate_blocks
from qudits_on_qubits.experiments.mitigation.zne import fold_cz_batch, linear_zne_extrapolate
from qudits_on_qubits.experiments.preparation import prepare_measurements
from qudits_on_qubits.reference_experiments import get_reference_experiment


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def load_state(directory):
    """Validate exact source files; never synthesize or silently substitute gates."""
    directory = Path(directory)
    e = np.load(directory / 'E.npy', allow_pickle=False)
    np.testing.assert_allclose(e, np.eye(4, 3), atol=1e-12, rtol=0)
    gates, evidence = {}, {}
    for name, embedding, target, tolerance in (
        ('F3', e, qutrit_fourier(), 1e-10),
        ('CZ3', np.kron(e, e), qutrit_cz(), 1e-5),
    ):
        path = directory / f'{name}_W.qpy'
        with path.open('rb') as handle:
            loaded = qpy.load(handle)
        if len(loaded) != 1:
            raise ValueError(f'{name}: expected exactly one circuit')
        gate = loaded[0]
        metrics = validate_code_space_gate(gate, embedding, target)
        allowed = {'u', 'u3', 'cx'} if name == 'F3' else {'u', 'u3', 'cz'}
        if (not set(gate.count_ops()) <= allowed or gate.num_clbits
            or not np.isfinite([metrics.E_norm, metrics.L_norm]).all()
            or max(metrics.E_norm, metrics.L_norm) > tolerance
            or (gate.metadata or {}).get('gate_library') != 'code_space_optimized_v1'
            or (name == 'F3' and metrics.N_2q > 2)):
            raise ValueError(f'{name}: invalid optimized gate: {metrics}')
        gates[name] = gate
        evidence[name] = {**asdict(metrics), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    circuit = QuantumCircuit(4)
    circuit.compose(gates['F3'], [0, 1], inplace=True)
    circuit.compose(gates['F3'], [2, 3], inplace=True)
    circuit.compose(gates['CZ3'], range(4), inplace=True)
    reference = get_reference_experiment('two_qutrit')
    expected = np.kron(e, e) @ reference.state.statevector()
    fidelity = abs(np.vdot(expected, Statevector(circuit).data)) ** 2
    if fidelity < 1 - 1e-9:
        raise ValueError('Gate-built state differs from the Bell reference')
    artifacts = BasisArtifacts(directory, 'two_qutrit', circuit, e, {}, evidence)
    return artifacts, {**evidence, 'state_fidelity': float(fidelity)}


def measurement_mapping(circuit):
    """Read native ISA wire indices in classical-bit order."""
    mapping = {}
    for item in circuit.data:
        if item.operation.name == 'measure':
            bit = circuit.find_bit(item.clbits[0]).index
            if bit in mapping:
                raise ValueError('Repeated classical measurement')
            mapping[bit] = circuit.find_bit(item.qubits[0]).index
    if set(mapping) != set(range(circuit.num_clbits)) or circuit.num_clbits != 4:
        raise ValueError('Expected exactly four final measurements')
    result = tuple(mapping[i] for i in range(4))
    if len(set(result)) != 4:
        raise ValueError('Measurement qubits must be distinct')
    return result


def physical_isa(circuit):
    """Materialize exactly the physical loci selected by IQM's serializer.

    IQM sometimes stores physical indices in initial_layout rather than wire
    positions. A layout-free output prevents later copies from reinterpreting it.
    """
    circuit = normalize_circuit_bit_indices(circuit)
    layout = circuit.layout
    if layout is not None and layout.initial_layout.get_registers() == set(circuit.qregs):
        indices = layout.initial_layout.get_virtual_bits()
    else:
        indices = {bit: circuit.find_bit(bit).index for bit in circuit.qubits}
    result = QuantumCircuit(circuit.num_qubits, circuit.num_clbits,
                            name=circuit.name, global_phase=circuit.global_phase)
    for item in circuit.data:
        result.append(item.operation, [indices[q] for q in item.qubits],
                      [circuit.find_bit(c).index for c in item.clbits])
    return result


def _pauli(circuit, qubit, x, z):
    # Native equatorial pi rotations: Z uses X then Y, up to global phase.
    if x:
        circuit.append(RGate(math.pi, math.pi / 2 if z else 0), [qubit])
    elif z:
        circuit.append(RGate(math.pi, 0), [qubit])
        circuit.append(RGate(math.pi, math.pi / 2), [qubit])


def twirl_cz(circuit, rng):
    """Independent Pauli twirl around each native CZ, without recompilation."""
    output = circuit.copy_empty_like()
    for item in circuit.data:
        qubits = [circuit.find_bit(q).index for q in item.qubits]
        bits = [circuit.find_bit(c).index for c in item.clbits]
        if item.operation.name == 'cz':
            a, b, c, d = (int(v) for v in rng.integers(0, 2, 4))
            _pauli(output, qubits[0], a, b)
            _pauli(output, qubits[1], c, d)
            output.append(item.operation, qubits)
            _pauli(output, qubits[0], a, b ^ c)
            _pauli(output, qubits[1], c, d ^ a)
        else:
            output.append(item.operation, qubits, bits)
    return output


def prepare_campaign(artifacts, schedule, adapter, transpilation, instances, seed):
    """Compile once; then fold and twirl native gates, retaining block identity."""
    if type(instances) is not int or instances < 1 or schedule.config.shots_per_draw % instances:
        raise ValueError('shots_per_draw must be divisible by twirling instances')
    if schedule.state != 'two_qutrit' or not schedule.complete:
        raise ValueError('A complete two_qutrit schedule is required')
    prepared = prepare_measurements(artifacts, schedule)
    compiled = tuple(physical_isa(circuit)
                     for circuit in adapter.compile(prepared.circuits, transpilation).circuits)
    if len(compiled) != len(prepared.circuits):
        raise ValueError('Compilation changed the circuit count')
    if any(set(q.count_ops()) - {'r', 'cz', 'measure', 'barrier'} for q in compiled):
        raise ValueError('Campaign requires native r/CZ IQM circuits (Garnet)')
    mappings = [measurement_mapping(q) for q in compiled]
    catalog = prepared.metadata['catalog_index_by_block_id']
    rng = np.random.default_rng(seed)
    batches = []
    # Factor one of the suppressed run is reused in the ZNE fit.
    for arm, factor in [('RAW', 1), ('MM_TWIRL_DD', 1), ('MM_TWIRL_DD', 3), ('MM_TWIRL_DD', 5)]:
        native = fold_cz_batch(compiled, factor)
        copies = 1 if arm == 'RAW' else instances
        circuits, records = [], []
        for block in schedule.blocks:
            index = catalog[block.block_id]
            for instance in range(copies):
                circuit = native[index].copy() if arm == 'RAW' else twirl_cz(native[index], rng)
                if measurement_mapping(circuit) != mappings[index]:
                    raise ValueError('Transformation changed measurement mapping')
                if circuit.count_ops().get('cz', 0) != factor * compiled[index].count_ops().get('cz', 0):
                    raise ValueError('CZ folding was changed')
                circuits.append(circuit)
                records.append({'block_id': block.block_id, 'instance': instance,
                                'mapping': list(mappings[index])})
        batches.append({'arm': arm, 'factor': factor, 'circuits': circuits,
                        'records': records, 'shots': schedule.config.shots_per_draw // copies})
    # Full 16-state assignment matrix for each measured physical tuple.
    calibrations = {}
    for mapping in sorted(set(mappings)):
        circuits = []
        for basis in range(16):
            circuit = QuantumCircuit(compiled[0].num_qubits, 4)
            for bit, physical in enumerate(mapping):
                if basis & (1 << bit):
                    circuit.append(RGate(math.pi, 0), [physical])
                circuit.measure(physical, bit)
            circuits.append(circuit)
        calibrations[mapping] = circuits
    return prepared, batches, calibrations


def count_vector(counts, shots):
    result = np.zeros(16)
    for key, value in counts.items():
        if (not isinstance(key, str) or len(key) != 4 or set(key) - {'0', '1'}
            or isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0):
            raise ValueError('Invalid four-bit counts')
        result[int(key, 2)] += value
    if result.sum() != shots or shots <= 0:
        raise ValueError('Counts do not match requested shots')
    return result / shots


def assignment_matrix(counts, shots):
    if len(counts) != 16:
        raise ValueError('Expected all 16 calibration preparations')
    matrix = np.column_stack([count_vector(c, shots) for c in counts])
    if not np.isfinite(np.linalg.cond(matrix)) or np.linalg.cond(matrix) > 100:
        raise ValueError('Readout assignment matrix is ill-conditioned; recalibrate')
    return matrix


def execute_campaign(directory, schedule, evidence, batches, calibrations, backend,
                     *, allow_hardware=False, calibration_shots=4096, max_circuits=100):
    """Explicit opt-in only. Save requests before submit and job IDs before wait.

    Existing directories are never resubmitted. Interrupted jobs are retrieved
    by their saved job IDs using the notebook recovery cell.
    """
    if allow_hardware is not True:
        raise PermissionError('Hardware execution is disabled')
    for value in (calibration_shots, max_circuits):
        if type(value) is not int or value < 1:
            raise ValueError('Shot and batch limits must be positive integers')
    from iqm.iqm_client import CircuitCompilationOptions, DDMode, STANDARD_DD_STRATEGY
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / 'schedule.json', schedule.to_safe_dict())
    write_json(directory / 'gates.json', evidence)
    write_json(directory / 'backend.json', {
        'name': str(backend.name), 'calibration_set_id': str(getattr(backend, 'calibration_set_id', '')),
    })
    requests = []
    for index, (mapping, circuits) in enumerate(calibrations.items()):
        requests.append({'name': f'calibration_{index}', 'circuits': circuits,
                         'shots': calibration_shots, 'dd': False, 'mapping': list(mapping)})
    for batch in batches:
        for start in range(0, len(batch['circuits']), max_circuits):
            requests.append({'name': f"{batch['arm']}_{batch['factor']}_{start}",
                             'circuits': batch['circuits'][start:start + max_circuits],
                             'shots': batch['shots'], 'dd': batch['arm'] != 'RAW',
                             'arm': batch['arm'], 'factor': batch['factor'],
                             'records': batch['records'][start:start + max_circuits]})
    # All artifacts are frozen before the first external write.
    for request in requests:
        name = request['name']
        with (directory / f'{name}.qpy').open('wb') as handle:
            qpy.dump(request['circuits'], handle)
        document = {k: v for k, v in request.items() if k != 'circuits'}
        document['qpy_sha256'] = hashlib.sha256((directory / f'{name}.qpy').read_bytes()).hexdigest()
        write_json(directory / f'{name}.request.json', document)
    write_json(directory / 'index.json', [r['name'] for r in requests])
    for request in requests:
        name = request['name']
        options = CircuitCompilationOptions(
            dd_mode=DDMode.ENABLED if request['dd'] else DDMode.DISABLED,
            dd_strategy=STANDARD_DD_STRATEGY if request['dd'] else None,
        )
        write_json(directory / f'{name}.submission.json', {'status': 'submission_unknown'})
        job = backend.run(request['circuits'], shots=request['shots'], circuit_compilation_options=options)
        write_json(directory / f'{name}.submission.json', {'status': 'submitted', 'job_id': str(job.job_id())})
        result = job.result()
        counts = [result.get_counts(i) for i in range(len(request['circuits']))]
        for item in counts:
            count_vector(item, request['shots'])
        write_json(directory / f'{name}.counts.json', counts)
    return directory


def analyze_campaign(directory, schedule, metadata):
    """Offline analysis; preserve signed corrected weights, no clipping."""
    directory = Path(directory)
    if json.loads((directory / 'schedule.json').read_text()) != schedule.to_safe_dict():
        raise ValueError('Analysis schedule does not match the saved campaign')
    requests = []
    for name in json.loads((directory / 'index.json').read_text()):
        request = json.loads((directory / f'{name}.request.json').read_text())
        if hashlib.sha256((directory / f'{name}.qpy').read_bytes()).hexdigest() != request['qpy_sha256']:
            raise ValueError('Circuit checkpoint hash mismatch')
        counts = json.loads((directory / f'{name}.counts.json').read_text())
        requests.append((request, counts))
    matrices = {tuple(r['mapping']): assignment_matrix(c, r['shots'])
                for r, c in requests if 'mapping' in r}
    grouped = {}
    for request, counts in requests:
        if 'arm' not in request:
            continue
        key = (request['arm'], request['factor'])
        blocks = grouped.setdefault(key, {})
        for record, count in zip(request['records'], counts, strict=True):
            probability = count_vector(count, request['shots'])
            corrected = np.linalg.solve(matrices[tuple(record['mapping'])], probability)
            raw, quasi = blocks.setdefault(record['block_id'], (Counter(), np.zeros(16)))
            raw.update(count)
            quasi += corrected * request['shots']
    rows = []
    raw_report = None
    for (arm, factor), blocks in grouped.items():
        if set(blocks) != {b.block_id for b in schedule.blocks}:
            raise ValueError('Incomplete block evidence')
        if any(sum(raw.values()) != schedule.config.shots_per_draw for raw, _ in blocks.values()):
            raise ValueError('Incomplete twirling evidence')
        if arm == 'RAW':
            raw_report = evaluate_blocks(get_reference_experiment('two_qutrit'), schedule,
                {k: v[0] for k, v in blocks.items()}, metadata['qutrit_bit_indices_by_setting'],
                metadata['encoding_outcome_map'])
        for corrected in ([False, True] if arm == 'RAW' else [True]):
            pooled = {}
            for block in schedule.blocks:
                raw, quasi = blocks[block.block_id]
                values = {format(i, '04b'): float(p) for i, p in enumerate(quasi)} if corrected else raw
                bucket = pooled.setdefault(block.settings, Counter())
                # Counter.update retains negative quasi-counts (Counter + drops them).
                bucket.update(values)
            value = compute_bell_value_from_counts(pooled, metadata['terms'],
                metadata['qutrit_bit_indices_by_setting'], discard_leakage=True,
                renormalize_after_discard=False, **decoding_kwargs_from_metadata(metadata))
            rows.append({'variant': 'MM' if arm == 'RAW' and corrected else arm,
                         'factor': factor, 'real': value.real, 'imag': value.imag})
    scaled = sorted((r for r in rows if r['variant'] == 'MM_TWIRL_DD'), key=lambda r: r['factor'])
    intercept, fit = linear_zne_extrapolate([r['factor'] for r in scaled],
        [complex(r['real'], r['imag']) for r in scaled])
    rows.append({'variant': 'MM_TWIRL_DD_ZNE', 'factor': 0, 'real': intercept.real, 'imag': intercept.imag})
    result = {'rows': rows, 'raw_block_analysis': raw_report, 'zne_fit': fit.to_safe_dict(),
              'mitigated_uncertainty': None, 'estimator': 'unconditional_zero_for_leakage'}
    write_json(directory / 'analysis.json', result)
    return result
