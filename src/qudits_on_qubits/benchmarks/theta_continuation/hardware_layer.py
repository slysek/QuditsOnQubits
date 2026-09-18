"""Add a frozen IQM topology layer to an existing theta campaign, without submission."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .artifacts import RunStore, atomic_write_json, fingerprint, runtime_provenance
from .encoding import theta_embedding
from .hardware_compilation import compile_case, compile_gate


PROFILES = {
    'two_qutrit': (27, 28, 19, 20),
    'ghz3': (27, 28, 19, 20, 35, 36),
    'ame43': (27, 28, 19, 20, 35, 36, 11, 12),
}


def validate_snapshot(snapshot: dict) -> dict:
    """Validate a local topology snapshot; never connect to a provider."""
    if not isinstance(snapshot, dict):
        raise ValueError('Snapshot must be a JSON object')
    result = deepcopy(snapshot)
    names, edges = result.get('qubit_names'), result.get('edges')
    if (result.get('schema_version') != 1 or result.get('backend') != 'emerald'
            or not isinstance(result.get('calibration_set_id'), str)
            or not result['calibration_set_id'].strip()
            or not isinstance(names, list) or len(names) < 4
            or any(not isinstance(name, str) or not name.strip() for name in names)
            or len(set(names)) != len(names)
            or result.get('native_gates') != ['r', 'cz', 'measure']
            or not isinstance(edges, list) or not edges):
        raise ValueError('Invalid Emerald topology snapshot')
    normalized = []
    for edge in edges:
        if (not isinstance(edge, list) or len(edge) != 2
                or any(type(q) is not int or not 0 <= q < len(names) for q in edge)
                or edge[0] == edge[1]):
            raise ValueError('Invalid snapshot edge')
        normalized.append(tuple(sorted(edge)))
    if len(set(normalized)) != len(normalized):
        raise ValueError('Duplicate snapshot edge')
    result['edges'] = [list(edge) for edge in sorted(normalized)]
    fingerprint(result)  # Also rejects NaN/Infinity in any extra snapshot field.
    return result


def hardware_profile(snapshot: dict, state_name: str) -> dict:
    """Return a fixed, connected induced subgraph with adjacent input pairs."""
    if state_name not in PROFILES:
        raise ValueError('Unsupported state profile')
    physical = PROFILES[state_name]
    if max(physical) >= len(snapshot['qubit_names']):
        raise ValueError('Snapshot lacks required physical qubits')
    positions = {q: i for i, q in enumerate(physical)}
    edges = sorted(tuple(sorted((positions[a], positions[b]))) for a, b in snapshot['edges']
                   if a in positions and b in positions)
    required = {(i, i + 1) for i in range(0, len(physical), 2)}
    visited = {0}
    while True:
        updated = visited | {b for a, b in edges if a in visited} | {a for a, b in edges if b in visited}
        if updated == visited:
            break
        visited = updated
    if not required <= set(edges) or len(visited) != len(physical):
        raise ValueError('Snapshot does not support the fixed connected qutrit profile')
    return dict(physical_qubits=list(physical), qubit_names=[snapshot['qubit_names'][q] for q in physical],
                edges=[list(edge) for edge in edges],
                policy='Fixed induced subgraph and initial placement for all theta; no error-metric ranking')


def _inventory(root: Path) -> dict:
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob('*')) if path.is_file()}


def _provenance() -> dict:
    result = runtime_provenance()
    package = Path(__file__).resolve().parents[2]
    for path in sorted((package / 'bell_measurements').glob('*.py')):
        result['source_hashes'][str(path.relative_to(package.parent))] = hashlib.sha256(path.read_bytes()).hexdigest()
    from importlib.metadata import version
    result['versions']['iqm-client'] = version('iqm-client')
    return result


def run_hardware_layer(source_run, output_dir, snapshot, *, seeds=(0, 1, 2), indices=None,
                       states=None, resume=False, progress=print) -> dict:
    """Compile saved gates/states into a separate hash-checked, resumable layer."""
    source_path, output_path = Path(source_run).resolve(), Path(output_dir).resolve()
    if output_path.is_relative_to(source_path) or source_path.is_relative_to(output_path):
        raise ValueError('Hardware layer must be separate from the original source directory')
    if (not isinstance(seeds, (list, tuple)) or not seeds
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError('Use distinct nonnegative integer transpiler seeds')
    selected_states = list(PROFILES) if states is None else list(states)
    if (not selected_states or len(set(selected_states)) != len(selected_states)
            or any(state not in PROFILES for state in selected_states)):
        raise ValueError('Use distinct supported states')
    if isinstance(snapshot, (str, Path)):
        snapshot = json.loads(Path(snapshot).read_text(encoding='utf-8'))
    snapshot = validate_snapshot(snapshot)
    profiles = {state: hardware_profile(snapshot, state) for state in selected_states}
    gate_profile = hardware_profile(snapshot, 'two_qutrit')
    source = RunStore.open(source_path)
    if source.manifest.get('benchmark') not in ('theta_threshold_reassessment_v1', 'theta_continuation_v1'):
        raise ValueError('Expected a theta benchmark source')
    if fingerprint({k: v for k, v in source.manifest.items() if k != 'fingerprint'}) != source.manifest['fingerprint']:
        raise ValueError('Source manifest fingerprint mismatch')
    available = source.completed_point_indices()
    selected = available if indices is None else list(indices)
    if (not selected or any(type(i) is not int or i not in available for i in selected)
            or len(set(selected)) != len(selected) or 0 not in selected):
        raise ValueError('Select existing distinct point indices including baseline index 0')
    selected.sort()
    before = _inventory(source_path)
    # Validate every selected source bundle before creating output.
    points, full = {}, {}
    for index in selected:
        point = source.read_bundle(f'points/{index:05d}')
        meta = point['metadata']
        theta = meta.get('theta')
        if (meta.get('index') != index or meta.get('correct') is not True
                or isinstance(theta, bool) or not isinstance(theta, (float, int))
                or not math.isfinite(theta) or not 0 <= theta <= np.pi / 4):
            raise ValueError('Invalid or rejected source theta point')
        e = point['arrays'].get('E.npy')
        if e is None or e.shape != (4, 3) or not np.allclose(e, theta_embedding(theta), atol=1e-13, rtol=0):
            raise ValueError('Source encoding differs from theta identity')
        if not {'f3_optimal.qpy', 'cz3_selected.qpy'} <= point['circuits'].keys():
            raise ValueError('Source point lacks selected F3/CZ3')
        points[index] = point
        for state in selected_states:
            bundle = source.read_bundle(f'circuits/{state}/{index:05d}')
            m = bundle['metadata']
            if (m.get('index') != index or m.get('state_name') != state or m.get('theta') != theta
                    or m.get('success') is not True or 'graph_state_transpiled.qpy' not in bundle['circuits']
                    or ('gate_point_fingerprint' in m and m['gate_point_fingerprint'] != fingerprint(point['complete']))):
                raise ValueError('Source full-circuit identity mismatch')
            full[state, index] = bundle
    payload = dict(benchmark='theta_iqm_hardware_layer_v1', schema_version=1,
        source_run=str(source_path), source_fingerprint=source.manifest['fingerprint'],
        source_inventory_fingerprint=fingerprint(before), source_file_count=len(before),
        snapshot=snapshot, profiles=profiles, gate_profile=gate_profile,
        config=dict(indices=selected, states=selected_states, seeds=list(seeds)), provenance=_provenance(),
        hardware_submitted=False, cost_scope='Before DD/ZNE; native R/CZ and explicit terminal readout',
        baseline_scope='theta=0 from the same F3/CZ3 gate library, not the separately shortened hardware baseline')
    manifest = {**payload, 'fingerprint': fingerprint(payload)}
    if resume:
        store = RunStore.open(output_path, expected_fingerprint=manifest['fingerprint'])
        if json.loads((output_path / 'source_inventory.json').read_text()) != before:
            raise ValueError('Saved source inventory changed')
    else:
        store = RunStore.create(output_path, manifest)
        atomic_write_json(output_path / 'source_inventory.json', before)
    total = len(selected) * (2 + len(selected_states))
    done = 0
    for index in selected:
        point = points[index]
        theta = point['metadata']['theta']
        for gate, filename, edges in (('f3', 'f3_optimal.qpy', [[0, 1]]),
                                       ('cz3', 'cz3_selected.qpy', gate_profile['edges'])):
            relative = f'gates/{gate}/{index:05d}'
            if (output_path / relative / 'complete.json').exists():
                store.read_bundle(relative)
            else:
                data = compile_gate(point['circuits'][filename], edges, seeds=tuple(seeds))
                data['metadata'].update(index=index, theta=theta, gate=gate,
                    source_bundle_fingerprint=fingerprint(point['complete']))
                store.write_bundle(relative, **data)
            done += 1
            if progress:
                progress(f'{done}/{total}: {gate}, theta index {index}')
        for state in selected_states:
            relative = f'circuits/{state}/{index:05d}'
            if (output_path / relative / 'complete.json').exists():
                store.read_bundle(relative)
            else:
                bundle = full[state, index]
                data = compile_case(state, bundle['circuits']['graph_state_transpiled.qpy'],
                                    point['arrays']['E.npy'], profiles[state]['edges'], seeds=tuple(seeds))
                data['metadata'].update(index=index, theta=theta, state_name=state,
                    source_bundle_fingerprint=fingerprint(bundle['complete']),
                    source_saved_fidelity=bundle['metadata'].get('fidelity'),
                    physical_profile=profiles[state])
                store.write_bundle(relative, **data)
            done += 1
            if progress:
                progress(f'{done}/{total}: {state}, theta index {index}')
    if _inventory(source_path) != before:
        raise ValueError('Source artifacts changed while computing hardware layer')
    from .hardware_report import generate_hardware_report
    report = generate_hardware_report(output_path)
    result = dict(status='complete', cases=len(full), gate_cases=2 * len(selected),
                  source_files_unchanged=len(before), hardware_submitted=False,
                  fingerprint=manifest['fingerprint'], report=str(report))
    atomic_write_json(output_path / 'completion.json', result)
    return result
