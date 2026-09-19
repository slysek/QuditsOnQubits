"""Side-by-side reports for the additive, frozen-topology theta layer."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .artifacts import RunStore, atomic_write_json, fingerprint


def _csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list,dict)) else value
                             for key, value in row.items()})


def _costs(prefix, values):
    return {f'{prefix}_{key}': values[key] for key in ('n_cz','n_1q','depth','cz_depth','size')}


def _number(value):
    if value is None:
        return '—'
    return f'{value:.6g}' if isinstance(value, float) else str(value)


def _table(headers, rows):
    return ['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |',
            *['| '+' | '.join(_number(value) for value in row)+' |' for row in rows]]


def generate_hardware_report(directory) -> Path:
    """Read validated output bundles; retain old costs and add hardware columns."""
    directory = Path(directory).resolve()
    store = RunStore.open(directory)
    manifest = store.manifest
    if (manifest.get('benchmark') != 'theta_iqm_hardware_layer_v1'
            or fingerprint({k:v for k,v in manifest.items() if k!='fingerprint'}) != manifest['fingerprint']):
        raise ValueError('Invalid hardware-layer manifest')
    states, indices = manifest['config']['states'], manifest['config']['indices']
    gates, settings, summary = [], [], []
    for index in indices:
        for gate in ('f3','cz3'):
            m = store.read_bundle(f'gates/{gate}/{index:05d}')['metadata']
            gates.append(dict(gate=gate,index=index,theta=m['theta'],theta_over_pi=m['theta']/np.pi,
                **_costs('original',m['original']), **_costs('native',m['native']), **_costs('routed',m['routed']),
                full_operator_error=m['full_operator_error'],routing=m['routing'],
                qpy_path=f'gates/{gate}/{index:05d}/routed.qpy'))
        for state in states:
            relative = f'circuits/{state}/{index:05d}'
            m = store.read_bundle(relative)['metadata']
            rows = m['per_setting']
            for row in rows:
                settings.append(dict(state=state,index=index,theta=m['theta'],theta_over_pi=m['theta']/np.pi,
                    setting_index=row['index'],labels=row['labels'],
                    **_costs('native',row['native']), **_costs('routed',row['routed']),
                    local_measurement_n_cz=row['routed']['n_cz']-m['bell_prefix_routed']['n_cz'],
                    bell_prefix_routed_n_cz=m['bell_prefix_routed']['n_cz'],
                    repair_swaps=m['routing']['bell']['repair_count'],
                    routing_cz_delta=row['routed']['n_cz']-row['native']['n_cz'],
                    max_probability_error=row['max_probability_error'],
                    ideal_contribution=row['ideal_contribution'],
                    physical_qubits=m['physical_profile']['physical_qubits'],
                    routing=m['routing']['bell'],
                    qpy_path=f'{relative}/bell_routed_{row["index"]:03d}.qpy'))
            row = dict(state=state,index=index,theta=m['theta'],theta_over_pi=m['theta']/np.pi,
                **_costs('preparation_original',m['preparation_original']),
                **_costs('preparation_native',m['preparation_native']),
                **_costs('preparation_routed',m['preparation_routed']),
                **_costs('bell_prefix_native',m['bell_prefix_native']),
                **_costs('bell_prefix_routed',m['bell_prefix_routed']),
                bell_repair_swaps=m['routing']['bell']['repair_count'],
                bell_repair_n_cz=3*m['routing']['bell']['repair_count'],
                settings=len(rows),ideal_bell=m['ideal_bell'],source_fidelity=m['source_fidelity'],
                max_probability_error=m['max_probability_error'],routing=m['routing'],
                physical_qubits=m['physical_profile']['physical_qubits'],
                qpy_path=f'{relative}/preparation_routed.qpy')
            for stage in ('native','routed'):
                for cost in ('n_cz','n_1q','depth','cz_depth'):
                    values=[r[stage][cost] for r in rows]
                    row.update({f'bell_{stage}_{cost}_min':min(values),
                                f'bell_{stage}_{cost}_mean':float(np.mean(values)),
                                f'bell_{stage}_{cost}_max':max(values)})
            row['preparation_routing_cz_delta']=row['preparation_routed_n_cz']-row['preparation_native_n_cz']
            row['bell_routing_cz_delta_mean']=row['bell_routed_n_cz_mean']-row['bell_native_n_cz_mean']
            summary.append(row)
    baselines={row['state']:row for row in summary if row['index']==0}
    for row in summary:
        row['bell_cz_mean_minus_theta0']=row['bell_routed_n_cz_mean']-baselines[row['state']]['bell_routed_n_cz_mean']
    ranking={state:sorted([r for r in summary if r['state']==state and r['index']!=0],
                         key=lambda r:(r['bell_routed_n_cz_mean'],r['bell_routed_depth_max'],
                                       r['bell_routed_n_1q_mean'],r['index'])) for state in states}
    _csv(directory/'gates.csv',gates)
    _csv(directory/'bell_settings.csv',settings)
    _csv(directory/'summary.csv',summary)
    atomic_write_json(directory/'ranking.json',dict(
        criterion='mean native CZ in routed full Bell settings, then maximum depth, mean R, index',
        setting_weights='uniform resource summary over distinct Bell settings, not a shot allocation',
        states={state:[dict(index=r['index'],theta=r['theta'],mean_cz=r['bell_routed_n_cz_mean'],
                           max_depth=r['bell_routed_depth_max']) for r in rows] for state,rows in ranking.items()}))
    _plot(directory, summary, states)
    source=Path(manifest['source_run'])
    lines=['# Additional IQM layer for the theta benchmark','',
        '**The existing benchmark remains unchanged.** This report adds hardware costs for the same saved gates and states.', '',
        f'[Original report](<{(source/"report.md").as_posix()}>) · [Original full circuits CSV](<{(source/"full_circuits.csv").as_posix()}>)', '',
        f'Theta points: **{len(indices)}**. State preparations: **{len(summary)}**. Full Bell settings: **{len(settings)}**. F3/CZ3 blocks: **{len(gates)}**. Hardware jobs: **0**.', '',
        '## What is compared','',
        '1. Original U/CZ preparation with all-to-all connectivity — counts preserved from saved QPY files.',
        '2. Native R/CZ preparation with all-to-all connectivity.',
        '3. R/CZ preparation after routing on the selected IQM subgraph.',
        '4. Full Bell circuits: shared preparation, local basis rotations, decoding and readout; costs before and after routing.', '',
        'The 1q columns for U and R refer to different gate sets. Full Bell circuit depth includes readout; preparation depth excludes measurements. CZ depth counts CZ layers. Costs are before DD/ZNE and exclude pulses added by the server.', '',
        'In full Bell circuits, the final local preparation gates are combined with the local measurement and compiled together. Therefore, the full Bell cost need not equal the sum of the complete preparation and a separately compiled basis. The CSV separates the shared prefix (`bell_prefix_routed_n_cz`), local measurement blocks (`local_measurement_n_cz`) and repair SWAPs (3 CZ each, already included in the prefix). Changing one party’s basis does not change the circuits of the other parties.', '',
        '**The theta=0 baseline is a circuit from the same F3/CZ3 library. It is not the shortened hardware Bell baseline of 5–7 CZ from a separate experiment.**', '',
        '## Topology and comparison rules','',
        f'Backend: **{manifest["snapshot"]["backend"]}**. Snapshot calibration: `{manifest["snapshot"]["calibration_set_id"]}`.',
        f'Transpiler seeds: `{manifest["config"]["seeds"]}`. Every angle for a given state uses the same physical qubits and initial mapping.', '',
        *_table(['State','Physical qubits, wire order'],
                [(state,', '.join(manifest['profiles'][state]['qubit_names'])) for state in states]), '',
        'Routing preserves the final permutation. Output renumbering alone does not require a physical SWAP. For Bell measurements, we append the shortest SWAP sequence found that ensures connectivity within each qutrit’s qubit pair; a return to the original numbering is not required. All operations between parties finish before the local setting-dependent blocks.', '',
        'This comparison uses a fixed topology and a finite compilation budget. It is not a global routing minimum, a ranking by calibration errors, a pulse schedule or a prediction of hardware Bell values.', '',
        '## Candidates by full cost after routing','',
        'The ranking uses the mean CZ count across distinct Bell settings, followed by maximum depth and mean R count. The mean summarizes resources with equal setting weights; it is not a shot allocation plan. The baseline is shown separately.', '']
    for state in states:
        chosen=[baselines[state],*ranking[state][:3]]
        lines += [f'### {state}', '', *_table(
            ['index','theta/pi','Preparation CZ, original','Preparation CZ, IQM','Bell CZ, IQM min–max','Bell CZ, mean','delta from theta=0'],
            [(r['index'],r['theta_over_pi'],r['preparation_original_n_cz'],r['preparation_routed_n_cz'],
              f'{r["bell_routed_n_cz_min"]}–{r["bell_routed_n_cz_max"]}',r['bell_routed_n_cz_mean'],r['bell_cz_mean_minus_theta0']) for r in chosen]), '']
    lines += ['## All angles', '', *_table(
        ['state','index','theta/pi','CZ prep U/CZ','CZ prep IQM','Bell CZ IQM min–max','max Bell depth','histogram error'],
        [(r['state'],r['index'],r['theta_over_pi'],r['preparation_original_n_cz'],r['preparation_routed_n_cz'],
          f'{r["bell_routed_n_cz_min"]}–{r["bell_routed_n_cz_max"]}',r['bell_routed_depth_max'],r['max_probability_error']) for r in summary]), '',
        '![Costs across theta points](hardware_costs.png)', '',
        '[Plot PDF](hardware_costs.pdf) · [All state costs CSV](summary.csv) · [Each Bell setting CSV](bell_settings.csv) · [F3/CZ3 gates CSV](gates.csv) · [Ranking JSON](ranking.json)', '',
        '## Validation and artifacts','',
        'Source and new QPY/NPY files are read through a store that checks SHA256. Preparation compilation is checked against the actual input state, full F3/CZ3 gates against the entire operator, and measurements against all probabilities in classical-bit order. Leakage remains in the denominator and has zero weight. Ideal agreement confirms compilation correctness, not noise robustness.', '',
        'The `qpy_path` column in each CSV points to the actual circuit; files use compact subgraph indices. The corresponding physical qubit numbers and names, permutations and repair SWAPs are recorded in the bundle metadata and manifest.', '',
        f'Layer fingerprint: `{manifest["fingerprint"]}`. Source files checked for changes: {manifest["source_file_count"]}.', '']
    path=directory/'report.md'
    path.write_text('\n'.join(lines),encoding='utf-8')
    return path


def _plot(directory, rows, states):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(len(states),1,figsize=(10,3.6*len(states)),squeeze=False,constrained_layout=True)
    for state,ax in zip(states,axes[:,0]):
        selected=sorted((r for r in rows if r['state']==state),key=lambda r:r['theta'])
        x=[r['theta']*180/np.pi for r in selected]
        for key,label,color in [('preparation_original_n_cz','Preparation: original','#617789'),
                                ('preparation_routed_n_cz','Preparation: IQM','#d58d34'),
                                ('bell_routed_n_cz_mean','Full Bell: IQM, mean','#187c78')]:
            ax.plot(x,[r[key] for r in selected],'.-',label=label,color=color)
        ax.fill_between(x,[r['bell_routed_n_cz_min'] for r in selected],
                        [r['bell_routed_n_cz_max'] for r in selected],color='#187c78',alpha=.15)
        ax.set(title=state,xlabel='theta [degrees]',ylabel='CZ count')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(directory/'hardware_costs.png',dpi=180)
    fig.savefig(directory/'hardware_costs.pdf')
    plt.close(fig)
