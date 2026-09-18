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
    lines=['# Dodatkowa warstwa IQM dla benchmarku theta','',
        '**Dotychczasowy benchmark pozostaje bez zmian.** Ten raport dodaje koszty sprzętowe do tych samych zapisanych bramek i stanów.', '',
        f'[Oryginalny raport](<{(source/"report.md").as_posix()}>) · [Oryginalne pełne obwody CSV](<{(source/"full_circuits.csv").as_posix()}>)', '',
        f'Punkty theta: **{len(indices)}**. Przygotowania stanów: **{len(summary)}**. Pełne ustawienia Bella: **{len(settings)}**. Bloki F3/CZ3: **{len(gates)}**. Zadań sprzętowych: **0**.', '',
        '## Co porównujemy','',
        '1. Oryginalne przygotowanie U/CZ przy pełnej łączności — zachowane liczby z zapisanych QPY.',
        '2. Natywne przygotowanie R/CZ przy pełnej łączności.',
        '3. Przygotowanie R/CZ po routingu na wybranym podgrafie IQM.',
        '4. Pełne obwody Bella: wspólne przygotowanie, lokalne obroty baz, dekodowanie i odczyt; koszty przed i po routingu.', '',
        'Kolumny 1q dla U i R dotyczą różnych zbiorów bramek. Głębokość pełnego obwodu Bella obejmuje odczyt; głębokość przygotowania nie obejmuje pomiarów. CZ depth liczy warstwy CZ. Koszty są przed DD/ZNE i bez impulsów dopisywanych przez serwer.', '',
        'W pełnych obwodach Bella końcowe lokalne bramki przygotowania łączymy z lokalnym pomiarem i kompilujemy razem. Dlatego koszt pełnego Bella nie musi być sumą całego przygotowania i osobno skompilowanej bazy. CSV rozdziela wspólny prefiks (`bell_prefix_routed_n_cz`), lokalne bloki pomiarowe (`local_measurement_n_cz`) oraz naprawcze SWAP-y (3 CZ każdy, już wliczone w prefiks). Zmiana bazy jednej strony nie zmienia obwodu pozostałych stron.', '',
        '**Baseline theta=0 oznacza obwód z tej samej biblioteki F3/CZ3. Nie jest skróconym sprzętowym baseline’em Bella o 5–7 CZ z odrębnego eksperymentu.**', '',
        '## Topologia i zasady porównania','',
        f'Backend: **{manifest["snapshot"]["backend"]}**. Kalibracja migawki: `{manifest["snapshot"]["calibration_set_id"]}`.',
        f'Seedy transpilera: `{manifest["config"]["seeds"]}`. Każdy kąt danego stanu ma te same fizyczne kubity i początkowe przypisanie.', '',
        *_table(['Stan','Fizyczne kubity, kolejność przewodów'],
                [(state,', '.join(manifest['profiles'][state]['qubit_names'])) for state in states]), '',
        'Routing zachowuje końcową permutację. Samo inne numerowanie wyjść nie wymaga fizycznego SWAP-a. Dla pomiarów Bella dołączamy najkrótszą znalezioną sekwencję SWAP-ów zapewniającą połączenie wewnątrz każdej pary kutrytu; nie wymagamy powrotu do oryginalnej numeracji. Wszystkie operacje między stronami kończą się przed lokalnymi blokami zależnymi od ustawienia.', '',
        'Jest to porównanie przy zamrożonej topologii i skończonym budżecie kompilacji. Nie jest globalnym minimum routingu, rankingiem według błędów kalibracji, harmonogramem impulsów ani prognozą wartości Bella na sprzęcie.', '',
        '## Kandydaci według pełnego kosztu po routingu','',
        'Ranking wykorzystuje średnią liczbę CZ po różnych ustawieniach Bella, następnie maksymalną głębokość i średnią liczbę R. Średnia jest opisem zasobów z równymi wagami ustawień, nie planem shotów. Baseline pokazano oddzielnie.', '']
    for state in states:
        chosen=[baselines[state],*ranking[state][:3]]
        lines += [f'### {state}', '', *_table(
            ['indeks','theta/pi','CZ przygotowania, oryginał','CZ przygotowania, IQM','CZ Bella, IQM min–max','CZ Bella, średnia','delta do theta=0'],
            [(r['index'],r['theta_over_pi'],r['preparation_original_n_cz'],r['preparation_routed_n_cz'],
              f'{r["bell_routed_n_cz_min"]}–{r["bell_routed_n_cz_max"]}',r['bell_routed_n_cz_mean'],r['bell_cz_mean_minus_theta0']) for r in chosen]), '']
    lines += ['## Wszystkie kąty', '', *_table(
        ['stan','indeks','theta/pi','CZ prep U/CZ','CZ prep IQM','CZ Bella IQM min–max','max depth Bella','błąd histogramu'],
        [(r['state'],r['index'],r['theta_over_pi'],r['preparation_original_n_cz'],r['preparation_routed_n_cz'],
          f'{r["bell_routed_n_cz_min"]}–{r["bell_routed_n_cz_max"]}',r['bell_routed_depth_max'],r['max_probability_error']) for r in summary]), '',
        '![Koszty przy kolejnych theta](hardware_costs.png)', '',
        '[Wykres PDF](hardware_costs.pdf) · [Wszystkie koszty stanów CSV](summary.csv) · [Każde ustawienie Bella CSV](bell_settings.csv) · [Bramki F3/CZ3 CSV](gates.csv) · [Ranking JSON](ranking.json)', '',
        '## Walidacja i artefakty','',
        'Źródłowe oraz nowe QPY/NPY są odczytywane przez magazyn sprawdzający SHA256. Kompilację przygotowania sprawdzamy względem rzeczywistego wejściowego stanu, pełne F3/CZ3 względem całego operatora, a pomiary według wszystkich prawdopodobieństw w kolejności klasycznych bitów. Leakage pozostaje w mianowniku i ma wagę zero. Zgodność idealna potwierdza poprawność kompilacji, nie odporność na szum.', '',
        'Kolumna `qpy_path` w każdym CSV wskazuje rzeczywisty obwód; pliki używają kompaktowych indeksów podgrafu. Odpowiadające numery i nazwy fizycznych kubitów, permutacje oraz naprawcze SWAP-y zapisano w metadanych pakietu i manifeście.', '',
        f'Fingerprint warstwy: `{manifest["fingerprint"]}`. Plików źródłowych objętych kontrolą niezmienności: {manifest["source_file_count"]}.', '']
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
        for key,label,color in [('preparation_original_n_cz','Przygotowanie: oryginał','#617789'),
                                ('preparation_routed_n_cz','Przygotowanie: IQM','#d58d34'),
                                ('bell_routed_n_cz_mean','Pełny Bell: IQM, średnia','#187c78')]:
            ax.plot(x,[r[key] for r in selected],'.-',label=label,color=color)
        ax.fill_between(x,[r['bell_routed_n_cz_min'] for r in selected],
                        [r['bell_routed_n_cz_max'] for r in selected],color='#187c78',alpha=.15)
        ax.set(title=state,xlabel='theta [stopnie]',ylabel='Liczba CZ')
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(directory/'hardware_costs.png',dpi=180)
    fig.savefig(directory/'hardware_costs.pdf')
    plt.close(fig)
