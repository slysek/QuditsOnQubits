"""Offline uncertainty for the randomized three-scale Bell campaign.

The frozen setting schedule is conditioned on. Scores include zero-weight leakage.
Shots and setting-stratified blocks are two alternative sampling assumptions;
never add their variances, which would double-count shot noise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import chi2, norm, t

SCALES = np.array([1., 3., 5.])
MODEL_WEIGHTS = {
    'linear': np.array([13/12, 1/3, -5/12]),
    'quadratic': np.array([15/8, -5/4, 3/8]),
    'linear_1_3': np.array([1.5, -.5, 0.]),
    'linear_1_5': np.array([1.25, 0., -.25]),
    'linear_3_5': np.array([0., 2.5, -1.5]),
}


def block_statistics(counts, settings, weights):
    counts, settings, weights = np.asarray(counts), np.asarray(settings), np.asarray(weights)
    if (counts.ndim != 3 or counts.shape[0] != 3 or weights.ndim != 2
        or settings.shape != (counts.shape[1],) or counts.shape[2] != weights.shape[1]
        or counts.dtype.kind not in 'iuf' or not np.isfinite(counts).all()
        or np.any(counts < 0) or np.any(counts != np.floor(counts))
        or settings.dtype.kind not in 'iu' or np.any(settings < 0)
        or np.any(settings >= len(weights)) or weights.dtype.kind not in 'iuf'
        or not np.isfinite(weights).all()):
        raise ValueError('Expected finite real weights, integer counts and setting indices at three scales')
    shots = counts.sum(axis=2)
    frequencies = np.bincount(settings, minlength=len(weights))
    if np.any(shots < 2) or np.any(frequencies < 2):
        raise ValueError('Need at least two shots per block and two blocks per setting')
    probabilities = counts / shots[:, :, None]
    scores = np.sum(probabilities * weights[settings], axis=2)
    # Sample variance of each block mean, including covariance of all Bell terms.
    shot_components = np.maximum(0., np.sum(probabilities * weights[settings]**2, axis=2) - scores**2) / (shots - 1)
    block_components = np.stack([
        np.var(scores[:, settings == j], axis=1, ddof=1) / n
        for j, n in enumerate(frequencies)
    ], axis=1)
    return dict(
        bell=np.sum(scores / frequencies[settings], axis=1), scores=scores,
        frequencies=frequencies, shot_variance=np.sum(shot_components / frequencies[settings]**2, axis=1),
        block_variance_components=block_components,
        hoeffding_range_sum=float(np.sum(np.ptp(weights, axis=1)**2 / frequencies)),
    )


def bootstrap(counts, settings, weights, *, draws=20000, seed=20260914):
    stats = block_statistics(counts, settings, weights)
    if (type(draws) is not int or draws < 200 or type(seed) is not int or seed < 0):
        raise ValueError('Need at least 200 bootstrap draws and a nonnegative integer seed')
    rng = np.random.default_rng(seed)
    shot_draws, block_draws = np.zeros((draws, 3)), np.zeros((draws, 3))
    counts, settings, weights = np.asarray(counts), np.asarray(settings), np.asarray(weights)
    for s in range(3):
        for b, j in enumerate(settings):
            n = int(np.sum(counts[s, b]))
            sampled = rng.multinomial(n, counts[s, b] / n, size=draws)
            shot_draws[:, s] += (sampled @ weights[j]) / n / stats['frequencies'][j]
        # Independently resample each scale: equal block IDs do not establish
        # synchronized experiments when scale jobs were executed sequentially.
        for j, n in enumerate(stats['frequencies']):
            values = stats['scores'][s, settings == j]
            block_draws[:, s] += values[rng.integers(0, n, size=(draws, n))].mean(axis=1)
    return dict(shot=shot_draws, block=block_draws)


def _interval(value, se, quantile):
    return [float(value - quantile * se), float(value + quantile * se)]


def _contrast(c, stats, resamples):
    c = np.asarray(c)
    value = float(c @ stats['bell'])
    shot_se = float(np.sqrt(c**2 @ stats['shot_variance']))
    components = c[:, None]**2 * stats['block_variance_components']
    variance = float(np.sum(components))
    denominator = float(np.sum(components**2 / (stats['frequencies'][None, :] - 1)))
    df = variance**2 / denominator if denominator > 0 else None
    block_se = float(np.sqrt(variance))
    result = dict(value=value, weights=c.tolist(),
        shot=dict(se=shot_se, ci95=_interval(value, shot_se, norm.ppf(.975))),
        block=dict(se=block_se, welch_df=df,
            ci95=_interval(value, block_se, t.ppf(.975, df) if df is not None else 0.)),
        independent_block_hoeffding_ci95=_interval(value, float(np.sqrt(
            .5 * np.log(40) * np.sum(c**2) * stats['hoeffding_range_sum'])), 1.),
    )
    if resamples is not None:
        for kind in ('shot', 'block'):
            samples = resamples[kind] @ c
            result[kind]['bootstrap_se'] = float(np.std(samples, ddof=1))
            result[kind]['bootstrap_percentile_ci95'] = np.quantile(samples, [.025, .975]).tolist()
    return result


def summarize(stats, resamples, *, bound, ideal):
    if not np.isfinite([bound, ideal]).all():
        raise ValueError('Bell bound and ideal reference must be finite')
    models = []
    for name, c in MODEL_WEIGHTS.items():
        row = _contrast(c, stats, resamples)
        row.update(model=name, residual_dof=1 if name == 'linear' else 0,
            delta_from_classical=float(row['value'] - bound),
            error_vs_ideal=float(row['value'] - ideal),
            interpretation='conditional on model; statistical interval excludes extrapolation bias')
        row['shot']['signed_distance_from_bound_se'] = (
            row['delta_from_classical'] / row['shot']['se'] if row['shot']['se'] > 0 else None)
        models.append(row)
    curvature = _contrast(np.array([1., -2., 1.]), stats, resamples)
    shot_se, block_se = curvature['shot']['se'], curvature['block']['se']
    statistic = (curvature['value'] / shot_se)**2 if shot_se > 0 else None
    df = curvature['block']['welch_df']
    return dict(
        schema_version=1, scales=SCALES.tolist(), classical_bound=float(bound), ideal_bell=float(ideal),
        setting_frequencies=stats['frequencies'].tolist(),
        scale_statistics=[dict(scale=float(SCALES[s]), **_contrast(np.eye(3)[s], stats, resamples)) for s in range(3)],
        models=models,
        linearity=dict(contrast=curvature, quadratic_coefficient=curvature['value']/8,
            quadratic_coefficient_ci95=(np.array(curvature['shot']['ci95'])/8).tolist(),
            shot_chi_square=statistic, residual_dof=1,
            shot_chi_square_p=float(chi2.sf(statistic, 1)) if statistic is not None else None,
            block_welch_p=float(2*t.sf(abs(curvature['value'])/block_se, df)) if block_se > 0 and df is not None else None),
        model_sensitivity=dict(linear_minus_quadratic=_contrast(MODEL_WEIGHTS['linear']-MODEL_WEIGHTS['quadratic'], stats, resamples),
            bias_identified=False,
            explanation='Model spread is sensitivity, not a bias bound or an independent error bar. Adding c*(x-1)*(x-3)*(x-5) preserves data and changes the zero intercept by -15*c.'),
        assumptions=[
            'Condition on the saved setting schedule; sum nine within-setting block means; keep leakage in the denominator.',
            'Shot intervals assume independent shots within each block and independent blocks/scales; no serial drift coverage.',
            'Block intervals instead assume independent exchangeable blocks within each setting/scale; small strata limit accuracy.',
            'The two variance estimates are alternatives and are not added; empirical block variance can fall below shot variance.',
            'Percentile block bootstrap is exploratory and has finite-sample downward variance bias, severe for settings with two blocks.',
            'Welch intervals approximate normal block scores with estimated stratum variances; zero empirical variance is not certainty.',
            'Intervals are individual 95% statistical intervals, not simultaneous intervals and not confidence statements about an unvalidated noiseless intercept.',
            'Three scale jobs were sequential: scale dependence is confounded with drift; no inferred paired cross-scale covariance.',
            'CZ folding does not scale all R/readout errors; the zero-fold extrapolation need not remove those errors.',
            'Quadratic has three parameters at three points and no residual degrees of freedom for validation.',
            'Error relative to ideal B=6 combines remaining hardware and mitigation errors; it does not isolate model bias.',
        ],
        bootstrap=dict(draws=0 if resamples is None else len(resamples['shot']),
            method='per-block multinomial; independent setting-stratified block percentile bootstrap'),
    )


def _read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_hash(root, name, expected, provenance):
    path = (root/name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Evidence path leaves its directory')
    if not path.is_file() or _sha(path) != expected:
        raise ValueError(f'SHA256 mismatch: {path}')
    provenance[str(path)] = expected


def load_run(directory, source):
    """Verify archived evidence before interpreting the specific DD+ZNE run."""
    directory, source = Path(directory).resolve(), Path(source).resolve()
    audit, protocol, source_protocol = (_read(directory/'completion_audit.json'),
        _read(directory/'protocol.json'), _read(source/'protocol.json'))
    if (audit['status'] != 'complete' or protocol['zne_factors'] != [1, 3, 5]
        or protocol['variant'] != 'DD_ZNE' or protocol['twirling'] is not False
        or protocol['dd_mode'] != 'enabled'):
        raise ValueError('Expected completed DD+ZNE without twirling at scales 1,3,5')
    provenance = {str(directory/name): _sha(directory/name) for name in ('completion_audit.json', 'protocol.json')}
    provenance[str(source/'protocol.json')] = _sha(source/'protocol.json')
    _check_hash(directory, 'analysis.json', audit['analysis_sha256'], provenance)
    for name, digest in protocol['input_sha256'].items():
        _check_hash(directory, name, digest, provenance)
    for name, digest in audit['hardware_sha256'].items():
        _check_hash(directory/'hardware', name, digest, provenance)
    for name in ('bell_weights.npy', 'schedule.json', 'evidence.json'):
        _check_hash(source, name, source_protocol['input_sha256'][name], provenance)
    schedule, analysis = _read(directory/'schedule.json'), _read(directory/'analysis.json')
    if schedule != _read(source/'schedule.json') or schedule != _read(directory/'hardware/schedule.json'):
        raise ValueError('Frozen setting schedules differ')
    if _read(directory/'evidence.json') != _read(source/'evidence.json'):
        raise ValueError('Ideal-reference evidence differs')
    blocks = schedule['blocks']
    labels = [(f'A{a}', f'B{b}') for a in range(3) for b in range(3)]
    settings = np.array([labels.index(tuple(block['settings'])) for block in blocks])
    ids = [block['block_id'] for block in blocks]
    if len(set(ids)) != len(ids) or len(ids) != protocol['draws']:
        raise ValueError('Invalid schedule block identifiers')
    counts = []
    for factor in (1, 3, 5):
        request = _read(directory/f'hardware/DD_{factor}_0.request.json')
        records = request['records']
        if (request['dd'] is not True or request['factor'] != factor or request['arm'] != 'DD'
            or request['shots'] != protocol['shots_per_draw_per_scale']
            or [r['block_id'] for r in records] != ids
            or any(r['instance'] != 0 or r['mapping'] != protocol['selection']['measurement_map_c0_to_c3'] for r in records)):
            raise ValueError('Hardware request disagrees with the saved protocol')
        raw = _read(directory/f'hardware/DD_{factor}_0.counts.json')
        if not isinstance(raw, list) or len(raw) != len(blocks):
            raise ValueError('Missing measurement blocks')
        values = []
        for item in raw:
            if (not isinstance(item, dict) or any(
                not isinstance(key, str) or len(key) != 4 or set(key)-set('01')
                or type(value) is not int or value < 0 for key, value in item.items())
                or sum(item.values()) != request['shots']):
                raise ValueError('Invalid four-bit counts or shot total')
            values.append([item.get(f'{k:04b}', 0) for k in range(16)])
        counts.append(values)
    counts = np.array(counts, dtype=np.int64)
    weights = np.load(source/'bell_weights.npy', allow_pickle=False)
    stats = block_statistics(counts, settings, weights)
    expected = np.array([row['bell'] for row in analysis['rows']])
    if (not np.allclose(stats['bell'], expected, atol=1e-12, rtol=0)
        or abs(MODEL_WEIGHTS['linear'] @ stats['bell'] - analysis['bell']) > 1e-12
        or int(counts.sum()) != audit['total_verified_shots']):
        raise ValueError('Archived Bell estimates or shot budget cannot be reproduced')
    return dict(counts=counts, settings=settings, weights=weights, analysis=analysis,
        ideal=float(_read(source/'evidence.json')['ideal_bell']), provenance=provenance)


def write_outputs(output, result, stats, resamples):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output/'analysis.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    np.savez_compressed(output/'bootstrap.npz', **resamples)
    linear, quadratic = result['models'][:2]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), constrained_layout=True)
    x = np.linspace(0, 5.25, 240)
    axes[0].axvspan(0, 1, color='#eef2f6', label='Obszar ekstrapolacji')
    axes[0].errorbar(SCALES, stats['bell'], yerr=norm.ppf(.975)*np.sqrt(stats['shot_variance']),
        fmt='o', color='#202b37', capsize=4, label='IQM DD: 95% CI zliczeń', zorder=5)
    for degree, color, label in [(1, '#086f87', 'Liniowy'), (2, '#c85428', 'Kwadratowy')]:
        axes[0].plot(x, np.polynomial.polynomial.polyval(x, np.polynomial.polynomial.polyfit(SCALES, stats['bell'], degree)), color=color, label=label)
    axes[0].axhline(result['classical_bound'], color='#6b4b87', ls='--', label='Granica klasyczna')
    axes[0].set(xlabel='Skala foldingu CZ', ylabel='Wartość Bella', title='Dane i kształt modelu', xlim=(-.15, 5.3))
    axes[0].legend(fontsize=8, loc='lower left')
    labels = ['Liniowy: 1, 3, 5', 'Kwadratowy: 1, 3, 5', 'Liniowy: 1, 3', 'Liniowy: 1, 5', 'Liniowy: 3, 5']
    for i, row in enumerate(result['models']):
        axes[1].errorbar(row['value'], i, xerr=norm.ppf(.975)*row['shot']['se'], fmt='o',
            color='#c85428' if i == 1 else '#086f87', capsize=4)
    axes[1].axvline(result['classical_bound'], color='#6b4b87', ls='--')
    axes[1].axvline(result['ideal_bell'], color='#87939e', ls=':', label='Ideał: 6')
    axes[1].set(yticks=range(5), yticklabels=labels, xlabel='Ekstrapolowane B(0)', title='Wrażliwość na model i wybór skal')
    axes[1].invert_yaxis()
    axes[1].legend(fontsize=8, loc='lower right')
    for ax in axes:
        ax.grid(alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('DD + ZNE | 50 bloków × 1000 shotów × 3 skale', fontsize=14)
    fig.savefig(output/'zne_uncertainty.png', dpi=180)
    fig.savefig(output/'zne_uncertainty.pdf')
    plt.close(fig)
    lines = [
        '# DD + ZNE: niepewność i błąd modelu', '',
        '**Wniosek: wynik powyżej granicy klasycznej zależy od modelu. Te dane nie dają stabilnego potwierdzenia naruszenia po ZNE.**', '',
        f"IQM Emerald, DD włączone, twirling wyłączony. Trzy skale CZ: 1, 3, 5; po 50 bloków × 1000 shotów, łącznie 150 000. Granica klasyczna: {result['classical_bound']:.9f}; idealny obwód: B = 6.", '',
        '## Niepewność statystyczna', '',
        'Poniższe ± oznacza jeden błąd standardowy (1σ), a CI to przybliżony, dwustronny przedział 95%. Zakładamy niezależność shotów i bloków, ustalony harmonogram oraz poprawność wybranego modelu. Przedziały nie zawierają błędu modelu ani dowolnego dryfu.', '',
        '| Estymator | B | SE zliczeń | 95% CI zliczeń |', '|---|---:|---:|---|',
    ]
    for row in result['scale_statistics'] + result['models']:
        label = f"Skala {row['scale']:g}" if 'scale' in row else row['model']
        ci = row['shot']['ci95']
        lines.append(f"| {label} | {row['value']:.6f} | {row['shot']['se']:.6f} | [{ci[0]:.6f}, {ci[1]:.6f}] |")
    delta = result['model_sensitivity']['linear_minus_quadratic']
    lines += ['', f"Dla pierwotnego liniowego ZNE: B(0) = {linear['value']:.6f} ± {linear['shot']['se']:.6f}. Odległość od granicy to {linear['shot']['signed_distance_from_bound_se']:.2f} SE **wyłącznie przy założeniu poprawnego modelu liniowego**.", '',
        '## Kontrola bootstrapem i zmienność bloków', '',
        f"Wykonano {result['bootstrap']['draws']} replik dla każdej metody. Multinomialny bootstrap losuje zliczenia wewnątrz każdego zachowanego bloku. Drugi bootstrap losuje całe bloki osobno w każdej kombinacji ustawienia i skali; zachowuje nierówne liczności ustawień.", '',
        '| Model | SE bootstrap zliczeń | 95% CI bootstrap zliczeń | SE między blokami | 95% CI Welch między blokami |', '|---|---:|---|---:|---|',
    ]
    for row in (linear, quadratic):
        sc, bc = row['shot']['bootstrap_percentile_ci95'], row['block']['ci95']
        lines.append(f"| {row['model']} | {row['shot']['bootstrap_se']:.6f} | [{sc[0]:.6f}, {sc[1]:.6f}] | {row['block']['se']:.6f} | [{bc[0]:.6f}, {bc[1]:.6f}] |")
    lines += ['', 'Wariancje zliczeń i bloków to alternatywne oszacowania; nie dodajemy ich. Zmienność bloków bywa mniejsza od wariancji zliczeń wskutek małej próby. To nie dowodzi braku dryfu. Liczności ustawień A0B0…A2B2: '+str(result['setting_frequencies'])+'. A1B1 ma tylko dwa bloki. Percentylowy bootstrap bloków jest diagnostyczny: jego wariancja jest obciążona w dół przy małych licznościach (dla dwóch bloków o czynnik 1/2). Dlatego główna tabela bloków używa wariancji z korektą n−1 i przybliżenia Welcha.', '',
        f"Przedział Hoeffdinga dla liniowej kombinacji średnich, dopuszczający dowolną zależność shotów wewnątrz bloków, wynosi {linear['independent_block_hoeffding_ci95']}. Nadal wymaga niezależności bloków, dotyczy ważonej średniej na zmierzonych skalach i nie obejmuje obciążenia ekstrapolacji.", '',
        '## Dopasowanie i błąd modelu', '',
        f"Kontrast liniowości D = B(1) − 2B(3) + B(5) = {result['linearity']['contrast']['value']:.6f}, SE = {result['linearity']['contrast']['shot']['se']:.6f}. Dla modelu liniowego D powinno wynosić zero. Test χ² z 1 stopniem swobody daje p = {result['linearity']['shot_chi_square_p']:.6f}; przy estymacji z bloków test Welcha daje p = {result['linearity']['block_welch_p']:.6f}. Są oznaki odstępstwa od linii prostej przy przyjętych założeniach statystycznych. Wysokie R² samo nie waliduje modelu ([NIST](https://www.itl.nist.gov/div898/handbook/pmd/section4/pmd44.htm)).", '',
        f"Kwadratowe ZNE daje {quadratic['value']:.6f}, czyli {quadratic['delta_from_classical']:+.6f} względem granicy. Różnica liniowy − kwadratowy wynosi {delta['value']:.6f}; przewaga liniowego nad granicą wynosi tylko {linear['delta_from_classical']:.6f}. Ta różnica modeli jest miarą wrażliwości, **nie oszacowanym prawdziwym błędem systematycznym ani gwarantowaną granicą tego błędu**.", '',
        'Model kwadratowy ma trzy parametry i trzy punkty: interpoluje je dokładnie, pozostawiając zero stopni swobody do walidacji. Nie uznajemy go automatycznie za poprawny. Dopisanie c(λ−1)(λ−3)(λ−5) zachowuje wszystkie zmierzone punkty, lecz zmienia B(0) o −15c. Bez dodatkowych założeń o kształcie funkcji nie ma identyfikowalnego błędu samej ekstrapolacji. Taki przykład dotyczy braku ograniczenia narzuconego przez trzy punkty; fizyczne ograniczenia Bella nadal obowiązują.', '',
        f"Względem znanego ideału B=6 liniowe ZNE ma błąd {linear['error_vs_ideal']:+.6f} ({abs(linear['error_vs_ideal'])/result['ideal_bell']*100:.3f}%), a kwadratowe {quadratic['error_vs_ideal']:+.6f} ({abs(quadratic['error_vs_ideal'])/result['ideal_bell']*100:.3f}%). To całkowity pozostały błąd estymacji; nie da się go z tych danych rozdzielić na błędny kształt ZNE, dryf i nieskalowane błędy sprzętu.", '',
        'Skale zrealizowano w trzech kolejnych zadaniach. Wspólne identyfikatory bloków nie oznaczają jednoczesnych pomiarów: skala jest splątana z czasem wykonania. Dodatkowo folding zwielokrotnił CZ, pozostawiając liczbę bramek R i pomiarów bez zmian. Ekstrapolacja do λ=0 nie musi usuwać tych pozostałych błędów. Rozdzielenie skalowania szumu i wyboru ekstrapolatora jest podstawą metody ([Giurgica-Tiron i in.](https://arxiv.org/abs/2005.10921)).', '',
        '## Metoda i odtwarzalność', '',
        'Dla ustawienia j z n_j blokami obliczamy średnią z jego bloków, następnie sumujemy dziewięć średnich. W każdym bloku wynik to m_sb = Σ_k w_jk p_sbk. Waga stanów poza podprzestrzenią qutritu wynosi zero, ale ich zliczenia pozostają w mianowniku; nie ma postselekcji ani korekcji odczytu.', '',
        'Var(m_sb) = [Σ_k w_jk² p_sbk − m_sb²]/(N_sb−1). Var(B_s) = Σ_b Var(m_sb)/n_j(b)². Cały rzeczywisty wynik danego bitstringu jest liczony jedną wagą, więc nie pomijamy kowariancji algebraicznych składników Bella. Dla modelu liniowego wagi skal to [13/12, 1/3, −5/12]; dla kwadratowego [15/8, −5/4, 3/8]. Nie zastępujemy pierwotnej regresji nieważonej regresją ważoną.', '',
        'Wariancja z bloków: Σ_j s²_j/n_j, gdzie s²_j jest wariancją wyników bloków dla danego ustawienia i skali. Dla kontrastów propagujemy niezależne skale, a stopnie swobody przybliżamy wzorem Welcha–Satterthwaite’a. Test liniowości opiera się bezpośrednio na kontraście D i wariancji zliczeń; odpowiada testowi reszt ważonego dopasowania liniowego.', '',
        'Zweryfikowano SHA256 zapisanych danych sprzętowych i wejściowych, odtworzono trzy wartości Bella oraz pierwotne liniowe ZNE. analysis.json zawiera hashe wejść i skryptu, założenia oraz pełne wyniki. bootstrap.npz zawiera repliki. Przedziały są indywidualne 95%; dobór alternatywnych modeli jest analizą eksploracyjną, nie wcześniej ustaloną procedurą potwierdzania naruszenia.', '',
        'Kolejny pomiar, jeśli będzie potrzebny: więcej skal blisko 1, przeplatane skale, więcej niezależnych bloków i niezależna walidacja modelu. W tej analizie nie uruchamiano nowych zadań IQM.', '',
        '![Porównanie modeli ZNE](zne_uncertainty.png)', '',
    ]
    (output/'report.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--draws', type=int, default=20000)
    parser.add_argument('--seed', type=int, default=20260914)
    args = parser.parse_args()
    loaded = load_run(args.run, args.source)
    stats = block_statistics(loaded['counts'], loaded['settings'], loaded['weights'])
    samples = bootstrap(loaded['counts'], loaded['settings'], loaded['weights'], draws=args.draws, seed=args.seed)
    result = summarize(stats, samples, bound=loaded['analysis']['classical_bound'], ideal=loaded['ideal'])
    result['bootstrap']['seed'] = args.seed
    result['provenance'] = loaded['provenance']
    result['provenance'][str(Path(__file__).resolve())] = _sha(Path(__file__))
    result['versions'] = dict(numpy=np.__version__, scipy=__import__('scipy').__version__)
    write_outputs(args.output, result, stats, samples)
    print(json.dumps(dict(output=str(args.output), linear=result['models'][0],
        quadratic=result['models'][1], linearity=result['linearity']), indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
