"""Five-scale Bell analysis using actual moments of fractional folding variants."""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
from scipy.stats import chi2,norm,t
from scripts import iqm_bell_interleaved_zne as run
from scripts.iqm_randomized_bell_campaign import count_vector,write_json

BOUND=5.638155724715451


def statistics(samples,weights):
    weights=np.asarray(weights,dtype=float)
    if weights.shape!=(9,16) or not np.isfinite(weights).all():raise ValueError('Invalid Bell weights')
    if len(samples)!=1000:raise ValueError('Expected all 1000 variant records')
    probabilities=np.full((5,50,4,16),np.nan);shots=np.zeros((5,50,4),dtype=int)
    actual=np.full((5,50,4),np.nan);settings=np.full(50,-1,dtype=int)
    for r in samples:
        try:s=run.SCALES.index(r['scale'])
        except ValueError as error:raise ValueError('Unexpected scale') from error
        b,i,j=r['block_id'],r['instance'],r['setting']
        if (type(b) is not int or type(i) is not int or type(j) is not int
            or not 0<=b<50 or not 0<=i<4 or not 0<=j<9
            or shots[s,b,i]!=0 or settings[b] not in (-1,j)
            or type(r['shots']) is not int or r['shots']<2):raise ValueError('Invalid or duplicate block/variant')
        actual[s,b,i]=r['actual_scale']
        if (not np.isfinite(actual[s,b,i]) or actual[s,b,i]<1
            or actual[s,b,i]!=r['native_cz']/r['parent_cz']):raise ValueError('Incorrect actual scale')
        probabilities[s,b,i]=count_vector(r['counts'],r['shots'])
        shots[s,b,i]=r['shots'];settings[b]=j
    if not np.isfinite(probabilities).all() or len(np.unique(shots))!=1:raise ValueError('Incomplete or unequal variant shot budget')
    frequencies=np.bincount(settings,minlength=9)
    if np.any(frequencies<2):raise ValueError('Need two blocks per setting for block uncertainty')
    if not np.allclose(actual.mean(axis=2),np.array(run.SCALES)[:,None],atol=1e-12,rtol=0):
        raise ValueError('Unbalanced actual scales')
    score=np.sum(probabilities*weights[settings][None,:,None,:],axis=3)
    variance=np.maximum(0.,np.sum(probabilities*weights[settings][None,:,None,:]**2,axis=3)-score**2)/(shots-1)
    block_values=score.mean(axis=2)
    block_variance=variance.sum(axis=2)/16
    values=np.stack([block_values[:,settings==j].mean(axis=1) for j in range(9)])
    variances=np.stack([block_variance[:,settings==j].sum(axis=1)/frequencies[j]**2 for j in range(9)])
    moments=np.ones((9,5,3))
    for j in range(9):
        for degree in (1,2):
            per_block=(actual[:,settings==j]**degree).mean(axis=2)
            if not np.allclose(per_block,per_block[:,:1],atol=1e-12,rtol=0):raise ValueError('Inconsistent scale moments within setting')
            moments[j,:,degree]=per_block.mean(axis=1)
    return dict(values=values,variances=variances,moments=moments,probabilities=probabilities,
        shots=shots,settings=settings,frequencies=frequencies,weights=weights,
        block_values=block_values,actual=actual)


def _gof(y,variance,design):
    if np.any(variance<=0):return dict(chi_square=None,dof=len(y)-design.shape[1],p=None)
    root=np.sqrt(variance)
    beta=np.linalg.lstsq(design/root[:,None],y/root,rcond=None)[0]
    statistic=float(np.sum((y-design@beta)**2/variance));df=len(y)-design.shape[1]
    return dict(chi_square=statistic,dof=df,p=float(chi2.sf(statistic,df)))


def bell_linear_gof(stats):
    return _gof(stats['values'].sum(axis=0),stats['variances'].sum(axis=0),np.vander(run.SCALES,2,increasing=True))


def fit_polynomial(stats,degree,resamples=None):
    if degree not in (1,2):raise ValueError('Use linear or quadratic models')
    designs=stats['moments'][:,:,:degree+1]
    maps=np.stack([np.linalg.pinv(X) for X in designs]);coefficients=maps[:,0,:]
    betas=np.einsum('jks,js->jk',maps,stats['values'])
    value=float(betas[:,0].sum());se=float(np.sqrt(np.sum(coefficients**2*stats['variances'])))
    block_components=[]
    for j,n in enumerate(stats['frequencies']):
        block_contrasts=coefficients[j]@stats['block_values'][:,stats['settings']==j]
        block_components.append(float(np.var(block_contrasts,ddof=1)/n))
    block_components=np.array(block_components);block_v=float(block_components.sum())
    den=float(np.sum(block_components**2/(stats['frequencies']-1)))
    df=block_v**2/den if den>0 else None
    interval=lambda error,q:[float(value-q*error),float(value+q*error)]
    setting_fits=[_gof(y,v,X) for y,v,X in zip(stats['values'],stats['variances'],designs,strict=True)]
    total_chi=sum(r['chi_square'] for r in setting_fits) if all(r['chi_square'] is not None for r in setting_fits) else None
    dof=9*(5-degree-1)
    result=dict(model='linear' if degree==1 else 'quadratic_actual_moments',value=value,
        weights_by_setting=coefficients.tolist(),parameters_by_setting=betas.tolist(),
        shot_se=se,shot_ci95=interval(se,norm.ppf(.975)),delta_from_classical=value-BOUND,
        signed_distance_from_bound_se=(value-BOUND)/se if se>0 else None,error_vs_ideal=value-6,
        paired_block_se=float(np.sqrt(block_v)),paired_block_welch_df=df,
        paired_block_ci95=interval(np.sqrt(block_v),t.ppf(.975,df) if df is not None else 0),
        setting_gof=dict(chi_square=total_chi,dof=dof,p=float(chi2.sf(total_chi,dof)) if total_chi is not None else None),
        dof_per_setting=5-degree-1)
    if resamples is not None:
        for kind,draws in resamples.items():
            b=np.sum(draws*coefficients[None,:,:],axis=(1,2))
            result[kind+'_bootstrap_se']=float(np.std(b,ddof=1))
            result[kind+'_bootstrap_ci95']=np.quantile(b,[.025,.975]).tolist()
    return result


def bootstrap(stats,*,draws=20000,seed=20260915):
    if type(draws) is not int or draws<200 or type(seed) is not int or seed<0:raise ValueError('Invalid bootstrap configuration')
    rng=np.random.default_rng(seed);shot=np.zeros((draws,9,5));paired=np.zeros_like(shot)
    for s in range(5):
        for b,j in enumerate(stats['settings']):
            for i in range(4):
                n=int(stats['shots'][s,b,i])
                counts=rng.multinomial(n,stats['probabilities'][s,b,i],size=draws)
                shot[:,j,s]+=(counts@stats['weights'][j])/n/(4*stats['frequencies'][j])
    for j,n in enumerate(stats['frequencies']):
        values=stats['block_values'][:,stats['settings']==j].T
        paired[:,j,:]=values[rng.integers(0,n,size=(draws,n))].mean(axis=1)
    return dict(shot=shot,paired_block=paired)


def summarize(stats,resamples,*,draws,seed):
    models=[fit_polynomial(stats,k,resamples) for k in (1,2)]
    rows=[]
    for s,scale in enumerate(run.SCALES):
        value=float(stats['values'][:,s].sum());se=float(np.sqrt(stats['variances'][:,s].sum()))
        rows.append(dict(scale=scale,value=value,shot_se=se,shot_ci95=[value-norm.ppf(.975)*se,value+norm.ppf(.975)*se]))
    return dict(schema_version=1,status='complete',scales=list(run.SCALES),classical_bound=BOUND,ideal_bell=6,
        total_shots=int(stats['shots'].sum()),setting_frequencies=stats['frequencies'].tolist(),
        scale_rows=rows,models=models,bell_linear_gof=bell_linear_gof(stats),
        linear_minus_quadratic=models[0]['value']-models[1]['value'],
        actual_scale_moments_by_setting=stats['moments'].tolist(),bootstrap=dict(draws=draws,seed=seed),
        assumptions=[
            'Independent identically distributed shots within each physical circuit variant; independent shot noise across circuits.',
            'Condition on saved settings and folding choices; average four variants, then blocks within each setting, then sum nine settings.',
            'Fractional variants differ; estimate each variance before aggregation. No pooled-multinomial assumption across different variants.',
            'Quadratic fits use measured actual-scale second moments per setting, not the square of the requested average scale.',
            'Actual CZ count ratio is a noise proxy; coherent/site-dependent errors and unscaled R/readout errors can violate either model.',
            'Paired block estimates preserve cross-scale covariance within setting but assume independent exchangeable blocks; shared job drift is not covered.',
            'Some settings have only two blocks; percentile block bootstrap underestimates variance at small sample size and is exploratory.',
            'CIs are individual approximate 95% statistical intervals, conditional on the chosen model; they do not bound extrapolation bias.',
            'Reported global linear test uses five Bell values and 3 residual degrees of freedom; setting tests assess all nine response curves separately.',
            'Five scales allow model checking but do not identify a model-independent noiseless value.',
        ])


def write_report(output,result,stats):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    write_json(output/'analysis.json',result)
    linear,quadratic=result['models'];rows=result['scale_rows']
    fig,ax=plt.subplots(figsize=(9.5,5.6),constrained_layout=True)
    x=np.linspace(0,5.2,250)
    ax.axvspan(0,1,color='#edf1f5',label='Obszar ekstrapolacji')
    ax.errorbar(run.SCALES,[r['value'] for r in rows],yerr=[norm.ppf(.975)*r['shot_se'] for r in rows],fmt='o',color='#1b2936',capsize=4,label='IQM DD: 95% CI zliczeń',zorder=5)
    for model,color,label in [(linear,'#08758a','Liniowy'),(quadratic,'#c2562b','Kwadratowy (rzeczywiste momenty)')]:
        parameters=np.sum(model['parameters_by_setting'],axis=0)
        ax.plot(x,np.polynomial.polynomial.polyval(x,parameters),color=color,label=label)
        ax.errorbar([0],[model['value']],yerr=[norm.ppf(.975)*model['shot_se']],fmt='s',color=color,capsize=4)
    ax.axhline(BOUND,color='#75528c',ls='--',label=f'Granica klasyczna: {BOUND:.5f}')
    ax.axhline(6,color='#87939e',ls=':',label='Idealny obwód: 6')
    ax.set(xlabel='Skala foldingu CZ',ylabel='Wartość Bella',title='DD + ZNE: 5 przeplatanych skal, 250 000 shotów',xlim=(-.2,5.3))
    ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False);ax.legend(fontsize=9,loc='best')
    fig.savefig(output/'zne_comparison.png',dpi=180);fig.savefig(output/'zne_comparison.pdf');plt.close(fig)
    lines=['# IQM: DD + ZNE z pięcioma przeplatanymi skalami','',
        'Skale: **1, 1.5, 2, 3, 5**. DD: STANDARD_DD_STRATEGY IQM. Twirling i korekcja odczytu wyłączone. Zachowany harmonogram 50 losowań; 1000 shotów na losowanie i skalę. Łącznie **250 000 shotów**.','',
        '## Wyniki','',f'Granica klasyczna: **{BOUND:.9f}**. Ideał: **6**. Niepewności są warunkowe względem przyjętego modelu i założeń o niezależnych zliczeniach.','',
        '| Skala | Bell | SE (1σ) | 95% CI |','|---|---:|---:|---|']
    for row in rows:lines.append(f"| {row['scale']:g} | {row['value']:.6f} | {row['shot_se']:.6f} | [{row['shot_ci95'][0]:.6f}, {row['shot_ci95'][1]:.6f}] |")
    lines+=['','| Model | B(0) | SE (1σ) | 95% CI | Różnica od granicy |','|---|---:|---:|---|---:|']
    for model in result['models']:lines.append(f"| {model['model']} | {model['value']:.6f} | {model['shot_se']:.6f} | [{model['shot_ci95'][0]:.6f}, {model['shot_ci95'][1]:.6f}] | {model['delta_from_classical']:+.6f} |")
    p=result['bell_linear_gof']
    lines+=['',f"Test liniowości sumy Bella: χ² = {p['chi_square']}, df = {p['dof']}, **p = {p['p']}**. To test zgodności danych z prostą, nie prawdopodobieństwo prawdziwości modelu ani test samego naruszenia Bella.",'',
        f"Różnica ekstrapolacji liniowej i kwadratowej: **{result['linear_minus_quadratic']:+.6f}**. Jest to wrażliwość na model, nie oszacowany prawdziwy błąd systematyczny.",'',
        f"Dodatkowy test kwadratowego modelu wszystkich dziewięciu odpowiedzi ustawień: χ² = {quadratic['setting_gof']['chi_square']}, df = 18 (po 2 na ustawienie), p = {quadratic['setting_gof']['p']}. To silniejsze założenie niż test samej sumy Bella; tych p nie należy bezpośrednio porównywać jako rankingu modeli.",'',
        '## Przeplatanie i częściowy folding','',
        'Każde z 10 zadań zawiera pięć kolejnych losowań i wszystkie pięć skal. Dla każdego losowania i skali przygotowano cztery warianty po 250 shotów, następnie kolejność 100 obwodów w zadaniu przetasowano. Nie wykonywano wszystkich pomiarów jednej skali w osobnym zadaniu. To ogranicza powiązanie skali z czasem, ale nie usuwa dowolnego dryfu.','',
        'Każda oryginalna bramka CZ ma dokładnie zadaną średnią liczbę powtórzeń w czterech wariantach. Dodawane są pary CZ·CZ, bez zmiany bramek R i mapowania pomiarów. Przykład przy 5 CZ i skali 1.5: warianty mają skale 1.4, 1.4, 1.4, 1.8 (w permutowanej kolejności). Zapisujemy każdą rzeczywistą skalę. Idealne prawdopodobieństwa wszystkich transformacji sprawdzono lokalnie.','',
        'Dla kwadratowej odpowiedzi średnia z wariantów zależy od średniej λ², a nie tylko od (średniej λ)². Dlatego dopasowujemy odpowiedzi ustawień z macierzą [1, E(λ), E(λ²)] i sumujemy ich wyrazy wolne. Model liniowy pozostaje pierwotną nieważoną regresją pięciu wartości Bella. Krzywa kwadratowa na wykresie przedstawia odpowiedź modelu dla jednej dokładnej skali; punkty ułamkowe są mieszaninami wariantów.','',
        '## Niepewność i ograniczenia','',
        f"Bootstrap: {result['bootstrap']['draws']} replik, seed {result['bootstrap']['seed']}. Losowanie multinomialne osobno dla każdego fizycznego wariantu. Wariancję liczymy przed agregacją, nie traktujemy czterech różnych obwodów jako jednej próby o wspólnym rozkładzie.",'',
        'Estymator: średnia czterech wariantów, średnia bloków danego ustawienia, suma dziewięciu ustawień. Zdarzenia poza podprzestrzenią qutritu mają wagę zero, ale pozostają w mianowniku. Nie ma postselekcji. Liczności ustawień: '+str(result['setting_frequencies'])+'.','',
        'Dodatkowe przedziały z parowanych bloków zachowują korelacje między skalami danego losowania. Zakładają niezależne bloki, więc nie obejmują dryfu wspólnego dla całego zadania. Przy dwóch blokach w jednym ustawieniu bootstrap bloków jest diagnostyczny i może zaniżać wariancję. Pełne liczby i założenia znajdują się w analysis.json.','',
        'Folding CZ skaluje tylko część błędów. Liczby bramek R i pomiarów pozostają stałe, a liczba CZ jest jedynie przybliżeniem poziomu szumu. Przedział statystyczny nie obejmuje błędu ekstrapolacji. Powyższe wartości same nie stanowią testu Bella zamykającego luki eksperymentalne.','']
    if 'comparison' in result:
        lines+=['## Porównanie z poprzednim benchmarkiem','',
            f"Poprzednie liniowe ZNE (1,3,5, skale wykonywane seriami): {result['comparison']['previous_linear']:.6f}; nowe: {linear['value']:.6f}; różnica: {result['comparison']['delta_linear']:+.6f}. To osobne wykonania sprzętowe i inny plan próbkowania; różnica nie izoluje efektu samego przeplatania.",'']
    lines+=['![Dane i modele](zne_comparison.png)','']
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')


def finish(directory,*,draws=20000,seed=20260915):
    directory=Path(directory);protocol,schedule,jobs,weights=run.load_prepared(directory)
    stats=statistics(run.read_samples(directory),weights)
    replicas=bootstrap(stats,draws=draws,seed=seed);result=summarize(stats,replicas,draws=draws,seed=seed)
    previous=directory.parent/'iqm_emerald_dd_zne_50x1000_20260914/analysis.json'
    if previous.exists():
        old=run.prior.read_json(previous)
        result['comparison']=dict(previous_linear=old['bell'],delta_linear=result['models'][0]['value']-old['bell'],previous_sha256=run.prior.sha256(previous))
    result['protocol']=protocol
    result['input_sha256']={str(p.relative_to(directory)):run.prior.sha256(p) for p in directory.rglob('*') if p.is_file() and (p.parent==directory/'hardware' or p.name in protocol['input_sha256'] or p.name in ('protocol.json','protocol.sha256'))}
    result['source_sha256']={str(Path(__file__).resolve()):run.prior.sha256(Path(__file__)),str(Path(run.__file__).resolve()):run.prior.sha256(Path(run.__file__))}
    output=directory/'analysis';write_report(output,result,stats)
    np.savez_compressed(output/'bootstrap.npz',**replicas)
    audit=dict(status='complete',total_shots=result['total_shots'],job_ids=[run.prior.read_json(directory/f'hardware/job_{i:03d}.submission.json')['job_id'] for i in range(10)],
        finished_at=run.now(),input_sha256=result['input_sha256'],output_sha256={p.name:run.prior.sha256(p) for p in output.iterdir() if p.is_file()})
    write_json(directory/'completion_audit.json',audit)
    return result
