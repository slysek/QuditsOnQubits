"""Statistical tests for interleaved fractional-noise mixtures."""
import numpy as np
import pytest
from scripts import iqm_bell_interleaved_analysis as analysis
from scripts import iqm_bell_interleaved_zne as run
from tests.test_iqm_bell_interleaved_zne import parents,schedule


def synthetic_samples():
    plan=run.make_plan(parents(),schedule(),seed=21,check_ideal=False)
    samples=[]
    for job in plan:
        for r in job['records']:
            x=r['actual_scale'];prob=.7-.025*x-.003*x*x;n=1000000
            ones=round(n*prob)
            samples.append(dict(**r,shots=n,counts={'0000':ones,'1111':n-ones}))
    weights=np.zeros((9,16));weights[:,0]=1
    return samples,weights


@pytest.fixture(scope='module')
def data():return synthetic_samples()


def test_actual_moments_recover_quadratic_zero_intercept(data):
    samples,weights=data;stats=analysis.statistics(samples,weights)
    row=analysis.fit_polynomial(stats,2)
    assert row['value']==pytest.approx(6.3,abs=1e-5)
    assert row['setting_gof']['dof']==18
    assert row['setting_gof']['p']>.99
    # Treating the fractional mixtures as exact nominal scales introduces bias.
    nominal=np.polynomial.polynomial.polyfit(run.SCALES,stats['values'].sum(axis=0),2)[0]
    assert abs(nominal-6.3)>1e-5
    linear=analysis.fit_polynomial(stats,1)
    expected=np.polynomial.polynomial.polyfit(run.SCALES,stats['values'].sum(axis=0),1)[0]
    assert linear['value']==pytest.approx(expected)
    assert analysis.bell_linear_gof(stats)['p']<.05


def test_variance_keeps_different_folding_variants_separate(data):
    samples,weights=data;stats=analysis.statistics(samples,weights)
    j=0;s=1;chosen=[r for r in samples if r['setting']==j and r['scale']==1.5]
    nblocks=len(chosen)//4
    variance=sum((r['counts']['0000']/r['shots'])*(1-r['counts']['0000']/r['shots'])/(r['shots']-1) for r in chosen)/(16*nblocks**2)
    assert stats['variances'][j,s]==pytest.approx(variance)
    row=analysis.fit_polynomial(stats,1)
    c=np.linalg.pinv(np.vander(run.SCALES,2,increasing=True))[0]
    assert row['shot_se']**2==pytest.approx(np.sum(stats['variances']*c*c))


def test_missing_duplicate_and_invalid_records_rejected(data):
    samples,weights=data
    for damaged in (samples[:-1],samples+[samples[0]]):
        with pytest.raises(ValueError):analysis.statistics(damaged,weights)
    damaged=[dict(r) for r in samples];damaged[0]['actual_scale']+=.2
    with pytest.raises(ValueError,match='scale'):analysis.statistics(damaged,weights)
    damaged=[dict(r) for r in samples];damaged[0]['counts']={'0000':-1}
    with pytest.raises(ValueError):analysis.statistics(damaged,weights)


def test_bootstrap_reproducibility_and_output(data,tmp_path):
    samples,weights=data;stats=analysis.statistics(samples,weights)
    a=analysis.bootstrap(stats,draws=300,seed=8);b=analysis.bootstrap(stats,draws=300,seed=8)
    for key in a:np.testing.assert_array_equal(a[key],b[key])
    result=analysis.summarize(stats,a,draws=300,seed=8)
    assert result['models'][1]['shot_bootstrap_ci95'][0]<6.3<result['models'][1]['shot_bootstrap_ci95'][1]
    analysis.write_report(tmp_path,result,stats)
    assert (tmp_path/'report.md').stat().st_size>1000
    assert (tmp_path/'zne_comparison.png').stat().st_size>10000
