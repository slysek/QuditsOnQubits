from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares, minimize_scalar
from scipy.stats import t

from scripts.iqm_bell_short import analyze, count_vector, folding_block_size


MODELS = ('linear', 'quadratic', 'exponential', 'exponential_offset')
X = np.array([1., 2., 3.])


def predict(model, parameters, x):
    x = np.asarray(x, dtype=float)
    if model in ('linear', 'quadratic'):
        return np.polynomial.polynomial.polyval(x, parameters)
    if model == 'exponential':
        amplitude, rate = parameters
        return amplitude * np.exp(-rate * x)
    if model == 'exponential_offset':
        offset, amplitude, rate = parameters
        return offset + amplitude * np.exp(-rate * x)
    raise ValueError(f'Unsupported model: {model}')


def fit_offset_exponential(x, y):
    if np.ptp(y) < 1e-12:
        raise ValueError('Offset exponential rate is not identifiable for constant data')
    span = float(x[-1] - x[0])
    unit_x = (x - x[0]) / span

    def projected(rate):
        design = np.column_stack((np.ones(len(x)), np.exp(-rate * unit_x)))
        parameters = np.linalg.lstsq(design, y, rcond=None)[0]
        return float(np.sum((design @ parameters - y) ** 2)), parameters

    candidates = [minimize_scalar(lambda r: projected(r)[0], bounds=bounds,
        method='bounded', options={'xatol': 1e-10}) for bounds in ((-20., -1e-6), (1e-6, 20.))]
    fit = min(candidates, key=lambda candidate: candidate.fun)
    if not fit.success or abs(fit.x) < 1e-4 or abs(fit.x) > 19.99:
        raise ValueError('Offset exponential is singular or reaches the numerical rate boundary')
    _, (offset, amplitude) = projected(fit.x)
    rate = fit.x / span
    return np.array([offset, amplitude * np.exp(rate * x[0]), rate])


def fit_model(model, y, *, scales=X):
    y = np.asarray(y, dtype=float)
    x = np.asarray(scales, dtype=float)
    if (x.ndim != 1 or len(x) < 3 or y.shape != x.shape or not np.isfinite(y).all()
        or not np.isfinite(x).all() or np.any(np.diff(x) <= 0) or x[0] <= 0):
        raise ValueError('Expected finite data at at least three distinct increasing positive scales')
    if model in ('linear', 'quadratic'):
        degree = 1 if model == 'linear' else 2
        return np.polynomial.polynomial.polyfit(x, y, degree)
    if model == 'exponential_offset':
        if len(x) != 3 or not np.allclose(np.diff(x), x[1] - x[0]):
            return fit_offset_exponential(x, y)
        first, second = np.diff(y)
        if abs(first) < 1e-12 or abs(second) < 1e-12:
            raise ValueError('Offset exponential is not identifiable for flat differences')
        ratio = second / first
        if ratio <= 0 or np.isclose(ratio, 1., rtol=1e-8, atol=1e-12):
            raise ValueError('No identifiable real offset exponential for these differences')
        rate = -np.log(ratio) / (x[1] - x[0])
        at_first_scale = first / (ratio - 1)
        return np.array([y[0] - at_first_scale, at_first_scale * np.exp(rate * x[0]), rate])
    if model != 'exponential':
        raise ValueError(f'Unsupported model: {model}')
    if np.any(y <= 0):
        raise ValueError('Positive-amplitude exponential requires positive data')
    slope, intercept = np.polyfit(x, np.log(y), 1)
    with np.errstate(over='raise', invalid='raise'):
        fit = least_squares(lambda p: predict(model, p, x) - y,
            [np.exp(intercept), -slope],
            jac=lambda p: np.column_stack((np.exp(-p[1] * x), -p[0] * x * np.exp(-p[1] * x))),
            ftol=1e-11, xtol=1e-11, gtol=1e-11, max_nfev=200)
    if not fit.success or not np.isfinite(fit.x).all() or fit.x[0] <= 0:
        raise ValueError('Exponential fit did not converge')
    return fit.x


def covariance_floor(empirical, floor):
    empirical, floor = np.asarray(empirical, float), np.asarray(floor, float)
    if (empirical.ndim != 2 or empirical.shape[0] != empirical.shape[1] or floor.shape != empirical.shape
        or not np.isfinite(empirical).all() or not np.isfinite(floor).all()
        or not np.allclose(empirical, empirical.T)
        or not np.allclose(floor, np.diag(np.diag(floor))) or np.any(np.diag(floor) < 0)):
        raise ValueError('Expected a symmetric covariance and nonnegative diagonal shot floor')
    diagonal = np.sqrt(np.maximum(np.diag(floor), 1e-15))
    values, vectors = np.linalg.eigh(empirical / np.outer(diagonal, diagonal))
    return np.outer(diagonal, diagonal) * ((vectors * np.maximum(values, 1.)) @ vectors.T)


def campaign_statistics(samples, weights, *, scales=(1, 2, 3)):
    analyze(samples, weights, scales=scales)
    block_size = folding_block_size(scales)
    count = len(scales)
    repeats = max(s['repeat'] for s in samples) + 1
    result = {}
    for arm in ('DD', 'TWIRLING'):
        values = np.zeros((repeats, count))
        shot_variance = np.zeros(count)
        for sample in samples:
            if sample['arm'] != arm:
                continue
            index = list(scales).index(sample['scale'])
            p = count_vector(sample['counts'], sample['shots'])
            w = weights[sample['setting']]
            value = float(w @ p)
            values[sample['repeat'], index] += value
            shot_variance[index] += max(0., float(w * w @ p - value * value)) / sample['shots']
        result[arm] = dict(blocks=values.reshape(-1, block_size, count).mean(axis=1),
                           shot_covariance=np.diag(shot_variance / repeats ** 2), scales=list(scales),
                           repeats_per_block=block_size)
    return result


def compare_statistics(statistics, *, draws=2000, seed=20260910):
    if type(draws) is not int or draws < 100:
        raise ValueError('Use at least 100 uncertainty draws')
    rng = np.random.default_rng(seed)
    rows, data, curves = [], {}, {}
    grid = np.linspace(0., 3.2, 161)
    for arm, values in statistics.items():
        blocks = np.asarray(values['blocks'], dtype=float)
        x = np.asarray(values.get('scales', X), dtype=float)
        if blocks.ndim != 2 or blocks.shape[1] != len(x) or len(blocks) < 4 or not np.isfinite(blocks).all():
            raise ValueError('Expected at least four finite repeat blocks at the specified scales')
        fit_model('linear', blocks.mean(axis=0), scales=x)
        mean = blocks.mean(axis=0)
        covariance = covariance_floor(np.cov(blocks, rowvar=False) / len(blocks), values['shot_covariance'])
        df = len(blocks) - 1
        errors = np.sqrt(np.diag(covariance))
        data[arm] = dict(scales=x.tolist(), values=mean.tolist(), standard_errors=errors.tolist(),
                         ci95_half_width=(t.ppf(.975, df) * errors).tolist(), covariance=covariance.tolist(),
                         repeat_blocks=len(blocks), repeats_per_block=values.get('repeats_per_block', 2))
        fluctuations = rng.standard_normal((draws, len(x))) @ np.linalg.cholesky(covariance).T
        simulated = mean + fluctuations / np.sqrt(rng.chisquare(df, draws) / df)[:, None]
        curves[arm] = {}
        for model in MODELS:
            count = 2 if model in ('linear', 'exponential') else 3
            row = dict(arm=arm, model=model, parameters=None, bell_zero=None, ci95_low=None, ci95_high=None,
                       rss=None, residual_dof=len(x)-count, failed_draws=None, status='not_identifiable',
                       decaying_to_finite_offset=None)
            try:
                parameters = fit_model(model, mean, scales=x)
            except (ValueError, FloatingPointError, RuntimeError):
                rows.append(row)
                continue
            row.update(parameters=parameters.tolist(), bell_zero=float(predict(model, parameters, 0.)),
                       rss=float(np.sum((predict(model, parameters, x) - mean) ** 2)),
                       status='interpolating' if count == len(x) else 'ok')
            if model == 'exponential_offset':
                row['decaying_to_finite_offset'] = bool(parameters[2] > 0)
            curves[arm][model] = predict(model, parameters, grid).tolist()
            zero = []
            for draw in simulated:
                try:
                    fitted = fit_model(model, draw, scales=x)
                    value = float(predict(model, fitted, 0.))
                    if np.isfinite(value):
                        zero.append(value)
                except (ValueError, FloatingPointError, RuntimeError):
                    pass
            row['failed_draws'] = draws - len(zero)
            if row['failed_draws']:
                row['status'] = 'unstable'
            else:
                row['ci95_low'], row['ci95_high'] = np.quantile(zero, [.025, .975]).tolist()
            rows.append(row)
    return dict(rows=rows, data=data, grid=grid.tolist(), curves=curves, draws=draws, seed=seed,
        method='Ordinary least squares on Bell values; approximate 95% intervals from joint multivariate Student-t propagation across interleaved repeat blocks, with a covariance floor from shot noise. Intervals exclude model bias. No interval is reported when any uncertainty draw cannot be fitted.',
        warnings=['A model with as many parameters as scales interpolates and cannot be validated by its residuals.',
                  'An exponential with a free offset may have no finite large-scale asymptote.',
                  'Offset exponential fits with more than three points search rates with abs(k) * (max_scale - min_scale) <= 20 and reject singular or boundary solutions.',
                  'Intermediate scales average partial folds; nonlinear noise scaling is a model assumption.'],
        unidentified_polynomials=dict(minimum_degree=len(x),
            reason=f'With {len(x)} scales, polynomials of degree >= {len(x)} are not uniquely determined; adding a * product(x - scale) leaves measured points unchanged.'))


def compare_models(samples, weights, *, draws=2000, seed=20260910, scales=(1, 2, 3)):
    return compare_statistics(campaign_statistics(samples, weights, scales=scales), draws=draws, seed=seed)


def plot_comparison(result):
    import matplotlib.pyplot as plt
    labels = dict(linear='Liniowa', quadratic='Kwadratowa', exponential='Exp', exponential_offset='Exp + offset')
    figure, axes = plt.subplots(1, len(result['data']), figsize=(12, 4), squeeze=False)
    for ax, (arm, data) in zip(axes[0], result['data'].items()):
        ax.axvspan(0, 1, color='grey', alpha=.08)
        ax.errorbar(data['scales'], data['values'], yerr=data['ci95_half_width'], fmt='ko', capsize=4, label='Dane: 95%')
        for model, values in result['curves'][arm].items():
            line, = ax.plot(result['grid'], values, label=labels[model], linestyle='--' if model == 'exponential_offset' else '-')
            ax.plot(0, values[0], 'o', color=line.get_color())
        ax.axhline(6, color='grey', linestyle=':', label='Teoria')
        ax.set(xlabel='Skala CZ', ylabel='Bell', title=arm, xlim=(-.05, 3.2))
        ax.legend(fontsize=8)
    figure.tight_layout()
    return figure
