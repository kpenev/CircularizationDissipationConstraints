"""Posterior CDF uncertainty from Dynesty run realizations.

The installed Dynesty 3.1.0 has no simulate_run(); its documented equivalent
is jitter_run(resample_run(results)). No likelihood function is called here.
"""

import numpy as np
from dynesty import utils as dyfunc

QUANTILE_LEVELS = np.array(
    [0.02275013194817921, 0.10, 0.15865525393145707, 0.16,
     0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.84,
     0.8413447460685429, 0.90, 0.9772498680518208],
    dtype=float,
)
TIDAL_PERIODS_DAYS = np.geomspace(1.0, 50.0, 30)


def evaluate_tidal_lgq(physical, names, tidal_periods=TIDAL_PERIODS_DAYS):
    """Evaluate the sampled tidal law, including its positive-alpha 50d cap.

    Rows are physical posterior samples; columns are tidal periods. This
    configuration has no spin-frequency law or inertial enhancement.
    """
    periods = np.asarray(tidal_periods, dtype=float)
    if periods.ndim != 1 or not np.all(np.isfinite(periods)) or np.any(periods <= 0):
        raise ValueError('Tidal periods must be a positive finite vector')
    q0 = physical[:, names.index('lgQ_min')]
    alpha = physical[:, names.index('lgQ_powerlaw')]
    pbreak = physical[:, names.index('lgQ_break_period')]
    if np.any(pbreak <= 0) or np.any(pbreak > 50):
        raise ValueError('The current tidal law requires 0 < Pbreak <= 50 days')
    if 'lgQ_inertial_boost' in names:
        if np.any(physical[:, names.index('lgQ_inertial_boost')] != 0):
            raise ValueError('Frequency-only Q diagnostic requires inertial boost zero')
    effective_periods = np.where(alpha[:, None] > 0,
                                 np.minimum(periods[None, :], 50.0),
                                 periods[None, :])
    return q0[:, None] + np.maximum(
        0.0, alpha[:, None] * np.log10(effective_periods / pbreak[:, None])
    )


def _normalized_weights(weights):
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError('Posterior weights must be a finite nonnegative vector')
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0:
        raise ValueError('Posterior weights must have positive finite sum')
    return weights / total


def weighted_cdf_at(locations, samples, weights):
    """Weighted empirical CDF at one or more fixed parameter locations."""
    locations = np.asarray(locations, dtype=float)
    samples = np.asarray(samples, dtype=float)
    weights = _normalized_weights(weights)
    if samples.ndim != 1 or samples.size != weights.size or not np.all(np.isfinite(samples)):
        raise ValueError('Samples and weights must be matching finite vectors')
    if not np.all(np.isfinite(locations)):
        raise ValueError('CDF locations must be finite')
    order = np.argsort(samples, kind='stable')
    cumulative = np.cumsum(weights[order])
    positions = np.searchsorted(samples[order], locations, side='right')
    return np.where(positions > 0, cumulative[np.maximum(positions - 1, 0)], 0.0)


def _simulate_combined(results, rng):
    """Combined path and prior-volume uncertainty, preserving sample indices."""
    if hasattr(dyfunc, 'simulate_run'):
        simulated, indices = dyfunc.simulate_run(results, rstate=rng, return_idx=True)
    else:
        bootstrapped, indices = dyfunc.resample_run(
            results, rstate=rng, return_idx=True
        )
        simulated = dyfunc.jitter_run(bootstrapped, rstate=rng)
    return simulated, np.asarray(indices, dtype=int)


def estimate_posterior_cdf_uncertainty(results, prior_transform,
                                       n_realizations=128,
                                       confidence=0.95,
                                       tolerance=0.01,
                                       seed=None,
                                       levels=QUANTILE_LEVELS):
    """Stop on max CDF standard deviation over physical parameters and Q(P).

    The 95th-percentile error is retained only as a diagnostic. Thresholds
    remain fixed across combined nested-sampling uncertainty realizations.
    """
    if n_realizations < 2:
        raise ValueError('CDF uncertainty requires at least two realizations')
    if not 0 < confidence < 1 or not 0 < tolerance < 1:
        raise ValueError('Confidence and tolerance must lie between 0 and 1')
    levels = np.asarray(levels, dtype=float)
    if levels.ndim != 1 or not np.all((levels > 0) & (levels < 1)):
        raise ValueError('CDF levels must be a vector of interior probabilities')
    samples = np.asarray(results.samples)
    if samples.ndim != 2 or len(samples) < 2:
        raise ValueError('At least two stored nested samples are needed')
    names = [name for name, _unit in prior_transform.parameter_order]
    physical = np.asarray([
        prior_transform(row)['parameters'] for row in samples
    ], dtype=float)
    if physical.shape != (len(samples), len(names)) or not np.all(np.isfinite(physical)):
        raise ValueError('Physical posterior samples must be a finite matrix')
    weights = _normalized_weights(results.importance_weights())
    if len(weights) != len(samples):
        raise ValueError('Posterior sample and weight counts differ')

    monitored = [i for i in range(physical.shape[1])
                 if not np.all(physical[:, i] == physical[0, i])]
    excluded_fixed = [names[i] for i in range(physical.shape[1]) if i not in monitored]
    if not monitored:
        raise ValueError('No varying physical parameter to monitor')
    tidal_lgq = evaluate_tidal_lgq(physical, names)
    observable_names = [names[i] for i in monitored] + [
        'lgQ_tidal_period_%0.8gd' % period for period in TIDAL_PERIODS_DAYS
    ]
    observable_metadata = [
        dict(observable_kind='physical_parameter',
             tidal_period_days=float('nan'), tidal_frequency_rad_per_day=float('nan'))
        for _ in monitored
    ] + [
        dict(observable_kind='tidal_lgQ', tidal_period_days=float(period),
             tidal_frequency_rad_per_day=float(2 * np.pi / period))
        for period in TIDAL_PERIODS_DAYS
    ]
    observables = np.column_stack((physical[:, monitored], tidal_lgq))
    nominal_quantiles = np.asarray([
        dyfunc.quantile(observables[:, i], levels, weights=weights)
        for i in range(observables.shape[1])
    ], dtype=float)
    nominal_cdfs = np.asarray([
        weighted_cdf_at(nominal_quantiles[k], observables[:, k], weights)
        for k in range(observables.shape[1])
    ])
    if not np.all(np.isfinite(nominal_quantiles)):
        raise ValueError('Nominal weighted quantiles are not finite')

    simulated_cdfs = np.empty((n_realizations, observables.shape[1], len(levels)))
    rng = np.random.default_rng(seed)
    for realization in range(n_realizations):
        simulated, indices = _simulate_combined(results, rng)
        if not np.array_equal(np.asarray(simulated.samples), samples[indices]):
            raise ValueError('Dynesty simulation sample indices do not match')
        sim_weights = _normalized_weights(simulated.importance_weights())
        if len(sim_weights) != len(indices):
            raise ValueError('Simulated sample and weight counts differ')
        for k in range(observables.shape[1]):
            simulated_cdfs[realization, k] = weighted_cdf_at(
                nominal_quantiles[k], observables[indices, k], sim_weights
            )

    absolute_errors = np.abs(simulated_cdfs - levels[None, None, :])
    upper_errors = np.quantile(absolute_errors, confidence, axis=0)
    means = np.mean(simulated_cdfs, axis=0)
    stds = np.std(simulated_cdfs, axis=0, ddof=1)
    if not np.all(np.isfinite(stds)):
        raise ValueError('CDF standard deviations must be finite')
    worst_k, worst_level = np.unravel_index(np.argmax(stds), stds.shape)
    details = []
    for k in range(observables.shape[1]):
        for level_index, level in enumerate(levels):
            details.append(dict(
                parameter=observable_names[k],
                **observable_metadata[k],
                probability=float(level),
                nominal_physical_quantile=float(nominal_quantiles[k, level_index]),
                nominal_empirical_cdf=float(nominal_cdfs[k, level_index]),
                simulated_cdf_mean=float(means[k, level_index]),
                simulated_cdf_std=float(stds[k, level_index]),
                p95_absolute_cdf_error=float(upper_errors[k, level_index]),
            ))
    statistic = dict(
        nreal=int(n_realizations),
        confidence=float(confidence),
        tolerance=float(tolerance),
        stopping_statistic='max_cdf_standard_deviation',
        cdf_std_ddof=1,
        global_sigma_cdf=float(stds[worst_k, worst_level]),
        global_p95_absolute_cdf_error=float(np.max(upper_errors)),
        worst_parameter=observable_names[worst_k],
        worst_tidal_period_days=observable_metadata[worst_k]['tidal_period_days'],
        worst_tidal_frequency_rad_per_day=(
            observable_metadata[worst_k]['tidal_frequency_rad_per_day']
        ),
        worst_probability=float(levels[worst_level]),
        converged=bool(stds[worst_k, worst_level] <= tolerance),
        excluded_fixed_parameters=excluded_fixed,
        monitored_parameter_count=len(monitored),
        monitored_tidal_frequency_count=len(TIDAL_PERIODS_DAYS),
        simulation_method='simulate_run' if hasattr(dyfunc, 'simulate_run')
                          else 'jitter_run(resample_run(results))',
    )
    return statistic, details
