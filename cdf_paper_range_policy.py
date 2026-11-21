"""Paper-defined tidal-Q CDF stopping policy, installed before sampler import.

Outside source/ to preserve the fingerprint used for worker search recovery.
Penev & Schussler 2022, section 4.1, printed p6154: select tidal periods
where the 97.7th-percentile lgQ is within 0.7 dex of its minimum. Check all
four diagnostic quantiles in the interval bounded by first/last selected
periods. Physical-parameter CDFs remain diagnostics only.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

POLICY = 'tidal_Q_cdf_std_paper_range_v1'
LEVELS = np.array([0.02275013194817921, 0.15865525393145707,
                   0.8413447460685429, 0.9772498680518208])
RANGE_DEX = 0.7


def apply_paper_range(statistic, details, tolerance):
    """Replace the stopping decision; preserve full-grid diagnostics."""
    tidal = [row for row in details if row['observable_kind'] == 'tidal_lgQ']
    tail = sorted(
        [row for row in tidal if np.isclose(row['probability'], LEVELS[-1],
                                           rtol=0, atol=1e-12)],
        key=lambda row: row['tidal_period_days'])
    if not tail:
        raise ValueError('No 97.7th-percentile curve for paper range selection')
    periods = np.asarray([row['tidal_period_days'] for row in tail])
    values = np.asarray([row['nominal_physical_quantile'] for row in tail])
    if (not np.all(np.isfinite(values)) or not np.all(np.isfinite(periods))
            or np.any(np.diff(periods) <= 0)):
        raise ValueError('Paper range requires finite quantiles on a unique ordered grid')
    cutoff = float(values.min() + RANGE_DEX)
    selected = values <= cutoff
    lower, upper = float(periods[selected][0]), float(periods[selected][-1])
    monitored = []
    for row in details:
        correct_level = bool(np.any(np.isclose(row['probability'], LEVELS,
                                               rtol=0, atol=1e-12)))
        used = (row['observable_kind'] == 'tidal_lgQ' and correct_level
                and lower <= row['tidal_period_days'] <= upper)
        row['used_for_stopping'] = bool(used)
        if used:
            monitored.append(row)
    expected = int(np.sum((periods >= lower) & (periods <= upper))) * len(LEVELS)
    if len(monitored) != expected:
        raise ValueError('Missing diagnostic quantiles inside informative period interval')
    sigmas = np.asarray([row['simulated_cdf_std'] for row in monitored])
    if not np.all(np.isfinite(sigmas)) or np.any(sigmas < 0):
        raise ValueError('CDF standard deviations must be finite and nonnegative')
    worst = monitored[int(np.argmax(sigmas))]
    result = dict(statistic)
    result.update(
        policy_version=POLICY,
        stopping_statistic='max_cdf_standard_deviation_paper_period_range',
        diagnostic_full_grid_max_sigma_cdf=statistic['global_sigma_cdf'],
        diagnostic_full_grid_max_p95_error=statistic['global_p95_absolute_cdf_error'],
        diagnostic_physical_parameter_count=statistic['monitored_parameter_count'],
        global_sigma_cdf=float(worst['simulated_cdf_std']),
        global_p95_absolute_cdf_error=max(float(row['p95_absolute_cdf_error']) for row in monitored),
        worst_parameter=worst['parameter'],
        worst_probability=float(worst['probability']),
        worst_tidal_period_days=float(worst['tidal_period_days']),
        worst_tidal_frequency_rad_per_day=float(worst['tidal_frequency_rad_per_day']),
        converged=bool(worst['simulated_cdf_std'] <= tolerance),
        monitored_parameter_count=0,
        monitored_tidal_frequency_count=expected // len(LEVELS),
        diagnostic_tidal_grid_count=len(periods),
        stopping_quantiles=LEVELS.tolist(),
        period_range_selection_quantile=float(LEVELS[-1]),
        period_range_threshold_dex=RANGE_DEX,
        period_range_minimum_upper_lgQ=float(values.min()),
        period_range_lgQ_cutoff=cutoff,
        informative_period_min_days=lower,
        informative_period_max_days=upper,
        exact_cutoff_selected_periods_days=periods[selected].tolist(),
        interior_above_cutoff_periods_days=periods[
            (periods >= lower) & (periods <= upper) & ~selected].tolist(),
        period_range_policy='closed interval between first and last selected grid periods',
        physical_parameters_used_for_stopping=False,
    )
    return result, details


def install(cdf):
    """Install before bayesian.nesty_sampling imports the estimator."""
    if getattr(cdf.estimate_posterior_cdf_uncertainty, '_paper_range_policy', False):
        return
    original = cdf.estimate_posterior_cdf_uncertainty
    cdf.TIDAL_PERIODS_DAYS = np.geomspace(1.0, 50.0, 100)
    cdf.evaluate_tidal_lgq.__defaults__ = (cdf.TIDAL_PERIODS_DAYS,)
    cdf.QUANTILE_LEVELS = LEVELS.copy()

    def estimate(results, prior_transform, n_realizations=128, confidence=.95,
                 tolerance=.01, seed=None, levels=None):
        if levels is not None and not np.array_equal(np.asarray(levels), LEVELS):
            raise ValueError('Paper range stopping policy requires its four diagnostic quantiles')
        statistic, details = original(
            results, prior_transform, n_realizations=n_realizations,
            confidence=confidence, tolerance=tolerance, seed=seed, levels=LEVELS)
        statistic, details = apply_paper_range(statistic, details, tolerance)
        print('CDF paper-range check: P=[%g,%g] days; max_sigma=%g; '
              'threshold=%g; converged=%s' % (
                  statistic['informative_period_min_days'],
                  statistic['informative_period_max_days'],
                  statistic['global_sigma_cdf'], tolerance, statistic['converged']), flush=True)
        # One durable record per system/check with the range and exact scope.
        diagnostic_directory = os.environ.get('FAILED_SOLUTIONS_DIR')
        if diagnostic_directory:
            path = Path(diagnostic_directory).parent / 'posterior_cdf_paper_range.jsonl'
            record = dict(statistic, recorded_at=datetime.now(timezone.utc).isoformat(),
                          slurm_job_id=os.environ.get('SLURM_JOB_ID'),
                          stored_samples=len(results.samples),
                          seed=seed)
            with path.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
        return statistic, details

    estimate._paper_range_policy = True
    cdf.estimate_posterior_cdf_uncertainty = estimate

