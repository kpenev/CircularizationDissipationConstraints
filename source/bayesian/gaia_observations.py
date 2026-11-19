"""Gaia observations; preserve source IDs as strings and supplied P-e covariance."""
import csv
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def symmetric_sigma(row, prefix, suffix):
    errors = [float(row[f'{prefix}_error_{side}_{suffix}']) for side in ('minus', 'plus')]
    if not all(math.isfinite(x) and x > 0 for x in errors):
        raise ValueError(f'Invalid uncertainties for {prefix}')
    return sum(errors)/2


def load_systems():
    filename = Path(os.environ.get('GAIA_SYSTEMS_CSV', ROOT/'data/gaia_systems.csv'))
    selection = ROOT/'data/selected_systems.csv'
    with selection.open() as stream:
        selected = {r['system_id'] for r in csv.DictReader(stream) if r['catalog'] == 'Gaia'}
    with filename.open() as stream:
        rows = list(csv.DictReader(stream))
    result = {}
    for row in rows:
        source_id = row['gaia_dr3_source_id']
        if source_id not in selected:
            continue
        if source_id in result or not source_id.isdecimal() or source_id == '1857469799576340864':
            raise ValueError(f'Duplicate or excluded Gaia source: {source_id}')
        means = {key: float(row[column]) for key, column in (
            ('mass1', 'M1_msun'), ('mass2', 'M2_msun'), ('age', 'age_gyr'),
            ('feh', 'feh_dex'), ('period', 'orbital_period_days'), ('eccentricity', 'eccentricity'))}
        sigmas = dict(mass1=symmetric_sigma(row,'M1','msun'),
                      mass2=symmetric_sigma(row,'M2','msun'),
                      age=symmetric_sigma(row,'age','gyr'),
                      feh=symmetric_sigma(row,'feh','dex'),
                      period=float(row['orbital_period_error_days']),
                      eccentricity=float(row['eccentricity_error']))
        rho = float(row['period_eccentricity_correlation'])
        covariance = float(row['period_eccentricity_covariance_days'])
        if not (all(math.isfinite(x) for x in means.values())
                and all(math.isfinite(x) and x > 0 for x in sigmas.values())
                and math.isfinite(rho) and abs(rho) < 1):
            raise ValueError(f'Invalid Gaussian data for {source_id}')
        if not math.isclose(covariance,rho*sigmas['period']*sigmas['eccentricity'],rel_tol=1e-6,abs_tol=1e-15):
            raise ValueError(f'Inconsistent P-e covariance for {source_id}')
        result[source_id] = dict(source_id=source_id, host=row['host_name'],
                                 means=means, sigmas=sigmas, rho=rho,
                                 covariance_days=covariance)
    if set(result) != selected:
        raise ValueError(f'Missing Gaia systems: {sorted(selected-set(result))}')
    return result


def get_system(source_id):
    return load_systems()[str(source_id)]


def conditional_eccentricity(observation, period):
    """Mean and sigma of the Gaussian e|P; P itself follows its Gaussian marginal."""
    m, s, rho = observation['means'], observation['sigmas'], observation['rho']
    return (m['eccentricity']+rho*s['eccentricity']/s['period']*(period-m['period']),
            s['eccentricity']*math.sqrt(1-rho*rho))


def eccentricity_distribution(mean, sigma):
    """Physical Gaussian likelihood, with negligible tails cut for legacy support logic.

    The inherited likelihood uses finite support endpoints to choose its
    eccentricity-map approximation. Retain +/-8 sigma (omitted Gaussian mass
    < 1.3e-15) and intersect with 0 <= e <= 1; normalize on that interval.
    """
    from scipy import stats
    lower, upper = max(0., mean-8*sigma), min(1., mean+8*sigma)
    if not lower < upper:
        raise ValueError('Negligible conditional eccentricity probability in [0,1]')
    return stats.truncnorm((lower-mean)/sigma,(upper-mean)/sigma,loc=mean,scale=sigma)
