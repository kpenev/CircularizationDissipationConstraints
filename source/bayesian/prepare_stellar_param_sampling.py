#!/usr/bin/env python3
"""Pickle approximations to marginalized CDFs of stellar parameters."""

from functools import partial
import re
from collections import namedtuple

from matplotlib import pyplot
from configargparse import ArgumentParser, DefaultsFormatter
from astropy import units as u
import scipy
from scipy.stats import norm

#TODO: make parsed command line match exactly system.
def parse_configuration():
    """Return the configuration to use per the command line."""

    def parse_quanitty_with_errors(value_str, units=1):
        """Parse a string like 5.0 +- 1.3 or 5.0 +0.2 -0.8 (space optional)."""

        value_str, error_str = value_str.rsplit('+', 1)
        plus_error_str, minus_error_str = error_str.rsplit('-', 1)

        ValueWithErrors = namedtuple('ValueWithErrors', ['value',
                                                         'plus_error',
                                                         'minus_error'])
        return ValueWithErrors(
            value=float(value_str) * units,
            plus_error=float(plus_error_str or minus_error_str) * units,
            minus_error=float(minus_error_str) * units
        )


    parser = ArgumentParser(
        description=__doc__,
        default_config_files=[],
        formatter_class=DefaultsFormatter,
        ignore_unknown_config_file_keys=False
    )

    parser.add_argument(
        'pickle_fname',
        help='The filename to pickle the calculated results.'
    )

    parser.add_argument(
        '--feh',
        type=parse_quanitty_with_errors,
        help='The measured [Fe/H] for the star as well as its estimated '
        'standard deviation(s), possibly asymmetric. If using command line '
        'system parameters, this argument must be specified.'
    )
    parser.add_argument(
        '--logg',
        type=parse_quanitty_with_errors,
        help='If known, the masured value of log10(g) at the surface of the '
        'star as well as its estimated standard deviation(s), possibly '
        'asymmetric.'
    )
    parser.add_argument(
        '--Teff',
        type=partial(parse_quanitty_with_errors, units=u.K),
        help='If known, the masured value of the effective temperature of the '
        'star, in Kelvin, as well as its estimated standard deviation(s), '
        'possibly asymmetric.'
    )
    parser.add_argument(
        '--mean-density', '--density', '--rho',
        type=partial(parse_quanitty_with_errors, units=u.g / u.cm**3),
        help='If known, the mesured mean stellar density in g/cm3, as well as '
        'its estimated standard deviation(s), possibly asymmetric.'
    )
    parser.add_argument(
        '--feh-max-cdf-step',
        type=float,
        default=0.01,
        help='The largest change in CDF([Fe/H]) allowed between consecutive '
        '[Fe/H] grid points.'
    )
    parser.add_argument(
        '--feh-max-step',
        type=float,
        default=0.05,
        help='The largest difference between neighboring [Fe/H] grid points.'
    )
    parser.add_argument(
        '--min-feh-pdf',
        type=float,
        default=1e-8,
        help='The range of [Fe/H] values considered is such that the '
        'PDF([Fe/H])/max(PDF([Fe/H])) is bigger than the specified value. '
        'Basically, the probability density of [Fe/H] outside the resulting '
        'bounds is approximated as zero.'
    )
    parser.add_argument(
        '--time-ode-max-step',
        type=float,
        default=0.1,
        help='The maximum steps the ODE solver that calculates the integral '
        'over time is allowed to take.'
    )
    parser.add_argument(
        '--time-ode-rtol',
        type=float,
        default=1e-6,
        help='The relative tolerance to impose of the time integral ODE '
        'solution.'
    )
    parser.add_argument(
        '--time-ode-atol',
        type=float,
        default=1e-8,
        help='The absolute tolerance to impose of the time integral ODE '
        'solution.'
    )
    parser.add_argument(
        '--age-cdf-interp-tolerance',
        type=float,
        default=1e-6,
        help='The maximum absolute difference between 2-D interpolation of time'
        ' CDF(t|M, [Fe/H]) based on grid with 1/4 of the points and the '
        'remaining 3/4 of the values. The mass and metallicity grid resolution '
        'is increased until this criterion is satisfied.'
    )
    parser.add_argument(
        '--mass-cdf-interp-tolerance',
        type=float,
        default=1e-6,
        help='The maximum absolute difference between 1-D interpolation of '
        'CDF(M|[Fe/H]) (marginalized over age) based on grid with 1/2 of the '
        'point and the calculated values at the remaining 1/2 of the points. '
        'The resolution of the metallicity grid is increased until this '
        'criterion is satisfied.'
    )
    parser.add_argument(
        '--num-parallel-processes', '-p',
        type=int,
        default=4,
        help='The number of simultaneous processes to use for the calculations.'
    )

    return parser.parse_args()

def get_initial_feh_grid(configuration):
    """
    Find the crudest [Fe/H] grid satisfying config ignoring interp. tolerances.
    """

    scaled_range = scipy.sqrt(-2.0 * scipy.log(configuration.min_feh_pdf))
    min_feh = (configuration.feh.value
               -
               scaled_range * configuration.feh.minus_error)
    max_feh = (configuration.feh.value
               +
               scaled_range * configuration.feh.plus_error)
    lower_feh, upper_feh = [min_feh], [max_feh]

    upper_dist = norm(configuration.feh.value, configuration.feh.plus_error)
    lower_dist = norm(configuration.feh.value, configuration.feh.minus_error)

    while min_feh < max_feh:
        min_cdf = lower_dist.cdf(min_feh)
        max_sf = upper_dist.sf(max_feh)

        min_feh = min(
            min_feh + configuration.feh_max_step,
            lower_dist.ppf(min_cdf + configuration.feh_max_cdf_step)
        )
        max_feh = max(
            max_feh - configuration.feh_max_step,
            lower_dist.isf(max_sf + configuration.feh_max_cdf_step)
        )
        if min_feh < max_feh:
            lower_feh.append(min_feh)
            upper_feh.append(max_feh)

    upper_feh.reverse()
    return scipy.array(lower_feh + [configuration.feh.value] + upper_feh)


def main(configuration):
    """Avoid polluting the global namespace."""

    feh_grid = get_initial_feh_grid(configuration)
    pyplot.plot(feh_grid, '.')
    pyplot.axhline(configuration.feh.value)
    pyplot.axhline(configuration.feh.value + configuration.feh.plus_error)
    pyplot.axhline(configuration.feh.value - configuration.feh.minus_error)
    pyplot.show()

if __name__ == '__main__':
    main(parse_configuration())
