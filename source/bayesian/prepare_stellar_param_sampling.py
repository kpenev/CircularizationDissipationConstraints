#!/usr/bin/env python3
"""Pickle approximations to marginalized CDFs of stellar parameters."""

from functools import partial
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
        '--max-discarded-feh-probability',
        type=float,
        default=1e-8,
        help='The range of [Fe/H] values considered is such that sampled [Fe/H]'
        ' values have at most the given probability of landing outside the '
        'grid.'
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

def get_initial_feh_grid(config):
    """
    Find the crudest [Fe/H] grid satisfying config ignoring interp. tolerances.
    """

    def get_half_grid_offsets(side):
        """Return the grid poinst on one side (plus or minus) of the median."""

        assert side in ['plus', 'minus']

        distribution = norm(scale=getattr(config.feh, side + '_error'))

        offset = distribution.isf(config.max_discarded_feh_probability
                                  /
                                  2.0)
        result = []
        while offset > 0:
            result.append(offset)
            current_sf = distribution.sf(offset)
            offset = max(offset - config.feh_max_step,
                         distribution.isf(current_sf + config.feh_max_cdf_step))

        result = scipy.array(result)
        if side == 'minus':
            return -result
        else:
            return result[ : : -1]

    return scipy.concatenate((get_half_grid_offsets('minus'),
                              [0],
                              get_half_grid_offsets('plus'))) + config.feh.value

def plot_feh_grid(config, feh_grid):
    """Display plots showing [Fe/H] grid was correctly generated."""

    pyplot.plot(feh_grid, '.')
    x_range = scipy.array([0, feh_grid.size - 1])
    pyplot.plot(x_range,
                x_range * config.feh_max_step + feh_grid[0],
                '-')
    pyplot.plot(x_range[::-1],
                -x_range * config.feh_max_step + feh_grid[-1],
                '-')
    pyplot.axhline(config.feh.value)
    pyplot.axhline(config.feh.value + config.feh.plus_error)
    pyplot.axhline(config.feh.value - config.feh.minus_error)
    pyplot.show()

    scaled_feh_diff = feh_grid - config.feh.value
    scaled_feh_diff[scaled_feh_diff > 0] /= config.feh.plus_error
    scaled_feh_diff[scaled_feh_diff < 0] /= config.feh.minus_error
    feh_cdf = norm.cdf(scaled_feh_diff)
    x_med = scipy.argmin(scipy.fabs(feh_grid - config.feh.value))
    pyplot.plot(feh_cdf, '.')
    pyplot.plot(x_range,
                0.5 + (x_range - x_med) * config.feh_max_cdf_step,
                '-')
    pyplot.show()

def main(config):
    """Avoid polluting the global namespace."""

    feh_grid = get_initial_feh_grid(config)
    plot_feh_grid(config, feh_grid)

if __name__ == '__main__':
    main(parse_configuration())
