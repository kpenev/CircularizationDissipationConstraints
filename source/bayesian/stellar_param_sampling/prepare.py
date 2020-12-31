#!/usr/bin/env python3
"""Pickle approximations to marginalized CDFs of stellar parameters."""

from functools import partial
from collections import namedtuple
import os.path
import sys

import matplotlib
from configargparse import ArgumentParser, DefaultsFormatter
from astropy import units as u
import numpy

from stellar_evolution.manager import StellarEvolutionManager

from star_sampler import StarSampler
from gaussian_log_likelihood import GaussianLogLikelihood
from continous_max_age import plot_max_age

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

    debug_plots = StarSampler.list_debug_plots()

    def parse_debug_plot(value_str):
        """Parse a debug plot argument (see help message for format)."""

        plot_type, filename = (entry.strip() for entry in value_str.split('='))
        assert plot_type in debug_plots
        return plot_type, filename

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
        '--stellar-evolution-interpolator-dir', '--interpolator-dir',
        default=(
            os.path.expanduser(
                '~/projects/git/poet/stellar_evolution_interpolators'
            )
        ),
        help='The directory to read stellar evolution interpolator from.'
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
        default=0.1,
        help='The largest change in CDF([Fe/H]) allowed between consecutive '
        '[Fe/H] grid points.'
    )
    parser.add_argument(
        '--feh-max-step',
        type=float,
        default=0.1,
        help='The largest difference allowed between neighboring [Fe/H] grid '
        'points.'
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
        '--mass-max-step',
        type=float,
        default=0.1,
        help='The largest difference allowed between neighboring mass grid '
        'points.'
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
        default=1e-4,
        help='The maximum absolute difference between 2-D interpolation of time'
        ' CDF(t|M, [Fe/H]) based on grid with 1/4 of the points (1/2 in each '
        'dimension) and the remaining 3/4 of the values. The mass-[Fe/H] grid '
        'resolution is increased until this criterion is '
        'satisfied.'
    )
    parser.add_argument(
        '--age-cdf-check-log-ages',
        type=lambda v: numpy.linspace(*v),
        default=numpy.linspace(-3, 1, 10),
        help='The log10 of the ages at which to check if the age CDF is well '
        'approximated by the current interpolation on mass and [Fe/H].'
    )
    parser.add_argument(
        '--mass-cdf-interp-tolerance',
        type=float,
        default=1e-4,
        help='The maximum absolute difference between 1-D interpolation of '
        'CDF(M|[Fe/H]) (marginalized over age) based on grid with 1/4 of the '
        'point and the calculated values at the remaining 3/4 of the points. '
        'The resolution of the mass-[Fe/H] grid is increased until this '
        'criterion is satisfied.'
    )
    parser.add_argument(
        '--grid-refine-algorithm',
        choices=['all', 'worst'],
        default='worst',
        help='Choose how interpolation grid for CDFs is going to be refined. If'
        " `'all'`, all indices in each dimension for which precision is not "
        "satisfied get new neighbors added. If `'worst'`, only the worst "
        'deviating mass and [Fe/H] entries get higher resolution at the next '
        'iteration.'
    )
    parser.add_argument(
        '--num-parallel-processes', '-p',
        type=int,
        default=4,
        help='The number of simultaneous processes to use for the calculations.'
    )
    parser.add_argument(
        '--debug-plot',
        action='append',
        type=parse_debug_plot,
        help=(
            'Enable another debugging plot. The format is '
            '<which_plot>=<filaneme>, with <which_plot> one of: '
            +
            ', '.join(debug_plots)
            +
            '. If <filename> is empty, the plot is displayed instead of saved.'
        )
    )
    parser.add_argument(
        '--debug-plot-dpi',
        type=int,
        default=300,
        help='The resoution to use for debugging plots.'
    )

    return parser.parse_args()

def main(config):
    """Avoid polluting the global namespace."""

#    numpy.set_printoptions(sys.maxsize)

    interpolator = StellarEvolutionManager(
        config.stellar_evolution_interpolator_dir
    ).get_interpolator_by_name(
        'default'
    )

#    plot_max_age(interpolator)

    matplotlib.rcParams['figure.dpi'] = config.debug_plot_dpi
    matplotlib.rcParams['figure.autolayout'] = True

    log_likelihood = GaussianLogLikelihood(
        mean=[5.0, 1.0, 0.0],
        covariance=[
            [0.50, 0.00, 0.00],
            [0.00, 0.20, 0.00],
            [0.00, 0.00, 0.50]
        ],
        interpolator=interpolator,
        rtol=config.time_ode_rtol,
        atol=config.time_ode_atol
    )

    plot_masses, plot_feh = numpy.meshgrid(
        numpy.linspace(0.98125, 0.98125, 1),
        numpy.linspace(-0.3, 0.3, 7)
    )
    log_likelihood.plot_age_cdf_integrand(
        mass=plot_masses.flatten() * u.M_sun,
        feh=plot_feh.flatten()
    )
#    sys.exit(1)

    StarSampler(log_likelihood, config)

if __name__ == '__main__':
    main(parse_configuration())
