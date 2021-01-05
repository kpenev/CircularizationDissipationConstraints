#!/usr/bin/env python3
"""Pickle approximations to marginalized CDFs of stellar parameters."""

from functools import partial
from collections import namedtuple
import os.path

import matplotlib
from matplotlib import pyplot
from configargparse import ArgumentParser, DefaultsFormatter
from astropy import units as u
import numpy
from numpy.random import rand

from stellar_evolution.manager import StellarEvolutionManager
from split_normal_distribution import split_normal

from star_sampler import StarSampler
from gaussian_log_likelihood import GaussianLogLikelihood

RandomQuantity = namedtuple('RandomQuantity', ['distribution', 'units'])

#TODO: make parsed command line match exactly system.
def parse_configuration():
    """Return the configuration to use per the command line."""

    def parse_quantity_with_errors(value_str, units=None):
        """Parse a string like 5.0 +- 1.3 or 5.0 +0.2 -0.8 (space optional)."""

        value_str, error_str = value_str.rsplit('+', 1)
        plus_error_str, minus_error_str = error_str.rsplit('-', 1)

        distribution = split_normal.freeze_error_bar(
            mode=float(value_str),
            abs_plus_error=float(plus_error_str or minus_error_str),
            abs_minus_error=float(minus_error_str)
        )
        return (
            distribution if units is None
            else RandomQuantity(distribution, units)
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
        type=parse_quantity_with_errors,
        help='The measured [Fe/H] for the star as well as its estimated '
        'standard deviation(s), possibly asymmetric.'
    )
    parser.add_argument(
        '--logg',
        type=parse_quantity_with_errors,
        help='If known, the masured value of log10(g) at the surface of the '
        'star as well as its estimated standard deviation(s), possibly '
        'asymmetric.'
    )
    parser.add_argument(
        '--Teff',
        type=partial(parse_quantity_with_errors, units=u.K),
        help='If known, the masured value of the effective temperature of the '
        'star, in Kelvin, as well as its estimated standard deviation(s), '
        'possibly asymmetric.'
    )
    parser.add_argument(
        '--mean-density', '--density', '--rho',
        type=partial(parse_quantity_with_errors, units=u.g / u.cm**3),
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

    interpolator = StellarEvolutionManager(
        config.stellar_evolution_interpolator_dir
    ).get_interpolator_by_name(
        'default'
    )

    matplotlib.rcParams['figure.dpi'] = config.debug_plot_dpi
    matplotlib.rcParams['figure.autolayout'] = True

    GaussianLogLikelihood.set_interpolator(interpolator)

    mean = numpy.array([5.0, 1.0, 0.0])
    covariance = numpy.array(
        [
            [+0.200, +0.010, -0.007],
            [+0.010, +0.040, +0.030],
            [-0.007, +0.030, +0.200]
        ]
    )

    log_likelihood = GaussianLogLikelihood(
        mean=mean,
        covariance=covariance,
        rtol=config.time_ode_rtol,
        atol=config.time_ode_atol,
        max_step=config.time_ode_max_step
    )

    plot_masses, plot_feh = numpy.meshgrid(
        numpy.linspace(0.98125, 0.98125, 1),
        numpy.linspace(-0.3, 0.3, 7)
    )
#    log_likelihood.plot_age_cdf_integrand(
#        mass=plot_masses.flatten() * u.M_sun,
#        feh=plot_feh.flatten()
#    )

    star_sampler = StarSampler(log_likelihood, config)
    samples = numpy.empty((10000, 3), dtype=numpy.float64)
    for i in range(samples.shape[0]):
        if i % 100 == 0:
            print(repr(i) + '/' + repr(samples.shape[0]), end='\r')
        samples[i] = star_sampler(rand(3))

    cdf_x = numpy.empty((2*samples.shape[0],),
                        dtype=samples.dtype)

    cdf_y = numpy.empty(cdf_x.shape, dtype=samples.dtype)
    cdf_y[::2] = (numpy.arange(samples.shape[0], dtype=numpy.float64)
                  /
                  samples.shape[0])
    cdf_y[1::2] = cdf_y[::2] + 1.0 / samples.shape[0]

    for var_index in range(3):
        cdf_x[::2] = numpy.sort(samples[:, var_index])
        cdf_x[1::2] = cdf_x[::2]

        pyplot.subplot(2, 3, var_index + 1)
        pyplot.plot(cdf_x, cdf_y, '-k')
        pyplot.title(['[Fe/H]', r'$M_\star\ [M_\odot]$', 't [Gyr]'][var_index])

        pyplot.subplot(2, 3, var_index + 4)
        pyplot.hist(samples[:, var_index],
                    bins=samples.shape[0] // 100,
                    density=True)
    pyplot.savefig('marginalized_test_samples.eps')

if __name__ == '__main__':
    main(parse_configuration())
