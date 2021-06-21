"""Useful functions for confiuring the stellar parameter sampling."""

import os.path

#False positive
#pylint: disable=import-error
from star_sampler import StarSampler
#pylint: enable=import-error

def add_star_sampler_config_args(parser):
    """Add arguments to parser specifying the sampler configuration."""

    debug_plots = StarSampler.list_debug_plots()
    def parse_debug_plot(value_str):
        """Parse a debug plot argument (see help message for format)."""

        plot_type, filename = (entry.strip() for entry in value_str.split('='))
        assert plot_type in debug_plots
        return plot_type, filename

    parser.add_argument(
        '--star-sampler-pickle-fname',
        default='star_sampler.pkl',
        help='The filename to pickle the fully set-up star sampler to/from.'
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
