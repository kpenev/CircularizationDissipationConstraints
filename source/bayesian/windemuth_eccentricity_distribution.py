#!/usr/bin/env python3

"""Define a class representing the eccentricity distribution of a W19 binary."""

from multiprocessing import Pool
from os import path
import logging

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy
from scipy.optimize import brentq
from scipy import stats
from scipy.interpolate import InterpolatedUnivariateSpline
from configargparse import ArgumentParser, DefaultsFormatter

from general_purpose_python_modules.eccentricity_kde_distro_gen import\
    eccentricity_noncircular_kde_distro_gen
from general_purpose_python_modules.multi_pickle import MultiPickle
from general_purpose_python_modules.approximate_1d_function import\
    approximate_1d_function
from stellar_evolution.manager import StellarEvolutionManager

from bayesian.windemuth_et_al_util import get_samples, get_available_kic


_logger = logging.getLogger(__name__)

_manual_kernel_widths = {
    10268903: (2e-4, 2e-5),
    6962018: (2e-5, 2e-6),
    11616200: (3e-5, 3e-6),
    4380283: (1.5e-5, 1.5e-6),
    11200773: (5e-4, 5e-5),
    10960995: (2e-4, 2e-5),
    5022440: (2e-4, 1e-5),
    5802470: (1e-4, 3e-6),
    3241344: (2e-4, 2e-5),
    11867071: (3e-3, 3e-4),
    3427776: (2e-3, 1e-5),
    9881258: (5e-4, 1e-5),
    10935310: (1e-3, 1e-4),
    10031409: (1.5e-4, 1e-6),
    9532421: (1e-3, 1e-4),
    3973504: (2e-3, 1e-5),
    8957954: (1e-5, 1e-6),
    6521542: (1e-4, 1e-6),
    4285087: (6e-5, 2e-6),
    4947726: (3e-4, 1e-5),
    7691527: (2e-4, 2e-5), #was (1e-4, 1e-5)
    6227560: (3e-4, 1e-5),
    8302455: (1e-4, 2e-6),
    12004679: (1e-4, 3e-6),
    7369523: (2e-4, 6e-6),
    9971475: (2e-5, 2e-6), #was (3e-6, 3e-7), (1e-6, 1e-7)
    7129465: (2e-4, 2e-6),
    5181455: (2e-4, 5e-6),
    8381592: (1e-3, 3e-5),
    7376500: (3e-4, 3e-5),
    8618226: (1e-4, 3e-6),
    9649222: (2e-4, 2e-5),
    10385682: (2e-5, 2e-6),
    9715925: (1e-3, 3e-5),
    10965963: (1e-3, 1e-4),
    7125636: (1e-3, 1e-4),
    5597970: (5e-5, 5e-6),
    7960547: (6e-4, 1e-5),
    8746310: (2e-4, 2e-6),
    7362852: (2e-4, 2e-5),
    7128918: (5e-4, 5e-6), #was (3e-4, 3e-6) (1e-4, 3e-6)
    12557713: (2e-3, 2e-4),
    4753988: (2e-3, 2e-4),
    10923260: (2e-4, 2e-5),
    3003991: (0.01, 0.01),
    6927629: (3e-5, 1e-6),
    8364119: (2e-4, 2e-5),
    6949550: (1e-4, 3e-6),
    5731312: (1e-3, 1e-4),
    3348093: (4e-4, 4e-5),
    9532123: (3e-4, 3e-5),
    9892471: (1e-3, 2e-5),# was (4e-4, 1e-5) (4e-4, 5e-6),
    2445134: (2e-4, 2e-5),
    11391181: (1e-3, 1e-4),
    4948863: (4e-4, 4e-5), #was (2e-4, 2e-5), (1e-4, 2e-5) (2e-4, 2e-5),
    9775253: (2e-5, 2e-6), #was (5e-6, 5e-7),
    4839180: (1.5e-4, 7e-6), #was (1e-4, 5e-6),
    5652260: (8e-4, 1e-5), #was (2e-4, 1e-5),
    6707942: (3e-3, 3e-4), #was (1e-3, 1e-4),
    9934208: (1e-3, 2e-5), #was (2e-4, 2e-5),
    7597703: (3e-4, 3e-6),
    5781192: (3e-4, 3e-5), #was (3e-4, 1e-5), default
    11232745: (2e-5, 2e-6),
    8984706: (2e-4, 1e-5), #was (1e-4, 5e-6), (5e-5, 5e-6),
    5393558: (3e-4, 1e-5), #was default
    11409698: (1e-4, 1e-4),#was (1e-6, 1e-4), (1e-5, 1e-6),
    11619964: (2e-3, 2e-4),
    9353182: (3e-5, 3e-6), #was default
    4352168: (3e-4, 3e-5), #was (3e-5, 3e-6), default
    6594972: (1e-4, 3e-6), #was default
    8621353: (1e-3, 1e-4), #was (3e-4, 3e-5), default
    6185717: (3e-4, 3e-5), #was default (5e-5, 5e-6), default
    8414159: (6e-5, 6e-6), #was (3e-5, 3e-6), default
    5871918: (3e-4, 3e-5), #was default
    10274244: (1e-3, 1e-4), #was default
    6301030: (1e-4, 3e-6), #was default
    11704044: (3e-5, 3e-6), #was (1e-5, 1e-6), default
    6359798: (5e-5, 5e-6), #was (3e-5, 3e-6), default
    7200102: (3e-4, 3e-5), #was default
    7118545: (3e-5, 3e-6), #was default
    12251779: (2e-4, 2e-5), #was default, (5e-5, 5e-6), default
    8008128: (1e-3, 1e-4), #was default
    4678171: (1e-3, 3e-5),
    8111622: (1e-4, 1e-5), #was (3e-5, 3e-6), default
    9965206: (1e-3, 4e-5), #was (2e-3, 2e-5), (1e-3, 1e-5), default
    5622250: (1e-3, 2e-5), #was default
    8879427: (1e-3, 1e-4), #was (2e-4, 2e-5)
    6421188: (1e-3, 1e-4), #was default
    5979863: (1e-3, 1e-4), #was default
    7987749: (1e-3, 5e-5), #was default
    8356054: (1e-3, 1e-5), #was default
    9001468: (2e-4, 2e-5), #was default
    6522750: (4e-4, 1e-5), #was (2e-4, 5e-6)
    6131659: (3e-5, 1e-6),
    12316447: (2e-4, 2e-5), #was (1e-4, 1e-5), (3e-5, 3e-6)
    10753734: (6e-5, 6e-6),
    9016295: (2e-3, 2e-4), #was (2e-4, 2e-5)
    10332789: (3e-4, 3e-5), #was default
    10518735: (5e-4, 5e-5), #was default
    4252226: (4e-4, 4e-5), # was (2e-4, 2e-5), (1e-4, 1e-5), (5e-5, 5e-6),
    5460835: (1e-3, 1e-4), # was default
    12017140: (6e-4, 6e-5), #was (3e-4, 3e-5), default
    4633434: (5e-3, 5e-4), #was default
    10711913: (2e-3, 2e-4), #was default
    10258558: (4e-3, 4e-5), #was (2e-3, 2e-5), (1e-3, 1e-5), default
    9838060: (3e-3, 3e-4),
    10849244: (1e-4, 1e-5), #was (3e-5, 3e-6)
    10215422: (1e-5, 1e-6),
    6672229: (2e-4, 2e-5), #was default
    5983348: (4e-4, 4e-5), #was default
    7767733: (2e-3, 2e-4), #was (1e-3, 1e-4), default
    12356914: (2e-3, 2e-4), #was default
    9632895: (2e-4, 2e-5), #was default
    2998124: (2e-4, 2e-5), #was default
    7136958: (3e-4, 3e-5), #was default
    6431670: (2e-4, 2e-5), #was default
    4847832: (2e-5, 2e-6),
    5003117: (2e-3, 2e-4), #was (2e-4, 2e-5)
    12644769: (6e-5, 6e-6), #was (3e-5, 3e-6)
    6042663: (3e-4, 3e-5), #was default
    7121885: (3e-4, 3e-5), #was default
    3757778: (1e-3, 1e-4), #was (6e-4, 6e-5), (3e-4, 3e-5), default
    8183389: (3e-3, 3e-4), #was (1e-3, 1e-4), default
    4247023: (4e-4, 4e-5), #was default
    9164836: (4e-4, 4e-5), #was default
    2442084: (1e-3, 1e-4), #was (3e-4, 3e-5), default
    12208887: (1e-3, 1e-4), #was (3e-4, 3e-5), default
    9714123: (1e-3, 1e-4), #was default
    9837544: (3e-5, 3e-6), #was (1e-5, 1e-6)
    8044608: (2e-4, 2e-5), #was (5e-5, 5e-6)
    8605074: (1e-3, 1e-4), #was default
    7620844: (1e-3, 1e-4), #was default
    10292238: (3e-4, 3e-5), #was default
    4824268: (3e-4, 3e-5), #was default
    7024530: (1e-3, 1e-4), #was default
    9839062: (1e-3, 1e-4), #was default
    8560285: (1e-3, 1e-4), #was default
    9110346: (7e-5, 7e-6),
    7732791: (2e-3, 1e-5),
    9656543: (7e-4, 1e-5),
    3834364: (2e-3, 1e-5),
    11228612: (4e-4, 3e-6),
    7798259: (5e-4, 5e-5), #was (3e-4, 3e-5), default
    11967004: (5e-4, 5e-5), #was (3e-4, 3e-5), default
    4579321: (3e-4, 3e-5), #was default
    5039441: (3e-4, 3e-5), #was default
    9119652: (3e-4, 3e-5), #was default
    6312521: (3e-4, 3e-5), #was default
    7605600: (3e-4, 3e-5), #was default
    10091257: (3e-4, 3e-5), #was default
    4276114: (5e-4, 5e-5), #was (3e-4, 3e-5), default
    7377033: (1e-3, 3e-5), #was (3e-4, 3e-5), default
    11403216: (3e-4, 3e-5), #was default
    11252617: (6e-4, 6e-5), #was (3e-4, 3e-5), default
    4346875: (6e-4, 6e-5), #was (3e-4, 3e-5), default
    10483644: (3e-4, 3e-5), #was default
    6546508: (6e-4, 6e-5), #was (3e-4, 3e-5), default
    8460600: (1e-3, 1e-4), #was (6e-4, 6e-5), (3e-4, 3e-5), default
    10020423: (3e-4, 3e-5), #was default
    9762519: (5e-4, 5e-5), #was (3e-4, 3e-5), default
    8543278: (3e-4, 3e-5), #was default
    9025914: (3e-4, 3e-5), #was default
    6029130: (3e-4, 3e-5), #was default
    2719873: (3e-4, 3e-5), #was default
    7624297: (3e-4, 3e-5), #was default
    10651945: (6e-4, 6e-5), #was (3e-4, 3e-5), default
    6947666: (1e-3, 1e-4), #was (3e-4, 3e-5), default
}


def _get_eccentricity_kernel_width(kic_id):
    """Return dict of eccentricity kernel widths for each KIC."""

    return _manual_kernel_widths.get(kic_id, (1e-4, 1e-5))


def _get_support(eccentricity_pdf, median_eccentricity):
    """Return the range outside of which the PDF(e) can be neglected."""

    def to_solve(e):
        pdf = eccentricity_pdf(e)
        return numpy.log10(pdf) + 6.0 if pdf > 0 else -numpy.inf

    if eccentricity_pdf(0.0) > 1e-6:
        min_e = 0.0
    else:
        min_e = brentq(to_solve, 0.0, median_eccentricity)

    if eccentricity_pdf(1.0) > 1e-6:
        max_e = 1.0
    else:
        max_e = brentq(to_solve, median_eccentricity, 1.0)

    _logger.debug('Approximate support: %s < e < %s',
                  repr(min_e),
                  repr(max_e))

    return min_e, max_e


def plot_eccentricity_histogram(e_samples, e_range, unzoomed_bins):
    """Plot a histogram of the given samples changing the prior to uniform e."""

    hist, bin_edges = numpy.histogram(
        e_samples,
        bins=int(numpy.ceil(
            unzoomed_bins
            *
            max(
                1,
                (e_samples[-1] - e_samples[0])
                /
                (e_range[1] - e_range[0])
            )
        )),
        density=True
    )

#    hist /= bin_edges[1:]**2 - bin_edges[:-1]**2
    hist /= (hist * (bin_edges[1:] - bin_edges[:-1])).sum()
    pyplot.bar(x=bin_edges[:-1],
               height=hist,
               width=bin_edges[1:] - bin_edges[:-1],
               align='edge',
               color='none',
               edgecolor='black')


def plot_ecosw_esinw_samples(w19_samples, axes, tight=False):
    """Plot the samples e sin(w) vs e cos(w) in the given axes."""

    axes.plot(w19_samples['ecosw'], w19_samples['esinw'], 'ok', markersize=0.5)
    if not tight:
        axes.axhline(y=0, linewidth=0.5)
        axes.axvline(x=0, linewidth=0.5)


def get_eccentricity_pdf(kic_id,
                         pickle_fname='windemuth_eccentricity_ditsros.pkl',
                         **approximation_config):
    """Setup the evaluation of the PDF for the given KIC ID."""


    assert 'func' not in approximation_config
    assert 'support' not in approximation_config
    assert 'grid_only' not in approximation_config
    approximation_config['KIC'] = kic_id
    approximation_config['ext'] = 0

    pickle_file = MultiPickle(pickle_fname)
    pickled = pickle_file.check_for_pickled(approximation_config)
    if pickled is not None:
        for param in ['min_grid_points',
                      'tolerance',
                      'min_grid_step',
                      'grid_refine_limit',
                      'grid_only',
                      'KIC']:
            if param in approximation_config:
                del approximation_config[param]

        _logger.debug('Unpickled: %s', repr(pickled))
        return InterpolatedUnivariateSpline(
            *pickled,
            **approximation_config
        )

    w19_samples = get_samples(kic_id)
    e_samples = numpy.sqrt(w19_samples['esinw']**2 + w19_samples['ecosw']**2)

    sin_kernel_width, cos_kernel_width = _get_eccentricity_kernel_width(kic_id)
    kde_distro = eccentricity_noncircular_kde_distro_gen(
        esinw_samples=w19_samples['esinw'],
        ecosw_samples=w19_samples['ecosw'],
        sin_kernel=stats.norm(scale=sin_kernel_width),
        cos_kernel=stats.norm(scale=cos_kernel_width),
        #stats.rdist(c=4, scale=sin_kernel_width),
        #stats.rdist(c=4, scale=cos_kernel_width),
        uniform_e_samples=True
    )

    del approximation_config['KIC']
    x_values, y_values = approximate_1d_function(
        kde_distro.pdf,
        _get_support(kde_distro.pdf, numpy.median(e_samples)),
        min_grid_step=1e-8,
        grid_only=True,
        **approximation_config
    )

    approximation_config['KIC'] = kic_id
    pickle_file.add_result(approximation_config, x_values, y_values)
    del approximation_config['KIC']
    return InterpolatedUnivariateSpline(x_values,
                                        y_values,
                                        **approximation_config)


def evaluate_eccentricity_pdf(kic_id):
    """Coumpute KDE data to plot for given KIC ID."""

    custom_zoom = {5979863: (0.03, 0.04),
                   4948863: (4e-3, 8e-3),
                   5652260: (0, 0.01),
                   7362852: (1e-3, 6e-3),
                   4947726: (0.162, 0.173),
                   3973504: (0.08, 0.14),
                   7025851: (0.0, 1e-3),
                   9881258: (0.0, 1e-3),
                   4380283: (5.0e-5, 1.5e-4),
                   6312521: (0.06, 0.1),
                   9110346: (0.0, 1e-4),
                   7732791: (0.0, 0.0003),
                   9656543: (0.0, 1e-3),
                   10960995: (0.0, 3e-4),
                   5022440: (0.0, 3e-4),
                   5802470: (0.0, 1e-4),
                   4815612: (0.0, 1e-4),
                   10031409: (0.0, 2e-4),
                   8957954: (0.0, 1e-4),
                   4285087: (0.0, 2e-4),
                   6227560: (0.0, 2e-3),
                   8302455: (0.0, 2e-4),
                   9971475: (0.0, 5e-4),
                   7129465: (0.0, 1e-4),
                   5181455: (0.0, 2e-4),
                   7376500: (0.41, 0.42),
                   8618226: (0.0, 5e-4),
                   9649222: (0.0, 2e-3),
                   5597970: (0.0, 2e-4),
                   8746310: (0.0, 2e-4),
                   10923260: (0.4060, 0.4150),
                   3003991: (0.0, 0.2),
                   6927629: (0.0, 1e-4),
                   6949550: (0.26477, 0.26493),
                   9892471: (0, 0.002),
                   9775253: (0, 3e-4),
                   7597703: (0, 2e-4),
                   11232745: (0, 1e-3),
                   8414159: (0, 0.002),
                   11499757: (0.26, 0.263),
                   11704044: (0, 2e-4),
                   8879427: (0.45, 0.465),
                   12316447: (0.368, 0.370),
                   7021177: (0.584, 0.596),
                   8572936: (0.5170, 0.5195),
                   8973000: (0.52, 0.53)}

    w19_samples = get_samples(kic_id)
    e_samples = numpy.sort(
        numpy.sqrt(w19_samples['esinw']**2
                   +
                   w19_samples['ecosw']**2)
    )
    plot_ranges = [
        (
            0,
            e_samples[-1]
        ),
        custom_zoom.get(kic_id, numpy.quantile(e_samples, [0.025, 0.975]))
    ]
    if (
            plot_ranges[1][1] - plot_ranges[1][0]
            >
            0.2 * (plot_ranges[0][1] - plot_ranges[0][0])
            and
            kic_id not in custom_zoom
    ):
        print(
            f'Full range ({plot_ranges[0][0]:g}, {plot_ranges[0][1]:g}) '
            f'and zoom range ({plot_ranges[1][0]:g}, {plot_ranges[1][1]:g})'
            ' comparable. No zoom plot necessary.'
        )
        plot_ranges = plot_ranges[:1]
    else:
        print(f'Adding zoom-in plot for KIC {kic_id:d}: '
              f'{plot_ranges[1][0]:g} < ef < {plot_ranges[1][1]:g}')

    plot_x = [numpy.linspace(*e_range, 1000) for  e_range in plot_ranges]
    plot_y = get_eccentricity_pdf(kic_id)(plot_x)

    return  plot_ranges, plot_x, plot_y


#No good way to simplify
#pylint: disable=too-many-locals
def plot_eccentricity_distribution(kic_id_list,
                                   plot_fname,
                                   bins,
                                   num_parallel_processes):
    """
    Plot KDE estimated eccentricity distribution and histogram for given KIC.

    Args:
        kic_id_list([int,...]):    The KIC identifiers of the Windemuth et. al.
            (2019) binaries to plot the eccentricity distribution of. A
            multi-page PDF is created with each KIC plot on a separate page.

        plot_fname(str):    The filename to save the plot. If empty, the plots
            are shown but not saved.

        bins(int or sequence of floats or str):    The bins to use for buliding
            the histogram to show. See `numpy.histogram` for details. The numpy.
            derived bins are then scaled by the inverse of the area
            corresponding to each bin.

    Returns:
        None
    """

    if plot_fname:
        pdf = PdfPages(plot_fname)

    with Pool(num_parallel_processes) as pool:
        kde_plot_data = pool.map(evaluate_eccentricity_pdf, kic_id_list)

    for kic_id, kde_plot_data in zip(kic_id_list, kde_plot_data):
        w19_samples = get_samples(kic_id)
        e_samples = numpy.sort(
            numpy.sqrt(w19_samples['esinw']**2
                       +
                       w19_samples['ecosw']**2)
        )
        print('Ecccentricity samples:\n' + repr(e_samples))

        tight = False
        for e_range, plot_x, plot_y in zip(*kde_plot_data):
            pyplot.subplot(111, position=(0.1, 0.1, 0.85, 0.85))
            plot_eccentricity_histogram(e_samples, e_range, bins)
            pyplot.plot(plot_x, plot_y, color='red')
            pyplot.xlim(*e_range)
            pyplot.suptitle(str(kic_id) + ' PDF($e_f$)')

            if (
                    plot_y[plot_x < 0.6 * e_range[0] + 0.4 * e_range[1]].max()
                    <
                    plot_y[plot_x > 0.4 * e_range[0] + 0.6 * e_range[1]].max()
            ):
                inset_location = 'upper left'
            else:
                inset_location = 'upper right'
            plot_ecosw_esinw_samples(
                get_samples(kic_id),
                inset_axes(pyplot.gca(),
                           width='35%',
                           height='35%',
                           loc=inset_location),
                tight=tight
            )
            tight = True

            if plot_fname:
                pdf.savefig()
                pyplot.close()
            else:
                pyplot.show()
    if plot_fname:
        pdf.close()
#pylint: enable=too-many-locals


def parse_command_line():
    """The command line arguments supplied when used as a script."""

    parser = ArgumentParser(
        description='Convenience tool for working with W19 binaries.',
        default_config_files=['w19_util.cfg'],
        args_for_writing_out_config_file=['--generate-config-file'],
        args_for_setting_config_path=['--config-file', '-c'],
        formatter_class=DefaultsFormatter,
        ignore_unknown_config_file_keys=False
    )

    parser.add_argument(
        'kic',
        help='The KIC for which to plot the eccentricity distribution. '
        'The plot will show binned eccentricity samples, with bin '
        'heights divided by the area of the annulus in (ecosw, esinw) '
        'space corresponding to each bin as well as the KDE estimated '
        'eccentricity distrubition.'
    )
    parser.add_argument(
        'plot_fname',
        help='The filename to save the plot as. Use empty string to show.'
    )

    parser.add_argument(
        '--stellar-evolution-interpolator-dir', '--interpolator-dir',
        default=(
            path.expanduser(
                '~/projects/git/poet/stellar_evolution_interpolators'
            )
        ),
        help='The directory to read stellar evolution interpolator from.'
    )
    parser.add_argument(
        '--e-distro-histogram-bins',
        type=int,
        default=30,
        help='The number of bins to use for plotting eccentricity ditribution.'
    )
    parser.add_argument(
        '--num-parallel-processes',
        type=int,
        default=4,
        help='The numbef or parallel processes to use for computing KDE '
        'estimates of eccentricity distributions for plotting.'
    )

    return parser.parse_args()


def main(config):
    """Avoid polluting global namespace."""

    interpolator = StellarEvolutionManager(
        config.stellar_evolution_interpolator_dir
    ).get_interpolator_by_name(
        'default'
    )

    logging.basicConfig(level=logging.DEBUG)

    available_kic = get_available_kic(interpolator, numpy.inf, 1.0)

    plot_eccentricity_distribution(
        kic_id_list=(
            available_kic if config.kic == 'all'
            else [int(config.kic)]
        ),
        plot_fname=config.plot_fname,
        bins=config.e_distro_histogram_bins,
        num_parallel_processes=config.num_parallel_processes
    )


if __name__ == '__main__':
    main(parse_command_line())
