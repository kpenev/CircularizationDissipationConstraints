#!/usr/bin/env python3

"""Create grid of color map plots of the constraints ready for article."""

from subprocess import call
from os import path, listdir
import hashlib
import pickle
from multiprocessing import Pool

import matplotlib
from matplotlib import pyplot, cm, colors, rcParams
from matplotlib.backends.backend_pdf import PdfPages
from configargparse import ArgumentParser, DefaultsFormatter
from scipy.stats import rdist, norm
from scipy.interpolate import interp1d
import numpy
from kde import KDEDistribution
from astropy.table import Table
from cdspyreadme import CDSTablesMaker

from emcee_autocorrelation import\
    max_likelihood_autocorr,\
    average_autocorr
from mcmc_quantile_convergence import get_raftery_lewis_diagnostics
from emcee_quantile_convergence import find_emcee_quantiles
from combined_mcmc_constraint import CombinedMCMCConstraint

from bayesian.visualize_emcee import\
    add_frequency_dependence_plot_config,\
    get_plot_data,\
    FrequencyDependencePlotter,\
    evaluate_lgq
#False positive, not sure why only m35 fails!
#pylint: disable=unused-import
from bayesian.m35_util import get_binary_data as get_m35_binary_data
#pylint: enable=unused-import
from bayesian.ngc6819_util import get_binary_data as get_ngc6819_binary_data
from bayesian.ngc188_util import get_binary_data as get_ngc188_binary_data
from bayesian.m35_util import get_photometry as get_m35_photometry
#pylint: enable=unused-import
from bayesian.ngc6819_util import get_photometry as get_ngc6819_photometry
from bayesian.ngc188_util import get_photometry as get_ngc188_photometry

from bayesian.cluster_util import select_binary_data

def parse_command_line(quantiles_only=False):
    """Parse command line for plotting configuration."""

    parser = ArgumentParser(
        description=__doc__,
        default_config_files=[
            path.splitext(__file__)[0] + '.cfg'
        ],
        formatter_class=DefaultsFormatter,
        ignore_unknown_config_file_keys=False
    )

    default_samples_dir = path.join(
        path.dirname(
            path.dirname(
                path.abspath(
                    __file__
                )
            )
        ),
        'samples'
    )

    parser.add_argument(
        '--samples-dir',
        default=default_samples_dir,
        help='The directory holding the HDF5 files with MCMC samples.'
    )
    parser.add_argument(
        '--data-pickle',
        default=path.join(
            default_samples_dir,
            'processed_sampling_data.pkl'
        )
    )
    parser.add_argument(
        '--combined-quantiles-pickle',
        default=path.join(
            default_samples_dir,
            'combined_quantiles.pkl'
        )
    )
    parser.add_argument(
        '--skip-download',
        default=False,
        action='store_true',
        help='If passed, does not download the latest samples from stampede2 '
        'and ganymede.'
    )
    parser.add_argument(
        '--convergence-ptide-grid',
        default=list(numpy.logspace(0, numpy.log10(50), 30)),
        nargs='+',
        type=float,
        help='The tidal periods at which to generate convergence plots.'
    )
    parser.add_argument(
        '--convergence-quantiles',
        default=list(norm.cdf([-2.0, -1.0, 1.0, 2.0])),
        nargs='+',
        type=float,
        help='The quantiles for which to display the lgQ evolution on the '
        'convergence plots.'
    )
    parser.add_argument(
        '--constraint-validity-threshold',
        default=0.7,
        type=float,
        help='Constraints are considered valid if the highest quantile does '
        'not exceed this value + its minimum over tidal period.'
    )
    parser.add_argument(
        '--convergence-combine-nsteps',
        default=10,
        type=int,
        help='How many steps to bin together when calculating quantiles for '
        'the convergence plots. A value of zero combines all steps previos to '
        'the x value instead of a fixed width window.'
    )
    parser.add_argument(
        '--burn-in-tolerance',
        type=float,
        default=1e-3,
        help='Automatically determined burn-in requires that the stationary '
        'probability for quantile indicator chains is reached to within this '
        'precision. See Raftery & Lewis (1995).'
    )
    parser.add_argument(
        '--variance-realizations',
        type=int,
        default=10000,
        help='Variance of the estimated CDF is calculated by generating this '
        'many independent random realizations of the best fit Markov process to'
        ' the thinned chain of number of walkers below each quantile.'
    )
    parser.add_argument(
        '--individual-plot-mode', '--individual-plots',
        choices=['pages', 'subplots'],
        default='subplots',
        help='Choose whether plots of individual systems will be saved each on '
        'a separate page in a multi-page PDF file (`pages`) or as grid of '
        'sub-plots in a single figure (`subplots`).'
    )
    parser.add_argument(
        '--combined-constraint-period-range', '--combined-prange',
        nargs=2,
        type=float,
        default=(2.0, 10.0),
        help='The range of tidal periods to show in the combined constraint '
        'plot.'
    )
    parser.add_argument(
        '--nthreads',
        default=16,
        help='The maximum number of parallel threads to use for calculating '
        'quantile diagnostics (the only slow part of preparing plotting data).'
    )
    parser.add_argument(
        '--mrt-author',
        default='Penev, K. and Schussler, J.',
        help='The author to specify for the generated data behind the figures '
        'tables.'
    )
    if not quantiles_only:
        add_frequency_dependence_plot_config(parser)

    return parser.parse_args()


def download_latest_samples(destination):
    """Download the latest samples files from stampede2 and ganymede."""

    for source in [
            'kxp174430@ganymede:~/M35*_powerlaw_alllock_*mcmc_samples.h5',
            'kpenev@stampede2.tacc.utexas.edu:~/'
            'NGC*_mcmc_powerlawlgQ_samples.h5'
    ]:
        call(['rsync', '-avz', '--progress', source, destination])


def add_preprocessed_data(samples_fname, preprocessed_data, result, config):
    """Add system data from loaded pickle to result."""

    if samples_fname in preprocessed_data:
        with open(
            path.join(config.samples_dir, samples_fname),
            'rb'
        ) as samples_f:
            if (
                hashlib.md5(samples_f.read()).hexdigest()
                ==
                preprocessed_data[samples_fname]['checksum']
                and
                preprocessed_data[samples_fname]['ptide_grid']
                ==
                config.convergence_ptide_grid
                and
                preprocessed_data[samples_fname]['quantiles']
                ==
                config.convergence_quantiles
            ):
                result[preprocessed_data[samples_fname]['system']] = (
                    preprocessed_data[samples_fname]['system_data']
                )
                return True
    return False


def add_orbit_and_photometry(sampling_data):
    """Add orbit data to `sampling_data`."""

    sb1_orbits = dict()
    photometry = dict()
    for cluster in ['M35', 'NGC6819', 'NGC188']:
        sb1_orbits[cluster] = globals()['get_'
                                        +
                                        cluster.lower()
                                        +
                                        '_binary_data']()[0]
        photometry[cluster] = globals()['get_'
                                        +
                                        cluster.lower()
                                        +
                                        '_photometry']()
    for binary in sampling_data.keys():
        cluster = binary.split('_')[0]
        id_column = ('PKM' if cluster == 'NGC188' else 'WOCS')
        binary_id = int(binary.split('_')[1])
        sampling_data[binary]['orbit'] = select_binary_data(
            sb1_orbits[cluster],
            None,
            id_column,
            binary_id
        )
        sampling_data[binary]['photometry'] = photometry[cluster][
            photometry[cluster][id_column] == binary_id
        ]


def get_single_quantile(cdf_value, ptide, samples, config):
    """Return the quantile and diagnostics for given ptide and CDF value."""

    return find_emcee_quantiles(
        evaluate_lgq(
            samples,
            numpy.array([ptide])
        ).reshape(
            samples['lgQ_min'].shape
        ),
        cdf_value,
        config.burn_in_tolerance,
        config.variance_realizations
    )


def get_quantiles(samples, config):
    """Return the quantiles with diagnostics based on the given samples."""

    with Pool(config.nthreads) as workers:
        flattened_quantiles = workers.starmap(
            get_single_quantile,
            [
                (cdf_value, ptide, samples, config)
                for ptide in config.convergence_ptide_grid
                for cdf_value in config.convergence_quantiles
            ]
        )
    nquantiles = len(config.convergence_quantiles)
    return [
        flattened_quantiles[p_i * nquantiles : p_i * nquantiles + nquantiles]
        for p_i in range(len(config.convergence_ptide_grid))
    ]


def get_sampling_data(config, add_quantiles=True):
    """Return dictionary index by system of samples, likelihood, & quantile."""


    if path.exists(config.data_pickle):
        with open(config.data_pickle, 'rb') as pickle_file:
            preprocessed_data = pickle.load(pickle_file)
    else:
        preprocessed_data = dict()

    result = dict()
    for samples_fname in listdir(config.samples_dir):
        if not path.splitext(samples_fname)[1] == '.h5':
            print('Skipping ' + repr(samples_fname))
            continue
        if add_preprocessed_data(samples_fname,
                                 preprocessed_data,
                                 result,
                                 config):
            print('Reusing pickled data for: ' + repr(samples_fname))
            continue
        try:
            print('Reading: ' + repr(samples_fname))
            system, samples, log_probability = get_plot_data(
                path.join(config.samples_dir, samples_fname),
                0,
                config.chain_condition
            )
        except AssertionError:
            continue
        print(system)
        result[system] = dict(samples=samples,
                              log_probability=log_probability)
        if add_quantiles:
            quantiles = get_quantiles(samples, config)
            result[system]['quantiles'] = quantiles
            with open(
                path.join(config.samples_dir, samples_fname),
                'rb'
            ) as samples_f:
                preprocessed_data[samples_fname] = dict(
                    checksum=hashlib.md5(samples_f.read()).hexdigest(),
                    ptide_grid=config.convergence_ptide_grid,
                    quantiles=config.convergence_quantiles,
                    system=system,
                    system_data=result[system]
                )
                with open(config.data_pickle, 'wb') as pickle_file:
                    pickle.dump(preprocessed_data, pickle_file)

    add_orbit_and_photometry(result)
    return result


def plot_single_diagnostic_period(quantile_data,
                                  diagnostic,
                                  axis,
                                  config,
                                  *,
                                  fmt='-',
                                  label=True,
                                  zorder=None):
    """Plot some diagnostic of the quantiles of lgQ(Ptide) vs Ptide."""

    diagnostic_ind = ['quantile', 'cdf', 'std', 'thin', 'burnin'].index(
        diagnostic
    )
    axis.set_xscale('log')
    axis.set_xlim(1, 50)

    kwargs=dict()
    if label is not None:
        kwargs['label'] = label
    if zorder is not None:
        kwargs['zorder'] = zorder

    quantile_labels =[
        'CDF={0:.3f}'.format(cdf) for cdf in config.convergence_quantiles
    ]
    description_quantity = dict(
        quantile="value of log10(Q')",
        std="standard deviation of the CDF",
        burnin="burnin period"
    )
    data_behind = Table(
        [config.convergence_ptide_grid],
        names=['Ptide'],
        dtype=[float],
        descriptions=["The tidal period at which Q' was evaluated"],
    )
    for quantile_ind, label in enumerate(quantile_labels):
        if label is True:
            kwargs['label'] = label

        plot_y = [
            quantile_data[ptide_ind][quantile_ind][diagnostic_ind]
            for ptide_ind in range(len(config.convergence_ptide_grid))
        ]

        data_behind[label] = plot_y
        data_behind[label].description = (
            "The {quantity} at the {label} quantile".format(
                quantity=description_quantity[diagnostic],
                label=label
            )
        )
        axis.plot(
            config.convergence_ptide_grid,
            plot_y,
            fmt,
            **kwargs
        )
    return data_behind


def get_valid_ptide_indices(quantiles, config):
    """Indicate well constrained entries of convergence Ptide grid."""

    upper_quantile_ind = numpy.argmax(config.convergence_quantiles)
    upper_quantile = numpy.array([entry[upper_quantile_ind][0]
                                  for entry in quantiles])
    if upper_quantile.min() > 9:
        return None
    return numpy.nonzero(
        upper_quantile
        <
        upper_quantile.min() + config.constraint_validity_threshold
    )[0]


def mark_valid_constraint_range(quantiles, axis, config):
    """Add two vertical lines showing Ptide range where constraint is valid."""

    try:
        first_valid, last_valid = get_valid_ptide_indices(
            quantiles,
            config
        )[[0,-1]]
        axis.axvline(x=config.convergence_ptide_grid[first_valid],
                     linewidth=1.5,
                     color='red',
                     zorder=30)
        axis.axvline(x=config.convergence_ptide_grid[last_valid],
                     linewidth=1.5,
                     color='red',
                     zorder=30)
    except TypeError:
        pass


def get_burnin(system_data, config, binary=''):
    """Return the burnin to use for this system."""

    burnin = 0
    valid_ptide_indices = get_valid_ptide_indices(system_data['quantiles'],
                                                  config)
    if valid_ptide_indices is None:
        valid_ptide_indices = numpy.arange(
            len(config.convergence_ptide_grid)
        )

    for ptide_ind in valid_ptide_indices:
        ptide_burnin = max(
            quantiles[4]
            for quantiles in system_data['quantiles'][ptide_ind]
        )
        burnin = max(burnin, ptide_burnin)
        if burnin > system_data['samples'].shape[0] - 100:
            print(
                'Burn-in period for %(binary)s leaves < %(minsteps)d steps,'
                'using last %(minsteps)d steps!'
                %
                dict(binary=binary, minsteps=100)
            )
            return system_data['samples'].shape[0] - 100
    print('%(binary)s burn-in: %(burnin)d'
          %
          dict(binary=binary, burnin=burnin))
    return burnin


def plot_single_lgq_period(binary, system_data, __, axis, config):
    """Plot the constraint for a single system."""

    axis.set_xscale('log')
    orig_axis = pyplot.gca()
    pyplot.sca(axis)
    lgq_range = (4.0 if binary.startswith('M35_') else 5.0, 12)
    config.combined_constraint_lgQ_grid = numpy.linspace(*lgq_range, 50)
    frequency_dependence_plotter = FrequencyDependencePlotter(1, config)
    frequency_dependence_plotter.add_chain(
        system_data['samples'][get_burnin(system_data, config, binary):],
        None
    )
    pyplot.ylim(*lgq_range)
    frequency_dependence_plotter.plot_combined_pdf_heat_map(
        xlabel=config.bottom,
        ylabel=config.left
    )
    data_behind = plot_single_diagnostic_period(system_data['quantiles'],
                                                'quantile',
                                                axis,
                                                config,
                                                fmt='-k',
                                                label=None,
                                                zorder=20)
    pyplot.axvline(x=min(system_data['orbit']['Per']),
                   color='black',
                   linewidth=3,
                   zorder=20)
    mark_valid_constraint_range(system_data['quantiles'],
                                axis,
                                config)
    decorate_subplot(
        axis,
        binary,
        r"$\log_{10}Q_\star'$",
        config,
        yticks=[4, 6, 8, 10, 12] if binary.startswith('M35') else [5, 7, 9, 11]
    )
    pyplot.sca(orig_axis)
    return data_behind


#pylint: disable=too-many-locals
def plot_single_convergence(*,
                            _,
                            system_data,
                            ptide,
                            ptide_ind,
                            quantity,
                            values,
                            combine_nsteps,
                            axis):
    """Plot a distribution quantity vs step number for a single system/Ptide."""

    pyplot.xscale('linear')

    shape = system_data['samples'].shape
    lgq_values = evaluate_lgq(
        system_data['samples'],
        numpy.array([ptide])
    ).reshape(
        shape
    )

    plot_x = numpy.arange(combine_nsteps + 1,
                          len(lgq_values))
    plot_y = numpy.empty((plot_x.size, len(values)),
                          dtype=float)

    for x_ind, lgq_values_end in enumerate(plot_x):
        lgq_values_start = (
            0 if combine_nsteps == 0
            else lgq_values_end - combine_nsteps - 1
        )
        kde = KDEDistribution(
            lgq_values[lgq_values_start: lgq_values_end].flatten(),
            rdist(c=4, scale=0.05)
        )
        plot_y[x_ind, :] = getattr(kde, quantity)(values)


    for dset_i in range(len(values)):
        axis.set_prop_cycle(None)
        min_x = 0
        for max_x, size in [
                (system_data['quantiles'][ptide_ind][dset_i][-1], 3),
                (
                    system_data['quantiles'][ptide_ind][dset_i][-1]
                    +
                    combine_nsteps,
                    6
                ),
                (plot_x.size, 9)
        ]:
            axis.plot(
                plot_x[min_x : max_x + 1],
                plot_y[min_x : max_x + 1, dset_i],
                '-x',
                markersize=size
            )
            min_x = max_x
#pylint: enable=too-many-locals


def plot_single_lgq_quantiles(binary, system_data, ptide, axis, config):
    """Plot lgQ @ quantiles vs step number for a single system/tidal period."""

    ptide_index = config.convergence_ptide_grid.index(ptide)
    plot_single_convergence(
        binary=binary,
        system_data=system_data,
        ptide=ptide,
        ptide_ind=ptide_index,
        quantity='ppf',
        values=config.convergence_quantiles,
        combine_nsteps=config.convergence_combine_nsteps,
        axis=axis
    )
    for quantile_diag in system_data['quantiles'][ptide_index]:
        axis.axhline(y=quantile_diag[0], zorder=0, color='black')
    decorate_subplot(axis,
                     binary,
                     r"$\log_{10}Q_\star'$",
                     config,
                     xlabel='emcee step')


def plot_single_quantiles_lgq(binary, system_data, ptide, axis, config):
    """Plot quantile(lgQ) vs step number for a single system/tidal period."""

    ptide_index = config.convergence_ptide_grid.index(ptide)
    plot_single_convergence(
        binary=binary,
        system_data=system_data,
        ptide=ptide,
        ptide_ind=ptide_index,
        quantity='cdf',
        values=[
            quantile_diag[0]
            for quantile_diag in system_data['quantiles'][ptide_index]
        ],
        combine_nsteps=config.convergence_combine_nsteps,
        axis=axis
    )
    for target_cdf, quantile_diag in zip(config.convergence_quantiles,
                                         system_data['quantiles'][ptide_index]):
        axis.axhspan(ymin=(target_cdf - quantile_diag[2]),
                     ymax=(target_cdf + quantile_diag[2]),
                     zorder=0,
                     facecolor='lightgray')
        extra_args = dict()
        for estimate in quantile_diag[1]:
            extra_args['color'] = axis.axhline(
                y=estimate,
                **extra_args
            ).get_color()
    decorate_subplot(axis,
                     binary,
                     r"$CDF(\log_{10}Q_\star')$",
                     config,
                     xlabel='emcee step')

def plot_single_autocorrelation(_, system_data, ptide, axis, __):
    """Plot the autocorrelation of lg[Q(Ptide)] for a single system/Ptide."""

    lgq_values = evaluate_lgq(
        system_data['samples'],
        numpy.array([ptide])
    ).reshape(
        system_data['samples'].shape
    )
    autocorr = average_autocorr(lgq_values.T, time=False)
    axis.plot(numpy.arange(autocorr.size), autocorr, '-o')


def plot_single_autocorrelation_time(_, system_data, ptide, axis, __):
    """Plot the autocorrelation of lg[Q(Ptide)] for a single system/Ptide."""

    lgq_values = evaluate_lgq(
        system_data['samples'],
        numpy.array([ptide])
    ).reshape(
        system_data['samples'].shape
    )
    plot_x = numpy.arange(max(int(0.1 * lgq_values.shape[0]), 10),
                          lgq_values.shape[0] - 1)
    plot_y = numpy.empty(plot_x.shape)
    for ind, eval_nsamples in enumerate(plot_x):
        plot_y[ind] = average_autocorr(lgq_values[:eval_nsamples].T)
    axis.plot(plot_x, plot_y, '-o')


def get_raftery_lewis_plot_data(binary_chain,
                                quantile_cdf,
                                num_walkers,
                                burn_in_tolerance):
    """Return the data to plot for the given quantile."""

    plot_data = [
        numpy.empty(num_walkers + 1, dtype=float),
        numpy.empty(num_walkers + 1, dtype=float),
        numpy.empty(num_walkers + 1, dtype=int),
        numpy.empty(num_walkers + 1, dtype=int),
    ]

    plot_data[0][:-1] = binary_chain.mean(axis=0) - quantile_cdf
    print('Mean binary chain (q=%.5f): %s'
          %
          (quantile_cdf, repr(binary_chain.mean())))
    for walker in range(num_walkers):
        (
            plot_data[1][walker],
            plot_data[2][walker],
            plot_data[3][walker]
        ) = get_raftery_lewis_diagnostics(binary_chain[:, walker],
                                          burn_in_tolerance)
    plot_data[1] /= num_walkers**0.5
    return plot_data


#pylint: disable=too-many-locals
def plot_single_raftery_lewis_diagnostics(_,
                                          system_data,
                                          ptide,
                                          axis_list,
                                          config):
    """Create plots of CDF(mean, var, thin, burnin) of quantiles per walker."""

    ptide_ind = config.convergence_ptide_grid.index(ptide)
    lgq_values = evaluate_lgq(system_data['samples'], numpy.array([ptide]))

    lgq_values = lgq_values.reshape(system_data['samples'].shape)

    axis_list[3].set_xscale('log')
    for quantile_diagnostics, quantile_cdf in zip(
            system_data['quantiles'][ptide_ind],
            config.convergence_quantiles
    ):
        binary_chain = (lgq_values < quantile_diagnostics[0]).astype(int)
        plot_data = get_raftery_lewis_plot_data(binary_chain,
                                                quantile_cdf,
                                                system_data['samples'].shape[1],
                                                config.burn_in_tolerance)
        print('Combined diagnostics: ' + repr(quantile_diagnostics))
#            print('Got plot data for q=%.4f, walker %d.'
#                  %
#                  (quantile, walker))
        label = 'q=%d%%' % int(numpy.round(100 * quantile_cdf))
        for axis, plot_quantity, combined in zip(axis_list,
                                                 plot_data,
                                                 quantile_diagnostics[1:]):

            plot_quantity[-1] = plot_quantity[:-1].max()
            plot_quantity = plot_quantity[numpy.isfinite(plot_quantity)]
            color = axis.step(
                numpy.sort(plot_quantity),
                (
                    numpy.arange(plot_quantity.size, dtype=float)
                    /
                    plot_quantity.size
                ),
                label=label,
                zorder=10
            )[0].get_color()

            if (
                    combined is not None
                    and
                    numpy.isfinite(numpy.asarray(combined)).any()
            ):
                if label is not None:
                    combined -= quantile_cdf
                for line_x in numpy.atleast_1d(combined):
                    axis.axvline(x=line_x, color=color, zorder=20)

            label=None

    axis_list[3].axvspan(xmin=axis_list[3].get_xlim()[0],
                         xmax=system_data['samples'].shape[0],
                         facecolor='lightgray',
                         zorder=0)

    for title, axis in zip(['Mean - q', 'Variance', 'Thin', 'Burnin'],
                           axis_list):
        axis.set_title(title)
#pylint: enable=too-many-locals


def decorate_subplot(axis,
                     binary,
                     ylabel,
                     config,
                     *,
                     xticks=None,
                     yticks=None,
                     xlabel='Tidal Period [days]'):
    """Make sure the given subplot displays appropriate ticks/labels etc."""

    if config.bottom:
        axis.tick_params(axis='x',
                         which='both',
                         bottom=True,
                         labelbottom=True)

        axis.get_xaxis().set_major_formatter(
            matplotlib.ticker.ScalarFormatter()
        )
        axis.get_xaxis().set_minor_formatter(
            matplotlib.ticker.ScalarFormatter()
        )

        if xticks is not None:
            axis.set_xticks(xticks)

        axis.set_xticklabels(
            [
                str(int(val) if val - int(val) == 0 else val)
                for val in axis.get_xticks(minor=False)
            ],
            minor=False
        )

        axis.set_xticklabels(
            [
                str(int(val)) if int(val) in [5, 50] else ''
                for val in axis.get_xticks(minor=True)
            ],
            minor=True
        )
        axis.set_xlabel(xlabel)

    if config.left:
        axis.set_ylabel(ylabel)
        if yticks is not None:
            axis.set_yticks(yticks)

    cluster, binary_id = binary.split('_')
    axis.set_title(
        ('PKM ' if cluster == 'NGC188' else 'WOCS ') + binary_id,
        pad=3
    )


def plot_single_burnin_period(binary, system_data, __, axis, config):
    """Plot the required burnin for each quantile of lgQ(Ptide) vs Ptide."""

    axis.set_yscale('log')
    data_behind = plot_single_diagnostic_period(
        system_data['quantiles'],
        'burnin',
        axis,
        config,
        label=(
            None if getattr(plot_single_burnin_period,
                            'labeled_cluster',
                            None) == binary.split('_')[0]
            else True
        )
    )

    axis.axhspan(ymax=system_data['samples'].shape[0], ymin=0, color='black')

    if config.individual_plot_mode == 'subplots':
        plot_single_burnin_period.labeled_cluster = binary.split('_')[0]

    mark_valid_constraint_range(system_data['quantiles'],
                                axis,
                                config)
    axis.set_ylim(10, 2000)
    decorate_subplot(axis, binary, 'emcee steps', config)
    if config.individual_plot_mode == 'pages':
        axis.legend()
    return data_behind


def plot_single_cdfstd_period(binary, system_data, __, axis, config):
    """Plot the required burnin for each quantile of lgQ(Ptide) vs Ptide."""

    axis.set_yscale('log')
    data_behind = plot_single_diagnostic_period(
        system_data['quantiles'],
        'std',
        axis,
        config,
        label=(
            None if getattr(plot_single_cdfstd_period,
                            'labeled_cluster',
                            None) == binary.split('_')[0]
            else True
        )
    )
    if config.individual_plot_mode == 'subplots':
        plot_single_cdfstd_period.labeled_cluster = binary.split('_')[0]
    mark_valid_constraint_range(system_data['quantiles'],
                                axis,
                                config)
    axis.set_ylim(1e-3, 0.03)
    decorate_subplot(axis, binary, r'$\sigma_{CDF}$', config)
    if config.individual_plot_mode == 'pages':
        axis.legend()
    return data_behind


def get_plotting_order(plot_data, restrict_to_cluster=None):
    """Split the given systems by cluster and order them  by orbital period."""

    result = dict()
    cluster_list = (['M35', 'NGC6819', 'NGC188'] if restrict_to_cluster is None
                    else [restrict_to_cluster])
    for cluster in cluster_list:
        period_binary = [
            (min(data['orbit']['Per']), binary)
            for binary, data in plot_data.items()
            if binary.startswith(cluster) and min(data['orbit']['Per']) < 50.0
        ]
        result[cluster] = [entry[1] for entry in sorted(period_binary)]

    if restrict_to_cluster is None or restrict_to_cluster == 'M35':
        for bad_binary in ['M35_15012',
                           'M35_41032',
                           'M35_49043',
                           'NGC188_4999']:
            result[bad_binary.split('_')[0]].remove(bad_binary)

    if restrict_to_cluster is None:
        return result
    return result[cluster]


def get_subplots(num_systems, plot_type, config):
    """Return a properly configured subplots for all systems in a cluster."""

    figure, axes = pyplot.subplots(
        6, 3,
        sharex='col',
        sharey=('row' if plot_type == 'burnin_period' else 'row'),
        gridspec_kw=dict(wspace=0.05, hspace=0.25),
        figsize=(8.5,11)
    )
    axes = axes.flatten()

    for rm_ax in axes[num_systems:]:
        pyplot.delaxes(rm_ax)
    axes = axes[:num_systems]

    if plot_type == 'lgQ_period':
        figure.colorbar(
            cm.ScalarMappable(
                norm=colors.LogNorm(vmin=config.heat_map_contrast, vmax=1)
            ),
            ax=axes,
            location='bottom',
            aspect=40,
            pad=0.07
        )

    return axes


def plot_single_tightest_lgq_posterior(binary, system_data, _, axis, config):
    """Plot PDF(lgQ) at the tidal period with smallest upper quantile."""

    upper_quantile_ind = numpy.argmax(config.convergence_quantiles)
    plot_ptide_ind = numpy.array(
        [entry[upper_quantile_ind][0] for entry in system_data['quantiles']]
    ).argmin()
    evaluated_lgq = evaluate_lgq(
        system_data['samples'],
        numpy.array([config.convergence_ptide_grid[plot_ptide_ind]])
    )
    plot_lgq = numpy.linspace(4.0, 12.0, 1000)
    pdf = CombinedMCMCConstraint(
        plot_lgq,
        config.combined_constraint_kernel_width
    )
    pdf.add_samples(evaluated_lgq)
    axis.plot(plot_lgq, pdf() / (pdf().sum() * (plot_lgq[1] - plot_lgq[0])))
    decorate_subplot(axis,
                     binary,
                     'PDF',
                     config,
                     xlabel=r"$\log_{10}Q_\star'$",
                     xticks=[4,6,8,10,12])
    axis.set_xticks([5,7,9,11], minor=True)
    axis.set_xticklabels(['']*4, minor=True)
    axis.grid(visible=True, which='both')


def save_data_behind_figure(data, title, filename, config):
    """Create a data behind the figure MRT file."""

    tablemaker = CDSTablesMaker()
    tablemaker.title = title
    tablemaker.author, tablemaker.authors = config.mrt_author.split(' and ', 1)
    tablemaker.addTable(data,
                        name=filename,
                        nullvalue=-1)
    tablemaker.toMRT()


def save_individual_data_behind_figure(data, plot_type, binary, config):
    """Return a ready-to-use table maker that can save to AAS MRT format."""

    cluster, binary_id = binary.split('_')
    quantity = dict(
        lgQ_period="Quantiles of log10(Q*') vs tidal period",
        burnin_period="Burn-in period for each quantile vs tidal period",
        cdfstd_period=("Standard deviation of the CDF at each quantile vs "
                       "tidal period")
    )
    title_info = dict(
        cluster=('NGC ' + cluster[3:] if cluster.startswith('NGC')
                 else 'M ' + cluster[1:]),
        id_type=('PKM' if cluster == 'NGC188' else 'WOCS'),
        binary_id=binary_id,
        quantity=quantity[plot_type]
    )
    save_data_behind_figure(
        data,
        '{quantity} for {cluster} binary {id_type} {binary_id}'.format_map(
            title_info
        ),
        binary + '_' + plot_type + '.mrt'
    )


def plot_individual_constraints(plot_data, config):
    """Create plots showing the lgQ(Ptide) constraint for indivdiual systems."""

    plotting_order = get_plotting_order(plot_data)

    config.combined_constraint_heat_map = 'log'

    plot_type_split = dict(
        lgQ_period=[None],
        lgQ_quantiles=config.convergence_ptide_grid,
        quantiles_lgQ=config.convergence_ptide_grid,
        autocorrelation=config.convergence_ptide_grid,
        autocorrelation_time=config.convergence_ptide_grid,
        raftery_lewis_diagnostics=config.convergence_ptide_grid,
        burnin_period=[None],
        cdfstd_period=[None],
        tightest_lgQ_posterior=[None],
    )

    for plot_type in [
            'lgQ_period',
            #'lgQ_quantiles',
            #'quantiles_lgQ',
            #'autocorrelation',
            #'autocorrelation_time',
            #'raftery_lewis_diagnostics',
            'burnin_period',
            'cdfstd_period',
            #'tightest_lgQ_posterior'
    ]:
        print('Plot type: ' + plot_type)
        for cluster in ['M35', 'NGC6819', 'NGC188']:
            print('\tCluster: ' + cluster)
            for plot_split in plot_type_split[plot_type]:
                print('\t\tSplit: ' + repr(plot_split))
                output_fname = cluster + '_' + plot_type
                if plot_split is not None:
                    output_fname += '_' + str(plot_split)
                output_fname += '.pdf'
                if config.individual_plot_mode == 'subplots':
                    axes = get_subplots(len(plotting_order[cluster]),
                                        plot_type,
                                        config)
                else:
                    pdf = PdfPages(output_fname)
                for binary_index, binary in enumerate(plotting_order[cluster]):
                    config.left = (
                        config.individual_plot_mode == 'pages'
                        or
                        binary_index % 3 == 0
                    )
                    config.bottom = (
                        config.individual_plot_mode == 'pages'
                        or
                        binary_index >= len(plotting_order[cluster]) - 3
                    )
                    print('\t\t\tBinary: ' + repr(binary))
                    if plot_type == 'raftery_lewis_diagnostics':
                        assert config.individual_plot_mode == 'pages'
                        data_behind = None
                        plot_single_raftery_lewis_diagnostics(
                            binary,
                            plot_data[binary],
                            plot_split,
                            pyplot.subplots(2, 2)[1].flatten(),
                            config
                        )
                        pyplot.figlegend()
                    else:
                        data_behind = globals()[
                            'plot_single_' + plot_type.lower()
                        ](
                            binary,
                            plot_data[binary],
                            plot_split,
                            (
                                axes[binary_index]
                                if config.individual_plot_mode == 'subplots'
                                else pyplot.gca()
                            ),
                            config
                        )
                    if config.individual_plot_mode == 'pages':
                        pyplot.suptitle(binary
                                        +
                                        ': %d steps'
                                        %
                                        plot_data[binary]['samples'].shape[0])
                        pdf.savefig()
                        pyplot.close()
                    elif plot_type in ['burnin_period', 'cdfstd_period']:
                        pyplot.figlegend(loc='upper center',
                                         ncol=4,
                                         borderaxespad=3)
                    if data_behind is not None:
                        save_individual_data_behind_figure(data_behind,
                                                           plot_type,
                                                           binary,
                                                           config)

                if config.individual_plot_mode == 'subplots':
                    pyplot.savefig(output_fname,
                                   bbox_inches='tight',
                                   pad_inches=0)
                else:
                    pdf.close()


def save_combined_data_behind_figure(data,
                                     included_clusters,
                                     ncombined,
                                     config):
    """Return a ready-to-use table maker that can save to AAS MRT format."""

    tablemaker = CDSTablesMaker()
    tablemaker.title = (
        (
            "Quantiles of the combined distribution of log10(Q*') vs tidal "
            "period for {0} binaries".format(
                ', '.join(included_clusters[:-1])
                +
                (',' if len(included_clusters) > 2 else '')
                +
                (
                    ' and ' + included_clusters[-1]
                    if len(included_clusters) > 1 else
                    included_clusters[0]
                )
            )
        )
    )
    tablemaker.author, tablemaker.authors = config.mrt_author.split(' and ', 1)
    tablemaker.addTable(
        data,
        name='_'.join(included_clusters + [str(ncombined), 'combined_lgQ.mrt'])
    )
    tablemaker.toMRT()


def plot_combined_constraints(plot_data, config):
    """Plot heat-map of joint constraint from all systems in a cluster."""

    orig_font_size = rcParams['font.size']
    rcParams['font.size'] = '24'
    config.combined_constraint_heat_map = 'log'
    include_binaries = get_plotting_order(plot_data)

    numpy.set_printoptions(precision=16, floatmode='fixed', linewidth=100)
    config.combined_constraint_lgQ_grid = numpy.linspace(4, 7, 100)
    config.ptide_grid = numpy.linspace(*config.combined_constraint_period_range,
                                       100)
    all_combined_plotter = FrequencyDependencePlotter(0, config)

    fully_combined_n = 1

    assert include_binaries['NGC188'][6] == 'NGC188_4904'
    include_binaries['NGC188'][6] = include_binaries['NGC188'][-1]
    include_binaries['NGC188'][-1] = 'NGC188_4904'

    selected_quantiles = dict(ptide_grid=config.ptide_grid)

    combined_clusters = []
    for cluster in ['NGC6819', 'NGC188', 'M35']:
        print(cluster + ':')
        combined_clusters.append(cluster)
        cluster_plotter = FrequencyDependencePlotter(
            len(include_binaries[cluster]),
            config
        )
        for nadded, binary in enumerate(include_binaries[cluster]):
            burnin = get_burnin(plot_data[binary], config, binary)
            samples = plot_data[binary]['samples']
            valid_ptide_indices = get_valid_ptide_indices(
                plot_data[binary]['quantiles'],
                config
            )
            if valid_ptide_indices is None:
                period_range = (-numpy.inf, numpy.inf)
            else:
                period_range = numpy.array(config.convergence_ptide_grid)[
                    valid_ptide_indices[[0,-1]]
                ]
            min_lgq = 4 if samples['lgQ_min'].min() < 5 else 5

            cluster_plotter.add_chain(
                samples[burnin:],
                None,
                (4 if samples['lgQ_min'].min() < 5 else 5, numpy.inf),
                period_range
            )
            all_combined_plotter.add_chain(
                samples[burnin:],
                None,
                (min_lgq, numpy.inf),
                period_range
            )

            for output_fname, plotter, included_clusters, ncombined in [
                    (
                        cluster + '_combined_lgQ_period_%d.pdf' % (nadded + 1),
                        cluster_plotter,
                        [cluster],
                        (nadded + 1)
                    ),
                    (
                        'fully_combined_lgQ_period_%d.pdf' % fully_combined_n,
                        all_combined_plotter,
                        combined_clusters,
                        fully_combined_n
                    )
            ]:
                pyplot.figure(figsize=(11, 8.5), dpi=300, tight_layout=True)
                pyplot.xscale('log')
                print('    Adding ' + repr(binary))
                pyplot.ylim(min_lgq, 6.0 if cluster == 'M35' else 7.0)
                pyplot.xlim(*config.combined_constraint_period_range)
                plotter.plot_combined_pdf_heat_map()
                data_behind = plotter.plot_combined_quantiles(
                    config.convergence_quantiles,
                    fmt='-k'
                )
#                pyplot.figlegend(loc='lower center',
#                                 ncol=2,
#                                 bbox_to_anchor=(0.5, 1.0))
                pyplot.gca().set_xticklabels(
                    [
                        str(int(val) if val - int(val) == 0 else val)
                        for val in pyplot.gca().get_xticks(minor=False)
                    ],
                    minor=False
                )
                pyplot.gca().set_xticklabels(
                    [
                        str(int(val) if val - int(val) == 0 else val)
                        for val in pyplot.gca().get_xticks(minor=True)
                    ],
                    minor=True
                )

                pyplot.colorbar()
                print('    Plotting')
                pyplot.savefig(output_fname,
                               bbox_inches='tight',
                               pad_inches=0.0)
                print('    Created: ' + repr(output_fname))
                save_combined_data_behind_figure(data_behind,
                                                 included_clusters,
                                                 ncombined,
                                                 config)
                pyplot.cla()
                pyplot.clf()

            fully_combined_n += 1
        if cluster == 'M35':
            selected_quantiles[cluster] = [
                cluster_plotter.combined_pdf.ppf(cdf)
                for cdf in config.convergence_quantiles
            ]
        else:
            selected_quantiles['NGC6819'] = selected_quantiles['NGC188'] = [
                all_combined_plotter.combined_pdf.ppf(cdf)
                for cdf in config.convergence_quantiles
            ]
    return selected_quantiles


def create_tightest_constraint_latex(data_behind, config):
    """Save individual tightest vs global constraints as latex table."""

    latex_columns = []
    for colname in data_behind.colnames:
        latex_columns.append(
            r'\multicolumn{{1}}{{c@{{{0}}}}}{{\textbf{{{1}}}}}'.format(
                (
                    r'\quad\quad\quad' if colname in ['PKM',
                                                      'WOCS',
                                                      'Ptide',
                                                      'CDF-1(97.7%)']
                    else ''
                ),
                colname.replace(
                    'CDF-1(', ''
                ).replace(
                    'Comb. ', ' '
                ).replace(
                    ')', ''
                ).replace(
                    '%', r'\%'
                )
            )
        )
    data_behind.rename_columns(
        data_behind.colnames,
        latex_columns
    )
    data_behind.write(
        cluster + '_individual_vs_combined_constraints.tex',
        format='latex',
        latexdict=dict(
            tabletype=None,
            col_align=(
                r'c@{\quad\quad\quad}c@{\quad\quad\quad}'
                +
                r'cccc@{\quad\quad\quad}'
                +
                'c'*(2 if cluster == 'M35' else 4)
            ),
            header_start=(
                r'&&\multicolumn{4}{c@{\quad\quad\quad}}{\textbf{Individual CDF$^{-1}$}} & '
                +
                r'\multicolumn{'
                +
                ('2' if cluster == 'M35' else '4')
                +
                r'}{c}{\textbf{Combined CDF$^{-1}$}}\\'
            ),
            header_end='\\hline\n\\hline'
        ),
        formats={
            colname: (
                '%6s' if ('{PKM}' in colname or '{WOCS}' in colname)
                else '%5.2f'
            )
            for colname in data_behind.colnames
        },
        overwrite=True
    )


def plot_tightest_constraints(plot_data, config, combined_quantiles=None):
    """Plot tightest constraints as error bars vs tidal period."""

    include_binaries = get_plotting_order(plot_data)
    offsets = numpy.zeros(len(config.convergence_ptide_grid), dtype=int)
    for cluster in ['M35', 'NGC6819', 'NGC188']:
        pyplot.xscale('log')
        lgq_range = ((4, 8) if cluster == 'M35' else (5, 9))
        cluster_quantiles = numpy.empty((len(config.convergence_quantiles),
                                         len(include_binaries[cluster])))
        plot_ptide = numpy.empty(len(include_binaries[cluster]))
        data_behind = Table(
            [
                [b.split('_')[1] for b in include_binaries[cluster]],
                numpy.empty(len(include_binaries[cluster]))
            ],
            names=[
                ('PKM' if cluster == 'NGC188' else 'WOCS'),
                'Ptide'
            ],
            descriptions=[
                "Binary identifier",
                (
                    "The tidal period at which the {0:.1f}% quantile of "
                    "log10(Q') of the binary is smallest."
                ).format(100.0 * max(config.convergence_quantiles))
            ]
        )
        for binary_ind, binary in enumerate(include_binaries[cluster]):
            upper_quantile_ind = numpy.argmax(config.convergence_quantiles)
            ptide_ind = numpy.array([
                entry[upper_quantile_ind][0]
                for entry in plot_data[binary]['quantiles']
            ]).argmin()
            data_behind['Ptide'][binary_ind] = (
                config.convergence_ptide_grid[ptide_ind]
            )
            plot_ptide[binary_ind] = (
                config.convergence_ptide_grid[ptide_ind]
                *
                1.1**(
                    numpy.ceil(offsets[ptide_ind] / 2)
                    *
                    (-1)**(offsets[ptide_ind] % 2 + 1)
                )
            )
            offsets[ptide_ind] += 1

            cluster_quantiles[:, binary_ind] = [
                q[0] for q in plot_data[binary]['quantiles'][ptide_ind]
            ]
            print(
                '%s: best Ptide = %s (%d), lgQ = %s'
                %
                (
                    binary,
                    repr(plot_ptide[binary_ind]),
                    ptide_ind,
                    repr(cluster_quantiles[:, binary_ind])
                )
            )

        cdf_labels = [
            'CDF-1({0:.1f}%)'.format(100.0 * cdf)
            for cdf in config.convergence_quantiles
        ]
        data_behind.add_columns(
            cluster_quantiles,
            names=cdf_labels
        )
        for label, cdf in zip(cdf_labels, config.convergence_quantiles):
            data_behind[label].description = (
                "The {0:.1f}% quantile of log10(Q') at the tidal period with "
                "smallest {1:.1f}% quantile."
            ).format(100.0 * cdf, 100.0 * max(config.convergence_quantiles))
        elinewidth = 2
        ecolor = None
        while cluster_quantiles.shape[0] > 1:
            ecolor = pyplot.errorbar(
                plot_ptide,
                0.5 * (cluster_quantiles[-1] + cluster_quantiles[0]),
                yerr=0.5 * (cluster_quantiles[-1] - cluster_quantiles[0]),
                fmt='none',
                elinewidth=elinewidth,
                ecolor=ecolor,
                label=(cluster if elinewidth == 2 else None),
            )[2][0].get_color()
            elinewidth += 2
            cluster_quantiles = cluster_quantiles[1:-1]
        if combined_quantiles is not None:
            for plot_y, label, cdf in zip(combined_quantiles[cluster],
                                          cdf_labels,
                                          config.convergence_quantiles):
                pyplot.plot(
                    combined_quantiles['ptide_grid'],
                    plot_y,
                    '-k',
                    label=label,
                    linewidth=3,
                    zorder=20
                )
                data_behind['Comb. ' + label] = interp1d(
                    combined_quantiles['ptide_grid'],
                    plot_y,
                    bounds_error=False,
                    fill_value=-1
                )(
                    plot_ptide
                )
                data_behind['Comb. ' + label].description = (
                    "The {0:.1f}% quantile of the combined log10(Q') constraint"
                    " at the tidal period with smallest {1:.1f}% quantile."
                ).format(100.0 * cdf,
                         100.0 * max(config.convergence_quantiles))

        pyplot.xlabel('Tidal Period [days]')
        pyplot.ylabel(r"$\log_{10}Q_\star'$")
        pyplot.ylim(*lgq_range)
#        pyplot.legend()
        pyplot.gca().set_xticks(range(2, 11))
        pyplot.gca().set_xticklabels([str(i) for i in range(2, 11)])

        pyplot.savefig(cluster + '_tightest_constraints.pdf',
                       bbox_inches='tight',
                       pad_inches=0)
        pyplot.clf()
        if cluster == 'M35':
            data_behind.remove_columns(['Comb. CDF-1(2.3%)',
                                        'Comb. CDF-1(15.9%)'])
        save_data_behind_figure(
            data_behind,
            'Comparison between the combined constraints and the individual '
            'constraints for the {0} binaries'.format(cluster),
            cluster + '_individual_vs_combined_constraints.mrt',
            config
        )
        create_tightest_constraint_latex(data_behind, config)


def main(config):
    """Avoid polluting global namespace."""

    if not config.skip_download:
        download_latest_samples(config.samples_dir)

    plot_data = get_sampling_data(config)
    combined_quantiles = None
#    plot_individual_constraints(plot_data, config)
    #combined_quantiles = plot_combined_constraints(plot_data, config)
    if (
            combined_quantiles is None
            and
            path.exists(config.combined_quantiles_pickle)
    ):
        with open(config.combined_quantiles_pickle, 'rb') as quanntile_pickle:
            combined_quantiles = pickle.load(quanntile_pickle)
    elif combined_quantiles is not None:
        with open(config.combined_quantiles_pickle, 'wb') as quanntile_pickle:
            pickle.dump(combined_quantiles, quanntile_pickle)
    plot_tightest_constraints(plot_data, config, combined_quantiles)


if __name__ == '__main__':
    main(parse_command_line())
