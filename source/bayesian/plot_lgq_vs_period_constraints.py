#!/usr/bin/env python3

"""Create grid of color map plots of the constraints ready for article."""

from subprocess import call
from os import path, listdir
import hashlib
import pickle
from multiprocessing import Pool

from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages
from configargparse import ArgumentParser, DefaultsFormatter
from scipy.stats import rdist
import numpy
from kde import KDEDistribution

from emcee_autocorrelation import\
    max_likelihood_autocorr,\
    average_autocorr
from mcmc_quantile_convergence import get_raftery_lewis_diagnostics
from emcee_quantile_convergence import find_emcee_quantiles

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
        '--skip-download',
        default=False,
        action='store_true',
        help='If passed, does not download the latest samples from stampede2 '
        'and ganymede.'
    )
    parser.add_argument(
        '--convergence-ptide-grid',
        default=[1.0, 2.0, 4.0, 8.0, 16.0],
        nargs='+',
        type=float,
        help='The tidal periods at which to generate convergence plots.'
    )
    parser.add_argument(
        '--convergence-quantiles',
        default=[0.1, 0.2, 0.4, 0.6, 0.8, 0.9],
        nargs='+',
        type=float,
        help='The quantiles for which to display the lgQ evolution on the '
        'convergence plots.'
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
        '--nthreads',
        default=16,
        help='The maximum number of parallel threads to use for calculating '
        'quantile diagnostics (the only slow part of preparing plotting data).'
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


def add_orbit_data(sampling_data):
    """Add orbit data to `sampling_data`."""

    sb1_orbits = dict()
    for cluster in ['M35', 'NGC6819', 'NGC188']:
        sb1_orbits[cluster] = globals()['get_'
                                        +
                                        cluster.lower()
                                        +
                                        '_binary_data']()[0]
    for binary in sampling_data.keys():
        cluster = binary.split('_')[0]
        sampling_data[binary]['orbit'] = select_binary_data(
            sb1_orbits[cluster],
            None,
            'PKM' if cluster == 'NGC188' else 'WOCS',
            int(binary.split('_')[1])
        )


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
        flattened_quantiles =workers.starmap(
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


def get_sampling_data(config):
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
        quantiles = get_quantiles(samples, config)
        result[system] = dict(samples=samples,
                              log_probability=log_probability,
                              quantiles=quantiles)
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

    add_orbit_data(result)
    return result


def plot_single_lgq_period(_, system_data, __, axis, config):
    """Plot the constraint for a single system."""

    pyplot.xscale('log')
    orig_axis = pyplot.gca()
    pyplot.sca(axis)
    frequency_dependence_plotter = FrequencyDependencePlotter(1, config)
    frequency_dependence_plotter.add_chain(system_data['samples'], None)
    pyplot.ylim(4.0, 12.0)
    frequency_dependence_plotter.plot_combined_pdf_heat_map()
    pyplot.plot(
        config.convergence_ptide_grid,
        [
            [
                system_data['quantiles'][ptide_ind][quantile_ind][0]
                for quantile_ind in range(len(config.convergence_quantiles))
            ]
            for ptide_ind in range(len(config.convergence_ptide_grid))
        ],
        'xk',
        zorder=20
    )
    pyplot.axvline(x=min(system_data['orbit']['Per']),
                   color='black',
                   linewidth=5,
                   zorder=20)

    pyplot.sca(orig_axis)


#pylint: disable=too-many-locals
def plot_single_convergence(*,
                            binary,
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

    try:
        print(
            '\t\t\t%s: autocorrelation = %f (%f), chain length = %d'
            %
            (
                binary,
                max_likelihood_autocorr(lgq_values.T, 1),
                average_autocorr(lgq_values.T, 1),
                shape[0]
            )
        )
    except ValueError:
        print('%s: Failed to determine autocorrelation, chain length = %d'
              %
              (binary, shape[0]))

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


def get_plotting_order(plot_data, restrict_to_cluster=None):
    """Split the given systems by cluster and order them  by orbital period."""

    result = dict()
    cluster_list = (['M35', 'NGC6819', 'NGC188'] if restrict_to_cluster is None
                    else [restrict_to_cluster])
    for cluster in cluster_list:
        period_binary = [
            (min(data['orbit']['Per']), binary)
            for binary, data in plot_data.items() if binary.startswith(cluster)
        ]
        result[cluster] = [entry[1] for entry in sorted(period_binary)]

    if restrict_to_cluster is None:
        return result
    return result[cluster]


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
        raftery_lewis_diagnostics=config.convergence_ptide_grid
    )

    for plot_type in [
            'lgQ_period',
            'lgQ_quantiles',
            'quantiles_lgQ',
            'autocorrelation',
            'autocorrelation_time',
            'raftery_lewis_diagnostics'
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
                    _, axes = pyplot.subplots(4, 5, sharex=True, sharey=True)
                    axes = axes.flatten()
                else:
                    pdf = PdfPages(output_fname)
                for binary_index, binary in enumerate(plotting_order[cluster]):
                    print('\t\t\tBinary: ' + repr(binary))
                    if plot_type == 'raftery_lewis_diagnostics':
                        assert config.individual_plot_mode == 'pages'
                        plot_single_raftery_lewis_diagnostics(
                            binary,
                            plot_data[binary],
                            plot_split,
                            pyplot.subplots(2, 2)[1].flatten(),
                            config
                        )
                        pyplot.figlegend()
                    else:
                        globals()['plot_single_' + plot_type.lower()](
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

                if config.individual_plot_mode == 'subplots':
                    pyplot.savefig(output_fname)
                else:
                    pdf.close()


def main(config):
    """Avoid polluting global namespace."""

    if not config.skip_download:
        download_latest_samples(config.samples_dir)

    plot_data = get_sampling_data(config)
    plot_individual_constraints(plot_data, config)


if __name__ == '__main__':
    main(parse_command_line())
