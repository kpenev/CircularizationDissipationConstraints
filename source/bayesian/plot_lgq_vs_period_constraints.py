#!/usr/bin/env python3

"""Create grid of color map plots of the constraints ready for article."""

from subprocess import call
from os import path, listdir

from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages
from configargparse import\
    ArgumentParser,\
    DefaultsFormatter,\
    Action as ArgparseAction
from scipy.stats import rdist
import numpy
from kde import KDEDistribution

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

def parse_command_line():
    """Parse command line for plotting configuration."""

    #Interface specified by argparse module.
    #pylint: disable=too-few-public-methods
    class ParseBurnIn(ArgparseAction):
        """Action for parsing burn-in arguments."""

        def __call__(self, parser, namespace, values, option_string=None):
            """Parse a list of burn-in arguments."""

            for arg in values:
                system = 'default'
                try:
                    burnin = int(arg)
                except ValueError:
                    system, burnin = arg.split(':')
                    burnin = int(burnin)
                getattr(namespace, self.dest)[system] = burnin
    #pylint: enable=too-few-public-methods



    parser = ArgumentParser(
        description=__doc__,
        default_config_files=[
            path.splitext(__file__)[0] + '.cfg'
        ],
        formatter_class=DefaultsFormatter,
        ignore_unknown_config_file_keys=False
    )

    parser.add_argument(
        '--samples-dir',
        default=path.join(
            path.dirname(
                path.dirname(
                    path.abspath(
                        __file__
                    )
                )
            ),
            'samples'
        ),
        help='The directory holding the HDF5 files with MCMC samples.'
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
        '--convergence-lgQ',
        default=numpy.arange(5.0, 13.0, 1.0),
        nargs='+',
        type=float,
        help='The lg(Q) values whose quantile evolution to display on the '
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
        '--burn-in',
        nargs='+',
        default=dict(default=0),
        action=ParseBurnIn,
        help='Specify burn-in to use for all or some of the systems. Isolated '
        'numbers set default burn-in (applied to systems for which no specific '
        'burn-in is set) and entries of the form <cluster_name>:<number> '
        'specify burn-in for a given system. If passed multiple times it is as '
        'if passed once with the combined list of arguments. Later values '
        'overwrite earlier ones.'
    )
    parser.add_argument(
        '--individual-plot-mode', '--individual-plots',
        choices=['pages', 'subplots'],
        default='subplots',
        help='Choose whether plots of individual systems will be saved each on '
        'a separate page in a multi-page PDF file (`pages`) or as grid of '
        'sub-plots in a single figure (`subplots`).'
    )
    add_frequency_dependence_plot_config(parser)

    return parser.parse_args()

def download_latest_samples(destination):
    """Download the latest samples files from stampede2 and ganymede."""

    for source in [
            'kxp174430@ganymede:~/*_powerlaw_alllock_*mcmc_samples.h5',
            'kpenev@stampede2.tacc.utexas.edu:~/*_mcmc_powerlawlgQ_samples.h5'
    ]:
        call(['rsync', '-avz', '--progress', source, destination])

def get_sampling_data(config):
    """Return dictionary with system keys containing the samples and logprob."""

    result = dict()
    for samples_fname in listdir(config.samples_dir):
        try:
            system, samples, log_probability = get_plot_data(
                path.join(config.samples_dir, samples_fname),
                0,
                config.chain_condition
            )
            burn_in = config.burn_in.get(system, config.burn_in['default'])
            result[system] = (samples[burn_in:], log_probability[burn_in:])
        except AssertionError:
            pass

    return result

def plot_single_lgq_period(samples, _, axis, config):
    """Plot the constraint for a single system."""

    pyplot.xscale('log')
    orig_axis = pyplot.gca()
    pyplot.sca(axis)
    frequency_dependence_plotter = FrequencyDependencePlotter(1, config)
    frequency_dependence_plotter.add_chain(samples, None)
    pyplot.ylim(4.0, 12.0)
    frequency_dependence_plotter.plot_combined_pdf_heat_map()
    pyplot.sca(orig_axis)


def plot_single_convergence(*,
                            samples,
                            ptide,
                            quantity,
                            values,
                            combine_nsteps,
                            axis):
    """Plot a distribution quantity vs step number for a single system/Ptide."""

    pyplot.xscale('linear')

    shape = samples['lgQ_min'].shape
    lgq_values = evaluate_lgq(samples, numpy.array([ptide])).reshape(shape)

    plot_x = numpy.arange(combine_nsteps + 1,
                          len(lgq_values))
    plot_y = numpy.empty((plot_x.size, len(values)),
                          dtype=float)

    print('Samples (%d x %d): ' % lgq_values.shape + repr(lgq_values))
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
        print('Progress: %d/%d' % (x_ind, plot_x.size))

    print('Plot y: ' + repr(plot_y))
    axis.plot(plot_x, plot_y, '-x')



def plot_single_lgq_quantiles(samples, ptide, axis, config):
    """Plot lgQ @ quantiles vs step number for a single system/tidal period."""

    plot_single_convergence(samples=samples,
                            ptide=ptide,
                            quantity='ppf',
                            values=config.convergence_quantiles,
                            combine_nsteps=config.convergence_combine_nsteps,
                            axis=axis)


def plot_single_quantiles_lgq(samples, ptide, axis, config):
    """Plot quantile(lgQ) vs step number for a single system/tidal period."""

    plot_single_convergence(samples=samples,
                            ptide=ptide,
                            quantity='cdf',
                            values=config.convergence_lgQ,
                            combine_nsteps=config.convergence_combine_nsteps,
                            axis=axis)


def get_plotting_order(system_list):
    """Split the given systems by cluster and order them  by orbital period."""

    result = dict()
    for cluster in ['M35', 'NGC6819', 'NGC188']:
        sb1_orbits = globals()['get_'
                               +
                               cluster.lower()
                               +
                               '_binary_data']()[0]
        period_binary = [
            (
                min(select_binary_data(sb1_orbits,
                                       None,
                                       'PKM' if cluster == 'NGC188' else 'WOCS',
                                       int(binary.split('_')[1]))['Per']),
                binary
            )
            for binary in system_list if binary.startswith(cluster)
        ]
        result[cluster] = [entry[1] for entry in sorted(period_binary)]

    return result

def plot_individual_constraints(plot_data, config):
    """Create plots showing the lgQ(Ptide) constraint for indivdiual systems."""

    plotting_order = get_plotting_order(plot_data.keys())

    config.combined_constraint_heat_map = 'log'

    plot_type_split = dict(lgQ_period=[None],
                           lgQ_quantiles=config.convergence_ptide_grid,
                           quantiles_lgQ=config.convergence_ptide_grid)

    for plot_type in ['lgQ_quantiles', 'quantiles_lgQ']:#['lgQ_period', 'lgQ_quantiles']:
        for cluster in ['M35', 'NGC6819', 'NGC188']:
            for plot_split in plot_type_split[plot_type]:
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
                    globals()['plot_single_' + plot_type.lower()](
                        plot_data[binary][0],
                        plot_split,
                        (
                            axes[binary_index]
                            if config.individual_plot_mode == 'subplots' else
                            pyplot.gca()
                        ),
                        config
                    )
                    if config.individual_plot_mode == 'pages':
                        pyplot.title(binary)
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
