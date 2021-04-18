#!/usr/bin/env python3

"""Create plots of emcee sampling results."""

from abc import ABC, abstractmethod
import logging

import matplotlib
from matplotlib import pyplot
from configargparse import\
    ArgumentParser,\
    DefaultsFormatter,\
    Action as ArgparseAction
import numpy
import emcee
import h5py
import pandas
from scipy import stats
from asteval import Interpreter

from visuals import make_corner_plot

def parse_command_line():
    """Parse the command line for what and how to plot."""

    #Interface specified by argparse module.
    #pylint: disable=too-few-public-methods
    class ParseGrid(ArgparseAction):
        """Action for parsing linspace arguments."""

        def __call__(self, parser, namespace, values, option_string=None):
            """Parse a grid option to a numpy.arary()."""

            setattr(namespace,
                    self.dest,
                    numpy.linspace(float(values[0]),
                                   float(values[1]),
                                   int(values[2])))
    #pylint: enable=too-few-public-methods

    parser = ArgumentParser(
        description=__doc__,
        default_config_files=['visualize_emcee.cfg'],
        formatter_class=DefaultsFormatter,
        ignore_unknown_config_file_keys=False
    )
    parser.add_argument(
        'samples_fnames',
        nargs='+',
        help='The filename(s) containing the stored emcee chain(s) to plot. For'
        ' now only frequency dependence plots support plotting multiple '
        'constraints on top of each other. All other plots only plot the chain '
        'from the last file.'
    )
    parser.add_argument(
        '--chain-name', '--chain',
        default=None,
        help='Select which chain in the given file to plot. By default, plots '
        'the longest chain in the input file. Append ":<number>" after a '
        'filename to specify a burn-in number of samples.'
    )
    parser.add_argument(
        '--corner-plot-fname', '--corner-plot', '--corner',
        default=None,
        help='If specified, a coner plot is created and saved with the given '
        'filename.'
    )
    parser.add_argument(
        '--trace-plot-fname', '--traces-plot', '--traces',
        default=None,
        help='If specified, a trace plot is generated and saved with the given '
        'filaname.'
    )
    parser.add_argument(
        '--frequency-dependence-plot-fname',
        '--frequency-dependence-plot',
        '--frequency',
        default=None,
        help='If specified, a plot of the confidence interval of lgQ vs tidal '
        'frequency is generated and saved with the given filename. Inertial '
        'mode enhancement is ignored.'
    )
    parser.add_argument(
        '--errorbar-plot',
        nargs=3,
        metavar=('FILENAME', 'XEXPR', 'YEXPR'),
        default=None,
        help='If specified, for each input system the X and Y expressions are '
        'evaluated from the chain to get confidence intervals (see '
        '--plot-confidence argument) and a plot of the results is saved under '
        'the given file name.'
    )
    parser.add_argument(
        '--max-traces-per-plot',
        type=int,
        default=5,
        help='The maximum number of traces to include in a trace plot (more '
        'walkers will results in more panels in the plot).'
    )
    parser.add_argument(
        '--plot-confidence',
        type=float,
        nargs='+',
        default=[stats.norm.cdf(1.0) - stats.norm.cdf(-1.0)],
        help='Specify the confidence to display in the dissipation vs tidal '
        'frequency plot (see --frequency-dependence-plot-fname argument).'
    )
    parser.add_argument(
        '--ptide-grid',
        nargs=3,
        default=numpy.linspace(1.0, 20.0, 100),
        action=ParseGrid,
        metavar=('MIN_PERIOD', 'MAX_PERIOD', 'RES'),
        help='Set the range and resolution of the tidal period to include in '
        'the frequency dependence plot (see --frequency-dependence-plot-fname '
        'argument).'
    )
    parser.add_argument(
        '--frequency-dependence-plot-no-lines',
        action='store_true',
        help='If passed, the frequency dependence plot will not include '
        'individual lines.'
    )
    parser.add_argument(
        '--log-x',
        action='store_true',
        help='Switch the x-axis to log-scale.'
    )
    return parser.parse_args()

def get_backend(samples_fname, chain_name):
    """Return the chain to plot."""

    if chain_name is None:
        with h5py.File(samples_fname, 'r') as chain_file:
            available_chains = list(chain_file.keys())
        longest_chain = 0
        for try_chain_name in available_chains:
            backend = emcee.backends.HDFBackend(samples_fname,
                                                name=try_chain_name,
                                                read_only=True)
            if backend.iteration > longest_chain:
                selected_backend = backend
                chain_name = try_chain_name
                longest_chain = backend.iteration
        if longest_chain == 0:
            return None, None
    else:
        selected_backend = emcee.backends.HDFBackend(samples_fname,
                                                     name=chain_name,
                                                     read_only=True)
    with h5py.File(samples_fname, 'r') as chain_file:
        system_name = chain_file[chain_name].attrs['system']
    return selected_backend, system_name

def get_confidence_interval(samples, confidence):
    """
    Return the confidence interval for quantities given sampled values.

    Args:
        samples(2-D array):    The first index should iterate over quantities to
            get confidence interval for, and the second index should iterate
            over values for those quantities at each sample in the MCMC chain.

        confidence(float):    The confidence level at which to find the
            interval.

    Returns:
        1-D array:
            The lower end of the confidence interval for earch input quantity.

        1-D array:
            The upper end of the confidence interval for earch input quantity.
    """

    samples = numpy.sort(samples, 1)

    num_exclude = int(
        numpy.floor(
            (1.0 - confidence) * samples.shape[1]
        )
    )
    lower = samples[:, :num_exclude + 1]
    upper = samples[:, -num_exclude-1:]
    selected_indices = numpy.argmin(upper - lower, axis=1)
    lower = numpy.take_along_axis(lower,
                                  selected_indices[:, None],
                                  axis=1).flatten()
    upper = numpy.take_along_axis(upper,
                                  selected_indices[:, None],
                                  axis=1).flatten()
    return lower, upper

def save_corner_plot(samples, log_probability, config):
    """Create the corner plot specified on the command line."""

    samples = samples.flatten()
    log_probability = log_probability.flatten()
    include_params = [
        param
        for param in samples.dtype.names
        if samples[param].min() != samples[param].max()
    ]
    plot_data_frame = pandas.DataFrame(samples[include_params])
    plot_data_frame.insert(len(include_params), 'lnP', log_probability)
    make_corner_plot(plot_data_frame, config.corner_plot_fname)

def save_trace_plot(samples, log_probability, config):
    """Create the trace plot specified on the command line."""

    num_panels = samples.shape[1] // config.max_traces_per_plot
    _, panel_list = pyplot.subplots(num_panels,
                                    1,
                                    sharex=True,
                                    figsize=[6.4, 2.4 * num_panels])
    finite_probability = numpy.isfinite(log_probability)
    for first, panel in zip(
            range(0, samples.shape[1], config.max_traces_per_plot),
            panel_list
    ):
        for trace in range(first, first + config.max_traces_per_plot):
            plot_y = samples['lgQ_min'][:, trace].flatten()
            plot_x = numpy.arange(plot_y.size)
            plot_solid = finite_probability[:, trace].flatten()
            plot_dotted = numpy.logical_not(plot_solid)
            plot_dotted[numpy.argmax(plot_solid)] = True
            solid_line = panel.plot(plot_x[plot_solid],
                                    plot_y[plot_solid],
                                    '-')[0]
            panel.plot(plot_x[plot_dotted],
                       plot_y[plot_dotted],
                       ':',
                       color=solid_line.get_color())


    pyplot.savefig(config.trace_plot_fname)

class FrequencyDependencePlotterBase(ABC):
    """
    Create a plot showing the constraint of lgQ vs tidal frequency.

    If multiple MCMC chain files are specified, all are overplotted on the same
    axes.
    """

    def __init__(self, config):

        self.config = config
        self.transparency = 2.0 / (len(config.plot_confidence)
                                   *
                                   len(config.samples_fnames))
        self._hatch_index = 0
        self._hatch_list = ['\\\\', '//', '||', '--', '...', 'oo', '+', 'x', '*', 'O']

    @abstractmethod
    def evaluate_lgq(self, samples):
        """
        Evaluate the dissipation model that was sampled as function of period.

        Args:
            samples:    The samples directly as returned from the backend.

        Returns:
            2-D array:
                The first index should corresponding to `self.config.ptide_grid`
                The second index should iterate over points in the MCMC chain.
        """

    def add_chain(self, samples, label):
        """Overplot another chain of samples."""

        evaluated_lgq = self.evaluate_lgq(samples)

        if not self.config.frequency_dependence_plot_no_lines:
            pyplot.plot(
                self.config.ptide_grid,
                evaluated_lgq[:, ::1],
                '-k',
                linewidth=5,
                alpha=0.01,
                zorder=0
            )

        for conf_index, confidence in enumerate(
                self.config.plot_confidence
        ):
            min_lgq, max_lgq = get_confidence_interval(evaluated_lgq,
                                                       confidence)
            plot_kwargs = dict(
                #alpha=self.transparency,
                edgecolor='black',
                facecolor='none',
                zorder=10,
                hatch=self._hatch_list[self._hatch_index
                                       %
                                       len(self._hatch_list)],
                linewidth=0
            )
            self._hatch_index += 1
            if conf_index == 0:
                plot_kwargs['label'] = label
            pyplot.fill_between(
                self.config.ptide_grid,
                min_lgq,
                max_lgq,
                **plot_kwargs
            ).get_facecolor()

    def save(self):
        """Save the plot to the file sepecified by the init configuration."""

        pyplot.xlabel(r'Orbital Period [days]')
        pyplot.ylabel(r"$\log_{10}Q_\star'$")
        pyplot.savefig(self.config.frequency_dependence_plot_fname)

class PowerlawLgQDependencePlotter(FrequencyDependencePlotterBase):
    """Plot lgQ constrant from chains assuming a single saturating powerlaw."""

    def evaluate_lgq(self, samples):

        return (
            samples['lgQ_min'].flatten()[None, :]
            +
            numpy.maximum(
                0.0,
                (
                    samples['lgQ_powerlaw'].flatten()[None, :]
                    *
                    numpy.log10(
                        self.config.ptide_grid[:, None]
                        /
                        samples['lgQ_break_period'].flatten()[None, :]
                    )
                )
            )
        )

def add_errorbar(samples, config):
    """Add a single error bar per --errorbar-plot configuration argument."""

    x_expression, y_expression = config.errorbar_plot[1:]
    evaluate = Interpreter()
    for quantity in samples.dtype.names:
        evaluate.symtable[quantity] = samples[quantity].flatten()

    x_samples = evaluate(x_expression)
    y_samples = evaluate(y_expression)
    print('X (%d): ' % x_samples.shape + repr(x_samples))
    line_width = 1
    for confidence in reversed(sorted(config.plot_confidence)):
        x = numpy.median(x_samples)
        y = numpy.median(y_samples)
        x_min, x_max = get_confidence_interval(numpy.atleast_2d(x_samples),
                                               confidence)
        y_min, y_max = get_confidence_interval(numpy.atleast_2d(y_samples),
                                               confidence)
        pyplot.errorbar(
            x=x,
            y=y,
            xerr=[[x - x_min], [x_max - x]],
            yerr=[[y - y_min], [y_max - y]],
            elinewidth=line_width,
            color='black'
        )
        line_width += 2

def main(config):
    """"Avoid polluting global namespace."""

    if config.log_x:
        pyplot.gca().set_xscale('log')

    frequency_dependence_plotter = PowerlawLgQDependencePlotter(config)
    for samples_fname in config.samples_fnames:
        burn_in = 0
        if ':' in samples_fname:
            samples_fname, burn_in = samples_fname.split(':')
            burn_in = int(burn_in)
        backend, system_name = get_backend(samples_fname, config.chain_name)
        if backend is None:
            print('Empty chain in %s, skipping!' % samples_fname)
            continue
        samples = backend.get_blobs(discard=burn_in)
        log_probability = backend.get_log_prob(discard=burn_in)
        if config.corner_plot_fname:
            save_corner_plot(samples, log_probability, config)
        if config.trace_plot_fname:
            save_trace_plot(samples, log_probability, config)
        if config.frequency_dependence_plot_fname:
            frequency_dependence_plotter.add_chain(
                samples,
                system_name.replace('_', ' ')
            )
        if config.errorbar_plot:
            add_errorbar(samples, config)

    pyplot.ylim(5.0, 12.0)
    pyplot.legend()

    if config.frequency_dependence_plot_fname:
        frequency_dependence_plotter.save()
    if config.errorbar_plot:
        pyplot.savefig(config.errorbar_plot[0])

if __name__ == '__main__':
    matplotlib.use('Agg')
    logging.basicConfig(level=logging.CRITICAL)
    main(parse_command_line())
