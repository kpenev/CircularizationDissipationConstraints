#!/usr/bin/env python3

"""Create plots of emcee sampling results."""

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

from visuals import make_corner_plot

def parse_command_line():
    """Parse the command line for what and how to plot."""

    class ParseGrid(ArgparseAction):
        """Action for parsing linspace arguments."""

        def __call__(self, parser, namespace, values, option_string=None):
            """Parse a grid option to a numpy.arary()."""

            setattr(namespace,
                    self.dest,
                    numpy.linspace(float(values[0]),
                                   float(values[1]),
                                   int(values[2])))

    parser = ArgumentParser(
        description=__doc__,
        default_config_files=['visualize_emcee.cfg'],
        formatter_class=DefaultsFormatter,
        ignore_unknown_config_file_keys=False
    )
    parser.add_argument(
        'samples_fname',
        help='The filename containing the stored emcee chain to plot.'
    )
    parser.add_argument(
        '--chain-name', '--chain',
        default=None,
        help='Select which chain in the given file to plot. By default, plots '
        'the longest chain in the input file.'
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
        '--constraint-plot-fname', '--constraint-plot', '--constraint',
        default=None,
        help='If specified, a plot of the confidence interval of lgQ vs tidal '
        'frequency is generated and saved with the given filename. Inertial '
        'mode enhancement is ignored.'
    )
    parser.add_argument(
        '--burn-in',
        type=int,
        default=0,
        help='The number of samples to discard as burn-in.'
    )
    parser.add_argument(
        '--max-traces-per-plot',
        type=int,
        default=5,
        help='The maximum number of traces to include in a trace plot (more '
        'walkers will results in more panels in the plot).'
    )
    parser.add_argument(
        '--constraint-plot-confidence',
        type=float,
        nargs='+',
        default=[stats.norm.cdf(1.0) - stats.norm.cdf(-1.0)],
        help='Specify the confidence to display in the dissipation vs tidal '
        'frequency plot (see --constraint-plot-fname argument).'
    )
    parser.add_argument(
        '--constraint-plot-ptide-grid',
        nargs=3,
        default=numpy.linspace(1.0, 20.0, 100),
        metavar=('MIN_PERIOD', 'MAX_PERIOD', 'RES'),
        help='Set the range and resolution of the tidal period to include in '
        'the constraint plot (see --constraint-plot-fname argument).'
    )
    parser.add_argument(
        '--constraint-plot-no-lines',
        action='store_true',
        help='If passed, the constraint plot will not include individual lines.'
    )
    return parser.parse_args()

def get_backend(config):
    """Return the chain to plot."""

    if config.chain_name is None:
        with h5py.File(config.samples_fname, 'r') as chain_file:
            available_chains = list(chain_file.keys())
        longest_chain = 0
        for chain_name in available_chains:
            backend = emcee.backends.HDFBackend(config.samples_fname,
                                                name=chain_name,
                                                read_only=True)
            if backend.iteration > longest_chain:
                selected_backend = backend
        return selected_backend

    return emcee.backends.HDFBackend(config.samples_fname,
                                     name=config.chain_name,
                                     read_only=True)

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

def save_dissipation_constraint_plot(samples, config):
    """Create a plot showing the constraint of lgQ vs tidal frequency."""

    evaluated_lgq = numpy.maximum(
        1.0,
        (
            config.constraint_plot_ptide_grid[:, None]
            /
            samples['lgQ_break_period'].flatten()[None, :]
        )**samples['lgQ_powerlaw'].flatten()[None, :]
    ) * samples['lgQ_min'].flatten()[None, :]

    if not config.constraint_plot_no_lines:
        pyplot.plot(
            config.constraint_plot_ptide_grid,
            evaluated_lgq[:,::1],
            '-k',
            linewidth=20,
            alpha=0.01
        )

    evaluated_lgq.sort(1)

    for confidence in config.constraint_plot_confidence:
        num_exclude = int(
            numpy.floor(
                (1.0 - confidence) * evaluated_lgq.shape[1]
            )
        )
        print('Excluding %d points' % num_exclude)
        min_lgq = evaluated_lgq[:, :num_exclude + 1]
        max_lgq = evaluated_lgq[:, -num_exclude-1:]
        selected_indices = numpy.argmin(max_lgq - min_lgq, axis=1)
        min_lgq = numpy.take_along_axis(min_lgq,
                                        selected_indices[:, None],
                                        axis=1)
        max_lgq = numpy.take_along_axis(max_lgq,
                                        selected_indices[:, None],
                                        axis=1)

        pyplot.plot(
            config.constraint_plot_ptide_grid,
            min_lgq,
            '-b'
        )
        pyplot.plot(
            config.constraint_plot_ptide_grid,
            max_lgq,
            '-r'
        )

    pyplot.xlabel(r'Orbital Period [days]')
    pyplot.ylabel(r"$\log_{10}Q_\star'$")
    pyplot.ylim(5.0, 12.0)
    pyplot.show()

def main(config):
    """"Avoid polluting global namespace."""

    backend = get_backend(config)
    samples = backend.get_blobs(discard=config.burn_in)
    log_probability = backend.get_log_prob(discard=config.burn_in)
    if config.corner_plot_fname:
        save_corner_plot(samples, log_probability, config)
    if config.trace_plot_fname:
        save_trace_plot(samples, log_probability, config)
    if config.constraint_plot_fname:
        save_dissipation_constraint_plot(samples, config)

if __name__ == '__main__':
    main(parse_command_line())
