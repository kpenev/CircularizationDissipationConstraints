#!/usr/bin/env python3

"""Create plots of emcee sampling results."""

from matplotlib import pyplot
from configargparse import ArgumentParser, DefaultsFormatter
import numpy
import emcee
import h5py
import pandas

from visuals import make_corner_plot

def parse_command_line():
    """Parse the command line for what and how to plot."""

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
        help='If supplied, a coner plot is created and saved with the given '
        'filename.'
    )
    parser.add_argument(
        '--trace-plot-fname', '--traces-plot', '--traces',
        default=None,
        help='If supplied, a trace plot is generated and saved with the given '
        'filaname.'
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

def main(config):
    """"Avoid polluting global namespace."""

    backend = get_backend(config)
    samples = backend.get_blobs(discard=config.burn_in)
    log_probability = backend.get_log_prob(discard=config.burn_in)
    if config.corner_plot_fname:
        save_corner_plot(samples, log_probability, config)
    if config.trace_plot_fname:
        save_trace_plot(samples, log_probability, config)

if __name__ == '__main__':
    main(parse_command_line())
