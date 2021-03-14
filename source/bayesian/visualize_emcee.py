#!/usr/bin/env python3

"""Create plots of emcee sampling results."""

from matplotlib import pyplot
from corner import corner
from configargparse import ArgumentParser, DefaultsFormatter
import emcee
import h5py
import pandas

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


def main(config):
    """"Avoid polluting global namespace."""

    backend = get_backend(config)
    samples = backend.get_blobs()
    if config.corner_plot_fname:
        samples = samples.flatten()
        include_params = [
            param
            for param in samples.dtype.names
            if samples[param].min() != samples[param].max()
        ]
        corner(pandas.DataFrame(samples[include_params]))
        pyplot.savefig(config.corner_plot_fname)

if __name__ == '__main__':
    main(parse_command_line())
