#!/usr/bin/env python3

"""Evolve a grid of initial orbital period/eccentircity to study envelope."""

import numpy

from configargparse import ArgumentParser, DefaultsFormatter

from orbital_evolution.command_line_util import\
    add_binary_config,\
    add_evolution_config,\
    run_evolution

def parse_configuration():
    """Return the configuration for the grid to run."""

    parser = ArgumentParser(
        description=__doc__,
        default_config_files=['initial_condition_grid.cfg'],
        formatter_class=DefaultsFormatter,
        ignore_unknown_config_file_keys=False
    )
    parser.add_argument(
        '--config', '-c',
        is_config_file=True,
        help='Config file to use instead of default.'
    )

    add_binary_config(parser, skip=('initial_orbital_period',
                                    'initial_eccentricity'))
    add_evolution_config(parser)
    parser.add_argument(
        '--eccentricity-grid', '--e-grid',
        type=float,
        nargs=3,
        default=(0, 0.5, 6),
        metavar=('MIN_ECC', 'MAX_ECC', 'NUM_ECC'),
        help='The eccentricies for which to calculate evolutions. The arguments'
        'are the same as numpy.linspace.'
    )
    parser.add_argument(
        '--orbital-period-grid', '--porb-grid',
        type=float,
        nargs=3,
        default=(3, 20, 35),
        metavar=('MIN_PER', 'MAX_PER', 'NUM_PER'),
        help='The orbital periods for which to calculate evolutions. The '
        'arguments are the same as numpy.linspace.'
    )
    parser.add_argument(
        '--plot-ages',
        type=float,
        nargs=3,
        default=(0.1, 10.0, 100),
        metavar=('MIN_AGE', 'MAX_AGE', 'NUM_AGES'),
        help='The ages at which to plot the eccentricity envelope. The '
        'arguments are the same as numpy.linspace.'
    )

    return parser.parse_args()

def main(config):
    """Avoid polluting global namespace."""

    required_ages = numpy.linspace(*config.plot_ages)
    period_eccentricity = zip(
        *(
            entry.flatten()
            for entry in numpy.meshgrid(
                numpy.linspace(*config.orbital_period_grid),
                numpy.linspace(*config.eccentricity_grid)
            )
        )
    )
    for period, eccentircity in period_eccentricity:
        print(repr(period) + '\t' + repr(eccentircity))

    config.initial_eccentricity = 0.3
    config.initial_orbital_period = 5.0
    print(run_evolution(config).format())

if __name__ == '__main__':
    main(parse_configuration())
