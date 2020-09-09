#!/usr/bin/env python3

"""Evolve a grid of initial orbital period/eccentircity to study envelope."""

from multiprocessing import Pool

import numpy
from configargparse import\
    ArgumentParser,\
    DefaultsFormatter,\
    Action as ArgparseAction

from orbital_evolution.command_line_util import\
    add_binary_config,\
    add_evolution_config,\
    run_evolution

def parse_configuration():
    """Return the configuration for the grid to run."""

    class ParseGrid(ArgparseAction):

        def __call__(self, parser, namespace, values, option_string):
            """Parse a grid option to a numpy.arary()."""

            setattr(namespace,
                    self.dest,
                    numpy.linspace(float(values[0]),
                                   float(values[1]),
                                   int(values[2])))

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
        nargs=3,
        action=ParseGrid,
        default=('0', '0.5', '6'),
        metavar=('MIN_ECC', 'MAX_ECC', 'NUM_ECC'),
        help='The eccentricies for which to calculate evolutions. The arguments'
        'are the same as numpy.linspace.'
    )
    parser.add_argument(
        '--orbital-period-grid', '--porb-grid',
        nargs=3,
        default=numpy.linspace(3.0, 20.0, 35),
        action=ParseGrid,
        metavar=('MIN_PER', 'MAX_PER', 'NUM_PER'),
        help='The orbital periods for which to calculate evolutions. The '
        'arguments are the same as numpy.linspace.'
    )
    parser.add_argument(
        '--plot-ages',
        nargs=3,
        default=numpy.linspace(0.1, 10.0, 100),
        action=ParseGrid,
        metavar=('MIN_AGE', 'MAX_AGE', 'NUM_AGES'),
        help='The ages at which to plot the eccentricity envelope. The '
        'arguments are the same as numpy.linspace.'
    )
    parser.add_argument(
        '--number-parallel-processes', '--num-parallel',
        type=int,
        default=1,
        help='How many parallel processes to use to carry out the calculations.'
    )

    return parser.parse_args()

class EvolveScenario:
    """
    Run the evolution for single set of ICs and extract plot data.

    Attrs:
        config:    The configuration parsed from the command line.

        required_ages:    The ages at which period and eccentricity are required
            for plotting.
    """

    def __init__(self, config):
        """Prepare to run evolutions using the given configuration."""

        self.config = config

    def __call__(self, initial_conditions):
        """
        Calculate and return the evolution for the given initial conditions.

        Args:
            initial_conditions(tuple):    The initial period and eccentricity
                to calculate the evolution for. The rest of the system
                parameters are from the configuration passed to __init__().

        Returns:
            tuple(array, array):
                The orbital period and eccentricity at the ages in
                `self.required_ages`
        """

        self.config.initial_orbital_period, self.config.initial_eccentricity = (
            initial_conditions
        )
        evolution = run_evolution(self.config,
                                  required_ages=self.config.plot_ages,
                                  required_ages_only=True)
        print('Calculated evolution for P0=%f, e0=%f'
              %
              initial_conditions)
        return (
            evolution.age,
            evolution.orbital_period,
            evolution.eccentricity
        )

def main(config):
    """Avoid polluting global namespace."""

    scenarios = zip(
        *(
            entry.flatten()
            for entry in numpy.meshgrid(
                config.orbital_period_grid,
                config.eccentricity_grid
            )
        )
    )
    evolve_scenario = EvolveScenario(config)

    if config.number_parallel_processes == 1:
        evolutions = [evolve_scenario(s) for s in scenarios]
    else:
        with Pool(cmdline_args.number_parallel_processes) as process_pool:
            evolutions = process_pool.map(scenarios)

    print(repr(evolutions))

if __name__ == '__main__':
    main(parse_configuration())
