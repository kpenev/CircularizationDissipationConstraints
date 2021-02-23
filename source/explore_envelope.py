#!/usr/bin/env python3

"""Evolve a grid of initial orbital period/eccentircity to study envelope."""

from multiprocessing import Pool, Manager
import pickle
import os.path
from os import makedirs

import numpy
from configargparse import\
    ArgumentParser,\
    DefaultsFormatter,\
    Action as ArgparseAction

from orbital_evolution.command_line_util import\
    add_binary_config,\
    add_evolution_config,\
    set_up_library,\
    run_evolution

def parse_configuration():
    """Return the configuration for the grid to run."""

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
        default=numpy.linspace(0.4, 0.5, 2),
        metavar=('MIN_ECC', 'MAX_ECC', 'NUM_ECC'),
        help='The eccentricies for which to calculate evolutions. The arguments'
        'are the same as numpy.linspace.'
    )
    parser.add_argument(
        '--orbital-period-grid', '--porb-grid',
        nargs=3,
        default=numpy.linspace(3.0, 20.0, 70),
        action=ParseGrid,
        metavar=('MIN_PER', 'MAX_PER', 'NUM_PER'),
        help='The orbital periods for which to calculate evolutions. The '
        'arguments are the same as numpy.linspace.'
    )
    parser.add_argument(
        '--plot-ages',
        nargs=3,
        default=numpy.linspace(0.1, 10.0, 1000),
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
    parser.add_argument(
        '--progress-pickle-fname', '--progress-fname',
        default='eccentricity_envelope.pickle',
        help='The name of a file to save/load progress to/from.'
    )

    return parser.parse_args()

#Intended to work as callable for multiprocessing
#pylint: disable=too-few-public-methods
class EvolveScenario:
    """
    Run the evolution for single set of ICs and extract plot data.

    Attrs:
        config:    The configuration parsed from the command line.

        required_ages:    The ages at which period and eccentricity are required
            for plotting.
    """

    def _pickle_evolution(self, evolution):
        """Add the given evolution to the progress pickle file."""

        self._progress_lock.acquire()
        with open(self.config.progress_pickle_fname, 'ab') as pickle_file:
            pickle.dump(self.config.initial_orbital_period, pickle_file)
            pickle.dump(self.config.initial_eccentricity, pickle_file)
            pickle.dump(evolution, pickle_file)
        self._progress_lock.release()

    def _unpickle_progress(self):
        """Return the set of initial conditions with pickled evolution."""

        result = set()
        if not os.path.exists(self.config.progress_pickle_fname):
            pickle_dir = os.path.dirname(self.config.progress_pickle_fname)
            if pickle_dir and not os.path.exists(pickle_dir):
                makedirs(pickle_dir)

            with open(self.config.progress_pickle_fname, 'wb') as pickle_file:
                pickle.dump(self.config, pickle_file)

            return result

        with open(self.config.progress_pickle_fname, 'rb') as pickle_file:
            pickled_config = dict(vars(pickle.load(pickle_file)))
            current_config = dict(vars(self.config))

            for var_name in ['eccentricity_grid',
                             'orbital_period_grid',
                             'plot_ages']:
                assert (pickled_config[var_name]
                        ==
                        current_config[var_name]).all()
                del pickled_config[var_name]
                del current_config[var_name]
            assert pickled_config == current_config

            try:
                while True:
                    result.add(
                        (pickle.load(pickle_file), pickle.load(pickle_file))
                    )
                    pickle.load(pickle_file)
            except EOFError:
                pass

        return result

    def __init__(self, config, progress_lock):
        """Prepare to run evolutions using the given configuration."""

        self.config = config
        self.interpolator = set_up_library(config)
        self._progress_lock = progress_lock
        self._progress = self._unpickle_progress()

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

        if initial_conditions in self._progress:
            return

        self.config.initial_orbital_period, self.config.initial_eccentricity = (
            initial_conditions
        )
        evolution = run_evolution(self.config,
                                  interpolator=self.interpolator,
                                  required_ages=self.config.plot_ages,
                                  required_ages_only=True)

        self._pickle_evolution(evolution)

        print('Calculated evolution for P0=%f, e0=%f'
              %
              initial_conditions)
#pylint: enable=too-few-public-methods

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

    pool_manager = Manager()
    evolve_scenario = EvolveScenario(config,
                                     #False positive
                                     #pylint: disable=no-member
                                     pool_manager.Lock())
                                     #pylint: enable=no-member

    if config.number_parallel_processes == 1:
        for initial_conditions in scenarios:
            evolve_scenario(initial_conditions)
    else:
        with Pool(config.number_parallel_processes) as process_pool:
            process_pool.map(evolve_scenario, scenarios)

if __name__ == '__main__':
    main(parse_configuration())
