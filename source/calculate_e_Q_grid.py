#!/usr/bin/env python3
#pylint: disable=invalid-name

"""Calculate final eccentricity vs Q on a grid of Q values for a system."""

import os.path
from multiprocessing import Pool, Manager

import numpy
from astropy import units
from configargparse import ArgumentParser, DefaultsFormatter

from stellar_evolution.manager import StellarEvolutionManager
from orbital_evolution.evolve_interface import library as\
    orbital_evolution_library

from planetary_system_io import read_nasa_planets
from binary_utils import calculate_secondary_mass

from io_utilities import\
    add_info_cmdline_args,\
    fix_system_units,\
    get_nasa_system
from current_eccentricity_calculator import CurrentEccentricityCalculator

def parse_command_line():
    """Return the parsed command line as attributes of an object."""

    parser = ArgumentParser(
        description=__doc__,
        default_config_files=['system.info'],
        formatter_class=DefaultsFormatter,
        ignore_unknown_config_file_keys=True
    )
    parser.add_argument(
        '--stellar-evolution-interpolator-dir', '--interpolator-dir',
        default=(
            os.path.expanduser(
                '~/projects/git/poet/stellar_evolution_interpolators'
            )
        ),
        help='The directory to read stellar evolution interpolator from.'
    )
    parser.add_argument(
        '--nasa-data',
        default=None,
        help='Read system data from a CSV file downloaded from the NASA '
        'exoplanet archive.'
    )
    parser.add_argument(
        '--progress-pickle', '--progress',
        default='progress.pickle',
        help='The filename to save final eccentricities as soon as calculated '
        'to allow continuing after interruption.'
    )
    parser.add_argument(
        '--num-parallel-processes',
        type=int,
        default=4,
        help='The number of simultaneous processes to use for parallel '
        'calculation of the evolution of systems.'
    )
    add_info_cmdline_args(parser)
    result = parser.parse_args()
    if result.system_info_file:
        fix_system_units(result)
    return result

def prepare_nasa_system(system, interpolator):
    """Add attributes to the given system to allow calculating its evolution."""

    print('Preparing NASA system for evolution:')
    print(repr(system))

    assert numpy.isfinite(system.orbital_period)

    if (
            numpy.isfinite(system.db_star_mass
                           *
                           system.db_star_age
                           *
                           system.db_star_radius)
    ):
        system.primary_mass = system.db_star_mass
        system.age = system.db_star_age
        system.primary_radius = system.db_star_radius
    else:
        mass_age_solutions = interpolator.change_variables(
            float(system.feh),
            teff=float(system.teff.to_value('K')),
            rho=float(system.star_density.to_value('g/cm3'))
        )
        assert mass_age_solutions

        star_mass, star_age = mass_age_solutions[0]
        #False positive
        #pylint: disable=no-member
        system.primary_mass = star_mass * units.M_sun
        system.age = star_age * units.Gyr
        #pylint: enable=no-member
        system.primary_radius = interpolator(
            'radius',
            star_mass,
            system.feh
        )(
            system.age.to_value('Gyr')
        ) * units.Unit('solRad')

    assert numpy.isfinite(system.primary_mass)

    if numpy.isfinite(system.db_planet_mass):
        system.secondary_mass = system.db_planet_mass
    else:
        assert numpy.isfinite(system.rv_semi_amplitude)
        system.secondary_mass = calculate_secondary_mass(
            primary_mass=system.primary_mass,
            orbital_period=system.orbital_period,
            rv_semi_amplitude=system.rv_semi_amplitude,
            eccentricity=system.eccentricity
        )
    if numpy.isfinite(system.db_planet_radius):
        system.secondary_radius = system.db_planet_radius
    else:
        assert numpy.isfinite(system.primary_radius)
        assert numpy.isfinite(system.planet_to_star_radius_ratio)
        system.secondary_radius = (system.planet_to_star_radius_ratio
                                   *
                                   system.primary_radius)

def main():
    """Calculate the grid specified on the command line."""

    cmdline_args = parse_command_line()
    interpolator = StellarEvolutionManager(
        cmdline_args.stellar_evolution_interpolator_dir
    ).get_interpolator_by_name(
        'default'
    )
    orbital_evolution_library.read_eccentricity_expansion_coefficients(
        cmdline_args.eccentricity_expansion_coefficients.encode('ascii')
    )

    if cmdline_args.nasa_data:
        systems = read_nasa_planets(cmdline_args.nasa_data,
                                    eliminate=(),
                                    add_units=True,
                                    need_ages=False)
        evolution_systems = []
        #Fales positive
        #pylint: disable=no-member
        for system_index in numpy.argwhere(systems.pl_discmethod
                                           ==
                                           'Transit')[:, 0]:
        #pylint: enable=no-member
            try:
                system = get_nasa_system(system_index, systems)
                prepare_nasa_system(system, interpolator)
                print('Prepared system: ' + repr(system))
                if system.primary_mass <= 1.2:
                    evolution_systems.append(system)
            except AssertionError:
                pass
    else:
        evolution_systems = [prepare_nasa_system(cmdline_args, interpolator)]

    grid_jobs = [
        (system, lgQ) for system in evolution_systems for lgQ in numpy.arange(3.0, 10.0, 1.0)
    ]

    pool_manager = Manager()
    calculate_current_eccentricity = CurrentEccentricityCalculator(
        initial_eccentricity=0.55,
        interpolator=interpolator,
        progress_pickle_fname=cmdline_args.progress_pickle,
        progress_lock=pool_manager.Lock()
    )
    with Pool(cmdline_args.num_parallel_processes) as magfit_pool:
        magfit_pool.map(
            calculate_current_eccentricity,
            grid_jobs
        )

if __name__ == '__main__':
    main()
