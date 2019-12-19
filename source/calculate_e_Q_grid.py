#!/usr/bin/env python3
#pylint: disable=invalid-name

"""Calculate final eccentricity vs Q on a grid of Q values for a system."""

from multiprocessing import Pool, Manager

import numpy
from astropy import units, constants
from configargparse import ArgumentParser, DefaultsFormatter

from stellar_evolution.manager import StellarEvolutionManager
from orbital_evolution.evolve_interface import library as\
    orbital_evolution_library

from planetary_system_io import read_nasa_planets
from binary_utils import calculate_secondary_mass

from .io_utilities import\
    get_nasa_system,\
    read_geller_et_al_2009_binaries,\
    read_milliman_et_al_2014_binaries,\
    init_progress_pickle
from .command_line_utilities import\
    fix_system_units,\
    add_info_cmdline_args,\
    add_assumptions_cmdline_args,\
    add_path_cmdline_args
from .current_eccentricity_calculator import CurrentEccentricityCalculator

def parse_command_line():
    """Return the parsed command line as attributes of an object."""

    parser = ArgumentParser(
        description=__doc__,
        default_config_files=['system.info'],
        formatter_class=DefaultsFormatter,
        ignore_unknown_config_file_keys=True
    )
    parser.add_argument(
        '--num-parallel-processes',
        type=int,
        default=4,
        help='The number of simultaneous processes to use for parallel '
        'calculation of the evolution of systems.'
    )
    add_info_cmdline_args(parser)
    add_assumptions_cmdline_args(parser)
    add_path_cmdline_args(parser)
    result = parser.parse_args()
    fix_system_units(result)
    return result

def fix_semimajor(system):
    """Calculate the semimajor axis if not already in the system."""

    if not numpy.isfinite(getattr(system, 'semimajor', numpy.nan)):
        system.semimajor = (
            constants.G * (system.primary_mass + system.secondary_mass)
            *
            (system.orbital_period)**2
            /
            (4.0 * numpy.pi**2)
        )**(1.0 / 3.0)

def prepare_nasa_system(system,
                        interpolator,
                        small_planet_density):
    """Add attributes to the given system to allow calculating its evolution."""

    def set_primary_properties():
        """Set the mass, age and radius of the primary."""

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

    def set_secondary_properties():
        """Set the mass and radius of the secondary."""

        if numpy.isfinite(system.db_planet_radius):
            system.secondary_radius = system.db_planet_radius
        else:
            assert numpy.isfinite(system.primary_radius)
            assert numpy.isfinite(system.planet_to_star_radius_ratio)
            system.secondary_radius = (system.planet_to_star_radius_ratio
                                       *
                                       system.primary_radius)

        if numpy.isfinite(system.db_planet_mass):
            system.secondary_mass = system.db_planet_mass
        else:
            if numpy.isfinite(system.rv_semi_amplitude):
                system.secondary_mass = calculate_secondary_mass(
                    primary_mass=system.primary_mass,
                    orbital_period=system.orbital_period,
                    rv_semi_amplitude=system.rv_semi_amplitude,
                    eccentricity=system.eccentricity
                )
            else:
                #False positive
                #pylint: disable=no-member
                assert system.secondary_radius < 5.0 * units.earthRad
                system.secondary_mass = (
                    small_planet_density
                    *
                    4.0 / 3.0 * numpy.pi * system.secondary_radius**3
                )
                system.assumed_default_density = True
                #pylint: enable=no-member

    def fix_eccentricity():
        """Use backup methods for calculating eccentricity if unknown."""

        if (
                (
                    not numpy.isfinite(system.eccentricity)
                    or
                    not bool(system.eccentricity)
                )
                and
                numpy.isfinite(system.impact_parameter)
                and
                numpy.isfinite(system.transit_duration)
        ):
            print('Applying eccentricity fallback to system: ' + repr(system))
            duration_anomaly = (
                (
                    (
                        (system.primary_radius + system.secondary_radius)**2
                        -
                        (system.impact_parameter * system.primary_radius)**2
                    )**0.5
                    /
                    (numpy.pi * system.semimajor)
                )
                *
                system.orbital_period
                /
                system.transit_duration
            ).to_value('')
            system.eccentricity = units.Quantity(
                numpy.abs((duration_anomaly**2 - 1)
                          /
                          (duration_anomaly**2 + 1))
            )
            system.eccentricity.plus_error = 1.0 - system.eccentricity
            system.eccentricity.minus_error = 0.0
            print('Eccentricity fallback used for ' + system.hostname)

    print('Preparing NASA system for evolution:')
    print(repr(system))

    assert numpy.isfinite(system.orbital_period.to_value('day'))

    set_primary_properties()
    set_secondary_properties()
    fix_eccentricity()
    fix_semimajor(system)

def main():
    """Calculate the grid specified on the command line."""

    cmdline_args = parse_command_line()
    progress = init_progress_pickle(cmdline_args)

    interpolator = StellarEvolutionManager(
        cmdline_args.stellar_evolution_interpolator_dir
    ).get_interpolator_by_name(
        'default'
    )
    orbital_evolution_library.read_eccentricity_expansion_coefficients(
        cmdline_args.eccentricity_expansion_coefficients.encode('ascii')
    )

    evolution_systems = []
    if cmdline_args.nasa_data:
        systems = read_nasa_planets(cmdline_args.nasa_data,
                                    eliminate=(),
                                    add_units=True,
                                    need_ages=False)
        #Fales positive
        #pylint: disable=no-member
        for system_index in numpy.argwhere(
                numpy.logical_and(
                    systems.pl_discmethod == 'Transit',
                    systems.pl_orbper < 50.0 * units.day
                )
        )[:, 0]:
        #pylint: enable=no-member
            try:
                system = get_nasa_system(system_index, systems)
                prepare_nasa_system(system,
                                    interpolator,
                                    cmdline_args.small_planet_density)
                print('Prepared system: ' + repr(system))
                #pylint: disable=no-member
                if system.primary_mass <= 1.2 * units.M_sun:
                    evolution_systems.append(system)
                #pylint: enable=no-member
            except AssertionError:
                pass
    if cmdline_args.use_binary_stars:
        evolution_systems.extend(
            filter(
                lambda s: s.orbital_period < 50.0 * units.day,
                (
                    read_geller_et_al_2009_binaries()
                    if cmdline_args.use_binary_stars.upper() == 'NGC188' else
                    read_milliman_et_al_2014_binaries()
                )
            )
        )

    if cmdline_args.orbital_period is not None:
        evolution_systems.append(
            prepare_nasa_system(cmdline_args,
                                interpolator,
                                cmdline_args.small_planet_density)
        )

    print('Evolution systems (%d): ' % len(evolution_systems)
          +
          repr(evolution_systems))

    grid_jobs = [
        (system, lgQ)
        for system in evolution_systems
        for lgQ in numpy.arange(3.0, 10.0, 1.0)
    ]

    pool_manager = Manager()
    calculate_current_eccentricity = CurrentEccentricityCalculator(
        initial_eccentricity=cmdline_args.initial_eccentricity,
        primary_lgQ=cmdline_args.stellar_lgQ,
        interpolator=interpolator,
        progress=progress,
        progress_pickle_fname=cmdline_args.progress_pickle,
        #False positive
        #pylint: disable=no-member
        progress_lock=pool_manager.Lock()
        #pylint: enable=no-member
    )
    if cmdline_args.num_parallel_processes == 1:
        for job in grid_jobs:
            calculate_current_eccentricity(job)
    else:
        with Pool(cmdline_args.num_parallel_processes) as process_pool:
            process_pool.map(
                calculate_current_eccentricity,
                grid_jobs
            )

if __name__ == '__main__':
    main()
