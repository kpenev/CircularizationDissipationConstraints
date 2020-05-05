#!/usr/bin/env python3
#pylint: disable=invalid-name

"""Calculate final eccentricity vs Q on a grid of Q values for a system."""

from multiprocessing import Pool, Manager, TimeoutError as MPTimeoutError
import sys
import os.path

import numpy
from astropy import units, constants
from configargparse import ArgumentParser, DefaultsFormatter

from stellar_evolution.manager import StellarEvolutionManager
from orbital_evolution.evolve_interface import library as\
    orbital_evolution_library

from planetary_system_io import read_nasa_planets
from binary_utils import calculate_secondary_mass

sys.path.append(
    os.path.join(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        ),
        'data'
    )
)

#Need to update module search paths before importing
#pylint: disable=wrong-import-position

#False positive
#pylint: disable=import-error
import praesepe_binaries
import hyades_binaries
#pylint: enable=import-error
from io_utilities import\
    get_nasa_system,\
    read_geller_et_al_2009_binaries,\
    read_milliman_et_al_2014_binaries,\
    init_progress_pickle,\
    format_hyades_praesepe_binaries
from command_line_utilities import\
    fix_system_units,\
    add_info_cmdline_args,\
    add_assumptions_cmdline_args,\
    add_path_cmdline_args
from current_eccentricity_calculator import CurrentEccentricityCalculator
#pylint: enable=wrong-import-position

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
        #False positive
        #pylint: disable=no-member
        system.semimajor = (
            constants.G * (system.primary_mass + system.secondary_mass)
            *
            (system.orbital_period)**2
            /
            (4.0 * numpy.pi**2)
        )**(1.0 / 3.0)
        #pylint: enable=no-member

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
                assert (
                    #False positive
                    #pylint: disable=no-member
                    system.secondary_radius < 5.0 * units.earthRad
                    #pylint: enable=no-member
                )
                system.secondary_mass = (
                    small_planet_density
                    *
                    4.0 / 3.0 * numpy.pi * system.secondary_radius**3
                )
                system.assumed_default_density = True

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

def get_evolution_systems(cmdline_args, interpolator):
    """Return a list of the systems to process."""

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
        if cmdline_args.use_binary_stars.upper() == 'NGC188':
            candidate_systems = read_geller_et_al_2009_binaries()
        elif cmdline_args.use_binary_stars.upper() == 'NGC6819':
            candidate_systems = read_milliman_et_al_2014_binaries()
        else:
            assert cmdline_args.use_binary_stars.upper() == 'PRAESEPE/HYADES'
            candidate_systems = (
                format_hyades_praesepe_binaries(
                    praesepe_binaries.read_systems(),
                    #False positive
                    #pylint: disable=no-member
                    age=670.0 * units.Myr,
                    #pylint: enable=no-member
                    feh=0.156,
                    resolve_secondary_mass_range=(
                        cmdline_args.resolve_secondary_mass_range
                    )
                )
                +
                format_hyades_praesepe_binaries(
                    hyades_binaries.systems,
                    #False positive
                    #pylint: disable=no-member
                    age=635.0 * units.Myr,
                    #pylint: enable=no-member
                    feh=0.146,
                    resolve_secondary_mass_range=(
                        cmdline_args.resolve_secondary_mass_range
                    )
                )
            )
        #False positive
        #pylint: disable=no-member
        evolution_systems.extend(
            filter(
                lambda s: (
                    s.orbital_period < 50.0 * units.day
                    and
                    numpy.isfinite(s.primary_mass)
                    and
                    numpy.isfinite(s.secondary_mass)
                ),
                candidate_systems
            )
        )
        #pylint: enable=no-member

    if cmdline_args.orbital_period is not None:
        evolution_systems.append(
            prepare_nasa_system(cmdline_args,
                                interpolator,
                                cmdline_args.small_planet_density)
        )

    print('Evolution systems (%d): ' % len(evolution_systems)
          +
          repr(evolution_systems))

    return evolution_systems

def get_jobs(cmdline_args, evolution_systems):
    """Return list of the jobs to process with CurrentEccentricityCalculator."""

    def parse_known_to_fail(line):
        """Parse a single line from the file of evolutions known to fail."""

        system, lgQ, initial_eccentricity = line.split()
        return system, float(lgQ), float(initial_eccentricity)

    grid_jobs = [
        (system, lgQ, cmdline_args.initial_eccentricity)
        for system in evolution_systems
        for lgQ in numpy.arange(3.0, 10.0, 1.0)
    ]

    with open(cmdline_args.known_to_fail, 'r') as known_to_fail_file:
        known_to_fail = {parse_known_to_fail(line)
                         for line in known_to_fail_file}

    to_delete = []
    for index, job in enumerate(grid_jobs):
        replacement_job = (job[0].hostname,) + job[1:]
        if replacement_job in known_to_fail:
            replaced = False
            for initial_eccentricity in\
                    cmdline_args.fallback_initial_eccentricity:
                replacement_job = replacement_job[:2] + (initial_eccentricity,)
                if replacement_job not in known_to_fail:
                    grid_jobs[index] = (job[0],) + replacement_job[1:]
                    replaced = True
                    break
            if not replaced:
                to_delete.append(index)

    for i in reversed(to_delete):
        del grid_jobs[i]

    return grid_jobs

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

    pool_manager = Manager()
    calculate_current_eccentricity = CurrentEccentricityCalculator(
        primary_lgQ=cmdline_args.stellar_lgQ,
        interpolator=interpolator,
        progress=progress,
        progress_pickle_fname=cmdline_args.progress_pickle,
        #False positive
        #pylint: disable=no-member
        progress_lock=pool_manager.Lock()
        #pylint: enable=no-member
    )
    job_list = get_jobs(cmdline_args,
                        get_evolution_systems(cmdline_args, interpolator))
    if cmdline_args.num_parallel_processes == 1:
        final_eccentricities = [calculate_current_eccentricity(job)
                                for job in job_list]
        with open(cmdline_args.known_to_fail, 'a') as known_to_fail_file:
            for ef, job in zip(final_eccentricities, job_list):
                if ef is None:
                    known_to_fail_file.write(
                        '%s %s %s\n'
                        %
                        (job[0].hostname, repr(job[1]), repr(job[2]))
                    )
    else:
        with Pool(cmdline_args.num_parallel_processes) as process_pool:
            final_eccentricity_iter = process_pool.imap(
                calculate_current_eccentricity,
                job_list,
            )

            for job in job_list:
                try:
                    ef = final_eccentricity_iter.next(timeout=3.6e4)
                except MPTimeoutError:
                    ef = None
                if ef is None:
                    with open(cmdline_args.known_to_fail, 'a') as \
                            known_to_fail_file:
                        known_to_fail_file.write(
                            '%s %s %s\n'
                            %
                            (job[0].hostname, repr(job[1]), repr(job[2]))
                        )

if __name__ == '__main__':
    main()
