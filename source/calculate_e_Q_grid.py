#!/usr/bin/env python3
#pylint: disable=invalid-name

"""Calculate final eccentricity vs Q on a grid of Q values for a system."""

from matplotlib import pyplot
import numpy

from astropy import units
from configargparse import ArgumentParser, DefaultsFormatter

from stellar_evolution.manager import StellarEvolutionManager
from orbital_evolution.transformations import phase_lag
from orbital_evolution.evolve_interface import library as\
    orbital_evolution_library

from planetary_system_io import read_nasa_planets
from reproduce_system import find_evolution
from binary_utils import calculate_secondary_mass

from exoplanet_io import\
    add_info_cmdline_args,\
    fix_system_units,\
    get_nasa_system
from plots import plot_eccentricity_vs_period, plot_star_solving


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
            '/home/kpenev/projects/git/poet/'
            'stellar_evolution_interpolators'
        ),
        help='The directory to read stellar evolution interpolator from.'
    )
    parser.add_argument(
        '--nasa-data',
        default=None,
        help='Read system data from a CSV file downloaded from the NASA '
        'exoplanet archive.'
    )
    add_info_cmdline_args(parser)
    result = parser.parse_args()
    if result.system_info_file:
        fix_system_units(result)
    return result

def get_current_eccentricity(system,
                             interpolator,
                             lgQ,
                             initial_eccentricity=0.55):
    """Return the ecc. at the current age of the given system given lg(Q)."""

    evolution = find_evolution(
        system=system,
        interpolator=interpolator,
        dissipation=dict(
            primary=None,
            secondary=dict(
                reference_phase_lag=phase_lag(lgQ),
                tidal_frequency_breaks=None,
                spin_frequency_breaks=None,
                tidal_frequency_powers=numpy.array([0.0]),
                spin_frequency_powers=numpy.array([0.0])
            )
        ),
        initial_eccentricity=initial_eccentricity,
        #False positive.
        #pylint: disable=no-member
        disk_period=(7.0 * units.day),
        disk_dissipation_age=(5e-3 * units.Gyr),
        #pylint: enable=no-member
        max_age=system.age
    )
    print(evolution.format())
    assert numpy.allclose(evolution.age[-1],
                          system.age.to_value('Gyr'),
                          rtol=1e-10,
                          atol=1e-10)
    return evolution.eccentricity[-1]

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
#        plot_eccentricity_vs_period(systems)
        system = get_nasa_system('WASP-156', systems)
    else:
        system = cmdline_args

    print('System:')
    print(repr(system))
    mass_age_solutions = interpolator.change_variables(
        float(system.feh),
        teff=float(system.teff.to_value('K')),
        rho=float(system.star_density.to_value('g/cm3'))
    )
#    plot_star_solving(interpolator, system)
    print('Mass age solutions: ' + repr(mass_age_solutions))
    if mass_age_solutions:
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
            system.age
        ) * units.Unit('solRad')
    else:
        system.primary_mass = system.db_star_mass
        system.age = system.db_star_age
        system.primary_radius = system.db_star_radius
    print('System:')
    print(repr(system))
    system.secondary_mass = calculate_secondary_mass(
        primary_mass=system.primary_mass,
        orbital_period=system.orbital_period,
        rv_semi_amplitude=system.rv_semi_amplitude,
        eccentricity=system.eccentricity
    )
    system.secondary_radius = (system.planet_to_star_radius_ratio
                               *
                               system.primary_radius)
    print('System:')
    print(repr(system))
    plot_lgQ = numpy.arange(3.0, 10.0, 1.0)
    plot_e = [get_current_eccentricity(system, interpolator, lgQ)
              for lgQ in plot_lgQ]
    pyplot.plot(plot_lgQ, plot_e, 'ok')
    pyplot.show()

if __name__ == '__main__':
    main()
