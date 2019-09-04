#!/usr/bin/env python3

"""Calculate final eccentricity vs Q on a grid of Q values for a system."""

from matplotlib import pyplot
from astropy import units
from configargparse import ArgumentParser, DefaultsFormatter
import numpy

from stellar_evolution.manager import StellarEvolutionManager
from planetary_system_io import read_nasa_planets

from exoplanet_io import add_info_cmdline_args, fix_system_units

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
            '/home/kpenev/projects/git/poet/stellar_evolution_interpolators'
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

def plot_eccentricity_vs_period(systems):
    """Create a plot of eccentricity vs period."""

    def plot_selection(selected, **kwargs):
        """Plot one selection of points."""

        errors = numpy.dstack(
            (
                numpy.abs(systems.pl_orbeccenerr2),
                numpy.abs(systems.pl_orbeccenerr1)
            )
        )[0].T
        limit_only = systems.pl_orbeccenlim > 0.5
        errors[0][limit_only] = systems.pl_orbeccen[limit_only]
        errors[1][limit_only] = 0.0
        pyplot.errorbar(
            systems.pl_orbper[selected],
            systems.pl_orbeccen[selected],
            yerr=errors[:, selected],
            **kwargs
        )

    dense = systems.pl_dens > 2.0
    fluffy = systems.pl_dens < 2.0
    unknown = numpy.logical_and(numpy.logical_not(dense),
                                numpy.logical_not(fluffy))
    pyplot.xscale('log')
    plot_selection(fluffy, fmt='or', markersize=10, label='fluffy')
    plot_selection(dense, fmt='sb', markersize=15, label='dense')
    plot_selection(unknown, fmt='vg', markersize=15, label='unknown')
    pyplot.plot([0.8, 5.0], [0, 0.6], '-k')
    pyplot.ylim((0, 0.6))
    pyplot.xlim((0.7, 10))
    pyplot.legend()
    pyplot.show()

def main():
    """Calculate the grid specified on the command line."""

    cmdline_args = parse_command_line()
    interpolator = StellarEvolutionManager(
        cmdline_args.stellar_evolution_interpolator_dir
    ).get_interpolator_by_name(
        'default'
    )
    if cmdline_args.nasa_data:
        systems = read_nasa_planets(cmdline_args.nasa_data,
                                    eliminate=(),
                                    add_units=True,
                                    need_ages=False)
        plot_eccentricity_vs_period(systems)
    if cmdline_args.system_info_file:
        star_mass, star_age = interpolator.change_variables(
            cmdline_args.feh,
            teff=cmdline_args.teff.to_value('K'),
            rho=cmdline_args.star_density.to_value('g/cm3')
        )[0]
        #False positive
        #pylint: disable=no-member
        cmdline_args.star_mass = star_mass * units.M_sun
        cmdline_args.age = star_age * units.Gyr
        #pylint: enable=no-member
        print(repr(cmdline_args))

if __name__ == '__main__':
    main()
