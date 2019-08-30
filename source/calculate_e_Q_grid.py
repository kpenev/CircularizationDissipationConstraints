#!/usr/bin/env python3

"""Calculate final eccentricity vs Q on a grid of Q values for a system."""

from configargparse import ArgumentParser, DefaultsFormatter

from astropy import units

from planetary_system_io import QuantityWithErrors
from stellar_evolution.manager import StellarEvolutionManager

def add_system_cmdline_args(parser):
    """Add arguments to a parser to extract required system parameters."""

    parser.add_argument(
        '--config-file', '-c',
        is_config_file=True,
        help='Specify a configuration file in liu of using command line '
        'options. Any option can still be overriden on the command line. '
        'Default: %(default)s'
    )
    parser.add_argument(
        '--star-density', '--rho-star', '--ISOrho',
        type=QuantityWithErrors,
        help='stellar density'
    )
    parser.add_argument(
        '--planet-to-star-radius-ratio', '--LCrprstar',
        type=QuantityWithErrors,
        help='Rp/R* from lightcurve.'
    )
    parser.add_argument(
        '--teff', '--SMEiiteff',
        type=QuantityWithErrors,
        help='Final SME, stellar effective temperature.',
    )
    parser.add_argument(
        '--feh', '--SMEiizfeh',
        type=QuantityWithErrors,
        help='Final SME, stellar metallicity.'
    )
    parser.add_argument(
        '--logg', '--SMEiilogg',
        type=QuantityWithErrors,
        help='Final SME, stellar surface gravity.'
    )
    parser.add_argument(
        '--rv-semi-amplitude', '--RVK',
        type=QuantityWithErrors,
        help='RV semi-amplitude [m/s]'
    )
    parser.add_argument(
        '--eccentricity', '--RVeccen', '-e',
        type=QuantityWithErrors,
        help='eccentricity.'
    )
    parser.add_argument(
        '--semimajor-to-rstar-ratio', '--PPar',
        type=QuantityWithErrors,
        help='relative orbital radius (a/R*)'
    )

def fix_system_units(cmdline_args):
    """Add units to the system parameters."""
    cmdline_args.star_density *= units.Unit('g/cm3')
    cmdline_args.teff *= units.Unit('K')
    cmdline_args.rv_semi_amplitude *= units.Unit('m/s')

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
    add_system_cmdline_args(parser)
    result = parser.parse_args()
    fix_system_units(result)
    return result

def main():
    """Calculate the grid specified on the command line."""

    cmdline_args = parse_command_line()
    interpolator = StellarEvolutionManager(
        cmdline_args.stellar_evolution_interpolator_dir
    ).get_interpolator_by_name(
        'default'
    )
    star_mass, star_age = interpolator.change_variables(
        cmdline_args.feh,
        teff=cmdline_args.teff.to_value('K'),
        rho=cmdline_args.star_density.to_value('g/cm3')
    )[0]
    cmdline_args.star_mass = star_mass * units.M_sun
    cmdline_args.age = star_age * units.Gyr
    print(repr(cmdline_args))

if __name__ == '__main__':
    main()
