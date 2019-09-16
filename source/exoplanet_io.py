"""Utilities for input and output of exoplanet data."""

from types import SimpleNamespace

from astropy import units
import numpy

from planetary_system_io import QuantityWithErrors

def add_info_cmdline_args(parser):
    """Add arguments to a parser to parse required parameters from info file."""

    parser.add_argument(
        '--config-file', '-c',
        is_config_file=True,
        help='Specify a configuration file in liu of using command line '
        'options. Any option can still be overriden on the command line. '
        'Default: %(default)s'
    )
    parser.add_argument(
        '--eccentricity-expansion-coefficients', '--e-coef',
        default='eccentricity_expansion_coef.txt',
        help='The file to read eccentricity expansion coefficients from.'
    )
    parser.add_argument(
        '--system-info-file',
        is_config_file=True,
        help='A HAT style info file containing the system parameters for '
        'processing a single system.'
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
    parser.add_argument(
        '--orbital-period', '--Porb',
        type=QuantityWithErrors,
        help='The present day orbital period of the system.'
    )

def get_nasa_system(hostname, nasa_systems):
    """Return a single system from a parsed NASA exoplanets archive file."""

    def get_quantity(system_index, column_name):
        """Return a properly formatted Quantity instance with errors."""

        result = units.Quantity(
            getattr(nasa_systems, column_name)[system_index]
        )[0]
        result.plus_error = getattr(
            nasa_systems,
            column_name+'err1'
        )[system_index][0]
        result.minus_error = -getattr(
            nasa_systems,
            column_name+'err2'
        )[system_index][0]
        print('Result: ' + repr(result))
        return result

    system_index = numpy.where(nasa_systems.pl_hostname == hostname)[0]
    result = SimpleNamespace(
        star_density=get_quantity(system_index, 'st_dens'),
        db_star_mass=get_quantity(system_index, 'st_mass'),
        db_star_age=get_quantity(system_index, 'st_age'),
        db_star_radius=get_quantity(system_index, 'st_rad'),
        planet_to_star_radius_ratio=get_quantity(system_index,
                                                 'pl_ratror'),
        teff=get_quantity(system_index, 'st_teff'),
        feh=get_quantity(system_index, 'st_metfe'),
        logg=get_quantity(system_index, 'st_logg'),
        rv_semi_amplitude=get_quantity(system_index, 'pl_rvamp'),
        eccentricity=get_quantity(system_index, 'pl_orbeccen'),
        semimajor_to_rstar_ratio=get_quantity(system_index, 'pl_ratdor'),
        orbital_period=get_quantity(system_index, 'pl_orbper'),
    )
    return result

def fix_system_units(cmdline_args):
    """Add units to the system parameters."""

    cmdline_args.star_density *= units.Unit('g/cm3')
    cmdline_args.teff *= units.Unit('K')
    cmdline_args.rv_semi_amplitude *= units.Unit('m/s')
