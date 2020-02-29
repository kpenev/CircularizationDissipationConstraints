"""Functions for processing command line and config file user input."""

import os.path

from astropy import units
import numpy

from planetary_system_io import QuantityWithErrors

data_dir = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(
                __file__
            )
        )
    ),
    'data',
)

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

def fix_system_units(cmdline_args):
    """Add units to the system parameters."""

    for attribute, unit_str in [('density', 'g/cm3'),
                                ('teff', 'K'),
                                ('rv_semi_amplitude', 'm/s'),
                                ('small_planet_density', 'g/cm3')]:
        if getattr(cmdline_args, attribute, None) is not None:
            setattr(
                cmdline_args,
                attribute,
                getattr(cmdline_args, attribute) * units.Unit(unit_str)
            )

def add_assumptions_cmdline_args(parser):
    """Add arguments controlling what assumptions should be made."""

    parser.add_argument(
        '--small-planet-density',
        type=float,
        default=2.0,
        help='A default density in g/cm^3 to assume for small planets for which'
        ' mass is unknown'
    )
    parser.add_argument(
        '--stellar-lgQ',
        type=float,
        default=numpy.inf,
        help='The dissipation parameter to assume for the star in the system.'
    )
    parser.add_argument(
        '--initial-eccentricity',
        type=float,
        default=0.55,
        help='The initial eccentrcicity to assume.'
    )

def add_path_cmdline_args(parser):
    """Add arguments controlling where to find required inputs."""

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
        '--use-binary-stars',
        action='store',
        default=None,
        help='Read binary stellar system data from a default set of files for a'
        ' cluster.'
    )
    parser.add_argument(
        '--progress-pickle', '--progress',
        default='progress.pickle',
        help='The filename to save final eccentricities as soon as calculated '
        'to allow continuing after interruption.'
    )
