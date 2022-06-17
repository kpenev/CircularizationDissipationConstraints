"""Common command line parsing for bayeasian analysis."""

from collections import namedtuple
import os.path

import numpy
from configargparse import ArgumentParser, DefaultsFormatter

from split_normal_distribution import split_normal
from planetary_system_io import read_cds_pipe_table

#False positive (hanhdled in __init__.py).
#pylint: disable=import-error
#import praesepe_binaries
#import hyades_binaries
from command_line_utilities import data_dir
from bayesian import m35_util
from bayesian import hyadespraesepe_util
from bayesian.basic_util import default_logging_format
#pylint: enable=import-error

RandomQuantity = namedtuple('RandomQuantity', ['distribution', 'units'])

def parse_quantity_with_errors(value_str, quantity_units=None):
    """Parse a string like 5.0 +- 1.3 or 5.0 +0.2 -0.8 (space optional)."""

    value_str, error_str = value_str.rsplit('+', 1)
    plus_error_str, minus_error_str = error_str.rsplit('-', 1)

    distribution = split_normal.freeze_error_bar(
        mode=float(value_str),
        abs_plus_error=float(plus_error_str or minus_error_str),
        abs_minus_error=float(minus_error_str)
    )
    return (
        distribution if quantity_units is None
        else RandomQuantity(distribution, quantity_units)
    )


def add_dissipation_args(parser):
    """Add arguments for specifying the dissipation to the given parser."""

    dissipation = parser.add_argument_group(
        title='Dissipation',
        description='Dissipation is parametrized as: '
        "Q'=Qmin * max(1, (P/P0)**alpha) / boost"
        'Where `boost` is 1 for tidal terms outside the inertial mode range '
        'and usually < 1 in the inertial mode range.'
    )
    dissipation.add_argument(
        '--lgQ-min',
        nargs=2,
        type=float,
        default=(4.0, 12.0),
        help='The range to use for the uniform prior in the log10(`Qmin`) '
        'parameter.'
    )
    dissipation.add_argument(
        '--lgQ-break-period',
        nargs=2,
        type=float,
        default=None,
        help='The range to use for the break period (`P0`) assuming uniform '
        'prior in log(P0). If not specified, the dissipation is assumed not to '
        'depend on period, hence the `--lgQ-powerlaw-range` argument is '
        'ignored.'
    )
    dissipation.add_argument(
        '--lgQ-powerlaw',
        nargs=2,
        type=float,
        default=None,
        help='The range to use for the uniform prior for the powerlaw '
        'dependence (`alpha`) of the dissipation. If not specified, the '
        'dissipation is assumed not to depend on period, hence the '
        '`--lgQ-break-period-range` argument is ignored.'
    )
    dissipation.add_argument(
        '--lgQ-inertial-boost',
        nargs=2,
        type=float,
        default=(1.0, 1.0),
        help='The range to assume for log10(`boost`) dissipation argument '
        '(boost of dissipation in inertial mode range). If not specified, '
        'dissipation is not enhanced in the inertial mode range (i.e. `boost` '
        'is always equal to 1).'
    )


def add_cluster_args(parser):
    """Add arguments for selecting a cluster and specifying its properties."""

    open_clusters = ['NGC188', 'NGC6819', 'Praesepe/Hyades', 'M35']

    cluster = parser.add_argument_group(
        title='Open cluster',
        description='Arguments to select a cluster by name and define its '
        'observed properties.'
    )
    cluster.add_argument(
        '--cluster',
        choices=open_clusters,
        help='The name of the cluster being anaalyzed.'
    )
    cluster.add_argument(
        '--cluster-age',
        type=parse_quantity_with_errors,
        default=None,
        help='The age of the cluster, as well as its estimated '
        'standard deviation(s), possibly asymmetric.'
    )
    cluster.add_argument(
        '--cluster-feh',
        type=parse_quantity_with_errors,
        default=None,
        help='The measured [Fe/H] for the cluster as well as its estimated '
        'standard deviation(s), possibly asymmetric.'
    )


def add_stellar_spindown_args(parser, num_stars):
    """Add arguments to the parser to define priors for the spindown params."""

    assert num_stars in [1, 2]

    spin = parser.add_argument_group(
        title='Stellar spin evolution parameters',
        description='Arguments to specify the priors to assume for the '
        'parameters controlling the spin evolution of isolated stars. '
        'Specifying a the same value for upper and lower bound of any option '
        'results in a fixed value assumed for the corresponding parameter.'
    )
    spin.add_argument(
        '--disk-dissipation-age',
        type=float,
        nargs=2,
        default=(5.0, 5.0),
        help='The prior distribution to assume for the disk dissipation age is '
        'uniform within the given range in Myrs.'
    )
    for component, _ in zip(['primary', 'secondary'], range(num_stars)):
        spin.add_argument(
            '--%s-disk-lock-period' % component,
            type=float,
            nargs=2,
            default=(5.0, 5.0),
            help='The prior distribution to assume for the period (in days) to '
            'which the surface spin of stars is locked until the disk '
            'dissipates.'
        )
        spin.add_argument(
            '--%s-wind-strength' % component,
            type=float,
            nargs=2,
            default=(0.17, 0.17),
            help='The strength factor of the rate at which stars lose angular '
            'momentum to magnetically launched wind.'
        )
        spin.add_argument(
            '--%s-wind-saturation' % component,
            type=float,
            nargs=2,
            default=(2.45, 2.45),
            help='The frequency, in rad/day, above which the scaling of angular'
            ' momentum loss with spin changes from cubic to linear.'
        )
        spin.add_argument(
            '--%s-core-envelope-coupling-timescale' % component,
            type=float,
            nargs=2,
            default=(10.0, 10.0),
            help='The timescale, in Myrs, on which the core and the envelope '
            'converge toward solid body rotation.'
        )


def add_primary_args(parser, properties):
    """
    Add arguments for specifying a set of primary star properties.

    Args:
        parser:    The parser to add the arguments to (added as a new group).

        properties:    An iterable of the names of the properties to add. Must
            be a sub-set of `'logg'`, `'Teff'`, `'feh'`, and/or `'rho'`. The
            order is ignored.
    """

    primary = parser.add_argument_group(
        title='Primary star',
        description='Arguments for specifying observational constraints on '
        'primary star properties.'
    )
    for name, description in [
            (
                'feh',
                'The measured [Fe/H] for the primary star'
            ),
            (
                'logg',
                'The masured value of log10(g) at the surface of the star'
            ),
            (
                'Teff',
                'The masured value of the effective temperature of the primary '
                'star'
            ),
            (
                'rho',
                'The mesured mean density of the primary star in g/cm3'
            )
    ]:
        if name in properties:
            primary.add_argument(
                '--primary-' + name,
                type=parse_quantity_with_errors,
                help=(description
                      +
                      ', as well as its estimated standard deviation(s), '
                      'possibly asymmetric.')
            )

def get_binary_ids():
    """Return dictionary indexed by cluster containing binary IDs."""

    binaries = dict()
    binaries['NGC188'] = numpy.concatenate((
        read_cds_pipe_table(
            os.path.join(
                data_dir,
                'Geller_et_al_2009_WIYN_single_lined_orbits.tsv'
            )
        )['PKM'],
        read_cds_pipe_table(
            os.path.join(
                data_dir,
                'Geller_et_al_2009_WIYN_double_lined_orbits.tsv'
            )
        )['PKM']
    ))

    binaries['NGC6819'] = numpy.concatenate((
        read_cds_pipe_table(
            os.path.join(
                data_dir,
                'Milliman_et_al_2014_WIYN_single_lined_orbits.tsv'
            )
        )['WOCS'],
        read_cds_pipe_table(
            os.path.join(
                data_dir,
                'Milliman_et_al_2014_WIYN_double_lined_orbits.tsv'
            )
        )['WOCS']
    ))

    binaries['HyadesPraesepe'] = numpy.concatenate([
        [
            int(id)
            for id in systems['ID'].to_numpy()
        ]
        for systems in hyadespraesepe_util.get_binary_data()
    ])

    binaries['M35'] = numpy.concatenate([
        systems['WOCS'].to_numpy()
        for systems in m35_util.get_binary_data()
    ])

    return binaries


def add_binary_selection_args(parser):
    """Add an argument to parses to choose a binary system to process."""

    binaries = get_binary_ids()

    parser.add_argument(
        'system',
        choices=list(
            cluster_name + '_' + str(system)
            for cluster_name, cluster_systems in binaries.items()
            for system in cluster_systems
        ),
        help='Select the system to analyze.'
    )

def add_sampling_parameters(parser):
    """Add parameters configuring how to perform the sampling."""

    sampling = parser.add_argument_group(
        title='Sampling parameters.',
        description='Configuration for how to perform the sampling.'
    )
    sampling.add_argument(
        '--initial-eccentricity',
        type=float,
        nargs=2,
        default=(0.5, 0.5),
        help='The range to use for the uniform prior on initial eccentricity.'
    )
    parser.add_argument(
        '--sampling',
        choices=['mcmc', 'nested', 'prior'],
        default='mcmc',
        help='Which sampling method to use. MCMC sampling useng the `emcee` '
        'package and nested sampling used `dynesty`.'
    )
    sampling.add_argument(
        '--mcmc-nwalkers',
        type=int,
        default=12,
        help='The number of walkers in the MCMC ensemble. Ignored if nested '
        'sampling is used.'
    )
    sampling.add_argument(
        '--mcmc-nsteps',
        type=int,
        default=1000,
        help='The number of MCMC steps to generate. Ignored if nested sampling '
        'is used. A value of zero results in attempting to flush any unsaved '
        'steps and exit'
    )
    sampling.add_argument(
        '--mcmc-min-initial-log-probability',
        type=float,
        default=-numpy.inf,
        help='The number of MCMC steps to generate. Ignored if nested sampling '
        'is used.'
    )
    sampling.add_argument(
        '--mcmc-starting-positions-only',
        action='store_true',
        help='If passed, the script only runs until it has found acceptable '
        'starting positions for all walkers.'
    )
    sampling.add_argument(
        '--num-parallel-processes',
        type=int,
        default=4,
        help='How many multiprocessing processes to use when parallel '
        'processing is available.'
    )
    sampling.add_argument(
        '--evolution-timeout',
        type=float,
        default=3600.0,
        help='The maximum amount of time a single evolution is allowed to run. '
        'If exceeded, the associated parameters are rejected.'
    )
    sampling.add_argument(
        '--track-final-eccentricity',
        action='store_true',
        help='If passed, the calculated final eccentricity is stored along with'
        ' the input parameters.'
    )
    sampling.add_argument(
        '--rvk-interpolation-accuracy',
        type=float,
        default=1e-4,
        help='The maximum error allowed in the interpolation of the radial '
        'velocity likelihood (lambda in notes) as a fraction of approximately '
        'the largest value. Comparison is to direct numerical integration.'
    )
    sampling.add_argument(
        '--rvk-show-interpolation',
        action='store_true',
        help='Enable dispylaying plots of the interpolation of the radial '
        'velocity semi-amplitude PDF as it is being refined.'
    )
    sampling.add_argument(
        '--initial-period-search-factor', '--period-search-factor',
        type=float,
        default=2.0,
        help='The factor by which to change the initial period guess while '
        'searching for a range surrounding the known present day orbital period'
    )
    sampling.add_argument(
        '--initial-period-scaled-guess', '--period-scaled-guess',
        type=float,
        default=2.0,
        help='The search for initial period to bracket the observed final '
        'period will start from this value multiplied by the final orbital '
        'period.'
    )
    sampling.add_argument(
        '--mcmc-recover-initial-conditions',
        action='store_true',
        help='If passed, log files matching the currently defined pattern are '
        'parsed for initial mcmc walker positions that resulted in acceptable '
        'log probability, to allow continuing previously interrupted run.'
    )
    sampling.add_argument(
        '--rv-likelihood-pickle-fname',
        default='pickles/%(system)s_rv_likelihood.pkl',
        help='Filename under which to store pickled RV likelihood '
        'interpolations. Can contain %%(system)s substitution which gets '
        'replaced by the system name, and %%(sampling) - replaced by the '
        'sampling method.'
    )
    sampling.add_argument(
        '--photometric-constraint-pickle-fname',
        default='pickles/%(system)s_photometric_constraints.pkl',
        help='Filename under which to store pickled photometric constraint.'
        'Can contain %%(system)s substitution which gets replaced by the '
        'system name, and %%(sampling) - replaced by the sampling method.'
    )
    sampling.add_argument(
        '--mass-sampling-pickle-fname',
        default='pickles/%(system)s_sample_sb1_masses.pkl',
        help='Filename under which to store pickled stellar mass sampler.'
        'Can contain %%(system)s substitution which gets replaced by the '
        'system name, and %%(sampling) - replaced by the sampling method.'
    )


def add_output_parameters(parser):
    """Add parameters controlling how and what output to generate."""

    output = parser.add_argument_group(
        title='Output parameters.',
        description='Configuration controlling how and what output to generate.'
    )
    output.add_argument(
        '--samples-fname',
        default='%(system)s_%(sampling)s_samples',
        help='The filename where to save generated MCMC samples. An extension '
        'is added automatically depending on the sampling method. If the file '
        'exists, sampling continues from the state saved in it. The filename '
        'can contain the following %% substitutions: %%(system)s - replaced by '
        'the name of the system, %%(sampling) - replaced by the sampling '
        'method.'
    )
    output.add_argument(
        '--fname-datetime-format',
        default='%Y%m%d%H%M%S',
        help='How to format date and time as part of filenames (e.g. when '
        'creating output files for multiprocessing.'
    )
    output.add_argument(
        '--std-out-err-fname',
        default='sampling_output/%(system)s_%(now)s_%(pid)d.outerr',
        help='Filename to redirect worker process stdout and stderr to during '
        'multiprocessing. Should include at least `%%(pid)d` (worker process '
        'id) substitution to avoid mangling, but may also include `%%(system)s`'
        ' (system name) and `%%(now)s` (approximate date and time the process '
        'started).'
    )
    output.add_argument(
        '--logging-fname',
        default='sampling_output/%(system)s_%(now)s_%(pid)d.log',
        help='Filename for log mesasges from sampling. Should include at least '
        '`%%(pid)d` (worker process id) substitution to avoid mangling during '
        'multiprocessing, but may also include `%%(system)s`'
        ' (system name) and `%%(now)s` (approximate date and time the process '
        'started).'
    )
    output.add_argument(
        '--logging-verbosity', '--verbosity',
        choices=['debug', 'info', 'warning', 'error', 'critical'],
        default='info',
        help='The lowest importance level of logging messages to issue.'
    )
    output.add_argument(
        '--logging-datetime-format',
        default=None,
        help='How to format date and time as part of filenames (e.g. when '
        'creating output files for multiprocessing.'
    )

    output.add_argument(
        '--logging-message-format', '--logging-format', '--log-fmt',
        default=default_logging_format,
        help='How to format logging messages. See python logging module '
        'documentation for details.'
    )

def parse_command_line(description,
                       config_fname,
                       *,
                       dissipation=False,
                       cluster=False,
                       primary_properties=(),
                       choose_binary=False,
                       spindown=0):
    """
    Parse the command line for a Bayesian run.

    Args:
        description(str):    The description to display in command line help
            message of the tool using the parsed command line.

        config_fname(str):    The name of the default config file.

        cluster(bool):    Whether to include command line arguments to select a
            an open cluster, and specify age and metallicity distributions to
            assume.

        primary_properties(iterable):    List of command line arguments for
            specifying observational constraints for the primary star
            properties. See :func:`add_primary_args()` for a list of all
            properties supported.
            .

        choose_binary(bool):    Whether to add argument for selecting a
            particular binary by name.

        spindown(int):    For how many components should spindown parameters be
            added (0, 1, or 2).

    Returns:
        argparse.Namespace:
            The parsed command line options.
    """

    parser = ArgumentParser(
        description=description,
        default_config_files=[config_fname],
        args_for_writing_out_config_file=['--generate-config-file'],
        args_for_setting_config_path=['--config-file', '-c'],
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
        '--eccentricity-expansion-coefficients', '--e-coef',
        default=os.path.expanduser(
            '~/projects/git/poet/eccentricity_expansion_coef_O400.sqlite'
        ),
        help='The file to read eccentricity expansion coefficients from.'
    )
    parser.add_argument(
        '--reseed-random-number-generator',
        default=False,
        action='store_true',
        help='In an ideal world, this should not be done. If this argument is '
        'passed, the internal random number generator object within ``emcee`` '
        'is re-seeded after the sampler is created. This is only useful if '
        'sampling encounters a state that causes extremely rare crashes '
        'that we don\'t have time to fix.'
    )

    add_sampling_parameters(parser)
    add_output_parameters(parser)
    if dissipation:
        add_dissipation_args(parser)
    if cluster:
        add_cluster_args(parser)
    if primary_properties:
        add_primary_args(parser, primary_properties)
    if spindown:
        add_stellar_spindown_args(parser, spindown)
    if choose_binary:
        add_binary_selection_args(parser)
    return parser.parse_args()
