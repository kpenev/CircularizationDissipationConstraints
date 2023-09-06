#!/usr/bin/env python3
"""I/O of Windemuth et. al. (2019) Kepler EB samples."""

from os import path
import logging
from subprocess import run
import platform

from matplotlib import pyplot
import numpy
from scipy import stats
import pandas
from astropy import units, constants
from configargparse import ArgumentParser, DefaultsFormatter

from stellar_evolution.manager import StellarEvolutionManager
from stellar_evolution.library_interface import library as stellar_evol_lib

from process_e_Q_grid import LinearEccentricityEnvelope

_data_dir = path.join(
    path.dirname(
        path.dirname(
            path.dirname(
                path.abspath(__file__)
            )
        )
    ),
    'data',
    'windemuth_et_al_19_samples'
)

eccentricity_envelope = LinearEccentricityEnvelope(min_period=1.0,
                                                   max_period=25.0,
                                                   min_eccenticity=0.02,
                                                   max_eccentricity=0.8)
_logger = logging.getLogger(__name__)


def get_max_likelihood_params():
    """Return the maximum likelihood estimates of the system parameters."""

    return pandas.read_csv(
        path.join(_data_dir, 'paper_maxlike_pars.dat'),
        header=0,
        sep=' ',
        index_col='KIC',
        escapechar='#'
    )


def get_samples(kic_id):
    """Return a pandas.DataFrame of the samples for the given EB (KIC Id)."""

    columns = ['Mtot',        #(Msun)
               'Mratio',
               'z',
               'tau',         #(log10yr)
               'dist',        #(pc)
               'E(B-V)',
               'P',           #(d)
               'tpe',         #(d)
               'esinw',
               'ecosw',
               'b',
               'q11',
               'q12',
               'q21',
               'q22',
               'SysLCerror',  #(ln)
               'SysSEDerror', #(ln)
               'E(B-V)unc',   #(ln)
               'distunc']     #(lnpc)
    raw_data = numpy.load(
        path.join(_data_dir, str(kic_id) + '.npz')
    )[
        'thinned_chain'
    ]
    if raw_data.shape[1] == 18:
        del columns[17]
    return pandas.DataFrame(raw_data, columns=columns)


def get_summary_data(kic_id=None):
    """Return the maximum likelihood and quantile posterior data."""

    maxlike_data = pandas.read_csv(
        path.join(_data_dir, 'paper_maxlike_pars.dat'),
        sep=r'\s+',
        index_col='#KIC'
    )
    stellar_data = pandas.read_csv(
        path.join(_data_dir, 'paper_stellar.posteriors'),
        header=1,
        sep=r'\s+',
        index_col='#KIC'
    )
    orbital_data = pandas.read_csv(
        path.join(_data_dir, 'paper_orbital.posteriors'),
        header=1,
        sep=r'\s+',
        index_col='#KIC'
    )

    maxlike_data.columns = ['maxlike_' + col for col in maxlike_data.columns]
    stellar_data.columns = ['posterior_' + col  for col in stellar_data.columns]
    orbital_data.columns = ['posterior_' + col  for col in orbital_data.columns]

    summary = maxlike_data.join([stellar_data, orbital_data])

    if kic_id is None:
        return summary.sort_values(by='maxlike_period')
    return summary.loc[kic_id]


def get_available_kic(interpolator=None, max_porb=numpy.inf, max_e=numpy.inf):
    """
    Return a list of the KIC which can be sampled.

    Args:
        interpolator(stellar evolution interpolator or None):    If not None, an
            additional check is performed to verify age is within POET range and
            that the convective moment of inertia is always positive.
    """

    def check_iconv(mass, feh, age):
        """Check if given star has always positive Iconv up to max_age."""

        iconv = interpolator('ICONV', mass, feh)
        return (
            iconv.max_age > age
            and
            iconv(numpy.linspace(10.0**(-0.5), age, 100)).min() > 0
            and
            iconv(numpy.logspace(-0.5, numpy.log10(age), 100)).min() > 0
        )

    data = get_summary_data()

    valid = data['maxlike_morph'] < 0.5
    _logger.info('Morphology cut leaves %d binaries', valid.sum())

    valid = numpy.logical_and(
        valid,
        data['posterior_tau(log10yr)'] + data['posterior_tau-sigma'] > 8.5
    )
    _logger.info('Age cut leaves %d binaries', valid.sum())

    valid = numpy.logical_and(
        valid,
        data['posterior_m1(msun)'] + data['posterior_m1+sigma'] < 1.2
    )
    valid = numpy.logical_and(
        valid,
        data['posterior_m2(msun)'] + data['posterior_m2+sigma'] < 1.2
    )
    _logger.info('Mass cut leaves %d binaries', valid.sum())

    logg = numpy.log10(
        numpy.minimum(
            (
                constants.G
                *
                data['posterior_m1(msun)'].to_numpy() * units.M_sun
                /
                (data['posterior_r1(rsun)'].to_numpy() * units.R_sun)**2
            ).to_value(units.cm / units.s**2),
            (
                constants.G
                *
                (data['posterior_m2(msun)'].to_numpy() * units.M_sun)
                /
                (data['posterior_r2(rsun)'].to_numpy() * units.R_sun)**2
            ).to_value(units.cm / units.s**2)
        )
    )
    valid = numpy.logical_and(valid, logg > 4.0)
    _logger.info('Log10(g) cut leaves %d binaries', valid.sum())

    valid = numpy.logical_and(valid, data['maxlike_period'] <= max_porb)
    _logger.info('Period cut leaves %d binaries', valid.sum())

    valid = numpy.logical_and(
        valid,
        (data['maxlike_esinw']**2 + data['maxlike_ecosw']**2) <= max_e
    )
    _logger.info('Eccentricity cut leaves %d binaries', valid.sum())

    if interpolator is not None:
        valid = numpy.logical_and(
            valid,
            numpy.vectorize(interpolator.feh_in_range)(
                numpy.vectorize(stellar_evol_lib.feh_from_z)(
                    data['posterior_z'].to_numpy()
                )
            )
        )
        _logger.info('[Fe/H] cut leaves %d binaries', valid.sum())
        #False positive
        #pylint: disable=no-member
        data = data.iloc[valid.to_numpy()]
        #pylint: enable=no-member
        valid = numpy.ones(data.index.size, dtype=bool)
        for i in range(valid.size):
            info = data.iloc[[i]]
            valid[i] = check_iconv(
                float(info['posterior_m1(msun)']),
                float(stellar_evol_lib.feh_from_z(info['posterior_z'])),
                10.0**(float(info['posterior_tau(log10yr)'])
                       +
                       float(info['posterior_tau+sigma'])
                       -
                       9.0)
            )
        _logger.info('Iconv cut leaves %d binaries', valid.sum())

    return data.index[valid].to_numpy()


def plot_eccentricity_vs_period(plot_fname, available_kic):
    """Plot or show period-eccentricity envelope and data it is based on."""

    plot_data = numpy.empty(len(available_kic), dtype=[('KIC', int),
                                                       ('P_median', float),
                                                       ('P_min', float),
                                                       ('P_max', float),
                                                       ('e_median', float),
                                                       ('e_min', float),
                                                       ('e_max', float),
                                                       ('M1_median', float),
                                                       ('M1_min', float),
                                                       ('M1_max', float),
                                                       ('tau_median', float),
                                                       ('tau_min', float),
                                                       ('tau_max', float),
                                                       ('Mratio_median', float),
                                                       ('Mratio_min', float),
                                                       ('Mratio_max', float)])
    target_quantiles = stats.norm().cdf((-1.0, 0.0, 1.0))
    for i, kic in enumerate(available_kic):
        samples = get_samples(kic)
        plot_data['KIC'] = kic
        for quantity in ['P', 'tau', 'Mratio']:
            (
                plot_data[f'{quantity}_min'][i],
                plot_data[f'{quantity}_median'][i],
                plot_data[f'{quantity}_max'][i]
            ) = numpy.percentile(samples[quantity].array, target_quantiles)
        (
            plot_data['e_min'][i],
            plot_data['e_median'][i],
            plot_data['e_max'][i]
        ) = numpy.percentile(
            numpy.sqrt(
                samples['esinw'].array**2
                +
                samples['ecosw'].array**2
            ),
            target_quantiles
        )
        (
            plot_data['M1_min'][i],
            plot_data['M1_median'][i],
            plot_data['M1_max'][i]
        ) = numpy.percentile(
            samples['Mtot']
            /
            (1.0 + numpy.minimum(samples['Mratio'], 1.0 / samples['Mratio'])),
            target_quantiles
        )

    selected = plot_data['Mratio_median'] > 0.5
    for label in ['q > 0.5', 'q <= 0.5']:
        pyplot.xscale('log')
        pyplot.errorbar(
            plot_data['P_median'][selected],
            plot_data['e_median'][selected],
            numpy.stack((
                plot_data['e_median'] - plot_data['e_min'],
                plot_data['e_max'] - plot_data['e_median']
            ))[:, selected],
            numpy.stack((
                plot_data['P_median'] - plot_data['P_min'],
                plot_data['P_max'] - plot_data['P_median']
            ))[:, selected],
            fmt='o',
            markerfacecolor='none',
            label=label
        )
        selected = numpy.logical_not(selected)

    envelope_x = 10.0**numpy.linspace(-1.0, 3, 1000)
    pyplot.plot(envelope_x,
                eccentricity_envelope(envelope_x),
                '-k')
    pyplot.legend()
    if not plot_fname:
        pyplot.show()
    else:
        pyplot.savefig(plot_fname)


def generate_slurm_scripts(hpc, available_kic, slurm_dir, sampling_mode):
    """Create slurm scripts for a given HPC cluster to sample W19 systems."""

    if hpc == 'ganymede':
        sys_per_node = 1
    elif hpc == 'stampede':
        sys_per_node = 3
    else:
        assert hpc == 'ls6'
        sys_per_node = 8
    kic_by_node = numpy.zeros(
        (int(numpy.ceil(available_kic.size / sys_per_node)), sys_per_node),
        dtype=available_kic.dtype
    )
    kic_by_node.ravel()[:available_kic.size] = available_kic
    slurm_generator = path.join(slurm_dir, 'generate_slurm.sh')
    for node_kic in kic_by_node:
        run(
            [slurm_generator, 'W19', sampling_mode, hpc]
            +
            [str(kic) for kic in node_kic],
            check=True
        )


def parse_command_line():
    """The command line arguments supplied when used as a script."""

    parser = ArgumentParser(
        description='Convenience tool for working with W19 binaries.',
        default_config_files=['w19_util.cfg'],
        args_for_writing_out_config_file=['--generate-config-file'],
        args_for_setting_config_path=['--config-file', '-c'],
        formatter_class=DefaultsFormatter,
        ignore_unknown_config_file_keys=False
    )
    parser.add_argument(
        '--stellar-evolution-interpolator-dir', '--interpolator-dir',
        default=(
            path.expanduser(
                '~/projects/git/poet/stellar_evolution_interpolators'
            )
        ),
        help='The directory to read stellar evolution interpolator from.'
    )
    parser.add_argument(
        '--max-porb',
        type=float,
        default=numpy.inf,
        help='Filter the W19 systems to only those with maximum likelihood '
        'orbital period less than the specified value.'
    )
    parser.add_argument(
        '--max-eccentricity', '--max-e',
        type=float,
        default=numpy.inf,
        help='Filter the W19 systems to only those with maximum likelihood '
        'present day eccentricity less than the specified value.'
    )
    parser.add_argument(
        '--create-pe-plot',
        default=None,
        help='If specified, a period-eccentricity plot will be created and '
        'saved with the given filename. Use empty string to show the plot '
        'instead of saving.'
    )

    parser.add_argument(
        '--list-valid-systems',
        action='store_true',
        help='If passed, output a list of systems for which circularization '
        'analysis can be performed, possibly imposing orbital period and '
        'eccentricity cuts (see --max-porb and --max-eccentricity arguments).'
    )
    parser.add_argument(
        '--generate-slurm-scripts', '--slurms',
        default=None,
        help='If passed, slurm scripts for circularization analysis of all '
        'valid systems are generated for the sampling mode specified and '
        'for the HPC cluster specified by --hpc.'
    )
    parser.add_argument(
        '--slurm-dir',
        default=path.join(
            path.dirname(path.dirname(path.abspath(__file__))),
            'slurm',
        ),
        help='The directory containing the slurm utilities for each HPC '
        'cluster.'
    )
    parser.add_argument(
        '--hpc',
        choices=['ganymede', 'stampede', 'ls6'],
        default=None,
        help='The HPC cluster to generate slurm scripts for. By default it '
        'is automatically determined from the host name (assuming you are '
        'running this on the cluster you need slurm scripts for).'
    )
    config = parser.parse_args()
    if config.hpc is None:
        hostname = platform.node()
        for hpc in ['ganymede', 'stampede', 'ls6']:
            if hpc in hostname:
                assert config.hpc is None
                config.hpc = hpc
    return config


def main(config):
    """Avoid polluting global namespace."""

    interpolator = StellarEvolutionManager(
        config.stellar_evolution_interpolator_dir
    ).get_interpolator_by_name(
        'default'
    )

    logging.basicConfig(level=logging.INFO)

    available_kic = get_available_kic(interpolator,
                                      config.max_porb,
                                      config.max_eccentricity)

    if config.list_valid_systems:
        print('\n'.join(map(repr, available_kic)))

    if config.create_pe_plot is not None:
        plot_eccentricity_vs_period(config.create_pe_plot, available_kic)

    if config.generate_slurm_scripts:
        generate_slurm_scripts(config.hpc,
                               available_kic,
                               config.slurm_dir,
                               config.generate_slurm_scripts)


if __name__ == '__main__':
    main(parse_command_line())
