#!/usr/bin/env python3
"""I/O of Windemuth et. al. (2019) Kepler EB samples."""

from os import path
import logging
from subprocess import run
import platform
from collections import defaultdict

from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy
import scipy.stats
import pandas
from astropy import units, constants
from configargparse import ArgumentParser, DefaultsFormatter

from stellar_evolution.manager import StellarEvolutionManager
from stellar_evolution.library_interface import library as stellar_evol_lib

from process_e_Q_grid import LinearEccentricityEnvelope
from eccentricity_distro import eccentricity_kde_distro_gen

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

def get_eccentricity_kernel_widths():
    """Return dict of eccentricity kernel widths for each KIC."""


    result =  defaultdict(lambda: 5e-4)
    for kic, kernel_width in [
            (10268903, 1.5e-4),
            (6962018, 8e-6),
            (11616200, 2e-5),
            (4380283, 1e-5),
            (9110346, 2e-5),
            (7732791, 5e-5),
            (5039441, 1e-4),
            (9656543, 1e-4),
            (3834364, 1.2e-3),
            (11228612, 2e-4),
            (10960995, 3e-6),
            (3241344, 2e-4),
            (5022440, 3e-6),
            (5802470, 2e-6),
            (4815612, 1e-6),
            (7377033, 1e-3),
            (11867071, 3e-3),
            (3427776, 2e-3),
            (10935310, 1.5e-3),
            (10031409, 5e-6),
            (9532421, 1.5e-3),
            (3973504, 1.5e-3),
            (8957954, 2e-6),
            (6521542, 2e-4),
            (11252617, 1e-3),
            (4285087, 1e-5),
            (7025851, 4e-5),
            (4346875, 1.2e-3),
            (7691527, 2e-4),
            (6227560, 1e-4),
            (8302455, 1e-5),
            (12004679, 3e-4),
            (7369523, 1e-4),
            (9971475, 1.5e-5),
            (7129465, 2e-6),
            (5181455, 5e-6),
            (8381592, 1e-3),
            (7376500, 2e-4),
            (8618226, 2e-5),
            (9649222, 5e-6),
            (6546508, 8e-4),
            (10385682, 1e-4),
            (8460600, 1.5e-3),
            (7125636, 7e-4),
            (8580438, 5e-5),
            (5597970, 5e-6),
            (8746310, 1e-5),
            (7362852, 7e-5),
            (12557713, 2e-3),
            (4753988, 3e-4),
            (10923260, 1e-4),
            (3003991, 2.0e-2),
            (6927629, 3e-6),
            (8364119, 3e-4),
            (6949550, 3e-6),
            (9532123, 3e-4),
            (9892471, 3e-5),
            (2445134, 1.5e-4),
            (4948863, 1.5e-4),
            (9775253, 5e-7),
            (4839180, 4e-5),
            (5652260, 1e-4),
            (6707942, 1.7e-3),
            (7597703, 3e-6),
            (11232745, 2e-5),
            (8984706, 1.3e-4),
            (11409698, 1e-4),
            (9353182, 3e-5),
            (6594972, 1e-4),
            (9025914, 1e-4),
            (9665503, 1e-4),
            (6185717, 2e-4),
            (8414159, 2.5e-5),
            (6301030, 1e-5),
            (11499757, 5e-5),
            (11704044, 3e-6),
            (6359798, 2e-5),
            (7118545, 3e-5),
            (12251779, 8e-5),
            (4678171, 7e-5),
            (8111622, 2e-5),
            (5622250, 5e-5),
            (8879427, 3e-4),
            (5979863, 1e-3),
            (9001468, 2e-4),
            (6522750, 2e-4),
            (6131659, 5e-5),
            (12316447, 3e-5),
            (7624297, 2e-4),
            (10992733, 8e-5),
            (7021177, 1.5e-4),
            (10753734, 3e-5),
            (10711913, 1.5e-3),
            (10518735, 1e-4),
            (9016295, 6e-4),
            (10258558, 2.5e-4),
            (4252226, 1e-4),
            (4633434, 2e-3),
            (12017140, 1e-4),
            (9838060, 2e-3),
            (6672229, 2.5e-4),
            (7821010, 2e-4),
            (10849244, 5e-5),
            (10215422, 1e-6),
            (5983348, 6e-4),
            (7767733, 1.5e-3),
            (10651945, 1.5e-4),
            (4773155, 8e-5),
            (5553624, 2e-4),
            (12356914, 2e-3),
            (8572936, 2e-5),
            (8973000, 1e-4),
            (2998124, 1e-4),
            (6431670, 1e-5),
            (4847832, 1e-5),
            (8183389, 1e-3),
            (5003117, 1.2e-3),
            (12644769, 1e-4),
            (8553907, 1e-4),
            (12217907, 2e-4),
            (7541502, 3e-5),
            (10420279, 7e-5),
            (4247023, 1.5e-4),
            (9164836, 1e-4),
            (8610483, 1.5e-4),
            (9714123, 1e-3),
            (9837544, 1e-5),
            (8560285, 1.5e-3),
            (8044608, 5e-5),
            (10292238, 1.5e-4),
            (4824268, 3e-4),
            (8760135, 5e-5),
            (9839062, 1e-3)
    ]:
        result[kic] = kernel_width
    return result

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
    _logger.info('Morphology cut leaves {:d} binaries'.format(valid.sum()))

    valid = numpy.logical_and(
        valid,
        data['posterior_tau(log10yr)'] + data['posterior_tau-sigma'] > 8.5
    )
    _logger.info('Age cut leaves {:d} binaries'.format(valid.sum()))

    valid = numpy.logical_and(
        valid,
        data['posterior_m1(msun)'] + data['posterior_m1+sigma'] < 1.2
    )
    valid = numpy.logical_and(
        valid,
        data['posterior_m2(msun)'] + data['posterior_m2+sigma'] < 1.2
    )
    _logger.info('Mass cut leaves {:d} binaries'.format(valid.sum()))

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
    _logger.info('Log10(g) cut leaves {:d} binaries'.format(valid.sum()))

    valid = numpy.logical_and(valid, data['maxlike_period'] <= max_porb)
    _logger.info('Period cut leaves {:d} binaries'.format(valid.sum()))

    valid = numpy.logical_and(
        valid,
        (data['maxlike_esinw']**2 + data['maxlike_ecosw']**2) <= max_e
    )
    _logger.info('Eccentricity cut leaves {:d} binaries'.format(valid.sum()))

    if interpolator is not None:
        valid = numpy.logical_and(
            valid,
            numpy.vectorize(interpolator.feh_in_range)(
                numpy.vectorize(stellar_evol_lib.feh_from_z)(
                    data['posterior_z'].to_numpy()
                )
            )
        )
        _logger.info('[Fe/H] cut leaves {:d} binaries'.format(valid.sum()))
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
        _logger.info('Iconv cut leaves {:d} binaries'.format(valid.sum()))

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
    target_quantiles = scipy.stats.norm().cdf((-1.0, 0.0, 1.0))
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


def plot_eccentricity_distribution(kic_id_list,
                                   plot_fname,
                                   bins):
    """
    Plot KDE estimated eccentricity distribution and histogram for given KIC.

    Args:
        kic_id_list([int,...]):    The KIC identifiers of the Windemuth et. al.
            (2019) binaries to plot the eccentricity distribution of. A
            multi-page PDF is created with each KIC plot on a separate page.

        plot_fname(str):    The filename to save the plot. If empty, the plots
            are shown but not saved.

        bins(int or sequence of floats or str):    The bins to use for buliding
            the histogram to show. See `numpy.histogram` for details. The numpy.
            derived bins are then scaled by the inverse of the area
            corresponding to each bin.

    Returns:
        None
    """


    custom_zoom = {7732791: (0.0, 0.005),
                   9656543: (0.0, 1e-3),
                   10960995: (0.0, 1e-4),
                   5022440: (0.0, 1e-4),
                   5802470: (0.0, 1e-4),
                   4815612: (0.0, 4e-5),
                   10031409: (0.0, 2e-4),
                   8957954: (0.0, 1e-4),
                   4285087: (0.0, 2e-4),
                   6227560: (0.0, 2e-3),
                   8302455: (0.0, 2e-4),
                   9971475: (0.0, 5e-4),
                   7129465: (0.0, 1e-4),
                   5181455: (0.0, 2e-4),
                   7376500: (0.41, 0.42),
                   8618226: (0.0, 5e-4),
                   9649222: (0.0, 2e-4),
                   5597970: (0.0, 2e-4),
                   8746310: (0.0, 2e-4),
                   10923260: (0.4, 0.42),
                   3003991: (0.0, 0.2),
                   6927629: (0.0, 1e-4),
                   6949550: (0.26477, 0.26493),
                   9892471: (0, 0.002),
                   9775253: (0, 3e-5),
                   7597703: (0, 2e-4),
                   11232745: (0, 1e-3),
                   8414159: (0, 0.002),
                   11499757: (0.26, 0.263),
                   11704044: (0, 2e-4),
                   8879427: (0.45, 0.465),
                   12316447: (0.368, 0.370),
                   7021177: (0.584, 0.596),
                   8572936: (0.5170, 0.5195),
                   8973000: (0.52, 0.53)}
    eccentricity_pdf_kernel_widths = get_eccentricity_kernel_widths()

    if plot_fname:
        pdf = PdfPages(plot_fname)
    for kic_id in kic_id_list:
        w19_samples = get_samples(kic_id)
        e_samples = numpy.sort(
            numpy.sqrt(w19_samples['esinw']**2
                       +
                       w19_samples['ecosw']**2)
        )
        print('Ecccentricity samples:\n' + repr(e_samples))
        kernel_width = eccentricity_pdf_kernel_widths[kic_id]
        kde_distro = eccentricity_kde_distro_gen(e_samples, kernel_width)
#        plot_x = numpy.linspace(e_samples[0] - 5.0 * kernel_width,
#                                e_samples[-1] + 5.0 * kernel_width,
#                                100)
#        pyplot.plot(plot_x, kde_distro.cdf(plot_x), color='red')
#        pyplot.axhline(y=0.1)
#        pyplot.axhline(y=0.9)
#        pyplot.show()
#        pyplot.clf()


        plot_ranges = [
            (
                0,
                e_samples[-1]
            ),
            custom_zoom.get(kic_id,
                            (kde_distro.ppf(0.025), kde_distro.ppf(0.975)))
        ]
        if (
            plot_ranges[1][1] - plot_ranges[1][0]
            >
            0.2 * (plot_ranges[0][1] - plot_ranges[0][0])
            and
            kic_id not in custom_zoom
        ):
            print(
                'Full range (%g, %g) and zoom range (%g, %g) comparable. No '
                'zoom plot necessary.'
                %
                (plot_ranges[0] + plot_ranges[1])
            )
            plot_ranges = plot_ranges[:1]
        else:
            print('Adding zoom-in plot for KIC %d: %g < ef < %g'
                  %
                  ((kic_id,) + plot_ranges[1]))


        for e_range in plot_ranges:
            hist, bin_edges = numpy.histogram(
                e_samples,
                bins=int(numpy.ceil(
                    bins
                    *
                    max(
                        1,
                        (e_samples[-1] - e_samples[0])
                        /
                        (e_range[1] - e_range[0])
                    )
                )),
                density=True
            )

            hist /= bin_edges[1:]**2 - bin_edges[:-1]**2
            hist /= (hist * (bin_edges[1:] - bin_edges[:-1])).sum()
            pyplot.bar(x=bin_edges[:-1],
                       height=hist,
                       width=bin_edges[1:] - bin_edges[:-1],
                       align='edge',
                       color='none',
                       edgecolor='black')

            plot_x = numpy.linspace(e_range[0], e_range[1], 300)
            plot_y = kde_distro.pdf(plot_x)
            pyplot.plot(plot_x, plot_y, color='red')
            pyplot.xlim(*e_range)
            pyplot.suptitle(str(kic_id) + ' PDF($e_f$)')

            if (
                plot_y[plot_x < 0.6 * e_range[0] + 0.4 * e_range[1]].max()
                <
                plot_y[plot_x > 0.4 * e_range[0] + 0.6 * e_range[1]].max()
            ):
                inset_location = 'upper left'
            else:
                inset_location = 'upper right'

            inset = inset_axes(pyplot.gca(),
                               width='35%',
                               height='35%',
                               loc=inset_location)

            inset.plot(w19_samples['ecosw'],
                       w19_samples['esinw'],
                       'ok',
                       markersize=0.5)
            inset.axhline(y=0, linewidth=0.5)
            inset.axvline(x=0, linewidth=0.5)

    #    pyplot.yscale('log')
            if plot_fname:
                pdf.savefig()
                pyplot.close()
            else:
                pyplot.show()
    if plot_fname:
        pdf.close()


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
        '--create-e-distro-plot',
        nargs=2,
        default=None,
        help='If specified, it should select a KIC and filename for the plot. '
        'A plot of the eccentricity distribution for the selected KIC will be '
        'created. The plot will show binned eccentricity samples, with bin '
        'heights divided by the area of the annulus in (ecosw, esinw) '
        'space corresponding to each bin as well as the KDE estimated '
        'eccentricity distrubition.'
    )
    parser.add_argument(
        '--e-distro-histogram-bins',
        type=int,
        default=30,
        help='The number of bins to use for plotting eccentricity ditribution.'
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
    if config.create_e_distro_plot is not None:
        plot_eccentricity_distribution(
            kic_id_list=(
                available_kic if config.create_e_distro_plot[0] == 'all'
                else [int(config.create_e_distro_plot[0])]
            ),
            plot_fname=config.create_e_distro_plot[1],
            bins=config.e_distro_histogram_bins
        )


if __name__ == '__main__':
    main(parse_command_line())
