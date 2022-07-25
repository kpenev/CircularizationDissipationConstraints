#!/usr/bin/env python3
"""I/O of Windemuth et. al. (2019) Kepler EB samples."""

from os import path
import logging

from matplotlib import pyplot
import numpy
import scipy.stats
import pandas
from astropy import units, constants

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


def get_summary_stats():
    """Return the posteriors median and 1-sigma quantiles."""

    return pandas.merge(
        pandas.read_csv(
            path.join(_data_dir, 'paper_orbital.posteriors'),
            header=1,
            sep=' ',
            index_col='KIC',
            escapechar='#'
        ),
        pandas.read_csv(
            path.join(_data_dir, 'paper_stellar.posteriors'),
            header=1,
            sep=' ',
            index_col='KIC',
            escapechar='#'
        ),
        left_index=True,
        right_index=True
    )


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


def get_summary_data():
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

    return maxlike_data.join([stellar_data, orbital_data])


def get_available_kic():
    """Return a list of the KIC identifiers for which samples are available."""

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

    return data.index[valid].to_numpy()


def plot_eccentricity_vs_period(plot_fname=None):
    """Plot or show period-eccentricity envelope and data it is based on."""

    available_kic = get_available_kic()

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
    if plot_fname is None:
        pyplot.show()
    else:
        pyplot.savefig(plot_fname)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    plot_eccentricity_vs_period('w19_period_eccentricity.pdf')
