#!/usr/bin/env python3
"""I/O of Windemuth et. al. (2019) Kepler EB samples."""

from os import path

from matplotlib import pyplot
import numpy
import scipy.stats
import pandas

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


def get_available_kic():
    """Return a list of the KIC identifiers for which samples are available."""

    ml_data = pandas.read_csv(
        path.join(_data_dir, 'paper_maxlike_pars.dat'),
        sep=r'\s+',
        index_col='#KIC'
    )

    return ml_data.index[(ml_data['morph'] < 0.5)].to_numpy()


if __name__ == '__main__':
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
                                                       ('tau_max', float)])
    target_quantiles = scipy.stats.norm().cdf((-1.0, 0.0, 1.0))
    for i, kic in enumerate(available_kic):
        samples = get_samples(kic)
        plot_data['KIC'] = kic
        for quantity in ['P', 'tau']:
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

    pyplot.xscale('log')
    age_selected = plot_data['tau_min'] < 8.5
    for label_pre in ['lg(t) < 8.5', 'lg(t) >= 8.5']:
        mass_selected = plot_data['M1_max'] > 1.2
        for label in ['M1>1.2', 'M1<=1.2']:
            selected = numpy.logical_and(age_selected, mass_selected)
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
                label=', '.join([label_pre, label])
            )
            mass_selected = numpy.logical_not(mass_selected)
        age_selected = numpy.logical_not(age_selected)

    envelope_x = 10.0**numpy.linspace(-1.0, 3, 1000)
    pyplot.plot(envelope_x,
                eccentricity_envelope(envelope_x),
                '-k')
    pyplot.legend()
    pyplot.show()
