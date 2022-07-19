#!/usr/bin/env python3
"""I/O of Windemuth et. al. (2019) Kepler EB samples."""

from os import path

import numpy
import pandas

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

    return pandas.DataFrame(
        numpy.load(
            path.join(_data_dir, str(kic_id) + '.npz')
        )[
            'thinned_chain'
        ],
        columns=['Mtot',        #(Msun)
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
    )


if __name__ == '__main__':
    samples = get_samples(5649206)
    print('Samples:\n{!r}'.format(samples))
    print('P stddev:' + repr(numpy.std(samples['P(d)'])))
