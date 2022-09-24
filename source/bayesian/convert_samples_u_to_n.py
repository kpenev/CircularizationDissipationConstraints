#!/usr/bin/env python3

"""Convert samples file from unit cube to independent normal sampling vars."""

from argparse import ArgumentParser
from os.path import exists
from shutil import copyfile

from scipy.stats import norm
import h5py

if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        'fnames',
        nargs=2,
        help='The filename to convert and the output filename.'
    )
    in_fname, out_fname = parser.parse_args().fnames
    assert not exists(out_fname)
    copyfile(in_fname, out_fname)

    with h5py.File(out_fname, 'r+') as outfile:
        for chain in outfile:
            niter = outfile[chain].attrs['iteration']
            if niter > 0:
                orig = outfile[chain]['chain'][:]
                print('Orig chain shape %s' % repr(orig.shape))
                print('Transforming first %d iterations' % niter)
                outfile[chain]['unit_cube_chain'] = orig[:niter]
                outfile[chain]['chain'][:niter] = norm.ppf(orig[:niter])
            orig = outfile[chain]['starting_positions']['unit_cube_values'][:]
            print('Transforming %dx%d starting positions' % orig.shape)
            outfile[
                chain
            ][
                'starting_positions'
            ][
                'independent_normal_values'
            ] = norm.ppf(orig)
