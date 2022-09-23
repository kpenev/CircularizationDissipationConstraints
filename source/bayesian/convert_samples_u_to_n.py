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
