#!/usr/bin/env python3

"""M35 specific functions required for circularization bayesian analysis."""

import os.path
import logging

from matplotlib import pyplot
import pandas
import numpy
from scipy import stats

from planetary_system_io import read_cds_pipe_table
from command_line_utilities import data_dir
from cmd_utils import CMDPhotometryInterpolator


def get_photometry():
    """Return a pandas DataFrame containing M35 V and B-V photometry."""

    return pandas.DataFrame(
        read_cds_pipe_table(
            os.path.join(
                data_dir,
                'Leiner_et_al_15_M35_rv_photometry.tsv'
            )
        )
    )

def get_binary_data(
        single_lined_orbits_fname=(
            os.path.join(
                data_dir,
                'Leiner_et_al_15_M35_sb1_orbits.tsv'
            )
        ),
        double_lined_orbits_fname=(
            os.path.join(
                data_dir,
                'Leiner_et_al_15_M35_sb2_orbits.tsv'
            )
        )
):
    """Return M35 SB1 and SB2 systems as :class:`pandas.DataFrame`s."""

    return (
        pandas.DataFrame(read_cds_pipe_table(single_lined_orbits_fname)),
        pandas.DataFrame(read_cds_pipe_table(double_lined_orbits_fname))
    )

def get_photometry_interpolator():
    """Return interpolator of photometry matching M35's age and [Fe/H]."""

    return CMDPhotometryInterpolator(
        os.path.join(
            data_dir,
            'CMD_150Myr_FeH-0.21dex_isochrone_Av0.62_UBVRIJHK.dat'
        ),
        9.8
    )

def plot_color_magnitude_diagram():
    """Plot the theoretical and observed CMD for M35 along with binaries."""

    interpolator = get_photometry_interpolator()

    measured_photometry = get_photometry()

    single_photometry = measured_photometry[measured_photometry['Mm'] == 'SM ']

    sb1_orbits, sb2_orbits = get_binary_data()

    sb1_data = pandas.merge(sb1_orbits, measured_photometry, on='WOCS')
    sb2_data = pandas.merge(sb2_orbits, measured_photometry, on='WOCS')

    interp_masses = interpolator.data[0]['Mini']
    predicted_photometry = interpolator(interp_masses)

    pyplot.plot(predicted_photometry[2] - predicted_photometry[3],
                -predicted_photometry[3],
                '-y',
                linewidth=3,
                zorder=10,
                label='Isochrone')

    marker_phot = interpolator(numpy.array([0.8, 1.6, 4.0]))
    pyplot.plot(marker_phot[2] - marker_phot[3],
                -marker_phot[3],
                'or',
                zorder=20,
                label='M=0.8, 1.6, 4.0 $M_\odot$')

    pyplot.plot(single_photometry['B-V'],
                -single_photometry['Vmag'],
                'ok',
                zorder=0,
                label='Single member')

    pyplot.plot(sb1_data['B-V'],
                -sb1_data['Vmag'],
                'og',
                zorder=1,
                label='SB1')
    pyplot.plot(sb2_data['B-V'],
                -sb2_data['Vmag'],
                'ob',
                zorder=1,
                label='SB2')
    pyplot.xlabel('B-V [mag]')
    pyplot.ylabel('-V [mag]')
    pyplot.legend()

    pyplot.show()

if __name__ == '__main__':
    plot_color_magnitude_diagram()
