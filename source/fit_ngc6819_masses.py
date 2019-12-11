#!/usr/bin/env python3

"""Find max likelinhood masses in NGC6819 from photometry and RVs."""

from matplotlib import pyplot
import scipy
import pandas

from planetary_system_io import read_cds_pipe_table
from cmd_photometry_interpolator import CMDPhotometryInterpolator
from mass_fitting import fit_binary_masses

def plot_bostanci_cmd(interpolator, photometry, distance_modulus):
    """Create a color-magnitude diagram of NGC 6819."""

#    single_member_phot = photometry[photometry['Mm'] == b'SM ']
    single_member_phot = photometry[photometry['Pmemb'] > 0.5]

    interp_masses = interpolator.data[0]['Mini']
    predicted_photometry = interpolator(interp_masses) + distance_modulus

    pyplot.plot(predicted_photometry[1] - predicted_photometry[2],
                -predicted_photometry[2],
                '-y',
                linewidth=5,
                zorder=10)


    pyplot.plot(single_member_phot['B-V'],
                -single_member_phot['Vmag'],
                'ok',
                zorder=0)
    pyplot.xlim(0, 2)
    pyplot.ylim(-20, -13)
    pyplot.show()

def plot_miliman_cmd(interpolator, photometry, distance_modulus):
    """Create a color-magnitude diagram of NGC 6819."""

    single_member_phot = photometry[photometry['Mm'] == b'SM ']

    interp_masses = interpolator.data[0]['Mini']
    predicted_photometry = interpolator(interp_masses) + distance_modulus

    pyplot.plot(predicted_photometry[2] - predicted_photometry[4],
                -predicted_photometry[2],
                '-y',
                linewidth=5,
                zorder=10)


    pyplot.plot(single_member_phot['V-I'],
                -single_member_phot['Vmag'],
                'ok',
                zorder=0)
    pyplot.xlim(0, 2)
    pyplot.ylim(-20, -13)
    pyplot.show()

def main():
    """Avoid polluting global namespace."""

    bostanci_interpolator =CMDPhotometryInterpolator(
        '../data/CMD_2.5Gyr_isochrone_Av0.35_FeH0.09_UBVRIJHK.dat'
    )
    milliman_interpolator =CMDPhotometryInterpolator(
        '../data/CMD_2.5Gyr_isochrone_Av0.5_FeH0.09_UBVRIJHK.dat'
    )
    milliman_photometry = read_cds_pipe_table(
        '../data/Milliman_et_al_14_NGC6819_RV_summary_with_VIphotometry.tsv'
    )

    single_lined_orbits = pandas.DataFrame(
        read_cds_pipe_table(
            '../data/Milliman_et_al_14_NGC6819_RV_single_lined_orbits.tsv'
        )
    )

    single_lined_data = pandas.merge(single_lined_orbits,
                                     pandas.DataFrame(milliman_photometry),
                                     on='WOCS')

    bostanci_photometry = read_cds_pipe_table(
        '../data/Bostanci_et_al_16_NGC6819_UBV_photometry.tsv'
    )
    plot_miliman_cmd(milliman_interpolator, milliman_photometry, 11.85)
    plot_bostanci_cmd(bostanci_interpolator, bostanci_photometry, 12.0)

if __name__ == '__main__':
    main()
