#!/usr/bin/env python3
"""Diagnostic plots for binary mas fitting for NGC6819."""

import os.path

from matplotlib import pyplot

#from planetary_system_io import read_cds_pipe_table
from cmd_photometry_interpolator import CMDPhotometryInterpolator
from fit_ngc6819_masses import fit_milliman
from io_utilities import read_milliman_data, data_dir

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

def plot_miliman_cmd(interpolator,
                     photometry,
                     distance_modulus,
                     single_lined_binaries,
                     double_lined_binaries):
    """Create a color-magnitude diagram of NGC 6819."""

    single_member_phot = photometry[photometry['Mm'] == b'SM ']

    interp_masses = interpolator.data[0]['Mini']
    predicted_photometry = interpolator(interp_masses) + distance_modulus

    for is_double_lined, binary_data in [(False, single_lined_binaries),
                                         (True, double_lined_binaries)]:
        for _, binary in binary_data.iterrows():
            binary_individual_photometry = interpolator(
                [binary['M1'], binary['M2']]
            ) + distance_modulus
            binary_combined_photometry = interpolator.get_binary_magnitudes(
                binary['M1'],
                binary['M2']
            ) + distance_modulus
            pyplot.plot(predicted_photometry[2] - predicted_photometry[4],
                        -predicted_photometry[2],
                        '-y',
                        linewidth=5,
                        zorder=10)
            pyplot.plot(single_member_phot['V-I'],
                        -single_member_phot['Vmag'],
                        'ok',
                        zorder=0)
            pyplot.plot(single_lined_binaries['V-I'].values,
                        -single_lined_binaries['Vmag'].values,
                        'or',
                        zorder=20,
                        markerfacecolor='none',
                        markeredgewidth=3)
            pyplot.plot(double_lined_binaries['V-I'].values,
                        -double_lined_binaries['Vmag'].values,
                        'og',
                        zorder=30,
                        markerfacecolor='none',
                        markeredgewidth=3)
            pyplot.plot(binary['V-I'],
                        -binary['Vmag'],
                        'x' + ('g' if is_double_lined else 'r'),
                        markersize=20,
                        markeredgewidth=5,
                        zorder=40)
            pyplot.plot(
                (
                    binary_individual_photometry[2]
                    -
                    binary_individual_photometry[4]
                ),
                -binary_individual_photometry[2],
                '+b',
                markersize=20,
                markeredgewidth=5,
                zorder=40
            )
            pyplot.plot(
                (
                    binary_combined_photometry[2]
                    -
                    binary_combined_photometry[4]
                ),
                -binary_combined_photometry[2],
                '+c',
                markersize=20,
                markeredgewidth=5,
                zorder=40
            )
            pyplot.title('WOCS: ' + repr(binary['WOCS']))
            pyplot.xlabel('V-I [mag]')
            pyplot.ylabel('V [mag]')
            pyplot.xlim(0, 2)
            pyplot.ylim(-17, -11)
            pyplot.show()

def plot_hole_cmd(interpolator, photometry, distance_modulus):
    """Create a color-magnitude diagram of NGC 6819."""

    print('Photometry: ' + repr(photometry))

    interp_masses = interpolator.data[0]['Mini']
    predicted_photometry = interpolator(interp_masses) + distance_modulus

    pyplot.plot(predicted_photometry[1] - predicted_photometry[2],
                -predicted_photometry[2],
                '-y',
                linewidth=5,
                zorder=10)

    pyplot.plot(photometry['B-V'],
                -photometry['Vmag'],
                'ok',
                zorder=0)
    pyplot.xlim(0, 2)
    pyplot.ylim(-20, -13)
    pyplot.show()



def main():
    """Avoid polluting global namespace."""

#    bostanci_interpolator = CMDPhotometryInterpolator(
#        os.path.join(
#            data_dir,
#            'CMD_2.5Gyr_isochrone_Av0.35_FeH0.09_UBVRIJHK.dat'
#        )
#    )
#    hole_photometry = read_cds_pipe_table(
#        os.path.join(
#            data_dir,
#            'Hole_et_al_2009_NGC6819_photometry.tsv'
#        )
#    )
#    (
#        milliman_photometry,
#        single_lined_data,
#        double_lined_data
#    ) = read_milliman_data()

#    bostanci_photometry = read_cds_pipe_table(
#        os.path.join(
#            data_dir,
#            'Bostanci_et_al_16_NGC6819_UBV_photometry.tsv'
#        )
#    )

    (
        milliman_photometry,
        single_lined_data,
        double_lined_data
    ) = read_milliman_data()

#    plot_hole_cmd(milliman_interpolator, hole_photometry, 11.85)
#    plot_bostanci_cmd(bostanci_interpolator, bostanci_photometry, 12.0)
    fit_milliman(single_lined_data,
                 double_lined_data)
#    bad_fits = [1010]

    print(80*'=' + '\nSingle line binaries\n' + 80 * '=')
    print(repr(single_lined_data))
    print(80*'=' + '\nDouble line binaries\n' + 80 * '=')
    print(repr(double_lined_data))

    plot_miliman_cmd(
        CMDPhotometryInterpolator(
            os.path.join(
                data_dir,
                'CMD_2.5Gyr_isochrone_Av0.45_FeH0.09_UBVRIJHK.dat'
            )
        ),
        milliman_photometry,
        11.85,
        single_lined_data,
        double_lined_data
    )

if __name__ == '__main__':
    main()
