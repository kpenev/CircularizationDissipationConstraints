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

def fit_milliman(single_lined_data,
                 double_lined_data,
                 interpolator,
                 distance_modulus=11.85):
    """
    Fit for the masses of the NGC6819 binaries using Milliman's photometry.

    Args:
        single_lined_data:    The data from Milliman et. al. 2014 for single
            lined binaries. Should include both RV info and photemtry.

        double_lined_data:    The data from Milliman et. al. 2014 for single
            lined binaries. Should include both RV info and photemtry.

        interpolator:     See ``photometry_interp`` argument to
            :func:`fit_binary_masses`.

        distance_modulus:     See same name argument to
            :func:`fit_binary_masses`.

    Returns:
        dict:
            Keys are the WOCS IDs of the binaries and the entries are 2-tuples
            containing the maximum likelihood primary and secondary mass for
            each binary.
    """

    for is_double_lined, binary_data in [(False, single_lined_data),
                                         (True, double_lined_data)]:
        for _, binary in binary_data.iterrows():
            if is_double_lined:
                rv_params = dict(
                    observed_mass_ratio=binary['q'],
                    mass_ratio_error=binary['e_q'],
                    observed_projected_primary_mass=binary['msini1'],
                    projected_primary_mass_error=binary['e_msini1']
                )
            else:
                rv_params = dict(
                    observed_mass_function=binary['f(m)'],
                    observed_mass_function_err=binary['e_f(m)']
                )

            result = fit_binary_masses(
                photometry_interp=interpolator,
                photometry=binary,
                distance_modulus=distance_modulus,
                min_mag_difference=(None if is_double_lined
                                    else {'V': 2.5}),
                magnitude_template='%(filchar)cmag',
                magnitude_error_template=('e_%(filchar)cmag'),
                color_template='%(filchar1)c-%(filchar2)c',
                color_error_template='e_%(filchar1)c-%(filchar2)c',
                **rv_params
            )

def main():
    """Avoid polluting global namespace."""

    bostanci_interpolator = CMDPhotometryInterpolator(
        '../data/CMD_2.5Gyr_isochrone_Av0.35_FeH0.09_UBVRIJHK.dat'
    )
    milliman_interpolator = CMDPhotometryInterpolator(
        '../data/CMD_2.5Gyr_isochrone_Av0.5_FeH0.09_UBVRIJHK.dat'
    )
    milliman_photometry = read_cds_pipe_table(
        '../data/Milliman_et_al_14_NGC6819_RV_summary_with_VIphotometry.tsv'
    )
    hole_photometry = read_cds_pipe_table(
        '../data/Hole_et_al_2009_NGC6819_photometry.tsv'
    )

    single_lined_orbits = pandas.DataFrame(
        read_cds_pipe_table(
            '../data/Milliman_et_al_14_NGC6819_RV_single_lined_orbits.tsv'
        )
    )
    double_lined_orbits = pandas.DataFrame(
        read_cds_pipe_table(
            '../data/Milliman_et_al_14_NGC6819_RV_double_lined_orbits.tsv'
        )
    )

    single_lined_data = pandas.merge(single_lined_orbits,
                                     pandas.DataFrame(milliman_photometry),
                                     on='WOCS')
    double_lined_data = pandas.merge(double_lined_orbits,
                                     pandas.DataFrame(milliman_photometry),
                                     on='WOCS')

    single_lined_data['e_Vmag'] = pandas.Series(0.02, single_lined_data.index)
    single_lined_data['e_V-I'] = pandas.Series(0.03, single_lined_data.index)

    double_lined_data['e_Vmag'] = pandas.Series(0.02, double_lined_data.index)
    double_lined_data['e_V-I'] = pandas.Series(0.03, double_lined_data.index)

    bostanci_photometry = read_cds_pipe_table(
        '../data/Bostanci_et_al_16_NGC6819_UBV_photometry.tsv'
    )
#    plot_hole_cmd(milliman_interpolator, hole_photometry, 11.85)
#    plot_miliman_cmd(milliman_interpolator, milliman_photometry, 11.85)
#    plot_bostanci_cmd(bostanci_interpolator, bostanci_photometry, 12.0)
    fit_milliman(single_lined_data, double_lined_data, milliman_interpolator)

if __name__ == '__main__':
    main()
