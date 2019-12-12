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
                [binary['m1_fit'], binary['m2_fit']]
            ) + distance_modulus
            binary_combined_photometry = interpolator.get_binary_magnitudes(
                binary['m1_fit'],
                binary['m2_fit']
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
        m1_series = pandas.Series(index=binary_data.index)
        m2_series = pandas.Series(index=binary_data.index)

        for binary_ind, binary in binary_data.iterrows():
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

            (m1_series[binary_ind], m2_series[binary_ind]) = fit_binary_masses(
                photometry_interp=interpolator,
                photometry=binary,
                distance_modulus=distance_modulus,
                min_mag_difference=(None if is_double_lined
                                    else {'V': 2.0}),
                magnitude_template='%(filchar)cmag',
                magnitude_error_template=('e_%(filchar)cmag'),
                color_template='%(filchar1)c-%(filchar2)c',
                color_error_template='e_%(filchar1)c-%(filchar2)c',
                **rv_params
            ).x
            print('WOCS %d: m1 = %f, m2 = %f' % (binary['WOCS'],
                                                 m1_series[binary_ind],
                                                 m2_series[binary_ind]))
        binary_data['m1_fit'] = m1_series
        binary_data['m2_fit'] = m2_series

def main():
    """Avoid polluting global namespace."""

    bostanci_interpolator = CMDPhotometryInterpolator(
        '../data/CMD_2.5Gyr_isochrone_Av0.35_FeH0.09_UBVRIJHK.dat'
    )
    milliman_interpolator = CMDPhotometryInterpolator(
        '../data/CMD_2.5Gyr_isochrone_Av0.45_FeH0.09_UBVRIJHK.dat'
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
#    plot_bostanci_cmd(bostanci_interpolator, bostanci_photometry, 12.0)
    fit_milliman(single_lined_data, double_lined_data, milliman_interpolator)
    print(80*'=' + '\nSingle line binaries\n' + 80 * '=')
    print(repr(single_lined_data))
    print(80*'=' + '\nDouble line binaries\n' + 80 * '=')
    print(repr(double_lined_data))
    plot_miliman_cmd(milliman_interpolator,
                     milliman_photometry,
                     11.85,
                     single_lined_data,
                     double_lined_data)
    bad_fits = [1010]

if __name__ == '__main__':
    main()
