#!/usr/bin/env python3

"""A test of binary stellar mass fitting using NGC188 from literature."""

import re

from matplotlib import pyplot
import scipy

from planetary_system_io import read_cds_pipe_table
from cmd_photometry_interpolator import CMDPhotometryInterpolator
from cmd_usno_photometry_interpolator import CMDUSNOPhotometryInterpolator
from mass_fitting import fit_binary_masses

def fit_all_binaries(interpolator,
                     ngc188_photometry,
                     ngc188_single_lined_orbits,
                     ngc188_double_lined_orbits,
                     ngc188_params,
                     distance_modulus=11.23):
    """Fit and report all binaries in NGC188 along with literature masses."""

    for is_double_lined, orbital_parameters in [
            (False, ngc188_single_lined_orbits),
            (True, ngc188_double_lined_orbits)
    ]:
        print(80 * '=')
        print(
            ('Double' if is_double_lined else 'Single')
            +
            ' lined binaries'
        )
        print(80 * '=')
        for binary in orbital_parameters:
            photometry = ngc188_photometry[
                ngc188_photometry['PKM'] == binary['PKM']
            ]

            answer = ngc188_params[
                ngc188_params['PKM'] == binary['PKM']
            ]
            if not answer:
                continue

            if is_double_lined:
                rv_params = dict(
                    observed_mass_ratio=binary['q'],
                    mass_ratio_error=binary['e_q'],
                    observed_projected_primary_mass=binary['msin3i1'],
                    projected_primary_mass_error=binary['e_msin3i1']
                )
            else:
                rv_params = dict(
                    observed_mass_function=binary['f(m)'],
                    observed_mass_function_err=binary['e_f(m)']
                )

            result = fit_binary_masses(photometry_interp=interpolator,
                                       photometry=photometry,
                                       distance_modulus=distance_modulus,
                                       min_mag_difference=(
                                           None if is_double_lined else
                                           dict(V=2.5)
                                       ),
                                       **rv_params)
            primary_m, secondary_m = result.x
            vmag_diff = (interpolator(secondary_m)[interpolator.filchars.index('V')]
                         -
                         interpolator(primary_m)[interpolator.filchars.index('V')])

            print(
                '%c Binary %d best fit masses: m1=%s (%s), m2=%s (%s), dV=%s' % (
                    ('v' if result.success else '*'),
                    binary['PKM'],
                    repr(primary_m),
                    answer['l_M1'][0].decode() + repr(answer['M1'][0]),
                    repr(secondary_m),
                    answer['l_M2'][0].decode() + repr(answer['M2'][0]),
                    vmag_diff
                )
            )

def plot_bad_binaries(interpolator,
                      ngc188_photometry,
                      ngc188_params,
                      *,
                      bad_binaries_fname='bad_binaries.txt',
                      distance_modulus=11.23):
    """Plot the fitting of binaries listed in the bad_binaries_fname."""

    def plot_fitting(binary_id, fit_m1, fit_m2):
        """Create a plot showing the fitting for a single binary."""

        cluster_members = ngc188_photometry[ngc188_photometry['Memb'] > 0.5]
        observed_ubvri = scipy.array((cluster_members["Umag"],
                                      cluster_members["Bmag"],
                                      cluster_members["Vmag"],
                                      cluster_members["Rmag"],
                                      cluster_members["Imag"]))
        interp_masses = interpolator.data[0]['Mini']
        predicted_ubvrijhk = interpolator(interp_masses)

        literature_params = ngc188_params[ngc188_params['PKM'] == binary_id]

        binary_photometry = ngc188_photometry[ngc188_photometry['PKM']
                                              ==
                                              binary_id]
        binary_photometry = [binary_photometry[filchar + 'mag']
                             for filchar in 'UBVRI']

        fit_photometry = (interpolator.get_binary_magnitudes(fit_m1, fit_m2)
                          +
                          distance_modulus)
        literature_binary_photometry = (
            interpolator.get_binary_magnitudes(literature_params['M1'],
                                               literature_params['M2'])
            +
            distance_modulus
        )
        fit_individual_photometry = (
            interpolator(scipy.array([fit_m1, fit_m2]))
            +
            distance_modulus
        )
        literature_individual_photometry = (
            interpolator(scipy.array([float(literature_params['M1']),
                                      float(literature_params['M2'])]))
            +
            distance_modulus
        )

        print('Fit individual star photometry: '
              +
              repr(fit_individual_photometry))

        print('Literature individual star photometry: '
              +
              repr(literature_individual_photometry))

        pyplot.title('Binary ' + repr(binary_id))
        for left in [1]:#range(5):
            for right in [2]:#range(left + 1, 5):
                pyplot.plot(observed_ubvri[left] - observed_ubvri[right],
                            -observed_ubvri[2],
                            'ok',
                            zorder=0)
                pyplot.plot(predicted_ubvrijhk[left] - predicted_ubvrijhk[right],
                            -predicted_ubvrijhk[2] - 11.23,
                            '-y',
                            linewidth=3,
                            zorder=10)
                pyplot.xlabel('%s - %s [mag]' % ('UBVRI'[left], 'UBVRI'[right]))
                pyplot.ylabel('-V [mag]')

                pyplot.plot(
                    [binary_photometry[left] - binary_photometry[right]],
                    [-binary_photometry[2]],
                    '+r',
                    markersize=20,
                    zorder=100
                )
                print('Target: ' + repr(([binary_photometry[left] - binary_photometry[right]],
                                         [-binary_photometry[2]])))
                pyplot.plot(
                    [fit_photometry[left] - fit_photometry[right]],
                    [-fit_photometry[2]],
                    'xg',
                    zorder=20,
                    markersize=20
                )
                pyplot.plot(
                    [
                        literature_binary_photometry[left]
                        -
                        literature_binary_photometry[right]
                    ],
                    [-literature_binary_photometry[2]],
                    'xc',
                    zorder=20,
                    markersize=20
                )
                for fit_single_phot, literature_single_phot in zip(
                        fit_individual_photometry.T,
                        literature_individual_photometry.T
                ):
                    pyplot.plot(
                        [fit_single_phot[left] - fit_single_phot[right]],
                        [-fit_single_phot[2]],
                        'sg',
                        zorder=30
                    )
                    pyplot.plot(
                        [literature_single_phot[left] - literature_single_phot[right]],
                        [-literature_single_phot[2]],
                        'oc',
                        zorder=40
                    )

                pyplot.ylim(-23, None)
                pyplot.show()

    bad_binary_line_rex = re.compile(
        r'(?P<success>[v*]) Binary (?P<binary_id>[0-9]+) best fit masses: '
        r'm1=(?P<m1>[0-9.e+-]+) .*, '
        r'm2=(?P<m2>[0-9.e+-]+) .*'
    )
    with open(bad_binaries_fname, 'r') as bad_binary_list:
        for line in bad_binary_list:
            parsed = bad_binary_line_rex.match(line)
            if parsed:
                plot_fitting(int(parsed['binary_id']),
                             float(parsed['m1']),
                             float(parsed['m2']))


def main():
    """Avoid polluting global scope."""

    interpolator = {
        'UBVRIJHK': CMDPhotometryInterpolator(
            '../data/CMD_7.0Gyr_FeH0dex_isochrone_Av0.2_UBVRIJHK.dat'
        ),
        'sdss': CMDPhotometryInterpolator(
            '../data/CMD_7.0Gyr_FeH0dex_isochrone_Av0.2_ugriz.dat'
        ),
        'usno': CMDUSNOPhotometryInterpolator(
            '../data/CMD_7.0Gyr_FeH0dex_isochrone_Av0.2_ugriz.dat'
        )
    }

    ngc188_photometry = {
        'UBVRIJHK': read_cds_pipe_table(
            '../data/Stetson_et_al_04_NGC188_UBVRI_photometry.tsv'
        ),
        'usno': read_cds_pipe_table(
            '../data/Fornal_et_al_2006_NGC188_photometry.tsv'
        )
    }
    read_cds_pipe_table(
        '../data/Stetson_et_al_04_NGC188_UBVRI_photometry.tsv'
    )
    ngc188_single_lined_binaries = read_cds_pipe_table(
        '../data/Geller_et_al_2009_WIYN_single_lined_orbits.tsv'
    )
    ngc188_double_lined_binaries = read_cds_pipe_table(
        '../data/Geller_et_al_2009_WIYN_double_lined_orbits.tsv'
    )
    ngc188_params = read_cds_pipe_table(
        '../data/Geller_et_al_2009_WIYN_physical_parameters.tsv'
    )

    for filter_set in ['usno', 'UBVRIJHK']:
        fit_all_binaries(interpolator[filter_set],
                         ngc188_photometry[filter_set],
                         ngc188_single_lined_binaries,
                         ngc188_double_lined_binaries,
                         ngc188_params)

        plot_bad_binaries(interpolator[filter_set],
                          ngc188_photometry[filter_set],
                          ngc188_params,
                          bad_binaries_fname='all_binary_fits.txt')

if __name__ == '__main__':
    main()
