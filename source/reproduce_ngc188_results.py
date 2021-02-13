#!/usr/bin/env python3

"""A test of binary stellar mass fitting using NGC188 from literature."""

import re
import os.path

from matplotlib import pyplot
import scipy

from planetary_system_io import read_cds_pipe_table
from cmd_utils import CMDPhotometryInterpolator, CMDUSNOPhotometryInterpolator
from mass_fitting import fit_binary_masses
from command_line_utilities import data_dir

def fit_all_binaries(interpolator,
                     ngc188_photometry,
                     ngc188_single_lined_orbits,
                     ngc188_double_lined_orbits,
                     ngc188_params,
                     distance_modulus=11.23,
                     observed_phot_template='%(filter)smag'):
    """Fit and report all binaries in NGC188 along with literature masses."""

    min_mag_difference_filchar = (
        'V' if 'V' in interpolator.available_filters else 'g'
    )
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
            if photometry.size == 0:
                print('No photometry for binary ' + repr(binary['PKM']))
                continue
            assert photometry.size == 1

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

            result = fit_binary_masses(
                photometry_interp=interpolator,
                photometry=photometry,
                distance_modulus=distance_modulus,
                min_mag_difference=(None if is_double_lined
                                    else {min_mag_difference_filchar: 2.5}),
                magnitude_template=observed_phot_template,
                magnitude_error_template=('e_' + observed_phot_template),
                **rv_params
            )
            primary_m, secondary_m = result.x
            mag_diff = (
                interpolator(secondary_m)[
                    interpolator.available_filters.index(
                        min_mag_difference_filchar
                    )
                ]
                -
                interpolator(primary_m)[
                    interpolator.available_filters.index(
                        min_mag_difference_filchar
                    )
                ]
            )

            if is_double_lined:
                mass_comparison = (
                    'q = %s (%s +- %s), min_m1 = %s +- %s'
                    %
                    (
                        repr(secondary_m / primary_m),
                        repr(binary['q']),
                        repr(binary['e_q']),
                        repr(binary['msin3i1']),
                        repr(binary['e_msin3i1']),
                    )
                )
            else:
                mass_comparison = (
                    'max(fm) = %s (%s +- %s)'
                    %
                    (
                        repr(secondary_m**3 / (primary_m + secondary_m)**2),
                        repr(binary['f(m)']),
                        repr(binary['e_f(m)'])
                    )
                )

            print(
                (
                    '%c Binary %d best fit masses: m1=%s (%s), m2=%s (%s), '
                    'dV=%s '
                ) % (
                    ('v' if result.success else '*'),
                    binary['PKM'],
                    repr(primary_m),
                    answer['l_M1'][0] + repr(answer['M1'][0]),
                    repr(secondary_m),
                    answer['l_M2'][0] + repr(answer['M2'][0]),
                    repr(mag_diff)
                )
                +
                mass_comparison
            )
            min_mag_difference = None

def plot_bad_binaries(interpolator,
                      ngc188_photometry,
                      ngc188_params,
                      *,
                      bad_binaries_fname='bad_binaries.txt',
                      distance_modulus=11.23,
                      observed_phot_template='%(filter)smag'):
    """Plot the fitting of binaries listed in the bad_binaries_fname."""

    def plot_fitting(binary_id, fit_m1, fit_m2):
        """Create a plot showing the fitting for a single binary."""

        try:
            cluster_members = ngc188_photometry[ngc188_photometry['Memb'] > 0.5]
        except ValueError:
            cluster_members = ngc188_photometry[ngc188_photometry['Mm'] > 0.5]
        observed_photometry = scipy.array([
            cluster_members[observed_phot_template % dict(filter=filter_name)]
            for filter_name in interpolator.available_filters[1:6]
        ])
        interp_masses = interpolator.data[0]['Mini']
        predicted_photometry = interpolator(interp_masses)[1:]

        literature_params = ngc188_params[ngc188_params['PKM'] == binary_id]

        binary_photometry = ngc188_photometry[
            ngc188_photometry['PKM'] == binary_id
        ]
        binary_photometry = [
            binary_photometry[observed_phot_template % dict(filter=filter_name)]
            for filter_name in interpolator.available_filters[1:6]
        ]

        fit_photometry = (interpolator.get_binary_magnitudes(fit_m1, fit_m2)
                          +
                          distance_modulus)
        literature_binary_photometry = (
            interpolator.get_binary_magnitudes(literature_params['M1'],
                                               literature_params['M2'])
            +
            distance_modulus
        )[1:]
        fit_individual_photometry = (
            interpolator(scipy.array([fit_m1, fit_m2]))[1:]
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
                pyplot.plot(
                    observed_photometry[left] - observed_photometry[right],
                    -observed_photometry[2],
                    'ok',
                    zorder=0
                )
                pyplot.plot(
                    predicted_photometry[left] - predicted_photometry[right],
                    -predicted_photometry[2] - 11.23,
                    '-y',
                    linewidth=3,
                    zorder=10
                )
                pyplot.xlabel(
                    '%s - %s [mag]'
                    %
                    (
                        interpolator.available_filters[left],
                        interpolator.available_filters[right]
                    )
                )
                pyplot.ylabel('-%c [mag]' % interpolator.available_filters[3])

                pyplot.plot(
                    [binary_photometry[left] - binary_photometry[right]],
                    [-binary_photometry[2]],
                    '+r',
                    markersize=20,
                    zorder=100,
                    markeredgewidth=5
                )
                print('Target: ' + repr(([binary_photometry[left] - binary_photometry[right]],
                                         [-binary_photometry[2]])))
                pyplot.plot(
                    [fit_photometry[left] - fit_photometry[right]],
                    [-fit_photometry[2]],
                    'xg',
                    zorder=20,
                    markersize=20,
                    markeredgewidth=5
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
                    markersize=20,
                    markeredgewidth=5
                )
                for fit_single_phot, literature_single_phot in zip(
                        fit_individual_photometry.T,
                        literature_individual_photometry.T
                ):
                    pyplot.plot(
                        [fit_single_phot[left] - fit_single_phot[right]],
                        [-fit_single_phot[2]],
                        'sg',
                        zorder=30,
                        markersize=10
                    )
                    pyplot.plot(
                        [literature_single_phot[left] - literature_single_phot[right]],
                        [-literature_single_phot[2]],
                        'oc',
                        zorder=40,
                        markersize=10
                    )

                pyplot.ylim(-23, None)
                pyplot.xlim(0, None)
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

def get_ngc188_usno_photometry():
    """Return a properly formatted field array with USNO filter photometry."""

    match_data = scipy.genfromtxt(
        os.path.join(
            data_dir,
            'Fornal_et_al_cross_Platais_et_al_NGC188_photometry.csv'
        ),
        names=True,
        dtype=None,
        delimiter=',',
        deletechars=''
    )
    photometry = read_cds_pipe_table(
        os.path.join(
            data_dir,
            'Fornal_et_al_2006_NGC188_photometry.tsv'
        )
    )

    result_dtype = [(name, (int if name == 'PKM' else dtype[0]))
                    for name, dtype in photometry.dtype.fields.items()]

    result = scipy.empty(photometry.shape, dtype=result_dtype)
    for result_index, phot_entry in enumerate(photometry):
        for colname in photometry.dtype.names:
            if colname == 'PKM':
                matched = (match_data['FTS'] == phot_entry['FTS'])
                if matched.any():
                    result[result_index][colname] = match_data['PKM_1'][
                        matched
                    ][0]
            else:
                result[result_index][colname] = phot_entry[colname]

    return result

def main():
    """Avoid polluting global scope."""

    interpolator = {
        'UBVRIJHK': CMDPhotometryInterpolator(
            os.path.join(
                data_dir,
                'CMD_7.0Gyr_FeH0dex_isochrone_Av0.2_UBVRIJHK.dat'
            )
        ),
        'sdss': CMDPhotometryInterpolator(
            os.path.join(
                data_dir,
                'CMD_7.0Gyr_FeH0dex_isochrone_Av0.2_ugriz.dat'
            )
        ),
        'usno': CMDUSNOPhotometryInterpolator(
            os.path.join(
                data_dir,
                'CMD_7.5Gyr_FeH0dex_isochrone_Av0.1.dat'
            )
        )
    }

    ngc188_photometry = {
        'UBVRIJHK': read_cds_pipe_table(
            os.path.join(
                data_dir,
                'Stetson_et_al_04_NGC188_UBVRI_photometry.tsv'
            )
        ),
        'usno': get_ngc188_usno_photometry()
    }
    read_cds_pipe_table(
        os.path.join(
            data_dir,
            'Stetson_et_al_04_NGC188_UBVRI_photometry.tsv'
        )
    )
    ngc188_single_lined_binaries = read_cds_pipe_table(
        os.path.join(
            data_dir,
            'Geller_et_al_2009_WIYN_single_lined_orbits.tsv'
        )
    )
    ngc188_double_lined_binaries = read_cds_pipe_table(
        os.path.join(
            data_dir,
            'Geller_et_al_2009_WIYN_double_lined_orbits.tsv'
        )
    )
    ngc188_params = read_cds_pipe_table(
        os.path.join(
            data_dir,
            'Geller_et_al_2009_WIYN_physical_parameters.tsv'
        )
    )

    distance_modulus = {'UBVRIJHK': 11.23,
                        'usno': 11.3}

    for filter_set in ['UBVRIJHK', 'usno']:
        observed_phot_template = (
            "%(filter)s'mag" if filter_set == 'usno'
            else "%(filter)smag"
        )
        fit_all_binaries(interpolator[filter_set],
                         ngc188_photometry[filter_set],
                         ngc188_single_lined_binaries,
                         ngc188_double_lined_binaries,
                         ngc188_params,
                         observed_phot_template=observed_phot_template,
                         distance_modulus=distance_modulus[filter_set])

        plot_bad_binaries(interpolator[filter_set],
                          ngc188_photometry[filter_set],
                          ngc188_params,
                          bad_binaries_fname='all_binary_fits.txt',
                          observed_phot_template=observed_phot_template,
                          distance_modulus=distance_modulus[filter_set])

if __name__ == '__main__':
    main()
