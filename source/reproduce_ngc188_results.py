#!/usr/bin/env python3

"""A test of binary stellar mass fitting using NGC188 from literature."""

import re
import os.path

from matplotlib import pyplot
import numpy
import pandas
from scipy import stats

from planetary_system_io import read_cds_pipe_table
from cmd_utils import CMDPhotometryInterpolator, CMDUSNOPhotometryInterpolator
from mass_fitting import fit_binary_masses
from command_line_utilities import data_dir

def get_photometry_distributions(photometry, min_stddev):
    """Return dictionary of all available photometry as normal distributions."""

    result = dict()
    for column, data in photometry.items():
        if column[1:] == 'mag' and numpy.isfinite(float(data)):
            result[column[0]] = stats.norm(
                loc=float(data),
                scale=max(float(photometry['e_' + column]), min_stddev)
            )
    return result

#TODO: break up?
#pylint: disable=too-many-locals
def fit_all_binaries(photometry_interpolators,
                     ngc188_photometry,
                     ngc188_single_lined_orbits,
                     ngc188_double_lined_orbits,
                     ngc188_params,
                     *,
                     observed_phot_template='%(filter)s'):
    """Fit and report all binaries in NGC188 along with literature masses."""

    min_mag_difference_filchar = 'V'
    fit_results = []
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

            answer = ngc188_params[
                ngc188_params['PKM'] == binary['PKM']
            ]
            if not answer.size:
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
                photometry_interpolators=photometry_interpolators,
                photometry=get_photometry_distributions(photometry),
                min_mag_difference=(None if is_double_lined
                                    else {min_mag_difference_filchar: 2.5}),
                magnitude_template=observed_phot_template,
                **rv_params
            )
            primary_m, secondary_m = result.x

            for interpolator in photometry_interpolators:
                if min_mag_difference_filchar in interpolator.available_filters:
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
                    break

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
            fit_results.append((binary['PKM'], primary_m, secondary_m))

    return fit_results
#pylint: enable=too-many-locals

def plot_binary_fit(interpolator,
                    ngc188_photometry,
                    ngc188_params,
                    fit_results,
                    *,
                    observed_phot_template='%(filter)smag'):
    """Plot the fitting of binaries listed in the bad_binaries_fname."""

    #TODO: Simplify
    #pylint: disable=too-many-locals
    def plot_fitting(binary_id, fit_m1, fit_m2):
        """Create a plot showing the fitting for a single binary."""

        try:
            cluster_members = ngc188_photometry[ngc188_photometry['Memb'] > 0.5]
        except ValueError:
            cluster_members = ngc188_photometry[ngc188_photometry['Mm'] > 0.5]

        get_mags = 'BV' if 'V' in interpolator.available_filters else 'gr'
        interp_indices = numpy.array([
            interpolator.available_filters.index(filchar)
            for filchar in get_mags
        ])

        observed_photometry = numpy.array([
            cluster_members[observed_phot_template % dict(filter=filter_name)]
            for filter_name in get_mags
        ])
        interp_masses = interpolator.data[0]['Mini']
        predicted_photometry = interpolator(interp_masses)[interp_indices]

        literature_params = ngc188_params[ngc188_params['PKM'] == binary_id]

        binary_photometry = ngc188_photometry[
            ngc188_photometry['PKM'] == binary_id
        ]
        binary_photometry = [
            binary_photometry[observed_phot_template % dict(filter=filter_name)]
            for filter_name in get_mags
        ]

        fit_photometry = interpolator.get_binary_magnitudes(
            fit_m1,
            fit_m2
        )[
            interp_indices
        ]
        literature_binary_photometry = (
            interpolator.get_binary_magnitudes(literature_params['M1'],
                                               literature_params['M2'])
        )[
            interp_indices
        ]
        fit_individual_photometry = (
            interpolator(numpy.array([fit_m1, fit_m2]))[interp_indices]
        )
        literature_individual_photometry = (
            interpolator(numpy.array([float(literature_params['M1']),
                                      float(literature_params['M2'])]))
        )[
            interp_indices
        ]

        print('Fit individual star photometry: '
              +
              repr(fit_individual_photometry))

        print('Literature individual star photometry: '
              +
              repr(literature_individual_photometry))

        pyplot.title('Binary ' + repr(binary_id))
        pyplot.plot(
            observed_photometry[0] - observed_photometry[1],
            -observed_photometry[1],
            'ok',
            zorder=0
        )
        pyplot.plot(
            predicted_photometry[0] - predicted_photometry[1],
            -predicted_photometry[1],
            '-y',
            linewidth=3,
            zorder=10
        )
        pyplot.xlabel('%s - %s [mag]' % tuple(get_mags))
        pyplot.ylabel('-%c [mag]' % get_mags[1])

        pyplot.plot(
            [binary_photometry[0] - binary_photometry[1]],
            [-binary_photometry[1]],
            '+r',
            markersize=20,
            zorder=100,
            markeredgewidth=5,
            label='Observed binary photometry'
        )
        print(
            'Target: '
            +
            repr((
                [binary_photometry[0] - binary_photometry[1]],
                [-binary_photometry[1]]
            ))
        )
        pyplot.plot(
            [fit_photometry[0] - fit_photometry[1]],
            [-fit_photometry[1]],
            'xg',
            zorder=20,
            markersize=20,
            markeredgewidth=5,
            label='Best fit binary photometry'
        )
        pyplot.plot(
            [
                literature_binary_photometry[0]
                -
                literature_binary_photometry[1]
            ],
            [-literature_binary_photometry[1]],
            'xc',
            zorder=20,
            markersize=20,
            markeredgewidth=5,
            label='Photometry from literature masses'
        )

        for skip_label, (
                fit_single_phot,
                literature_single_phot
        ) in enumerate(
            zip(fit_individual_photometry.T,
                literature_individual_photometry.T)
        ):
            pyplot.plot(
                [fit_single_phot[0] - fit_single_phot[1]],
                [-fit_single_phot[1]],
                'sg',
                zorder=30,
                markersize=10,
                **(dict() if skip_label
                   else dict(label='Best fit component phot'))
            )
            pyplot.plot(
                [
                    literature_single_phot[0]
                    -
                    literature_single_phot[1]
                ],
                [-literature_single_phot[1]],
                'oc',
                zorder=40,
                markersize=10,
                **(dict() if skip_label
                   else dict(label='Component phot for literature mass'))
            )

        pyplot.ylim(-23, None)
        pyplot.xlim(0, None)
        pyplot.figlegend()
        pyplot.show()
    #pylint: enable=too-many-locals

    for binary_fit in fit_results:
        plot_fitting(*binary_fit)

def get_ngc188_usno_photometry():
    """Return a properly formatted field array with USNO filter photometry."""

    match_data = numpy.genfromtxt(
        os.path.join(
            data_dir,
            'Fornal_et_al_cross_Platais_et_al_NGC188_photometry.csv'
        ),
        names=True,
        dtype=None,
        delimiter=',',
        deletechars='',
        encoding=None
    )
    photometry = read_cds_pipe_table(
        os.path.join(
            data_dir,
            'Fornal_et_al_2006_NGC188_photometry.tsv'
        )
    )

    result_dtype = [(name, (int if name == 'PKM' else dtype[0]))
                    for name, dtype in photometry.dtype.fields.items()]

    result = numpy.empty(photometry.shape, dtype=result_dtype)
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

    result.dtype.names = [unprime_usno_column_name(colname)
                          for colname in result.dtype.names]

    return result

def unprime_usno_column_name(colname):
    """Return the given column name without "'" if it is a photometry column."""

    if colname[0] in 'ugriz' and colname[2:] == 'mag':
        return colname[0] + 'mag'
    elif (
        colname[0] in 'fe'
        and
        colname[1] == '_'
        and
        colname[2] in 'ugriz'
        and
        colname[4:] == 'mag'
    ):
        return colname[:2] + colname[2] + 'mag'

    return colname

def main():
    """Avoid polluting global scope."""

    interpolator = {
        'UBVRIJHK': CMDPhotometryInterpolator(
            os.path.join(
                data_dir,
                'CMD_7.0Gyr_FeH0dex_isochrone_Av0.2_UBVRIJHK.dat'
            ),
            11.23
        ),
        'sdss': CMDPhotometryInterpolator(
            os.path.join(
                data_dir,
                'CMD_7.0Gyr_FeH0dex_isochrone_Av0.2_ugriz.dat'
            ),
            11.23
        ),
        'usno': CMDUSNOPhotometryInterpolator(
            os.path.join(
                data_dir,
                'CMD_7.5Gyr_FeH0dex_isochrone_Av0.1.dat'
            ),
            11.3
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

    merged_photometry = pandas.merge(
        pandas.DataFrame(ngc188_photometry['UBVRIJHK']),
        pandas.DataFrame(ngc188_photometry['usno']),
        on='PKM',
        how='outer'
    )

    fit_results = fit_all_binaries(
        [interpolator['UBVRIJHK'], interpolator['usno']],
        merged_photometry,
        ngc188_single_lined_binaries,
        ngc188_double_lined_binaries,
        ngc188_params
    )

    for filter_set in ['usno', 'UBVRIJHK']:
        plot_binary_fit(interpolator[filter_set],
                        ngc188_photometry[filter_set],
                        ngc188_params,
                        fit_results,
                        observed_phot_template='%(filter)smag')

if __name__ == '__main__':
    main()
