#!/usr/bin/env python3

"""A test of binary stellar mass fitting using hyadespraesepe from literature."""

import os.path

from matplotlib import pyplot
import numpy
import pandas

from planetary_system_io import read_cds_pipe_table,read_pipe_table_to_pandas
from bayesian import hyadespraesepe_util
from mass_fitting import fit_binary_masses
from command_line_utilities import data_dir

#TODO: break up?
#pylint: disable=too-many-locals
def fit_all_binaries(photometry_interpolators,
                     hyadespraesepe_photometry,
                     hyadespraesepe_single_lined_orbits,
                     hyadespraesepe_double_lined_orbits,
                     hyadespraesepe_params,
                     *,
                     observed_phot_template='%(filter)s'):
    """Fit and report all binaries in hyadespraesepe along with literature masses."""

    min_mag_difference_filchar = 'V'
    fit_results = []
    for is_double_lined, orbital_parameters in [
            (False, hyadespraesepe_single_lined_orbits),
            (True, hyadespraesepe_double_lined_orbits)
    ]:
        print(80 * '=')
        print(
            ('Double' if is_double_lined else 'Single')
            +
            ' lined binaries'
        )
        print(80 * '=')
        for binary in orbital_parameters:
            correct_interpolator={photometry_interpolators[0]}
            print(binary)
            #print(binary.dtype.names)
            #print(binary[0])
            #print(binary['ID_true'])
            #print(hyadespraesepe_photometry)
            #print(hyadespraesepe_photometry[0])
            #print(hyadespraesepe_photometry[:][:])
            if binary['ID']>=3001:
                correct_interpolator={photometry_interpolators[1]}
            correct_place=0
            for i in numpy.arange(binary['ID']):
                #print(hyadespraesepe_photometry[i][0])
                if hyadespraesepe_photometry[i][0] == binary['ID']:
                    correct_place=i
                    break
            #print(hyadespraesepe_photometry[i])
            photometry = hyadespraesepe_photometry[correct_place]
                #hyadespraesepe_photometry[0][0] == binary['ID']
            #]
            if photometry.size == 0:
                print('No photometry for binary ' + repr(binary['ID']))
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
            #print(binary)
            #print(photometry)
            result = fit_binary_masses(
                photometry_interpolators=correct_interpolator,
                photometry=hyadespraesepe_util.get_photometry_distributions(photometry),
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

            answer = hyadespraesepe_params[
                hyadespraesepe_params['ID'] == binary['ID']
            ]

            if answer.size:
                print(
                    (
                        '%c Binary %d best fit masses: m1=%s (%s), m2=%s (%s), '
                        'dV=%s '
                    ) % (
                        ('v' if result.success else '*'),
                        binary['ID'],
                        repr(primary_m),
                        answer['l_M1'][0] + repr(answer['M1'][0]),
                        repr(secondary_m),
                        answer['l_M2'][0] + repr(answer['M2'][0]),
                        repr(mag_diff)
                    )
                    +
                    mass_comparison
                )
            fit_results.append((binary['ID'], primary_m, secondary_m))

    return fit_results
#pylint: enable=too-many-locals

def plot_binary_fit(interpolator,
                    hyadespraesepe_photometry,
                    hyadespraesepe_params,
                    fit_results,
                    *,
                    observed_phot_template='%(filter)smag'):
    """Plot the fitting of binaries listed in the bad_binaries_fname."""

    #TODO: Simplify
    #pylint: disable=too-many-locals
    def plot_fitting(binary_id, fit_m1, fit_m2):
        """Create a plot showing the fitting for a single binary."""

        try:
            cluster_members = hyadespraesepe_photometry[hyadespraesepe_photometry['Memb'] > 0.5]
        except ValueError:
            cluster_members = hyadespraesepe_photometry[hyadespraesepe_photometry['Mm'] > 0.5]

        correct_interpolator=interpolator['hyades']
        if(binary_id>=3001):
            correct_interpolator=interpolator['praesepe']

        get_mags = 'BV' if 'V' in correct_interpolator.available_filters else 'gr'
        interp_indices = numpy.array([
            correct_interpolator.available_filters.index(filchar)
            for filchar in get_mags
        ])
        print(get_mags)
        #observed_photometry = numpy.array([
        #    cluster_members[observed_phot_template % dict(filter=filter_name)]
        #    for filter_name in get_mags
        #])
        BminusV=cluster_members['B-V']
        Vee=cluster_members['Vmag']
        Bee=BminusV+Vee
        observed_photometry = numpy.array([Bee, Vee])
        #
        interp_masses = correct_interpolator.data[0]['Mini']
        predicted_photometry = correct_interpolator(interp_masses)[interp_indices]

        literature_params = hyadespraesepe_params[hyadespraesepe_params['ID'] == binary_id]
        #print(hyadespraesepe_photometry.dtype.names)
        binary_photometry = hyadespraesepe_photometry[
            hyadespraesepe_photometry['ID'] == binary_id
        ]
        #binary_photometry = [
        #    binary_photometry[observed_phot_template % dict(filter=filter_name)]
        #    for filter_name in get_mags
        #]
        BminusV=binary_photometry['B-V']
        Vee=binary_photometry['Vmag']
        Bee=BminusV+Vee
        binary_photometry = numpy.array([Bee, Vee])
        #

        fit_photometry = correct_interpolator.get_binary_magnitudes(
            fit_m1,
            fit_m2
        )[
            interp_indices
        ]
        fit_individual_photometry = (
            correct_interpolator(numpy.array([fit_m1, fit_m2]))[interp_indices]
        )

        print('Fit individual star photometry: '
              +
              repr(fit_individual_photometry))

        if literature_params.size:
            literature_binary_photometry = (
                correct_interpolator.get_binary_magnitudes(literature_params['M1'],
                                                   literature_params['M2'])
            )[
                interp_indices
            ]
            literature_individual_photometry = (
                correct_interpolator(numpy.array([float(literature_params['M1']),
                                          float(literature_params['M2'])]))
            )[
                interp_indices
            ]
        else:
            literature_individual_photometry = numpy.full(
                fit_individual_photometry.shape,
                numpy.nan
            )
            literature_binary_photometry = numpy.full(
                fit_photometry.shape,
                numpy.nan
            )

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
        pyplot.savefig('fit_result_'+ repr(binary_id) + '.png')
        pyplot.clf()
    #pylint: enable=too-many-locals

    for binary_fit in fit_results:
        plot_fitting(*binary_fit)

def main():
    """Avoid polluting global scope."""

    interpolator = {
        'hyades': hyadespraesepe_util.get_photometry_interpolator(1001),
        'praesepe': hyadespraesepe_util.get_photometry_interpolator(3001)
    }

    hyadespraesepe_photometry = read_cds_pipe_table(
        os.path.join(
            data_dir,
            'HyadesPraesepe_photometry.tsv'
        )
    )

    hyadespraesepe_single_lined_binaries = read_cds_pipe_table(
        os.path.join(
            data_dir,
            'HyadesPraesepe_single.tsv'
        )
    )
    hyadespraesepe_double_lined_binaries = read_cds_pipe_table(
        os.path.join(
            data_dir,
            'HyadesPraesepe_double.tsv'
        )
    )
    hyadespraesepe_params = read_cds_pipe_table(
        os.path.join(
            data_dir,
            'HyadesPraesepe_parameter.tsv' # This is supposed to be empty
        )
    )

    #hyadespraesepe_photometry = pandas.merge(
    #    pandas.DataFrame(hyadespraesepe_photometry_a),
    #    pandas.DataFrame(hyadespraesepe_params),
    #    on='ID',
    #    how='outer'
    #)

    fit_results = fit_all_binaries(
        [interpolator['hyades'], interpolator['praesepe']],
        hyadespraesepe_photometry,
        hyadespraesepe_single_lined_binaries,
        hyadespraesepe_double_lined_binaries,
        hyadespraesepe_params
    )

    plot_binary_fit(interpolator,
                    hyadespraesepe_photometry,
                    hyadespraesepe_params,
                    fit_results,
                    observed_phot_template='%(filter)smag')
    print(fit_results)

if __name__ == '__main__':
    main()
