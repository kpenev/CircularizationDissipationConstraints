#!/usr/bin/env python3

"""Utilities for reading cluster data."""

import os.path
import logging
from multiprocessing import set_start_method
from time import time

from matplotlib import pyplot
import pandas
import numpy
from scipy import stats
from astropy import units

from planetary_system_io import read_cds_pipe_table
from command_line_utilities import data_dir
from cmd_utils import\
    CMDPhotometryInterpolator,\
    CMDUSNOPhotometryInterpolator
from process_e_Q_grid import LinearEccentricityEnvelope
from bayesian.photometric_constraint import\
    PhotometricConstraint,\
    plot_joint_pdf,\
    plot_m1_cdf,\
    plot_m2_cdf
from bayesian.rv_semiamplitude_constraint import RVSemiAmplitudeConstraint
from bayesian.eccentricity_likelihood import EccentricityLikelihood

_logger = logging.getLogger(__name__)

cluster_age_distribution = stats.norm(7.0, 0.5)
cluster_feh_distribution = stats.norm(0.21, 0.03)

def get_photometry_distributions(photometry, min_stddev=0.0):
    """Return dictionary of all available photometry as normal distributions."""

    result = dict()
    for column, data in photometry.items():
        if column[1:] == 'mag' and numpy.isfinite(float(data)):
            result[column[0]] = stats.norm(
                loc=float(data),
                scale=max(float(photometry['e_' + column]), min_stddev)
            )
    return result

def _unprime_usno_column_name(colname):
    """Return the given column name without "'" if it is a photometry column."""

    if colname[0] in 'ugriz' and colname[2:] == 'mag':
        return colname[0] + 'mag'

    if (
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

def get_usno_photometry():
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

    result.dtype.names = [_unprime_usno_column_name(colname)
                          for colname in result.dtype.names]

    return result

def get_photometry():
    """Return a pandas DataFrame containing NGC188 literature photometry."""

    return pandas.merge(
        pandas.DataFrame(
            read_cds_pipe_table(
                os.path.join(
                    data_dir,
                    'Stetson_et_al_04_NGC188_UBVRI_photometry.tsv'
                )
            )
        ),
        pandas.DataFrame(get_usno_photometry()),
        on='PKM',
        how='outer'
    )

def get_binary_data(
        single_lined_orbits_fname=(
            os.path.join(
                data_dir,
                'Geller_et_al_2009_WIYN_single_lined_orbits.tsv'
            )
        ),
        double_lined_orbits_fname=(
            os.path.join(
                data_dir,
                'Geller_et_al_2009_WIYN_double_lined_orbits.tsv'
            )
        ),
        physical_parameters_fname=(
            os.path.join(
                data_dir,
                'Geller_et_al_2009_WIYN_physical_parameters.tsv'
            )
        )
):
    """Return NGC188 SB1 and SB2 systems as :class:`pandas.DataFrame`s."""

    physical_parameters = pandas.DataFrame(
        read_cds_pipe_table(physical_parameters_fname)
    )

    return (
        pandas.merge(
            physical_parameters,
            pandas.DataFrame(
                read_cds_pipe_table(single_lined_orbits_fname)
            ),
            on='PKM',
            how='right'
        ),
        pandas.merge(
            physical_parameters,
            pandas.DataFrame(
                read_cds_pipe_table(double_lined_orbits_fname)
            ),
            on='PKM',
            how='right'
        )
    )

def get_photometry_interpolators():
    """Return a dictionary of interpolators of NGC188 photometry."""

    return {
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

def get_observed_orbit(binary_pkm_id):
    """Return pandas.DataFrames containing the orbital parameters of an SB1."""

    single_lined_data, _ = get_binary_data()

    selected = (single_lined_data['PKM'] == binary_pkm_id)
    if not selected.any():
        raise RuntimeError(
            'Only SB1 radial velocity constraints are supported at this time, '
            'and binary with PKM=%d is not single lined!'
            %
            binary_pkm_id
        )
    selected = single_lined_data[selected]
    _logger.info('Selected binary data:\n%s',
                 repr(selected.T))
    return selected

def get_photometric_constraint(binary_pkm_id):
    """Return a fully set-up photometric constraint for an NGC188 binary."""

    photometry = get_photometry()

    selected_photometry = get_photometry_distributions(
        photometry[photometry['PKM'] == binary_pkm_id],
        0.02
    )

    _logger.debug(
        'Selected photometry: ('
        +
        ', '.join(['%s: %s +- %s'] * len(selected_photometry)),
        *sum(
            (
                (
                    mag_col,
                    repr(distribution.kwds['loc']),
                    repr(distribution.kwds['scale'])
                )
                for mag_col, distribution in selected_photometry.items()
            ),
            ()
        )
    )

    interpolators = get_photometry_interpolators()

    return PhotometricConstraint(
        [interpolators['UBVRIJHK'], interpolators['usno']],
        selected_photometry,
        'photometric_constraints.pkl',
        min_magnitude_difference=dict(V=2.5)
    )

def get_rvk_constraint(observed_orbit,
                       num_parallel_processes,
                       interpolation_accuracy,
                       show_mismatch_plot):
    """Return fully set-up RV semi-amplitude constraint for an NGC188 binary."""

    signal_to_noise = float(observed_orbit['K']) / float(observed_orbit['e_K'])
    return RVSemiAmplitudeConstraint(
        observed_rvk=(
            stats.norm(
                loc=numpy.sqrt(
                    float(observed_orbit['K'])**2
                    +
                    float(observed_orbit['e_K'])**2
                ) * 1000.0,
                scale=float(observed_orbit['e_K']) * 1000.0
            )
            if signal_to_noise > 50 else
            stats.rice(
                b=signal_to_noise,
                scale=float(observed_orbit['e_K']) * 1000.0
            )
        ),
        max_discarded_probabiity=1e-6,
        interpolation_accuracy=interpolation_accuracy,
        num_parallel_processes=num_parallel_processes,
        pickle_fname='rvk_constraints.pkl',
        epsabs=0,
        epsrel=1e-8,
        limit=200,
        maxp1=200,
        show_mismatch_plot=show_mismatch_plot
    )

def alternative_eccentricity_envelope(orbital_period):
    """Return the envelope at the given orbital period."""

    gamma = 0.8
    beta = 0.25
    circularization_period = 13.8

    return numpy.maximum(
        0.05,
        0.6 * (
            1.0
            -
            numpy.exp(beta * (circularization_period - orbital_period))
        )
    )**gamma

def get_final_eccentricity_likelihood(observed_orbit):
    """Return :class:`EccentricityLikelihood` instance per the given orbit."""

    eccentricity_envelope = LinearEccentricityEnvelope(min_period=3.0,
                                                       max_period=20.0,
                                                       max_eccentricity=0.6)

    return EccentricityLikelihood(
        observed_eccentricity=stats.rice(
            b=float(observed_orbit['e']) / float(observed_orbit['e_e']),
            scale=float(observed_orbit['e_e'])
        ),
        envelope_eccentricity=eccentricity_envelope(
            float(observed_orbit['Per'])
        )
    )

def _test_photometric_constraint(binary_pkm_id):
    """Display plots showing the photometry based constraint."""

    single_lined_binaries, _ = get_binary_data()

    selected_binary = single_lined_binaries[
        single_lined_binaries['PKM'] == binary_pkm_id
    ]

    constraint = get_photometric_constraint(binary_pkm_id)

    plot_joint_pdf(
        constraint,
        (float(selected_binary['M1']), float(selected_binary['M2']))
    )
    plot_m1_cdf(constraint)
    plot_m2_cdf(constraint)

#TODO: split up
#pylint: disable=too-many-locals
def _test_rvk_constraint(binary_pkm_id):
    """Display plots showing the RV based constraint."""

    observed_orbit = get_observed_orbit(binary_pkm_id)
    rvk_constraint = get_rvk_constraint(
        observed_orbit=observed_orbit,
        num_parallel_processes=4,
        interpolation_accuracy=(1e-8, 1e-4),
        show_mismatch_plot=True
    )
    photometric_constraint = get_photometric_constraint(binary_pkm_id)

    single_lined_binaries, _ = get_binary_data()

    selected_binary = single_lined_binaries[
        single_lined_binaries['PKM'] == binary_pkm_id
    ]


    plots = dict(
        left=pyplot.subplot(131),
        middle=pyplot.subplot(132),
        right=pyplot.subplot(133)
    )
    for sub_plot in plots.values():
        sub_plot.set_xlabel(r'$M_2\ [M_\odot]$')
    plots['left'].set_title('$PDF(M_2 | M_1, e, P_{orb})$')
    plots['middle'].set_title(
        "$f'[K(M_1, M_2, P_{orb}, e)] / f'[K(M_1, M_2, P_{orb}, e=0.5)]$"
    )
    plots['right'].set_title('$CDF(M_2 | M_1, e, P_{orb})$')

    primary_mass = photometric_constraint.primary_mass_ppf(0.5) * units.M_sun
    secondary_mass_photometric_prior = (
        photometric_constraint.get_conditional_secondary_mass_distribution(
            primary_mass.to_value(units.M_sun)
        )
    )

    for eccentricity in numpy.linspace(0.5, 0, 6):

        rvk_constraint.prepare_secondary_sampling(
            primary_mass=primary_mass,
            orbital_period=float(selected_binary['Per']) * units.day,
            eccentricity=eccentricity,
            secondary_mass_prior=secondary_mass_photometric_prior
        )

        plot_m2 = numpy.linspace(0.15, 0.9, 300)

        start_time = time()
        plot_pdf = numpy.fromiter(
            (
                rvk_constraint.secondary_mass_pdf(m2)
                for m2 in  plot_m2 * units.M_sun
            ),
            dtype=float
        )
        _logger.debug('Calculating %d PDF values took %g minutes.',
                      plot_pdf.size,
                      (time() - start_time) / 60.0)
        start_time = time()
        plot_cdf = numpy.fromiter(
            (
                rvk_constraint.secondary_mass_cdf(m2)
                for m2 in plot_m2 * units.M_sun
            ),
            dtype=float
        )
        _logger.debug('Calculating %d CDF values took %g minutes.',
                      plot_cdf.size,
                      (time() - start_time) / 60.0)
        start_time = time()
        plot_fk_ratio = numpy.fromiter(
            (
                rvk_constraint.rv_semi_amplitude_pdf(
                    rvk_constraint.rv_semi_amplitude(
                        primary_mass=primary_mass,
                        secondary_mass=secondary_mass,
                        eccentricity=eccentricity
                    )
                )
                /
                rvk_constraint.rv_semi_amplitude_pdf(
                    rvk_constraint.rv_semi_amplitude(
                        primary_mass=primary_mass,
                        secondary_mass=secondary_mass,
                        eccentricity=0.5
                    )
                )
                for secondary_mass in plot_m2 * units.M_sun
            ),
            dtype=float
        )
        _logger.debug('Calculating %d f\'(K) ratios took %g minutes.',
                      plot_fk_ratio.size,
                      (time() - start_time) / 60.0)

        _logger.info('Finished calculations for e = %s',
                     repr(eccentricity))

        plots['left'].plot(plot_m2,
                           plot_pdf,
                           label='e = ' + str(eccentricity))
        plots['middle'].semilogy(plot_m2,
                                 plot_fk_ratio,
                                 label='e = ' + str(eccentricity))
        plots['right'].plot(plot_m2,
                            plot_cdf)
    plots['middle'].set_ylim((0.1, 10.0))
    pyplot.show()
#pylint: enable=too-many-locals

def plot_eccentricity_vs_period():
    """Show a period-eccentricity plot for NGC188."""

    def add_shifted_periods(binary_class):
        """Calculate the shifted orbital period for determining envelope."""

        m1_mult = 3.97405
        m2_mult = 2.5685
        m1_pwrlaw = 1.72462
        m2_pwrlaw = 1.55429

        binary_class.insert(
            len(binary_class.columns),
            'ShiftedPer',
            (
                binary_class['Per']
                +
                m1_mult * (1.0 - binary_class['M1']**m1_pwrlaw)
                +
                m2_mult * (1.0 - binary_class['M2']**m2_pwrlaw)
            )
        )

    binaries = get_binary_data()
    binaries = [
        binary_class[
            numpy.logical_and(
                binary_class['Prv'] > 50,
                binary_class['Ppm'] > 50
            )
        ]
        for binary_class in binaries
    ]

    for binary_class in binaries:
        add_shifted_periods(binary_class)

    print('Single lined binaries:\n' + repr(binaries[0]))
    print('Double lined binaries:\n' + repr(binaries[1]))

    pyplot.gca().set_xscale('log')

    for label, binary_class in zip(['SB1', 'SB2'], binaries):
        color = pyplot.errorbar(binary_class['ShiftedPer'],
                                binary_class['e'],
                                binary_class['e_e'],
                                fmt='o',
                                label=label)[0].get_color()
        pyplot.errorbar(binary_class['Per'],
                        binary_class['e'],
                        binary_class['e_e'],
                        fmt='o',
                        markeredgecolor=color,
                        markerfacecolor='none')
    envelope_x = 2.0**numpy.linspace(1, 6, 1000)
    pyplot.plot(envelope_x, alternative_eccentricity_envelope(envelope_x), '-k')
    pyplot.axhline(0.5)
    pyplot.ylim(0, 1.0)
    pyplot.xlim(2, 64)
    pyplot.xlabel('Orbital Period [d]')
    pyplot.ylabel('Eccentricity')
    pyplot.legend()
    pyplot.show()

if __name__ == '__main__':
    set_start_method('forkserver')
#    logging.basicConfig(level=logging.DEBUG)
    plot_eccentricity_vs_period()
    #_test_rvk_constraint(3732)
