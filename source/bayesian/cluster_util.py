"""Functions working with cluster data required for the circ. analysis."""

import logging
from time import time

from matplotlib import pyplot
from scipy import stats
import numpy
from astropy import units

from bayesian.eccentricity_likelihood import EccentricityLikelihood
from bayesian.rv_semiamplitude_constraint import RVSemiAmplitudeConstraint

_logger = logging.getLogger(__name__)

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

def select_binary_data(single_lined_data, _, id_column, binary_id):
    """Return pandas.DataFrames containing the orbital parameters of an SB1."""

    selected = (single_lined_data[id_column] == binary_id)
    if not selected.any():
        raise RuntimeError(
            'Only SB1 radial velocity constraints are supported at this time, '
            'and binary with %s=%d is not single lined!'
            %
            (id_column, binary_id)
        )
    selected = single_lined_data[selected]
    _logger.info('Selected binary data:\n%s', repr(selected.T))
    return selected

def plot_rvk_constraint(observed_orbit, photometric_constraint):
    """Display plots showing the RV based constraint."""

    rvk_constraint = get_rvk_constraint(
        observed_orbit=observed_orbit,
        num_parallel_processes=4,
        interpolation_accuracy=(1e-8, 1e-4),
        show_mismatch_plot=False
    )

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
            orbital_period=float(observed_orbit['Per']) * units.day,
            eccentricity=eccentricity,
            secondary_mass_prior=secondary_mass_photometric_prior
        )

        plot_m2 = numpy.linspace(
            rvk_constraint.secondary_mass_ppf(1e-5).to_value(units.M_sun),
            rvk_constraint.secondary_mass_ppf(1.0 - 1e-5).to_value(units.M_sun),
            300
        )

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

#TODO: figure out a better distribution for eccentricity
def get_final_eccentricity_likelihood(observed_orbit, eccentricity_envelope):
    """Return :class:`EccentricityLikelihood` instance per the given orbit."""

    return EccentricityLikelihood(
        observed_eccentricity=stats.norm(
            loc=float(observed_orbit['e']),
            scale=float(observed_orbit['e_e'])
        ),
        envelope_eccentricity=eccentricity_envelope(
            float(observed_orbit['Per'])
        )
    )

def plot_eccentricity_vs_period(binaries, eccentricity_envelope):
    """
    Show a period-eccentricity plot for a cluster.

    Args:
        binaries:    An iterable of two items: the single and double lined
            binaries respectively.
    """

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

    binaries = [
        binary_class[
            numpy.logical_and(
                numpy.logical_not(
                    binary_class.get('Prv', binary_class.get('PRV'))
                    <=
                    50
                ),
                numpy.logical_not(
                    binary_class.get(
                        'Ppm',
                        binary_class.get(
                            'PPM',
                            binary_class.get('PPM1')
                        )
                    )
                    <=
                    50
                )
            )
        ]
        for binary_class in binaries
    ]

    print('Single lined binaries:\n' + repr(binaries[0]))
    print('Double lined binaries:\n' + repr(binaries[1]))

    pyplot.gca().set_xscale('log')

    for label, binary_class in zip(['SB1', 'SB2'], binaries):
        color = pyplot.errorbar(binary_class['Per'],
                                binary_class['e'],
                                binary_class['e_e'],
                                fmt='ok',
                                label=label)[0].get_color()

#        if 'M1' in binary_class:
#            add_shifted_periods(binary_class)
#            pyplot.errorbar(binary_class['ShiftedPer'],
#                            binary_class['e'],
#                            binary_class['e_e'],
#                            fmt='o',
#                            markeredgecolor=color,
#                            markerfacecolor='none')
    envelope_x = 2.0**numpy.linspace(1, 6, 1000)
    pyplot.plot(envelope_x,
                eccentricity_envelope(envelope_x),
                '-k')
#    pyplot.axhline(0.5)
#    pyplot.axvspan(9, 16, color='red', alpha=0.3, zorder=-10)
#    pyplot.ylim(0, 0.7)
    pyplot.xlim(2, 100)
    pyplot.xlabel('Orbital Period [d]')
    pyplot.ylabel('Eccentricity')
#    pyplot.legend()
#    pyplot.show()

def plot_eccentricity_likelihood(observed_orbit, eccentricity_envelope):
    """Plot the likelihood of the final ecc. for a given system."""

    likelihood = get_final_eccentricity_likelihood(observed_orbit,
                                                   eccentricity_envelope)
    plot_e = numpy.linspace(0, 0.5, 1000)
    pyplot.plot(plot_e, numpy.vectorize(likelihood)(plot_e))
    pyplot.show()
