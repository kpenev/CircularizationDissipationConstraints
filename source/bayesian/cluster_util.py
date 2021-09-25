"""Functions working with cluster data required for the circ. analysis."""

import logging

from matplotlib import pyplot
from scipy import stats
import numpy
from astropy import units

from bayesian.eccentricity_likelihood import EccentricityLikelihood
from bayesian.approximate_rv_likelihood import ApproximateRVLikelihood
from binary_utils import rv_semi_amplitude_scale
from sample_sb1_masses import SampleSB1Masses

_logger = logging.getLogger(__name__)

def get_rv_likelihood(observed_orbit,
                      eccentricity_envelope,
                      num_parallel_processes,
                      interpolation_accuracy,
                      show_mismatch_plot):
    """Return fully set-up RV semi-amplitude constraint for an NGC188 binary."""

    signal_to_noise = float(observed_orbit['K']) / float(observed_orbit['e_K'])
    #TODO: find better observed eccentricity distribution
    return ApproximateRVLikelihood(
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
        observed_eccentricity=stats.norm(
            loc=float(observed_orbit['e']),
            scale=float(observed_orbit['e_e'])
        ),
        envelope_eccentricity=eccentricity_envelope(
            float(observed_orbit['Per'])
        ),
        max_discarded_probabiity=1e-6,
        tolerance=interpolation_accuracy,
        num_parallel_processes=num_parallel_processes,
        integration_options=dict(epsabs=0,
                                 epsrel=1e-8,
                                 limit=200,
                                 maxp1=200),
        min_grid_points=(100, 100),
        grid_refine_algorithm='1d',
        debug_plots=(dict(interpolation_performance='') if show_mismatch_plot
                     else None)
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

def plot_mass_pdfs(sample_stellar_masses,
                   m1_pdf_axis,
                   m2_pdf_axis,
                   linestyle='-'):
    """Plot PDF(M1) marginalized over M2 and PDF(M2|M1) on given axis."""

    m1_pdf_axis.set_title('$PDF(M_1 | e_{env}, P_{orb})$')
    m2_pdf_axis.set_title('$PDF(M_2 | M_1, e_{env}, P_{orb})$')

    m1_pdf_axis.set_xlabel(r'$M_1\ [M_\odot]$')
    m2_pdf_axis.set_xlabel(r'$M_2\ [M_\odot]$')

    med_primary_mass = sample_stellar_masses.primary_mass_ppf(0.5) * units.M_sun
    secondary_mass_distro = (
        sample_stellar_masses.get_conditional_secondary_mass_distribution(
            med_primary_mass.to_value(units.M_sun)
        )
    )

    print('Plotting for M1 = ' + repr(med_primary_mass))
    plot_data = dict(
        m1=numpy.linspace(
            sample_stellar_masses.primary_mass_ppf(1e-5),
            sample_stellar_masses.primary_mass_ppf(1.0-1e-5),
            30
        ),
        m2=numpy.linspace(
            secondary_mass_distro.ppf(1e-5),
            secondary_mass_distro.sf(1e-5),
            30
        )
    )
    plot_data['m1_pdf'] = numpy.vectorize(
        sample_stellar_masses.primary_mass_pdf
    )(
        plot_data['m1']
    )
    print('Calculated M1 PDF values')
    plot_data['m2_pdf'] = numpy.vectorize(
        secondary_mass_distro.pdf
    )(
        plot_data['m2']
    )
    print('Calculated M2 PDF values')
    m1_pdf_axis.semilogy(plot_data['m1'],
                         plot_data['m1_pdf'],
                         linestyle=linestyle,
                         color='b',
                         label='PDF(M1)')
    m2_pdf_axis.semilogy(plot_data['m2'],
                         plot_data['m2_pdf'],
                         linestyle=linestyle,
                         color='r',
                         label='PDF(M2|M1)')
    m1_pdf_axis.legend()
    m2_pdf_axis.legend()


def plot_rv_likelihood(observed_orbit,
                       eccentricity_envelope,
                       photometric_constraint):
    """Display plots showing the RV based constraint."""

    rv_likelihood = get_rv_likelihood(
        observed_orbit=observed_orbit,
        eccentricity_envelope=eccentricity_envelope,
        num_parallel_processes=24,
        interpolation_accuracy=1e-8,
        show_mismatch_plot=False
    )

    plots = dict(
        rv_likelihood=pyplot.subplot(131),
        rv_likelihood_ratio=pyplot.subplot(132),
        m2_pdf=pyplot.subplot(133)
    )
    plots['m1_pdf'] = pyplot.twiny(plots['m2_pdf'])

    for sub_plot in plots.values():
        sub_plot.set_xlabel(r'$M_2\ [M_\odot]$')
    plots['rv_likelihood'].set_title(
        r'$\lambda\left[e, K(M_1, M_2, P_{orb})\right]$'
    )
    plots['rv_likelihood_ratio'].set_title(
        r'$\frac{\lambda\left[e, K(M_1, M_2, P_{orb})\right]}'
        r'{\lambda\left[e_{env}, K(M_1, M_2, P_{orb})\right]}$'
    )

    sample_stellar_masses = SampleSB1Masses(
        rv_likelihood=rv_likelihood,
        photometric_constraint=photometric_constraint,
        orbital_period=float(observed_orbit['Per']),
        envelope_eccentricity=eccentricity_envelope(
            float(observed_orbit['Per'])
        )
    )

    plot_mass_pdfs(sample_stellar_masses,
                   plots['m1_pdf'],
                   plots['m2_pdf'],
                   '-')
    plot_mass_pdfs(photometric_constraint,
                   plots['m1_pdf'],
                   plots['m2_pdf'],
                   '--')

    med_primary_mass = sample_stellar_masses.primary_mass_ppf(0.5) * units.M_sun

    print('Plotting for M1 = ' + repr(med_primary_mass))
    plot_m2 = numpy.linspace(
        0.2,#secondary_mass_photometric_prior.ppf(1e-5),
        1.0,#secondary_mass_photometric_prior.sf(1e-5),
        30000
    )
    print('M2 = ' + repr(plot_m2))
    plot_rvk_scale = rv_semi_amplitude_scale(
        med_primary_mass,
        plot_m2 * units.M_sun,
        float(observed_orbit['Per']) * units.day
    ).to_value(units.m / units.s)

    print('K0 = ' + repr(plot_rvk_scale))


    plot_rv_likelihood_denom = rv_likelihood(
        0.5,#eccentricity_envelope(float(observed_orbit['Per'])),
        plot_rvk_scale
    )[0]

    for eccentricity in numpy.linspace(0.1, 0, 20):

        plot_rv_likelihood_numer = rv_likelihood(eccentricity,
                                                 plot_rvk_scale)[0]

        _logger.info('Finished calculations for e = %s',
                     repr(eccentricity))

        plots['rv_likelihood'].semilogy(
            plot_m2,
            plot_rv_likelihood_numer,
            label='e = ' + str(eccentricity)
        )
        plots['rv_likelihood_ratio'].semilogy(
            plot_m2,
            plot_rv_likelihood_numer / plot_rv_likelihood_denom,
            label='e = ' + str(eccentricity)
        )
#        plots['right'].plot(plot_m2,
#                            plot_cdf)
#    plots['middle'].set_ylim((0.1, 10.0))
    plots['rv_likelihood'].legend()
    plots['rv_likelihood_ratio'].legend()

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
