#!/usr/bin/env python3
"""NGC6811 specific functions required for circularization bayesian analysis."""

import os.path
import logging

from matplotlib import pyplot
import pandas
from scipy import stats
from astropy import units
import numpy

from planetary_system_io import read_cds_pipe_table
from command_line_utilities import data_dir
from cmd_utils import CMDPhotometryInterpolator
from process_e_Q_grid import LinearEccentricityEnvelope
from bayesian.photometric_constraint import PhotometricConstraint
from bayesian.cluster_util import\
    select_binary_data,\
    plot_rvk_constraint,\
    plot_eccentricity_vs_period,\
    get_final_eccentricity_likelihood,\
    get_rvk_constraint

#https://ui.adsabs.harvard.edu/abs/2011ApJ...729L..10B/abstract
cluster_age_distribution = stats.norm(2.4, 0.3)

#https://ui.adsabs.harvard.edu/abs/2001AJ....121..327B/abstract
cluster_feh_distribution = stats.norm(0.09, 0.03)

eccentricity_envelope = LinearEccentricityEnvelope(min_period=8.0,
                                                   max_period=14.0,
                                                   min_eccenticity=0.05,
                                                   max_eccentricity=0.6)

_logger = logging.getLogger(__name__)

def get_photometry_distributions(photometry):
    """Return dictionary of all available photometry as normal distributions."""

    return dict(
        V=stats.norm(
            loc=float(photometry['Vmag']),
            scale=0.036
        ),
        I=stats.norm(
            loc=float(photometry['Vmag'] - photometry['V-I']),
            scale=0.039
        )
    )

def get_photometry(
        photometry_fname=(
            os.path.join(
                data_dir,
                'Milliman_et_al_14_NGC6819_RV_summary_with_VIphotometry.tsv'
            )
        )
):
    """Read the photometry from Milliman et al (2014) as pandas frame."""

    return pandas.DataFrame(read_cds_pipe_table(photometry_fname))

def get_binary_data(
        single_lined_orbits_fname=(
            os.path.join(
                data_dir,
                'Milliman_et_al_14_NGC6819_RV_single_lined_orbits.tsv'
            )
        ),
        double_lined_orbits_fname=(
            os.path.join(
                data_dir,
                'Milliman_et_al_14_NGC6819_RV_double_lined_orbits.tsv'
            )
        )
):
    """Read the data from the Milliman et al (2014) tables as pandas frames."""

    photometry = get_photometry()

    single_lined_data = pandas.merge(
        pandas.DataFrame(
            read_cds_pipe_table(single_lined_orbits_fname)
        ),
        photometry,
        on='WOCS',
        how='left'
    )
    double_lined_data = pandas.merge(
        pandas.DataFrame(
            read_cds_pipe_table(double_lined_orbits_fname)
        ),
        photometry,
        on='WOCS'
    )
    return single_lined_data, double_lined_data

def get_observed_orbit(binary_wocs_id):
    """Return pandas.DataFrames containing the orbital parameters of an SB1."""

    return select_binary_data(*get_binary_data(), 'WOCS', binary_wocs_id)

def get_photometric_constraint(binary_wocs_id):
    """Return a fully set-up photometric constraint for an NGC6811 binary."""

    photometry = get_photometry()
    selected_photometry = get_photometry_distributions(
        photometry[photometry['WOCS'] == binary_wocs_id]
    )

    interpolator = CMDPhotometryInterpolator(
        os.path.join(
            data_dir,
            'CMD_2.5Gyr_isochrone_Av0.45_FeH0.09_UBVRIJHK.dat'
        ),
        distance_modulus=11.85
    )
    return PhotometricConstraint(
        [interpolator],
        selected_photometry,
        'photometric_constraints.pkl',
        min_magnitude_difference=dict(V=2.0)
    )

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    binary_id = 66004
    orbit = get_observed_orbit(binary_id)
    likelihood = get_final_eccentricity_likelihood(orbit, eccentricity_envelope)
    rvk_constraint = get_rvk_constraint(
        observed_orbit=orbit,
        num_parallel_processes=4,
        interpolation_accuracy=(1e-8, 1e-4),
        show_mismatch_plot=False
    )
    eccentricity = 0.00793845
    primary_mass = 1.211849036970599
    secondary_mass = 0.28038471334925863
    orbital_period = 2.277939947623581 * units.d
    rvk_pdf_init = lambda m2: rvk_constraint.rv_semi_amplitude_pdf(
        rvk_constraint.rv_semi_amplitude(
            eccentricity=0.5,
            primary_mass=primary_mass * units.M_sun,
            secondary_mass=secondary_mass * units.M_sun,
            orbital_period=orbital_period
        )
    )
    rvk_pdf_final = lambda m2: rvk_constraint.rv_semi_amplitude_pdf(
        rvk_constraint.rv_semi_amplitude(
            eccentricity=eccentricity,
            primary_mass=primary_mass * units.M_sun,
            secondary_mass=m2 * units.M_sun,
            orbital_period=orbital_period
        )
    )
    print(
        (
            'NGC6819 %d:\n'
            '\tf(e=%s) = %s\n'
            '\trvk_pdf(m1=%s, m2=%s, ef=%s) = %s\n'
            '\trvk_pdf(m1=%s, m2=%s, ef=%s) = %s'
        )
        %
        (
            binary_id,
            eccentricity, likelihood(eccentricity),
            primary_mass, secondary_mass, 0.5, rvk_pdf_init(secondary_mass),
            primary_mass, secondary_mass, eccentricity,
            rvk_pdf_final(secondary_mass)
        )
    )
    plot_m2 = numpy.linspace(secondary_mass, secondary_mass + 0.01, 100)
    plot_numerator = numpy.vectorize(rvk_pdf_final)(plot_m2)
    plot_denominator = numpy.vectorize(rvk_pdf_init)(plot_m2)
    pyplot.plot(plot_m2, plot_numerator / plot_denominator)
    pyplot.show()
    plot_eccentricity_vs_period(get_binary_data(), eccentricity_envelope)
#    plot_rvk_constraint(get_observed_orbit(57004),
#                        get_photometric_constraint(57004))
