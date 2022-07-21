#!/usr/bin/env python3
"""NGC6811 specific functions required for circularization bayesian analysis."""

import os.path
import logging

from matplotlib import pyplot#, rcParams
import pandas
from scipy import stats

from planetary_system_io import read_cds_pipe_table
from command_line_utilities import data_dir
from cmd_utils import CMDPhotometryInterpolator
from process_e_Q_grid import LinearEccentricityEnvelope
from bayesian.photometric_constraint import PhotometricConstraint
from bayesian.cluster_util import\
    select_binary_data,\
    plot_rv_likelihood,\
    plot_eccentricity_vs_period

#https://ui.adsabs.harvard.edu/abs/2011ApJ...729L..10B/abstract
cluster_age_distribution = stats.norm(2.4, 0.3)

#https://ui.adsabs.harvard.edu/abs/2001AJ....121..327B/abstract
cluster_feh_distribution = stats.norm(0.09, 0.03)

eccentricity_envelope = LinearEccentricityEnvelope(min_period=8.0,
                                                   max_period=14.0,
                                                   min_eccenticity=0.05,
                                                   max_eccentricity=0.6,
                                                   extrapolate_to_e=0.8)

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

def get_photometric_constraint(binary_wocs_id,
                               pickle_fname='photometric_constraints.pkl'):
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
        pickle_fname,
        min_magnitude_difference=dict(V=2.0)
    )

def _test_rv_likelihood(binary_wocs_id):
    """Display plots showing the RV based constraint."""

    observed_orbit = get_observed_orbit(binary_wocs_id)
    photometric_constraint = get_photometric_constraint(binary_wocs_id)
    plot_rv_likelihood(observed_orbit,
                       eccentricity_envelope,
                       photometric_constraint)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    plot_eccentricity_vs_period(
        get_binary_data(),
        eccentricity_envelope
    )
#    pyplot.text(s='NGC 6819',
#                x=2.2, y=0.68,
#                fontsize='large',
#                ha='left',
#                va='top',
#                fontweight='semibold')
    pyplot.savefig('ngc6819_period_eccentricity.pdf')
    #_test_rvk_constraint(3732)
#    _test_rv_likelihood(66004)
