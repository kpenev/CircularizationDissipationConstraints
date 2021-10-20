#!/usr/bin/env python3

"""M35 specific functions required for circularization bayesian analysis."""

import os.path
import logging

from matplotlib import pyplot
import pandas
import numpy
from scipy import stats

from planetary_system_io import read_pipe_table_to_pandas
from command_line_utilities import data_dir
from cmd_utils import CMDPhotometryInterpolator
from process_e_Q_grid import LinearEccentricityEnvelope
from bayesian.cluster_util import\
    select_binary_data,\
    plot_rv_likelihood
from bayesian.photometric_constraint import\
    PhotometricConstraint,\
    plot_joint_pdf

cluster_age_distribution = stats.norm(0.15, 0.03)
cluster_feh_distribution = stats.norm(-0.18, 0.03)

eccentricity_envelope = LinearEccentricityEnvelope(min_period=8.0,
                                                   max_period=12.0,
                                                   min_eccenticity=0.05,
                                                   max_eccentricity=0.6)

def get_photometry():
    """Return a pandas DataFrame containing M35 V and B-V photometry."""

    return read_pipe_table_to_pandas(
        os.path.join(
            data_dir,
            'Leiner_et_al_15_M35_rv_photometry.tsv'
        )
    )

def get_binary_data(
        single_lined_orbits_fname=(
            os.path.join(
                data_dir,
                'Leiner_et_al_15_M35_sb1_orbits.tsv'
            )
        ),
        double_lined_orbits_fname=(
            os.path.join(
                data_dir,
                'Leiner_et_al_15_M35_sb2_orbits.tsv'
            )
        )
):
    """Return M35 SB1 and SB2 systems as :class:`pandas.DataFrame`s."""

    photometry = get_photometry()
    return (
        pandas.merge(
            read_pipe_table_to_pandas(single_lined_orbits_fname),
            photometry,
            on='WOCS'
        ),
        pandas.merge(
            read_pipe_table_to_pandas(double_lined_orbits_fname),
            photometry,
            on='WOCS'
        )
    )

def get_photometry_interpolator():
    """Return interpolator of photometry matching M35's age and [Fe/H]."""

    return CMDPhotometryInterpolator(
        os.path.join(
            data_dir,
            'CMD_150Myr_FeH-0.18dex_isochrone_Av0.62_UBVRIJHK.dat'
        ),
        9.53
    )

def get_photometric_constraint(binary_wocs_id):
    """Return a fully set-up photometric constraint for an M35 binary."""

    selected_photometry = get_observed_orbit(binary_wocs_id).squeeze()
    print('Selected photometry: ' + repr(selected_photometry))
    selected_photometry = {
        'V': stats.norm(
            loc=selected_photometry['Vmag'],
            scale=(0.06 if selected_photometry['r_B-V'] == 1 else 0.03)
        ),
        'B': stats.norm(
            loc=(selected_photometry['B-V'] + selected_photometry['Vmag']),
            scale=(0.08 if selected_photometry['r_B-V'] == 1 else 0.04)
        )
    }

    interpolator = get_photometry_interpolator()
    return PhotometricConstraint(
        [interpolator],
        selected_photometry,
        'photometric_constraints.pkl',
        min_magnitude_difference=dict(V=1.0)
    )

def plot_color_magnitude_diagram():
    """Plot the theoretical and observed CMD for M35 along with binaries."""

    interpolator = get_photometry_interpolator()

    measured_photometry = get_photometry()

    single_photometry = measured_photometry[measured_photometry['Mm'] == 'SM']

    sb1_data, sb2_data = get_binary_data()

    interp_masses = interpolator.data[0]['Mini']
    predicted_photometry = interpolator(interp_masses)

    pyplot.plot(predicted_photometry[2] - predicted_photometry[3],
                -predicted_photometry[3],
                '-y',
                linewidth=3,
                zorder=10,
                label='Isochrone')

    marker_phot = interpolator(numpy.array([0.8, 1.6, 4.0]))
    pyplot.plot(marker_phot[2] - marker_phot[3],
                -marker_phot[3],
                'or',
                zorder=20,
                label=r'M=0.8, 1.6, 4.0 $M_\odot$')

    pyplot.plot(single_photometry['B-V'],
                -single_photometry['Vmag'],
                'ok',
                zorder=0,
                label='Single member')

    pyplot.plot(sb1_data['B-V'],
                -sb1_data['Vmag'],
                'og',
                zorder=1,
                label='SB1')
    pyplot.plot(sb2_data['B-V'],
                -sb2_data['Vmag'],
                'ob',
                zorder=1,
                label='SB2')
    pyplot.xlabel('B-V [mag]')
    pyplot.ylabel('-V [mag]')
    pyplot.legend()

def get_observed_orbit(binary_wocs_id):
    """Return pandas.DataFrames containing the orbital parameters of an SB1."""

    result = select_binary_data(*get_binary_data(), 'WOCS', binary_wocs_id)
    return  result

def _test_photometric_constraint(binary_wocs_id):
    """Display plots showing the photometry based constraint."""

    constraint = get_photometric_constraint(binary_wocs_id)

    plot_joint_pdf(constraint, (1.04939401, 0.577582))

def _test_rv_likelihood(binary_wocs_id):
    """Display plots showing the RV based constraint."""

    observed_orbit = get_observed_orbit(binary_wocs_id)

    photometric_constraint = get_photometric_constraint(binary_wocs_id)
    plot_rv_likelihood(observed_orbit,
                       eccentricity_envelope,
                       photometric_constraint)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    pandas.set_option('display.max_rows', None)
#    plot_color_magnitude_diagram()
#    pyplot.show()
#    plot_eccentricity_vs_period(get_binary_data(), eccentricity_envelope)
    pyplot.show()
#    _test_photometric_constraint(23043)
#    pyplot.show()
    _test_rv_likelihood(23043)
#    pyplot.show()
