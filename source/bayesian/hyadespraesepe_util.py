#!/usr/bin/env python3
"""Hyades/Praesepe specific functions required for circularization bayesian analysis."""

import os.path
import logging

from matplotlib import pyplot#, rcParams
import pandas
from scipy import stats

from planetary_system_io import read_cds_pipe_table,read_pipe_table_to_pandas
from command_line_utilities import data_dir
from cmd_utils import CMDPhotometryInterpolator
from process_e_Q_grid import LinearEccentricityEnvelope
from bayesian.cluster_util import\
    select_binary_data,\
    plot_eccentricity_vs_period
from bayesian.photometric_constraint import PhotometricConstraint

cluster_age_distribution = stats.norm(0.75, 0.15) #https://www.aanda.org/articles/aa/full_html/2018/08/aa32843-18/aa32843-18.html
cluster_feh_distribution = stats.norm(0.15, 0.08) #https://ui.adsabs.harvard.edu/abs/2016A%26A...585A.150N/abstract

eccentricity_envelope = LinearEccentricityEnvelope(min_period=1.75,
                                                   max_period=30.0,
                                                   min_eccenticity=0.01,
                                                   max_eccentricity=0.9)

def get_photometric_constraint(binary_id,
                               pickle_fname='photometric_constraints.pkl'):
    """Return a fully set-up photometric constraint for a Hyades/Praesepe binary."""

    photometry = get_photometry()
    selected_photometry = get_photometry_distributions(
        photometry[photometry['ID'] == binary_id]
    )
    
    interpolator = get_photometry_interpolator(binary_id)
    return PhotometricConstraint(
        [interpolator],
        selected_photometry,
        pickle_fname,
        min_magnitude_difference=dict(V=1.0)
    )

def get_photometry(
        photometry_fname=(
            os.path.join(
                data_dir,
                'HyadesPraesepe_photometry.tsv'
            )
        )
):
    """Read the photometry as pandas frame."""

    return pandas.DataFrame(read_pipe_table_to_pandas(photometry_fname))

def get_photometry_distributions(photometry):
    """Return dictionary of all available photometry as normal distributions."""
    
    return dict(
        V=stats.norm(
            loc=float(photometry['Vmag']),
            scale=0.1
        ),
        B=stats.norm(
            loc=float(photometry['B-V'] - photometry['Vmag']),
            scale=0.1
        )
    )

def get_photometry_interpolator(binary_id):
    """Return a dictionary of interpolators of Hyades/Praesepe photometry."""

    distance_modulus=3.334  #https://ui.adsabs.harvard.edu/abs/2009A%26A...497..209V/abstract
    name='CMD_0.7Gyr_isochrone_Av0.00_FeH0.15_UBVRIJHK.dat'
    if binary_id>=3001:
        distance_modulus=6.30
        name='CMD_0.7Gyr_isochrone_Av0.08_FeH0.15_UBVRIJHK.dat' #https://ui.adsabs.harvard.edu/abs/2019MNRAS.483.3098S/abstract
                                                                #https://www.aanda.org/articles/aa/full_html/2018/08/aa32843-18/aa32843-18.html
    
    return CMDPhotometryInterpolator(
        os.path.join(
            data_dir,
            name
        ),
        distance_modulus
    )

def get_observed_orbit(binary_id):
    """Return pandas.DataFrames containing the orbital parameters of an SB1."""

    return select_binary_data(*get_binary_data(), 'ID', binary_id)

def get_binary_data(
        single_lined_orbits_fname=(
            os.path.join(
                data_dir,
                'HyadesPraesepe_single.tsv'
            )
        ),
        double_lined_orbits_fname=(
            os.path.join(
                data_dir,
                'HyadesPraesepe_double.tsv'
            )
        )
):
    """Read the data from the tables as pandas frames."""

    photometry = get_photometry()

    single_lined_data = pandas.merge(
        pandas.DataFrame(
            read_pipe_table_to_pandas(single_lined_orbits_fname)
        ),
        photometry,
        on='ID'
    )
    double_lined_data = pandas.merge(
        pandas.DataFrame(
            read_pipe_table_to_pandas(double_lined_orbits_fname)
        ),
        photometry,
        on='ID'
    )
    return single_lined_data, double_lined_data

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    plot_eccentricity_vs_period(
        get_binary_data(),
        eccentricity_envelope
    )

    pyplot.savefig('HyadesPraesepe_period_eccentricity.pdf')