#!/usr/bin/env python3

"""NGC188 specific functions required for circularization bayesian analysis."""

import os.path
import logging
from multiprocessing import set_start_method

import pandas
import numpy
from scipy import stats

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
from bayesian.cluster_util import\
    select_binary_data,\
    plot_rvk_constraint,\
    plot_eccentricity_vs_period,\
    plot_eccentricity_likelihood

_logger = logging.getLogger(__name__)

cluster_age_distribution = stats.norm(7.0, 0.5)
cluster_feh_distribution = stats.norm(0.21, 0.03)
eccentricity_envelope = LinearEccentricityEnvelope(min_period=3.0,
                                                   max_period=20.0,
                                                   max_eccentricity=0.6)


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

    result = select_binary_data(*get_binary_data(), 'PKM', binary_pkm_id)
    if binary_pkm_id == 5015:
        return result.iloc[1]

    return  result

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

def _test_rvk_constraint(binary_pkm_id):
    """Display plots showing the RV based constraint."""

    observed_orbit = get_observed_orbit(binary_pkm_id)
    photometric_constraint = get_photometric_constraint(binary_pkm_id)
    plot_rvk_constraint(observed_orbit, photometric_constraint)

if __name__ == '__main__':
    set_start_method('forkserver')
    logging.basicConfig(level=logging.DEBUG)
    plot_eccentricity_likelihood(get_observed_orbit(4965),
                                 eccentricity_envelope)
    plot_eccentricity_vs_period(get_binary_data(), eccentricity_envelope)
    #_test_rvk_constraint(3732)
