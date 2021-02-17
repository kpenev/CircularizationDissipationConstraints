"""Utilities for reading cluster data."""

import os.path

import pandas
import numpy
from scipy import stats

from planetary_system_io import read_cds_pipe_table
from command_line_utilities import data_dir
from cmd_utils import CMDPhotometryInterpolator, CMDUSNOPhotometryInterpolator

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

    result.dtype.names = [_unprime_usno_column_name(colname)
                          for colname in result.dtype.names]

    return result

def get_ngc188_photometry():
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
        pandas.DataFrame(get_ngc188_usno_photometry()),
        on='PKM',
        how='outer'
    )

def get_ngc188_binaries():
    """Return NGC188 SB1 and SB2 systems as pandas DataFrames."""

    physical_parameters = pandas.DataFrame(
        read_cds_pipe_table(
            os.path.join(
                data_dir,
                'Geller_et_al_2009_WIYN_physical_parameters.tsv'
            )
        )
    )

    return (
        pandas.merge(
            pandas.DataFrame(
                read_cds_pipe_table(
                    os.path.join(
                        data_dir,
                        'Geller_et_al_2009_WIYN_single_lined_orbits.tsv'
                    )
                )
            ),
            physical_parameters,
            on='PKM',
            how='outer'
        ),
        pandas.merge(
            pandas.DataFrame(
                read_cds_pipe_table(
                    os.path.join(
                        data_dir,
                        'Geller_et_al_2009_WIYN_double_lined_orbits.tsv'
                    )
                )
            ),
            physical_parameters,
            on='PKM',
            how='outer'
        )
    )

def get_ngc188_photometry_interpolators():
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
