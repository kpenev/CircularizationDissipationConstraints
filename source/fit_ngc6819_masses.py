#!/usr/bin/env python3

"""Find max likelinhood masses in NGC6819 from photometry and RVs."""

import os.path

import pandas

from cmd_utils import CMDPhotometryInterpolator
from mass_fitting import fit_binary_masses
from command_line_utilities import data_dir

def fit_milliman(single_lined_data,
                 double_lined_data,
                 interpolator=None,
                 distance_modulus=11.85):
    """
    Fit for the masses of the NGC6819 binaries using Milliman's photometry.

    Args:
        single_lined_data:    The data from Milliman et. al. 2014 for single
            lined binaries. Should include both RV info and photemtry.

        double_lined_data:    The data from Milliman et. al. 2014 for single
            lined binaries. Should include both RV info and photemtry.

        interpolator:     See ``photometry_interp`` argument to
            :func:`fit_binary_masses`. Defaults to 2.5 Gyr interpolator of
            UBVRIJHK magnitudes.

        distance_modulus:     See same name argument to
            :func:`fit_binary_masses`.

        plot_photometry(numpy field array or ``None``):    Should plots showing
            the fits be generated and displayed to the user? If not ``None``, it
            should be set to a numpy field array containing the available V &
            V-I photometry for NGC6819 from Milliman et. al. (2014).

    Returns:
        dict:
            Keys are the WOCS IDs of the binaries and the entries are 2-tuples
            containing the maximum likelihood primary and secondary mass for
            each binary.
    """

    if interpolator is None:
        interpolator = CMDPhotometryInterpolator(
            os.path.join(
                data_dir,
                'CMD_2.5Gyr_isochrone_Av0.45_FeH0.09_UBVRIJHK.dat'
            ),
            distance_modulus
        )

    for is_double_lined, binary_data in [(False, single_lined_data),
                                         (True, double_lined_data)]:
        m1_series = pandas.Series(index=binary_data.index)
        m2_series = pandas.Series(index=binary_data.index)

        for binary_ind, binary in binary_data.iterrows():
            if is_double_lined:
                rv_params = dict(
                    observed_mass_ratio=binary['q'],
                    mass_ratio_error=binary['e_q'],
                    observed_projected_primary_mass=binary['msini1'],
                    projected_primary_mass_error=binary['e_msini1']
                )
            else:
                rv_params = dict(
                    observed_mass_function=binary['f(m)'],
                    observed_mass_function_err=binary['e_f(m)']
                )

            (m1_series[binary_ind], m2_series[binary_ind]) = fit_binary_masses(
                photometry_interpolators=[interpolator],
                photometry=binary,
                min_mag_difference=(None if is_double_lined
                                    else {'V': 2.0}),
                magnitude_template='%(filchar)cmag',
                color_template='%(filchar1)c-%(filchar2)c',
                **rv_params
            ).x
            print('WOCS %d: m1 = %f, m2 = %f' % (binary['WOCS'],
                                                 m1_series[binary_ind],
                                                 m2_series[binary_ind]))
        binary_data['M1'] = m1_series
        binary_data['M2'] = m2_series
