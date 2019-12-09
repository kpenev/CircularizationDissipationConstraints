#!/usr/bin/env python3

"""
Define a class that works with interpolated SDSS photometry from CMD isochrones.
"""

from matplotlib import pyplot
import scipy

from planetary_system_io import read_cds_pipe_table
from magnitude_transformations import sdss_to_usno

from cmd_photometry_interpolator import CMDPhotometryInterpolator

class CMDUSNOPhotometryInterpolator(CMDPhotometryInterpolator):
    """Interpolate SDSS photometry from CMD isochrones for a single cluster."""

    def __init__(self, isochrone_fname):
        """Interpolate within the given isochrone grid."""

        super().__init__(isochrone_fname)

        assert self.filchars == 'ugriz'

        print('Initializing USNO interpolation.')
        self.grid_mag = sdss_to_usno(self.grid_mag)

    def __call__(self, interp_mass):
        """Estimate UNSO u', g', r', i', z' photometry for given mass(es)."""

        return sdss_to_usno(
            super().__call__(
                scipy.array(interp_mass, copy=False, ndmin=1)
            )
        )

    def get_binary_magnitudes(self, primary_mass, secondary_mass):
        """Estimate UNSO u', g', r', i', z' for a binary, given mass(es)."""

        return sdss_to_usno(
            super().get_binary_magnitudes(primary_mass, secondary_mass)
        )

if __name__ == '__main__':
    ngc_188_photometry = read_cds_pipe_table(
        '../data/Fornal_et_al_2006_NGC188_photometry.tsv'
    )
    cluster_members = ngc_188_photometry[ngc_188_photometry['Mm'] > 0.5]
    observed_usno_ugriz = scipy.array((cluster_members["u'mag"],
                                       cluster_members["g'mag"],
                                       cluster_members["r'mag"],
                                       cluster_members["i'mag"],
                                       cluster_members["z'mag"]))
    pyplot.plot(observed_usno_ugriz[1] - observed_usno_ugriz[2],
                -observed_usno_ugriz[1],
                'ok')

    interpolator = CMDSDSSPhotometryInterpolator(
        '../data/CMD_7.5Gyr_FeH0dex_isochrone_Av0.1.dat'
    )
    interp_masses = interpolator.data[0]['Mini']

    predicted_sdss_ugriz = interpolator(interp_masses)

    predicted_usno_ugriz = interpolator.get_usno_magnitudes(interp_masses)
    predicted_q1_binary_usno_ugriz = (
        interpolator.get_binary_usno_magnitudes(
            interp_masses,
            interp_masses
        )
    )

    pyplot.plot(predicted_usno_ugriz[1] - predicted_usno_ugriz[2],
                -predicted_usno_ugriz[1] - 11.3, 'or', markersize=10)
    pyplot.plot(predicted_usno_ugriz[1] - predicted_usno_ugriz[2],
                -predicted_usno_ugriz[1] - 11.3, '-r', linewidth=3)
    pyplot.plot(
        (
            predicted_q1_binary_usno_ugriz[1]
            -
            predicted_q1_binary_usno_ugriz[2]
        ),
        -predicted_q1_binary_usno_ugriz[1] - 11.3,
        '-g',
        linewidth=3
    )

    pyplot.show()
