#!/usr/bin/env python3

"""
Define a class that works with interpolated SDSS photometry from CMD isochrones.
"""

import re

from matplotlib import pyplot
from astropy import units
import numpy

from planetary_system_io import read_cds_pipe_table
from magnitude_transformations import sdss_to_usno

from cmd_isochrone_interpolator import CMDInterpolator

class CMDSDSSPhotometryInterpolator(CMDInterpolator):
    """
    Interpolate SDSS photometry from CMD isochrones for a single cluster.

    Attributes:
        age(float):    The age of the isochrones.

        feh(float):    The metallicity ([Fe/H]) of the isochrones.

        min_mass(astropy Quantity):    The minimum stellar mass in the
            isochrones.

        max_mass(astropy Quantity):    The maximum stellar mass in the
            isochrones.

        extinction(float):    The assumed extinction (Av) applied to the
            isochrone by the CMD interface.
    """

    def _parse_header(self):
        """Extract the useful information from the isochrone file header."""

        extinction_parse_rex = re.compile(
            'photometry includes extinction of Av=(?P<Av>[^ ,]*)[, ]'
        )
        for line in self.header:
            if ':' in line:
                keyword, value = line[1:].strip().split(':', 1)
                if keyword.strip() == 'Photometric system':
                    assert value.strip() == 'SDSS <i>ugriz</i>'
                if keyword.strip() == 'Attention':
                    parsed_extinction = extinction_parse_rex.match(value.strip())
                    assert parsed_extinction
                    self.extinction = float(parsed_extinction['Av'])

    def __init__(self, isochrone_fname):
        """Interpolate within the given isochrone grid."""

        super().__init__(isochrone_fname)
        self._parse_header()
        assert len(self.data) == 1
        #False positive
        #pylint: disable=no-member
        self.min_mass = self.data[0]['Mini'][0] * units.M_sun
        self.max_mass = self.data[0]['Mini'][-1] * units.M_sun
        self.feh = self.data[0]['MH'][0]
        #pylint: enable=no-member
        assert numpy.unique(self.data[0]['logAge']).size == 1
        #False positive
        #pylint: disable=no-member
        self.age = 10.0**(self.data[0]['logAge'][0] - 9.0) * units.Gyr
        #pylint: enable=no-member

    def __call__(self, interp_mass):
        """Return the SDSS u, g, r, i, z photometry for the given mass(es)."""

        return (self.get_interpolated(mag_letter + 'mag', interp_mass, None)
                for mag_letter in 'ugriz')

if __name__ == '__main__':
    ngc_188_photometry = read_cds_pipe_table(
        '../data/Fornal_et_al_2006_NGC188_photometry.tsv'
    )
    cluster_members = ngc_188_photometry[ngc_188_photometry['Mm'] > 0.5]
    observed_usno_ugriz = numpy.array((cluster_members["u'mag"],
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

    predicted_sdss_ugriz = numpy.stack(interpolator(interp_masses))
    print('mag data: ' + repr(predicted_sdss_ugriz))
    predicted_usno_ugriz = sdss_to_usno(predicted_sdss_ugriz)

    pyplot.plot(predicted_usno_ugriz[1] - predicted_usno_ugriz[2],
                -predicted_usno_ugriz[1] - 11.3, 'or', markersize=10)
    pyplot.plot(predicted_usno_ugriz[1] - predicted_usno_ugriz[2],
                -predicted_usno_ugriz[1] - 11.3, '-r', linewidth=3)

    pyplot.show()
