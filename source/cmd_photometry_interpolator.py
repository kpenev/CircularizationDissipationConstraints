#!/usr/bin/env python3

"""
Define a class that works with interpolated photometry from CMD isochrones.
"""

import re

from matplotlib import pyplot
from astropy import units
import scipy

from planetary_system_io import read_cds_pipe_table

from cmd_isochrone_interpolator import CMDInterpolator

class CMDPhotometryInterpolator(CMDInterpolator):
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
        filchar_rex = re.compile(
            '.* <i>(?P<filchars>[a-zA-Z]*)</i>'
        )
        for line in self.header:
            if ':' in line:
                keyword, value = line[1:].strip().split(':', 1)
                if keyword.strip() == 'Photometric system':
                    print('Matching ' + repr(value.strip()))
                    parsed_filchars = filchar_rex.match(value.strip())
                    assert parsed_filchars
                    self.filchars = parsed_filchars['filchars']
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
        assert scipy.unique(self.data[0]['logAge']).size == 1
        #False positive
        #pylint: disable=no-member
        self.age = 10.0**(self.data[0]['logAge'][0] - 9.0) * units.Gyr
        #pylint: enable=no-member

        self.grid_mag = scipy.stack(
            self.data[0][filchar + 'mag'] for filchar in self.filchars
        )

    def __call__(self, interp_mass):
        """Return the available photometry for the given mass(es)."""

        return scipy.stack(
            self.get_interpolated(mag_letter + 'mag', interp_mass, None)
            for mag_letter in self.filchars
        )

    def get_binary_magnitudes(self, primary_mass, secondary_mass):
        """Estimate SDSS u, g, r, i, z for a binary, given mass(es)."""

        primary_mags = self(primary_mass)
        secondary_mags = self(secondary_mass)
        return -2.5 * scipy.log10(10.0**(-primary_mags/2.5)
                                  +
                                  10.0**(-secondary_mags/2.5))

if __name__ == '__main__':
    ngc_188_photometry = read_cds_pipe_table(
        '../data/Stetson_et_al_04_NGC188_UBVRI_photometry.tsv'
    )
    cluster_members = ngc_188_photometry[ngc_188_photometry['Memb'] > 0.5]
    observed_ubvri = scipy.array((cluster_members["Umag"],
                                  cluster_members["Bmag"],
                                  cluster_members["Vmag"],
                                  cluster_members["Rmag"],
                                  cluster_members["Imag"]))

    interpolator = CMDPhotometryInterpolator(
        '../data/CMD_7.5Gyr_FeH0dex_isochrone_Av0.2_UBVRIJHK.dat'
    )

    interp_masses = interpolator.data[0]['Mini']
    predicted_ubvrijhk = interpolator(interp_masses)

    predicted_q1_binary_ubvrijhk = (
        interpolator.get_binary_magnitudes(
            interp_masses,
            interp_masses
        )
    )

#    pyplot.plot(predicted_ubvrijhk[1] - predicted_ubvrijhk[2],
#                -predicted_ubvrijhk[2] - 11.23, 'or', markersize=10)

    for left in range(5):
        for right in range(left + 1, 5):
            pyplot.plot(observed_ubvri[left] - observed_ubvri[right],
                        -observed_ubvri[2],
                        'ok')
            pyplot.plot(predicted_ubvrijhk[left] - predicted_ubvrijhk[right],
                        -predicted_ubvrijhk[2] - 11.23, '-r', linewidth=3)
            pyplot.xlabel('%s - %s [mag]' % ('UBVRI'[left], 'UBVRI'[right]))
            pyplot.ylabel('-V [mag]')

            pyplot.show()
