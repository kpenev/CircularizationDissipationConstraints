"""
Define a class that works with interpolated SDSS photometry from CMD isochrones.
"""

from cmd_isochrone_interpolator import CMDInterpolator
from astropy import units

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

    def __init__(self, isochrone_fname):
        """Interpolate within the given isochrone grid."""

        super().__init__(isochrone_fname)
        self._parse_header()
