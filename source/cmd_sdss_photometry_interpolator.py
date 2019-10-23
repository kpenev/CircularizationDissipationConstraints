#!/usr/bin/env python3

"""
Define a class that works with interpolated SDSS photometry from CMD isochrones.
"""

import re

from matplotlib import pyplot
from astropy import units
import scipy
import scipy.optimize

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

        grid_usno_magnitudes:    The estimated magnitudes for the grid stars in
            the USNO 1m u', g', r', i', and z' system.
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
        assert scipy.unique(self.data[0]['logAge']).size == 1
        #False positive
        #pylint: disable=no-member
        self.age = 10.0**(self.data[0]['logAge'][0] - 9.0) * units.Gyr
        #pylint: enable=no-member

        self.grid_usno_mag = sdss_to_usno(
            scipy.stack(
                self.data[0][filchar + 'mag']
                for filchar in 'ugriz'
            )
        )


    def __call__(self, interp_mass):
        """Return the SDSS u, g, r, i, z photometry for the given mass(es)."""

        return (self.get_interpolated(mag_letter + 'mag', interp_mass, None)
                for mag_letter in 'ugriz')

    def get_usno_magnitudes(self, interp_mass):
        """Estimate UNSO u', g', r', i', z' photometry for given mass(es)."""

        return sdss_to_usno(scipy.stack(self(interp_mass)))

    def usno_best_fit_mass(self,
                           photometry,
                           magnitude_template="%(filchar)c'mag",
                           error_template="e_%(filchar)c'mag"):
        """
        Find best fit mass for a subset of USNO u', g', r', i', z' measurements.

        Args:
            photometry(dict):    A subset of u', g', r', i', z' magnitudes and
                errors measured for the star to fit. The key <-> magnitude or
                key <-> error correspondence is specified by the template
                arguments.

            magnitude_template(str):    A %(filchar)c-substitution template that
                should expand to the key giving a particular magnitude measured
                nominal value.

            error_template(str):    A %(filchar)c-substitution template that
                should expand to the key giving a particular magnitude error.


        Returns:
            The stellar mass which best reproduces the given measurements,
            assuming gaussian errors.
        """

        def get_magnitude(filchar):
            """Return the nominal measured magnitude in the given filter."""

            return photometry[magnitude_template % dict(filchar=filchar)]

        def get_error(filchar):
            """Return the measurement error estimate in the given filter."""

            return photometry[error_template % dict(filchar=filchar)]

        def check_magnitude(filchar):
            """Return True iff the given magnitude has a measurement & error."""

            return (
                magnitude_template % dict(filchar=filchar) in photometry
                and
                error_template % dict(filchar=filchar) in photometry
            )

        def get_square_diff(theoretical_usno_magnitudes):
            """
            Return the normalized square difference b/w theory and measurement.
            """

            grid_square_diff = scipy.zeros(theoretical_usno_magnitudes[0].shape,
                                           dtype=float)
            for filter_index, filter_character in enumerate('ugriz'):
                if check_magnitude(filter_character):
                    grid_square_diff += (
                        (
                            theoretical_usno_magnitudes[filter_index]
                            -
                            get_magnitude(filter_character)
                        )
                        /
                        get_error(filter_character)
                    )**2
            return grid_square_diff

        best_grid_index = get_square_diff(self.grid_usno_mag).argmin()
        min_search_mass = self.data[0]['Mini'][max(0, best_grid_index - 1)]
        max_search_mass = self.data[0]['Mini'][
            min(best_grid_index + 1, self.data[0]['Mini'].size)
        ]
        return scipy.optimize.minimize_scalar(
            lambda m:
            get_square_diff(
                self.get_usno_magnitudes(
                    scipy.full(fill_value=m, shape=1)
                )
            ),
            bounds=(min_search_mass, max_search_mass),
            method='bounded'
        ).x

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

    predicted_sdss_ugriz = scipy.stack(interpolator(interp_masses))

    predicted_usno_ugriz = sdss_to_usno(predicted_sdss_ugriz)

    mfit = interpolator.usno_best_fit_mass(
        {
            "g'mag": predicted_usno_ugriz[1][2],
            "r'mag": predicted_usno_ugriz[2][2],
            "e_g'mag": 0.01,
            "e_r'mag": 0.01
        }
    )
    print('Best fit mass = ' + repr(mfit))

    pyplot.plot(predicted_usno_ugriz[1] - predicted_usno_ugriz[2],
                -predicted_usno_ugriz[1] - 11.3, 'or', markersize=10)
    pyplot.plot(predicted_usno_ugriz[1] - predicted_usno_ugriz[2],
                -predicted_usno_ugriz[1] - 11.3, '-r', linewidth=3)

    pyplot.show()
