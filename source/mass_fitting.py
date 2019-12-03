#!/usr/bin/env python3

"""Functions to fit for single and binary star masses given photometry."""

import scipy

from cmd_sdss_photometry_interpolator import CMDSDSSPhotometryInterpolator

def usno_fit_single_mass(usno_photometry_interp,
                         photometry,
                         magnitude_template="%(filchar)c'mag",
                         error_template="e_%(filchar)c'mag"):
    """
    Find single star best fit mass for a subset of USNO u', g', r', i', z' mags.

    Args:
        usno_photometry_interp(CMDSDSSPhotometryInterpolator):    Object able to
            predict USNO magnitudes for a given stellar mass.

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

    best_grid_index = get_square_diff(
        usno_photometry_interp.grid_usno_mag
    ).argmin()
    min_search_mass = usno_photometry_interp.data[0]['Mini'][
        max(0, best_grid_index - 1)
    ]
    max_search_mass = usno_photometry_interp.data[0]['Mini'][
        min(
            best_grid_index + 1,
            usno_photometry_interp.data[0]['Mini'].size
        )
    ]
    return scipy.optimize.minimize_scalar(
        lambda m:
        get_square_diff(
            usno_photometry_interp.get_usno_magnitudes(
                scipy.full(fill_value=m, shape=1)
            )
        ),
        bounds=(min_search_mass, max_search_mass),
        method='bounded'
    ).x

def usno_fit_binary_masses(usno_photometry_interp,
                           photometry,
                           mass_function,
                           mass_function_error,
                           min_mag_difference=dict(),
                           magnitude_template="%(filchar)c'mag",
                           error_template="e_%(filchar)c'mag"):
    r"""
    Find the best fit masses for stars in a binary given RV and photometry.

    Args:
        usno_photometry_interp:    See same name argument to
            :func:`usno_fit_single_mass`.

        photometry:    See same name argument to :func:`usno_fit_single_mass`.

        mass_function(float):    The value of the mass function
            (:math:`\frac{M_2^2\sin^3i}{M_1+M_2)}`), presumably from RV
            measurements].

        mass_function_error(float):    An error estimate of the
            ``mass_function`` argument.

        min_mag_difference(dict):    Minimal difference in magnitudes between
            primary and secondary to impose on the result. Keys should be names
            of USNO filters.

        magnitude_template:    See same name argument to
            :func:`usno_fit_single_mass`.

        error_template:    See same name argument to
            :func:`usno_fit_single_mass`.

    Returns:
        (float, float):
            The maximum likelihood estimates of the two masses assume all errors
            are Gaussian.
    """

    def negative_log_likelihood(primary_mass, secondary_mass):
        """Return -log(likelihood) of the data given stellar masses."""

        primary_mags = usno_photometry_interp.get_usno_magnitudes(
            primary_mass
        )
        secondary_mags = usno_photometry_interp.get_usno_magnitudes(
            secondary_mass
        )

if __name__ == '__main__':
    interpolator = CMDSDSSPhotometryInterpolator(
        '../data/CMD_7.5Gyr_FeH0dex_isochrone_Av0.1.dat'
    )

    predicted_usno_ugriz = interpolator.get_usno_magnitudes(0.7)
    print('Predicted magnitudes: ' + repr(predicted_usno_ugriz))
    mfit = usno_fit_single_mass(
        interpolator,
        {
            "g'mag": predicted_usno_ugriz[1],
            "r'mag": predicted_usno_ugriz[2],
            "e_g'mag": 0.01,
            "e_r'mag": 0.01
        }
    )
    print('Best fit mass = ' + repr(mfit))
