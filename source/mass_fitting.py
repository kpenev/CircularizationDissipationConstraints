#!/usr/bin/env python3

"""Functions to fit for single and binary star masses given photometry."""

import scipy, scipy.integrate

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

def mass_function_log_likelihood(primary_mass,
                                 secondary_mass,
                                 observed_mass_function,
                                 observed_mass_function_err):
    r"""
    Return unnormalized log-likelihood of observing mass func. given masses.

    Args:
        primary_mass(float):    The mass of the primary star.

        secondary_mass(float):    The mass of the secondary star.

        observed_mass_function(float):    The observed value of the mass
            function.

        observed_mass_function_err(float):    Error estimate of the observed
            mass function, if a single value, the error is assumed to follow
            symmetric Gaussian distribution. If a 2-tuple the first value
            specifies the positive standard deviation and the second value
            specifies the negative one.

    Returns:
        float:
            The log-likelihood of observing the given value of the mass function
            if the true masses in the system are ``primary_mass`` and
            ``secondary_mass``, after marginalizing over inclination assumed to
            be distributed uniformly on a sphere.

            The expression is: :math:`\int_0^\pi/2 e^{-\frac{(\mu \sin^3i -
            \bar{\mu})^2}{2\sigma^2}} \sin i di`, where
            :math:`\mu\equiv\frac{M_2^3}{(M_1+M_2)^2}`, :math:`\bar{\mu}` is the
            measured value of the mass function and :math:`\sigma` is the
            estimated standard deviation in the mass function measurement.
    """

    predicted_mass_function = (secondary_mass**3
                               /
                               (primary_mass + secondary_mass)**2)
    try:
        sigma = float(observed_mass_function_err)
    except TypeError:
        sigma = observed_mass_function_err[
            0 if predicted_mass_function <= observed_mass_function else 1
        ]

    integrand = lambda i: (
        scipy.exp(
            -(
                predicted_mass_function * scipy.sin(i)**3
                -
                observed_mass_function
            )**2
            /
            (2.0 * sigma**2)
        )
        *
        scipy.sin(i)
    )
    return scipy.log(scipy.integrate.quad(integrand, 0, scipy.pi/2))

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
            primary and secondary to impose on the result. Keys should be
            filter characters, i.e. one of: ``'u'``, ``'g'``, ``'r``, ``'i``,
            ``'z``.

        magnitude_template:    See same name argument to
            :func:`usno_fit_single_mass`.

        error_template:    See same name argument to
            :func:`usno_fit_single_mass`.

    Returns:
        (float, float):
            The maximum likelihood estimates of the two masses assume all errors
            are Gaussian.
    """

    def negative_log_likelihood(masses):
        """Return -log(likelihood) of the data given stellar masses."""

        predicted_photometry = usno_photometry_interp.get_binary_magnitudes(
            *masses
        )
        result = -mass_function_log_likelihood(*masses)
        for filchar, predicted in zip('ugriz', predicted_photometry):
            try:
                observed = photometry[magnitude_template
                                      %
                                      dict(filchar=filchar)]
                stdev = photometry[error_template
                                   %
                                   dict(filchar=filchar)]
            except KeyError:
                continue

            try:
                sigma = float(stddev)
            except TypeError:
                sigma = stddev[0 if predicted <= observed else 1]

            result -= (observed - predicted)**2 / (2.0 * sigma**2)

        return result

    def mag_differences(masses):
        """Return the secondary - primary mags for min mag constraints."""

        magnitudes = usno_photometry_interp.get_usno_magnitudes(masses)
        mag_differences = magnitudes[:, 1] - magnitudes[:, 0]
        return [
            mag_differences['ugriz'.index(filchar)]
            for filchar in sorted(min_mag_difference.keys())
        ]

    return scipy.optimize.minimize(
        negative_log_likelihood,
        scipy.array([usno_photometry_interp.max_mass,
                     usno_photometry_interp.min_mass]),
        method='trust-constr',
        constraints=scipy.optimize.NonlinearConstraint(
            mag_differences,
            [
                min_mag_difference[filchar]
                for filchar in sorted(min_mag_difference.keys())
            ],
            scipy.inf
        )
    )

if __name__ == '__main__':
    print('LL: ' + repr(mass_function_log_likelihood(1.0, 0.5, 0.1, 0.1)))
    interpolator = CMDSDSSPhotometryInterpolator(
        '../data/CMD_7.5Gyr_FeH0dex_isochrone_Av0.dat'
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


