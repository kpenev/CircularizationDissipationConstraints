#!/usr/bin/env python3

"""Functions to fit for single and binary star masses given photometry."""

import scipy
import scipy.integrate

def fit_single_mass(photometry_interp,
                    photometry,
                    magnitude_template="%(filchar)c'mag",
                    error_template="e_%(filchar)c'mag"):
    """
    Find single star best fit mass for a subset of USNO u', g', r', i', z' mags.

    Args:
        photometry_interp(CMDPhotometryInterpolator):    Object able to
            predict relevant magnitudes for a given stellar mass or binary.

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

    def get_square_diff(theoretical_magnitudes):
        """
        Return the normalized square difference b/w theory and measurement.
        """

        grid_square_diff = scipy.zeros(theoretical_magnitudes[0].shape,
                                       dtype=float)
        for filter_index, filter_character in enumerate(
                photometry_interp.filchars
        ):
            if check_magnitude(filter_character):
                grid_square_diff += (
                    (
                        theoretical_magnitudes[filter_index]
                        -
                        get_magnitude(filter_character)
                    )
                    /
                    get_error(filter_character)
                )**2
        return grid_square_diff

    best_grid_index = get_square_diff(
        photometry_interp.grid_mag
    ).argmin()
    min_search_mass = photometry_interp.data[0]['Mini'][
        max(0, best_grid_index - 1)
    ]
    max_search_mass = photometry_interp.data[0]['Mini'][
        min(
            best_grid_index + 1,
            photometry_interp.data[0]['Mini'].size
        )
    ]
    return scipy.optimize.minimize_scalar(
        lambda m: get_square_diff(
            photometry_interp(scipy.full(fill_value=m, shape=1))
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
            mass function. If a single value, the error is assumed to follow
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

    if (
            isinstance(predicted_mass_function, float)
            and
            predicted_mass_function > observed_mass_function
    ):
        return 0.0

    try:
        sigma = float(observed_mass_function_err)
    except TypeError:
        sigma = observed_mass_function_err[0]

    return -(
        scipy.minimum(
            0,
            (
                predicted_mass_function
                -
                observed_mass_function
            )
        )**2
        /
        (2.0 * sigma**2)
    )

def double_lined_orbit_log_likelihood(primary_mass,
                                      secondary_mass,
                                      *,
                                      observed_mass_ratio,
                                      mass_ratio_error,
                                      observed_projected_primary_mass,
                                      projected_primary_mass_error):
    r"""
    Return unnormalized log-likelihood of observing SB2 RV params given masses.

    Args:
        primary_mass(float):    The mass of the primary star.

        secondary_mass(float):    The mass of the secondary star.

        observed_mass_ratio(float):    The nominal value of the mass ratio
            derived from RV observations.

        mass_ratio_error(float):    Estimate of the error in
            ``observed_mass_ratio``. If a single value, the error is assumed to
            follow symmetric Gaussian distribution. If a 2-tuple the first value
            specifies the positive standard deviation and the second value
            specifies the negative one.

        observed_projected_primary_mass(float):    The nominal value of
            :math:`M_1 \sin^3 i` derived from RV observations.

        projected_primary_mass_error(float):    Error estimate of the
            ``observed_projected_primary_mass``. See ``mass_ratio_error`` for
            comments on two-sided errors.

    Returns:
        float:
            The log-likelihood of observing the given value of the mass ratio
            and projected primary mass (:math:`M_1 \sin^3 i`), assuming gaussian
            error distribution, with possibly different standard deviations
            below and above the peak.
    """

    predicted_mass_ratio = secondary_mass / primary_mass
    try:
        mass_ratio_stddev = float(mass_ratio_error)
    except TypeError:
        if predicted_mass_ratio < observed_mass_ratio:
            mass_ratio_stddev = mass_ratio_error[1]
        else:
            mass_ratio_stddev = mass_ratio_error[0]

    mass_ratio_log_likelihood = -(
        (predicted_mass_ratio - observed_mass_ratio)**2
        /
        (2.0 * mass_ratio_stddev**2)
    )

    try:
        primary_mass_stddev = float(projected_primary_mass_error)
    except TypeError:
        primary_mass_stddev = projected_primary_mass_error[0]

    return (
        mass_ratio_log_likelihood
        -
        scipy.minimum(
            0,
            (primary_mass - observed_projected_primary_mass)**2
            /
            (2.0 * primary_mass_stddev**2)
        )
    )

def fit_binary_masses(*,
                      photometry_interp,
                      photometry,
                      distance_modulus,
                      min_mag_difference=None,
                      magnitude_template="%(filchar)cmag",
                      error_template="e_%(filchar)cmag",
                      **rv_parameters):
    r"""
    Find the best fit masses for stars in a binary given RV and photometry.

    Args:
        photometry_interp:    See same name argument to
            :func:`fit_single_mass`.

        photometry:    See same name argument to :func:`fit_single_mass`.

        distance_modulus(float):    The distance modulus to assume for the
            binary being fit.

        min_mag_difference(dict):    Minimal difference in magnitudes between
            primary and secondary to impose on the result. Keys should be
            filter characters.

        magnitude_template:    See same name argument to
            :func:`fit_single_mass`.

        error_template:    See same name argument to
            :func:`fit_single_mass`.

        rv_parameters:    Either the ``observed_mass_function`` and
            ``observed_mass_function_err`` arguments to
            :func:`mass_function_log_likelihood` if the binary is SB1 or the
            ``observed_mass_ratio``, ``mass_ratio_error``,
            ``observed_projected_primary_mass`, and
            ``projected_primary_mass_error`` arguments to
            :func:`double_lined_orbit_log_likelihood` for SB2 binaries.

    Returns:
        (float, float):
            The maximum likelihood estimates of the two masses assume all errors
            are Gaussian.
    """

    def negative_log_likelihood(masses):
        """Return -log(likelihood) of the data given stellar masses."""

        print('Calculating -LL(%s)' % repr(masses))

        predicted_photometry = scipy.array(
            photometry_interp.get_binary_magnitudes(*masses)
            +
            distance_modulus,
            copy=False
        )
        print('Predicted photometry: '
              +
              repr(predicted_photometry)
              +
              ', shape: '
              +
              repr(predicted_photometry.shape))
        if 'observed_projected_primary_mass' in rv_parameters:
            result = scipy.array(
                -double_lined_orbit_log_likelihood(*masses, **rv_parameters),
                copy=False
            )
        else:
            result = scipy.array(
                -mass_function_log_likelihood(*masses, **rv_parameters),
                copy=False
            )

        print('Mass based -LL: ' + repr(result))

        for filchar, predicted in zip(photometry_interp.filchars,
                                      predicted_photometry):
            print('Filter character: ' + repr(filchar))
            try:
                observed = photometry[magnitude_template
                                      %
                                      dict(filchar=filchar)]
                stddev = photometry[error_template % dict(filchar=filchar)]
            except ValueError:
                continue

            try:
                sigma = float(stddev)
            except TypeError:
                sigma = stddev[0 if predicted <= observed else 1]

            if (
                    not scipy.isfinite(sigma)
                    or
                    not scipy.isfinite(observed)
            ):
                continue

            update = scipy.isfinite(predicted)
            result[update] += (
                observed - predicted[update]
            )**2 / (
                2.0 * sigma**2
            )
            print('Result after %s filchar: %s' % (filchar, repr(result)))

        print('-LL(%s) = %s' % (repr(masses), repr(result)))
        return result

    def mag_difference_constraints(masses):
        """Return the secondary - primary mags for min mag constraints."""

        magnitudes = photometry_interp(masses)
        mag_differences = magnitudes[:, 1] - magnitudes[:, 0]
        defficiencies = [
            (
                mag_differences[photometry_interp.filchars.index(filchar)]
                -
                min_diff
            )
            for filchar, min_diff in min_mag_difference.items()
        ]
        return min(defficiencies)

    mass_grid = scipy.meshgrid(
        photometry_interp.data[0]['Mini'][1:-1],
        photometry_interp.data[0]['Mini'][1:-1]
    )


    log_likelihood_grid = negative_log_likelihood((
        mass_grid
    ))

    if min_mag_difference:
        grid_constraints = mag_difference_constraints(mass_grid)
        log_likelihood_grid[grid_constraints < 0] = scipy.inf
    best_indices = scipy.stack(
        reversed(
            sorted(
                scipy.unravel_index(scipy.argmin(log_likelihood_grid),
                                    log_likelihood_grid.shape)
            )
        )
    )
    best_masses = photometry_interp.data[0]['Mini'][best_indices]

    print('Best masses: ' + repr(best_masses))

    mass_bounds = scipy.optimize.Bounds(
        lb=photometry_interp.data[0]['Mini'][0] + 0.01,
        ub=photometry_interp.data[0]['Mini'][
            photometry_interp.data[0]['Mini'].size - 1,
        ] - 0.01,
        keep_feasible=True
    )
    return scipy.optimize.minimize(
        fun=negative_log_likelihood,
        x0=best_masses,
        bounds=mass_bounds,
        constraints=(
            dict(
                type='ineq',
                fun=mag_difference_constraints
            )
            if min_mag_difference else
            ()
        ),
        options=dict(maxiter=1e6, disp=True)
    )
