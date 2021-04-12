#!/usr/bin/env python3

"""Functions to fit for single and binary star masses given photometry."""

import numpy
import scipy.integrate

def fit_single_mass(photometry_interp,
                    photometry,
                    *,
                    magnitude_template="%(filter)s",
                    color_template=None):
    """
    Fit single star mass for mags/colors with both observed & predicted values.

    Args:
        photometry_interp(CMDPhotometryInterpolator):    Object able to
            predict relevant magnitudes for a given stellar mass or binary.

        photometry(dict):    The measured distributions of magnitudes and colors
            for the star to fit. The key <-> magnitude, key <-> color
            correspondence is specified by the template arguments. The values
            should be distributinos following the `scipy.stats.rv_continuous`
            interface.

        magnitude_template(str):    A %(filter)s-substitution template that
            should expand to the key giving a particular magnitude measured
            nominal value.

        color_template(None or str):    If None, individual magnitudes are fit.
            If not None, it is assumed that ``photometry`` contains color
            information (not just magnitudes), so masses are derived by fitting
            color--magnitude digrams. In the latter case, this argument should
            be contain a %(filter1)s and %(filter2)s substitutions, expanding
            to the key in ``photometry`` giving a particular measured color
            nominal value. Colors are always assumed to be magnitude in the
            bluer band minus the magnitude in the redder band.

    Returns:
        The stellar mass which best reproduces the given measurements,
        assuming gaussian errors.
    """

    def get_magnitude_term(theoretical_value, filter_name):
        """Return the negative log-likelihood for the given filter."""

        return -photometry[
            magnitude_template % dict(filter=filter_name)
        ].logpdf(
            theoretical_value
        )

    def get_color_term(theoretical_value, filter1, filter2):
        """Return the nominal measured color for the two given filters."""

        substitution = dict(filter1=filter1, filter2=filter2)
        return -photometry[
            color_template % substitution
        ].logpdf(
            theoretical_value
        )

    def check_magnitude(filter_name):
        """Return True iff the given magnitude has a measurement & error."""

        return magnitude_template % dict(filter=filter_name) in photometry

    def check_color(filter1, filter2):
        """Return True iff the given color has a measurement & error."""

        return color_template % dict(filter1=filter1,
                                     filter2=filter2) in photometry

    def get_square_diff(theoretical_magnitudes):
        """
        Return the normalized square difference b/w theory and measurement.
        """

        grid_square_diff = numpy.zeros(theoretical_magnitudes[0].shape,
                                       dtype=float)
        for filter_index, filter_character in enumerate(
                photometry_interp.available_filters
        ):
            if check_magnitude(filter_character):
                grid_square_diff += get_magnitude_term(
                    theoretical_magnitudes[filter_index],
                    filter_character
                )

        if color_template:
            for index1, filter1 in enumerate(
                    photometry_interp.available_filters
            ):
                for index2 in range(index1 + 1,
                                    len(photometry_interp.available_filters)):
                    filter2 = photometry_interp.available_filters[index2]
                    if check_color(filter1, filter2):
                        grid_square_diff += get_color_term(
                            (
                                theoretical_magnitudes[index1]
                                -
                                theoretical_magnitudes[index2]
                            ),
                            filter1,
                            filter2
                        )
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
            photometry_interp(numpy.full(fill_value=m, shape=1))
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
        numpy.minimum(
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
        numpy.minimum(
            0,
            (primary_mass - observed_projected_primary_mass)**2
            /
            (2.0 * primary_mass_stddev**2)
        )
    )

def fit_binary_masses(photometry_interpolators,
                      photometry,
                      *,
                      min_mag_difference=None,
                      magnitude_template="%(filter)s",
                      color_template=None,
                      **rv_parameters):
    r"""
    Find the best fit masses for stars in a binary given RV and photometry.

    Args:
        photometry_interpolators:    List of photometry interpolator objects
            (see `photometry_interp` argument to :func:`fit_single_mass`).

        photometry:    See same name argument to :func:`fit_single_mass`.

        min_mag_difference(dict):    Minimal difference in magnitudes between
            primary and secondary to impose on the result. Keys should be
            filter characters.

        magnitude_template:    See same name argument to
            :func:`fit_single_mass`.

        error_template:    See same name argument to :func:`fit_single_mass`.

        color_template:     See same name argument to :func:`fit_single_mass`.

        color_error_template:     See same name argument to
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

    def update_negative_log_likelihood(*,
                                       filter_name,
                                       predicted,
                                       result,
                                       rh_filter_name=None):
        """
        Add to result the negative log-likelihood for a single magnitude/color.

        Args:
            filter_name(string):    The filter to add the negative
                log-likelihood for (if ``rh_filter_name`` is ``None``), or the
                LH filter of the color difference being added.

            predicted(array):    The predicted value(s) for the magnitude or
                color.

            result(array):    The array to add the negative log-likelihood to.

            rh_filter_name(string or None):    If None the log-likelihood for
                a single magnitude is added, if not None, this is the RH filter
                defining the color to add the log-likelihood for.
        """

        try:
            if rh_filter_name is None:
                observed = photometry[magnitude_template
                                      %
                                      dict(filter=filter_name)]
            else:
                observed = photometry[color_template
                                      %
                                      dict(filter1=filter_name,
                                           filter2=rh_filter_name)]
        except (KeyError, ValueError):
            return

        if observed is None:
            return

        #False positive
        #pylint: disable=assignment-from-no-return
        update = numpy.isfinite(predicted)
        #pylint: enable=assignment-from-no-return
        result[update] -= observed.logpdf(predicted[update])

    def negative_log_likelihood(masses):
        """Return -log(likelihood) of the data given stellar masses."""

        result = numpy.zeros(numpy.atleast_1d(masses[0]).shape, dtype=float)
        if rv_parameters:
            numpy.array(
                -(
                    double_lined_orbit_log_likelihood(*masses, **rv_parameters)
                    if 'observed_projected_primary_mass' in rv_parameters else
                    mass_function_log_likelihood(*masses, **rv_parameters)
                ),
                copy=False,
                ndmin=1
            )

        for photometry_interp in photometry_interpolators:
            predicted_photometry = numpy.array(
                photometry_interp.get_binary_magnitudes(*masses),
                copy=False
            )


            for filter_index, (filter_name, predicted) in enumerate(
                    zip(photometry_interp.available_filters,
                        predicted_photometry)
            ):
                update_negative_log_likelihood(filter_name=filter_name,
                                               predicted=predicted,
                                               result=result)
                if color_template is not None:

                    for rh_filter_name, rh_predicted in zip(
                            photometry_interp.available_filters[
                                filter_index + 1
                                :
                            ],
                            predicted_photometry[filter_index + 1:]
                    ):
                        update_negative_log_likelihood(
                            filter_name=filter_name,
                            rh_filter_name=rh_filter_name,
                            predicted=(predicted - rh_predicted),
                            result=result
                        )
        return result

    def mag_difference_constraints(masses):
        """Return the secondary - primary mags for min mag constraints."""

        min_defficiency = numpy.inf

        for photometry_interp in photometry_interpolators:
            magnitudes = photometry_interp(masses)
            mag_differences = magnitudes[:, 1] - magnitudes[:, 0]

            for filter_name, min_diff in min_mag_difference.items():
                if filter_name in photometry_interp.available_filters:
                    min_defficiency = numpy.minimum(
                        min_defficiency,
                        (
                            mag_differences[
                                photometry_interp.available_filters.index(
                                    filter_name
                                )
                            ]
                            -
                            min_diff
                        )
                    )
        return min_defficiency

    def get_mass_grid(resolution=1e-10):
        """Pick a grid of masses to search for a starting point."""

        min_mass = max(
            photometry_interp.data[0]['Mini'][1]
            for photometry_interp in photometry_interpolators
        )
        max_mass = min(
            photometry_interp.data[0]['Mini'][-2]
            for photometry_interp in photometry_interpolators
        )

        mass_grid = numpy.concatenate([
            photometry_interp.data[0]['Mini']
            for photometry_interp in photometry_interpolators
        ])
        mass_grid = mass_grid[
            numpy.logical_and(mass_grid > min_mass, mass_grid < max_mass)
        ]
        mass_grid.sort()

        discard_indices = numpy.array([numpy.nan])
        while discard_indices.size:
            discard_indices = numpy.nonzero(mass_grid[1:] - mass_grid[:-1]
                                            <
                                            resolution)[0] + 1
            mass_grid[discard_indices - 1] = 0.5 * (
                mass_grid[discard_indices - 1]
                +
                mass_grid[discard_indices]
            )
            mass_grid = numpy.delete(mass_grid, discard_indices)

        return mass_grid

    def choose_starting_point(mass_grid):
        """Return a good starting point for the minimization."""

        mass_grid_2d = numpy.meshgrid(mass_grid, mass_grid)

        log_likelihood_grid = negative_log_likelihood(mass_grid_2d)

        if min_mag_difference:
            grid_constraints = mag_difference_constraints(mass_grid_2d)
            log_likelihood_grid[grid_constraints < 0] = numpy.inf

        best_masses = mass_grid[
            numpy.stack(
                reversed(
                    sorted(
                        numpy.unravel_index(numpy.argmin(log_likelihood_grid),
                                            log_likelihood_grid.shape)
                    )
                )
            )
        ]
        print('Starting masses: ' + repr(best_masses))

        return best_masses

    mass_grid = get_mass_grid()

    return scipy.optimize.minimize(
        fun=negative_log_likelihood,
        x0=choose_starting_point(mass_grid),
        bounds=scipy.optimize.Bounds(
            lb=mass_grid[0],
            ub=mass_grid[-1],
            keep_feasible=True
        ),
        constraints=(
            dict(
                type='ineq',
                fun=mag_difference_constraints
            )
            if min_mag_difference else
            ()
        ),
        options=dict(maxiter=1e6, disp=False)
    )
