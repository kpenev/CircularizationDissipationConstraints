#!/usr/bin/env python3

"""Functions to fit for single and binary star masses given photometry."""

import scipy, scipy.integrate

from planetary_system_io import read_cds_pipe_table
from cmd_photometry_interpolator import CMDPhotometryInterpolator
from cmd_usno_photometry_interpolator import CMDUSNOPhotometryInterpolator

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

def fit_binary_masses(photometry_interp,
                      photometry,
                      mass_function,
                      mass_function_error,
                      distance_modulus,
                      min_mag_difference=dict(),
                      magnitude_template="%(filchar)cmag",
                      error_template="e_%(filchar)cmag"):
    r"""
    Find the best fit masses for stars in a binary given RV and photometry.

    Args:
        photometry_interp:    See same name argument to
            :func:`fit_single_mass`.

        photometry:    See same name argument to :func:`fit_single_mass`.

        mass_function(float):    The value of the mass function
            (:math:`\frac{M_2^2\sin^3i}{M_1+M_2)}`), presumably from RV
            measurements].

        mass_function_error(float):    An error estimate of the
            ``mass_function`` argument.

        distance_modulus(float):    The distance modulus to assume for the
            binary being fit.

        min_mag_difference(dict):    Minimal difference in magnitudes between
            primary and secondary to impose on the result. Keys should be
            filter characters.

        magnitude_template:    See same name argument to
            :func:`fit_single_mass`.

        error_template:    See same name argument to
            :func:`fit_single_mass`.

    Returns:
        (float, float):
            The maximum likelihood estimates of the two masses assume all errors
            are Gaussian.
    """

    def negative_log_likelihood(masses):
        """Return -log(likelihood) of the data given stellar masses."""

        predicted_photometry = scipy.array(
            photometry_interp.get_binary_magnitudes(*masses)
            +
            distance_modulus,
            copy=False
        )
        result = scipy.array(
            -mass_function_log_likelihood(*masses,
                                          mass_function,
                                          mass_function_error),
            copy=False
        )
        for filchar, predicted in zip(photometry_interp.filchars,
                                      predicted_photometry):
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
        photometry_interp.data[0]['Mini'],
        photometry_interp.data[0]['Mini']
    )

    log_likelihood_grid = negative_log_likelihood((
        mass_grid
    ))
    best_indices = scipy.stack(
        reversed(
            sorted(
                scipy.unravel_index(scipy.argmin(log_likelihood_grid),
                                    log_likelihood_grid.shape)
            )
        )
    )
    best_masses = photometry_interp.data[0]['Mini'][best_indices]

    mass_bounds = scipy.optimize.Bounds(
        lb = photometry_interp.data[0]['Mini'][0] + 0.01,
        ub = photometry_interp.data[0]['Mini'][
            photometry_interp.data[0]['Mini'].size - 1,
        ] - 0.01,
        keep_feasible = True
    )
    return scipy.optimize.minimize(
        fun=negative_log_likelihood,
        x0=best_masses,
#        method='trust-constr',
        bounds=mass_bounds,
#        options=dict(gtol=0.0, maxiter=1e6, initial_tr_radius=0.1, verbose=3)
        constraints = (
            dict(
                type='ineq',
                fun=mag_difference_constraints
            )
            if min_mag_difference else
            ()
        ),
        options=dict(maxiter=1e6, disp=False)
    )

if __name__ == '__main__':
    interpolator = CMDPhotometryInterpolator(
        '../data/CMD_7.5Gyr_FeH0dex_isochrone_Av0.2_UBVRIJHK.dat'
    )

    m1, m2 = scipy.pi/3, scipy.pi/20
    fmass = (m2**3 / 3.0)/((m1+m2)**2)
    binary_mag = {
        fc + 'mag': mag for fc, mag in zip(
            interpolator.filchars,
            interpolator.get_binary_magnitudes(m1, m2)
        )
    }
    for fc in interpolator.filchars:
        binary_mag['e_%cmag' % fc] = 0.1
    print('Binary mag: ' + repr(binary_mag))
    print(
        repr(
            fit_binary_masses(interpolator,
                              binary_mag,
                              fmass,
                              0.1 * fmass,
                              0.0)
        )
    )
    print('Correct answer: m1=%s, m2=%s' % (repr(m1), repr(m2)))

    ngc_188_photometry = read_cds_pipe_table(
        '../data/Stetson_et_al_04_NGC188_UBVRI_photometry.tsv'
    )
    ngc_188_single_lined_binaries = read_cds_pipe_table(
        '../data/Geller_et_al_2009_WIYN_single_lined_orbits.tsv'
    )
    ngc_188_params = read_cds_pipe_table(
        '../data/Geller_et_al_2009_WIYN_physical_parameters.tsv'
    )

    for binary in ngc_188_single_lined_binaries:
        photometry = ngc_188_photometry[
            ngc_188_photometry['PKM'] == binary['PKM']
        ]

        answer = ngc_188_params[
            ngc_188_params['PKM'] == binary['PKM']
        ]
        if not answer:
            continue

        result = fit_binary_masses(interpolator,
                                   photometry,
                                   binary['f(m)'],
                                   binary['e_f(m)'],
                                   11.23,
                                   min_mag_difference=dict(V=2.5))
        m1, m2 = result.x
        dv = (interpolator(m2)[interpolator.filchars.index('V')]
              -
              interpolator(m1)[interpolator.filchars.index('V')])

        print(
            '%c Binary %d best fit masses: m1=%s (%s), m2=%s (%s), dV=%s' % (
                ('v' if result.success else '*'),
                binary['PKM'],
                repr(m1),
                answer['l_M1'][0].decode() + repr(answer['M1'][0]),
                repr(m2),
                answer['l_M2'][0].decode() + repr(answer['M2'][0]),
                dv
            )
        )
