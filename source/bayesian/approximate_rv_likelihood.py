"""Allow approximating the RV likelihood (lambda in notes)."""

from functools import partial

import numpy
from scipy import integrate
from scipy.optimize import root_scalar

from approximate_2d_function import Approximate2DFunction

class ApproximateRVLikelihood(Approximate2DFunction):
    """Specialize 2D approximation to RV likelihood."""

    def _pdf_inclination_integrand(self,
                                   cos_inclination,
                                   max_rv_semiamplitude):
        """
        The integrand for marginalizing the PDF over inclination.

        Args:
            cos_inclination:    The value of cos(i), where i is the angle
                between the orbital angular momentum and the line of sight.

            max_rv_semiamplitude:    The semiamplitude that would be observed
                for the given orbit if it were viewed edge on.

        Returns:
            float:
                PDF of observing RV given the specified orbit.
        """

        return self._observed_rvk.pdf(
            max_rv_semiamplitude
            *
            numpy.sqrt(1.0 - numpy.square(cos_inclination))
        )

    def _cdf_inclination_integrand(self, sin_inclination, max_rv_semiamplitude):
        """The integrand for marginalizing the CDF over inclination."""

        return (self._observed_rvk.cdf(max_rv_semiamplitude * sin_inclination)
                /
                numpy.sqrt(1.0 - numpy.square(sin_inclination)))

    def _sf_inclination_integrand(self, sin_inclination, max_rv_semiamplitude):
        """The integrand for marginalizing survival functn over inclination."""

        return (self._observed_rvk.sf(max_rv_semiamplitude * sin_inclination)
                /
                numpy.sqrt(1.0 - numpy.square(sin_inclination)))

    def _get_inclination_integration_breaks(self,
                                            max_rv_semiamplitude,
                                            cdf_step=0.01):
        """Return array of points integration must hit for accuracy."""

        result = (
            self._observed_rvk.ppf(numpy.arange(cdf_step, 1.0, cdf_step))
            /
            max_rv_semiamplitude
        )
        return result[result < 1.0]

    def _get_eccentricity_integration_breaks(self, cdf_step=0.01):
        """Return array of points integration must hit for accuracy."""

        result = self._observed_eccentricity.ppf(
            numpy.arange(cdf_step, 1.0, cdf_step)
        )
        return result[
            numpy.logical_and(result > 0, result < self._envelope_eccentricity)
        ]


    def _marginalize_inclination(self,
                                 max_rv_semiamplitude,
                                 integrand=None,
                                 value_only=True):
        """Marginalize PDF/CDF/SF for given max RV semi-amplitude over incl."""

        if max_rv_semiamplitude == 0:
            return 0.0

        points = self._get_inclination_integration_breaks(max_rv_semiamplitude)
        split = min(self._observed_rvk.isf(1e-6) / max_rv_semiamplitude, 1.0)
        if integrand is None:
            integrand = self._pdf_inclination_integrand

            split = numpy.sqrt(1.0 - numpy.square(split))
            points = numpy.sqrt(1.0 - numpy.square(points))


        integration_result = integrate.quad(
            integrand,
            0.0,
            split,
            args=(max_rv_semiamplitude,),
            points=points,
            full_output=True,
            **self._integration_options
        )
        result = numpy.array(integration_result[:2])
        if split < 1.0:
            integration_result = integrate.quad(
                integrand,
                split,
                1.0,
                args=(max_rv_semiamplitude,),
                points=points,
                full_output=True,
                **self._integration_options
            )
            result += numpy.array(integration_result[:2])

        result *= 2.0 / numpy.pi

        if value_only:
            return result[0]

        return result


    def _get_eccentricity_integrand(self, eccentricity, _, rvk_scale):
        """
        Calculate the likelihood marginalized over inclination for given eccent.

        Args:
            eccentricity(float):    The value to assume for the present day
                eccentircity.

            rvk_scale(float):    The value of the radial velocity semi-amplitude
                assuming a circular edge-on orbit for which to evaluate the
                integral of the likelihood.

        Returns:
            float:
                Product of the radial velocity PDF and eccentricity PDF
                integrated over all possible inclinations.
        """

        return (
            self._observed_eccentricity.pdf(eccentricity)
            *
            self._marginalize_inclination(
                rvk_scale / numpy.sqrt(1.0 - numpy.square(eccentricity))
            )
        )


    def _get_eccentircity_integral(self, rvk_scale):
        """
        Return a function that evaluates to the integral of likelihood vs eccen.

        Args:
            rvk_scale(float):    The value of the radial velocity semi-amplitude
                assuming a circular edge-on orbit for which to evaluate the
                integral of the likelihood.

        Returns:
            scipy.integrate.OdeSolution:
                The integral over eccentricity of the likelihood function in the
                range 0 to the argument of the function.
        """

        assert rvk_scale not in self._eccentricity_integrals

        solution = integrate.solve_ivp(
            fun=self._get_eccentricity_integrand,
            t_span=(0, self._envelope_eccentricity),
            y0=numpy.array([
                self._marginalize_inclination(rvk_scale)
                *
                self._observed_eccentricity.cdf(0.0)
            ]),
            t_eval=self._get_eccentricity_integration_breaks(),
            args=(rvk_scale,),
            dense_output=True,
            **self._solve_ivp_options
        )
        assert solution.status == 0
        return solution.sol


    def _evaluate_e_integral(self, e_grid, rvk_scale):
        """Evaluate the integral for the given rvk_scale for x_grid."""

        return self._eccentricity_integrals[rvk_scale](e_grid)


    def _calculate_function_values(self, x_grid, y_grid, workers):
        """
        More efficient evaluation over a grid of values.

        Args:
            x_grid:    The eccentricity values at which to evaluate the
                likelihood.

            y_grid:    The radial velocity semi-amplitude values for circular
                edge-on orbits at which to evaluate the likelihood.

        Returns:
            See `Approximate2DFunction._calculate_function_values()`.
        """

        assert x_grid.max() <= self._envelope_eccentricity

        new_rvk_scales = [
            rvk_scale for rvk_scale in y_grid
            if rvk_scale not in self._eccentricity_integrals
        ]
        new_eccentricity_integrals = workers.map(
            self._get_eccentircity_integral,
            new_rvk_scales
        )
        for rvk_scale, eccentricity_integral in zip(new_rvk_scales,
                                                    new_eccentricity_integrals):
            self._eccentricity_integrals[rvk_scale] = eccentricity_integral

        return numpy.dstack(
            workers.map(
                partial(self._evaluate_e_integral, x_grid),
                y_grid
            )
        )[0]


    def _get_integration_breaks(self, max_rv_semiamplitude, cdf_step=0.01):
        """Return array of points integration must hit for accuracy."""

        result = (
            self._observed_rvk.ppf(numpy.arange(cdf_step, 1.0, cdf_step))
            /
            max_rv_semiamplitude
        )
        return result[result < 1.0]


    def _upper_bound_equation(self, upper_bound, target_prob):
        """The equation to solve in order to find max RV semi-amplitude."""

        integral = integrate.quad(
            lambda s: (self._observed_rvk.sf(s * upper_bound)
                       /
                       numpy.sqrt(1.0 - numpy.square(s))),
            0,
            1,
            points=self._get_integration_breaks(upper_bound),
            **self._integration_options
        )
        self._logger.debug('Upper bound equation integral at %s: %s',
                           repr(upper_bound),
                           repr(integral))
        return integral[0] - target_prob


    def _get_rvk_upper_bound(self, target_prob):
        """Return RV semi-amplitude s.t. prob(larger values) < target_prob."""

        rvk_upper_bound = root_scalar(
            self._upper_bound_equation,
            args=(target_prob,),
            bracket=(
                0,
                (
                    self._observed_rvk.isf(target_prob / numpy.pi)
                    /
                    numpy.sin(target_prob / 2.0)
                )
            )
        )
        assert rvk_upper_bound.converged
        return rvk_upper_bound.root

    def _get_initial_grid(self):
        """Return an initial grid from which to start refining interpolation."""

        e_grid = self._observed_eccentricity.ppf(
            numpy.linspace(0, 1.0, self.configuration.min_grid_points[0])
        )
        assert e_grid[0] <= 0
        e_grid[0] = 0.0

        rvk_grid = self._observed_rvk.ppf(
            numpy.linspace(0, 1.0, self.configuration.min_grid_points[1])
        )
        assert rvk_grid[0] <= 0
        rvk_grid[0] = self.configuration.support[2]
        rvk_grid[-1] = self.configuration.support[3]

        return (
            e_grid[
                numpy.logical_and(
                    e_grid >= 0,
                    e_grid <= self._envelope_eccentricity
                )
            ],
            rvk_grid[rvk_grid >= 0]
        )

    def __init__(self,
                 *,
                 observed_rvk,
                 observed_eccentricity,
                 envelope_eccentricity,
                 max_discarded_probabiity,
                 min_grid_points=(100, 100),
                 tolerance=1e-6,
                 pickle_fname='rv_likelihood.pkl',
                 integration_options=None,
                 solve_ivp_options=None,
                 **approximation_options):
        """
        Configure the approximation.

        Args:
            observed_rvk(rv_continuous):    The empirical distribution of the RV
                semi-amplitude. Assumed independent of `observed_eccentricity`.

            observed_eccentricity(rv_continuous):    The empirical distribution
                of the orbital eccentricity. Assumed independent of
                `observed_rvk`.

            envelope_eccentricity(float):    The eccentricity of the envelope at
                evaluated for the current system.

            max_discarded_probabiity(float):    The tails of the RV
                semi-amplitude distribution with weight less than this are
                truncated to define the range for interpolation.

            min_grid_points:    See same name argument to
                `approximate_2d_function.__init__()`

            tolerance(float):    The maximum absolute deviation between the
                approximate value and directly numerically calculated one.

            pickle_fname:    See same name argument to
                `approximate_2d_function.__init__()`

            integration_options(dict):    Passed directly to
                :func:`integrate.quad()` to control integration. Must not
                include `points`.

            solve_ivp_options(dict):    Passed directly to
                :func:`integrate.solve_ivp()`, used to calculate the
                eccentricity dependence.

            **approximation_options:    Any additional options to pass to
                `approximate_2d_function.__init__()`

        Returns:
            None
        """

        self._logger.info(
            'Constructing RV likelihood for K = %s, e = %s, e_env = %s',
            repr(observed_rvk),
            repr(observed_eccentricity),
            repr(envelope_eccentricity)
        )

        self._observed_rvk = observed_rvk
        self._observed_eccentricity = observed_eccentricity
        self._envelope_eccentricity = envelope_eccentricity
        self._integration_options = integration_options or dict()
        self._solve_ivp_options = solve_ivp_options or dict()

        for opt_name, opt_value in [
                ('rtol', 1.0e-5),
                ('atol', 0.01 * tolerance),
                ('max_step', (envelope_eccentricity
                              /
                              min_grid_points[0]))
        ]:
            if opt_name not in self._solve_ivp_options:
                self._solve_ivp_options[opt_name] = opt_value

        self._eccentricity_integrals = dict()

        max_discarded_probabiity /= 4.0

        super().__init__(
            func='rv_likelihood',
            support=(
                0.0,
                envelope_eccentricity,
                observed_rvk.ppf(max_discarded_probabiity),
                self._get_rvk_upper_bound(max_discarded_probabiity)
            ),
            min_grid_points=min_grid_points,
            tolerance=tolerance,
            pickle_fname=pickle_fname,
            plot_labels=dict(function=r'$\lambda(e_{max}, K)$',
                             x=r'$e_{max}$',
                             y='K'),
            **approximation_options
        )
        self._eccentricity_integrals = None
