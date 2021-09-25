"""Define class for sampling from mass priors for SB1 systems."""

import logging

import numpy
from scipy import optimize
from astropy import units

from binary_utils import rv_semi_amplitude_scale, calculate_secondary_mass
from sample_binary_masses import SampleBinaryMasses

class SampleSB1Masses(SampleBinaryMasses):
    r"""
    Implement sampling from the priors of SB1 masses given age and [Fe/H].

    Sample from the joint PDF:

    .. math::
        f\left(M_1, M_2|t, \left[\frac{Fe}{H}\right] \right)
        \propto
        f_{phot}\left(M_1, M_2, t, \left[\frac{Fe}{H}\right]\right)
        \lambda\left[e_{env}, K_0(M_1, M_2, P_{orb})\right]
    """

    _logger = logging.getLogger(__name__)

    def joint_likelihood(self, secondary_mass, primary_mass, return_log=False):
        """The likelihood function we are sampling from."""

        if secondary_mass > primary_mass:
            return -numpy.inf if return_log else 0.0

        rv_likelihood = float(
            self._rv_likelihood(
                self._envelope_eccentricity,
                rv_semi_amplitude_scale(primary_mass,
                                        secondary_mass,
                                        self._orbital_period)
            )
        )

        if return_log:
            return (
                self._photometric_constraint.logpdf(primary_mass,
                                                    secondary_mass)
                +
                numpy.log(rv_likelihood)
            )

        return (
            self._photometric_constraint.pdf(primary_mass, secondary_mass)
            *
            rv_likelihood
        )

    def secondary_mass_quad_points(self, primary_mass):
        """
        Return points quad kwarg required for relaible marginalization over m2.

        Args:
            primary_mass(float):    The primary mass at which the marginalized
                PDF is being calculated in solar masses.
        """

        return numpy.concatenate(
            (
                self._photometric_constraint.secondary_mass_quad_points(
                    primary_mass
                ),
                [
                    calculate_secondary_mass(
                        primary_mass * units.M_sun,
                        self._orbital_period * units.day,
                        rvk * units.m / units.s
                    ).to_value(units.M_sun)
                    for rvk in self._rv_likelihood.observed_rvk.ppf(
                        numpy.linspace(0.01, 0.99, 99)
                    )
                ]
            )
        )


    def _init_max_likelihood(self):
        """Return the max likelihood M1, M2, and likelihood value."""

        guess_likelihood = -numpy.inf
        for primary_mass in numpy.linspace(
                *self._photometric_constraint.mass_range,
                100
        ):
            secondary_masses = self.secondary_mass_quad_points(primary_mass)
            likelihoods = numpy.vectorize(
                self.joint_likelihood
            )(
                secondary_masses,
                primary_mass
            )
            best_index = numpy.argmax(likelihoods)
            if likelihoods[best_index] > guess_likelihood:
                m1_guess = primary_mass
                m2_guess = secondary_masses[best_index]
                guess_likelihood = likelihoods[best_index]

        min_result = optimize.minimize(
            fun=lambda x: -self.joint_likelihood(*x),
            x0=[m2_guess, m1_guess],
            bounds=optimize.Bounds(*self._photometric_constraint.mass_range,
                                   keep_feasible=True),
            options=dict(maxiter=1e6, disp=False)
        )
        assert min_result.success
        self._logger.debug('Likelihood maximization result: %s',
                           repr(min_result))
        self.max_likelihood = dict(m2=min_result.x[0],
                                   m1=min_result.x[1],
                                   likelihood=float(-min_result.fun))


    def __init__(self,
                 *,
                 rv_likelihood,
                 photometric_constraint,
                 orbital_period,
                 envelope_eccentricity,
                 pickle_fname='sample_sb1_masses.pkl',
                 quad_precision=None):
        """Set up sampling given RV and photometric likelihoods."""

        self._rv_likelihood = rv_likelihood
        self._photometric_constraint = photometric_constraint
        self._orbital_period = orbital_period
        self._envelope_eccentricity = envelope_eccentricity
        if quad_precision is None:
            self._quad_precision = dict()
        else:
            self._quad_precision = dict(epsabs=quad_precision[0],
                                        epsrel=quad_precision[1])

        self._init_max_likelihood()

        super().__init__(
            mass_range=photometric_constraint.mass_range,
            m2_integration_config=self._quad_precision,
            m1_solve_ivp_config=dict(
                rtol=1e-6,
                atol=1e-9 * self.max_likelihood['likelihood'],
            ),
            pickle_fname=pickle_fname,
            likelihood_pickle_entries=(
                rv_likelihood,
                photometric_constraint,
                orbital_period,
                envelope_eccentricity,
                quad_precision
            )
        )
