"""Define base class for stellar parameter log-likelihood functions."""

from abc import ABC, abstractmethod

from scipy.integrate import solve_ivp
from astropy import units as u

class LogLikelihoodBase(ABC):
    """Common interface for log-likelihood functions of stellar paramseters."""

    @abstractmethod
    def _age_cdf_integrand(self, age, _, mass, feh):
        """Unnormalized posterior PDF excluding direct [Fe/H] measurement."""

    def __init__(self,
                 interpolator,
                 mass=1.0 * u.M_sun,
                 feh=0.0,
                 **solve_ivp_options):
        """
        Prepare the log-likelihood for use.

        Args:
            interpolator:     The stellar evolution interpolator on which the
                log-likelihood is based.

            mass(astropy quantity):     The default mass to use when evaluating
                the various PDF and CDF methods.

            feh(float):     The default [Fe/H] to assume when evaluating the
                various PDF and CDF methods.

            solve_ivp_options:     Directly passed to :func:`solve_ivp` when
                calculating the integral of the age PDF.
        """

        self.interpolator = interpolator
        self.default_mass = mass
        self.default_feh = feh
        self._solve_ivp_options = solve_ivp_options

        self._age_integrals = dict()

    def __setattr__(self, name, value):
        """Return the default mass assumed by conditional distributions."""

        if name == 'defalut_mass':
            assert (self.interpolator.track_masses[0]
                    <
                    value
                    <
                    self.interpolator.track_masses[-1])
        elif name == 'default_feh':
            assert (self.interpolator.track_feh[0]
                    <
                    value
                    <
                    self.interpolator.track_feh[-1])
        super().__setattr__(name, value)

    def age_integral(self, mass=None, feh=None):
        """
        Return age integral re-using results when possible.

        If mass and/or feh are not None, the default values are used.
        """

        if mass is None:
            mass = self.default_mass

        if feh is None:
            feh = self.default_feh

        if (mass, feh) not in self._age_integrals:
            radius = self.interpolator('radius',
                                       mass.to_value(u.M_sun),
                                       feh)
            solution = solve_ivp(self._age_cdf_integrand,
                                 (radius.min_age, radius.max_age),
                                 [0.0],
                                 args=(mass.to_value(u.M_sun), feh),
                                 dense_output=True,
                                 **self._solve_ivp_options)
            assert solution.success
            self._age_integrals[(mass, feh)] = solution.sol

        return self._age_integrals[(mass, feh)]
