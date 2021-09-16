"""Implement sampling of binary star masses."""

from abc import ABC, abstractmethod

import numpy
from scipy import integrate, optimize

from conditional_secondary_mass_distribution import\
    ConditionalSecondaryMassDistribution

class SampleBinaryMasses(ABC):
    """Allow sampling from joint distribution of the two masses in a binary."""

    def _check_constraints(self, primary_mass, secondary_mass):
        """Return True iff the given masses are allowed."""

        if (
                min(primary_mass, secondary_mass) < self.mass_range[0]
                or
                max(primary_mass, secondary_mass) > self.mass_range[1]
        ):
            return False

        return True


    @abstractmethod
    def _joint_likelihood(self, secondary_mass, primary_mass, return_log=False):
        """The likelihood to sample from (proportional to joint PDF)."""


    @abstractmethod
    def _secondary_mass_quad_points(self, primary_mass):
        """Points arg to quad required for reliable marginalization oven m2."""


    def _primary_mass_likelihood(self, primary_mass, *_):
        """The derivative of cumulative m1 likelihood."""

        return integrate.quad(
            self._joint_likelihood,
            self.mass_range[0],
            primary_mass,
            args=(primary_mass,),
            points=self._secondary_mass_quad_points(primary_mass),
            **self._m2_integration_config
        )[0]


    def _get_cumulative_m1_likelihood(self):
        """Return scaled version of CDF(m1) marginalized over m2."""

        result = integrate.solve_ivp(
            self._primary_mass_likelihood,
            self.mass_range,
            numpy.array([0.0]),
            dense_output=True,
            **self._m1_solve_ivp_config
        )
        assert result.success
        return result.sol


    @abstractmethod
    def _check_for_pickled(self, pickle_fname):
        """Check if given file contains a re-usable pickle for current setup."""


    @abstractmethod
    def _add_to_pickle_file(self, pickle_fname):
        """Pickle a fully set-up constaint to the given file for fast re-use."""


    def __init__(self,
                 mass_range,
                 m2_integration_config,
                 m1_solve_ivp_config,
                 pickle_fname):
        """Set-up the constraint."""

        self._m1_solve_ivp_config = dict(m1_solve_ivp_config)
        self._m2_integration_config = m2_integration_config
        self.mass_range = mass_range

        for kwarg, default in [('max_step', 1e-2),
                               ('method', 'LSODA')]:
            if kwarg not in self._m1_solve_ivp_config:
                self._m1_solve_ivp_config[kwarg] = default


        for kwarg, default in [('limit', 200)]:
            if kwarg not in self._m2_integration_config:
                self._m2_integration_config[kwarg] = default

        self._cumulative_m1_likelihood = self._check_for_pickled(pickle_fname)
        if self._cumulative_m1_likelihood is None:
            self._cumulative_m1_likelihood = (
                self._get_cumulative_m1_likelihood()
            )
            self._add_to_pickle_file(pickle_fname)

        self._norm = self._cumulative_m1_likelihood(self.mass_range[1])[0]
        self._secondary_norm = None


    def pdf(self, primary_mass, secondary_mass):
        """The joint PDF of component masses given observed colors and mags."""

        primary_mass = float(primary_mass)
        secondary_mass = float(secondary_mass)

        if(
                secondary_mass > primary_mass
                or
                secondary_mass < self.mass_range[0]
                or
                primary_mass > self.mass_range[1]
        ):
            return 0.0
        return (
            self._joint_likelihood(secondary_mass, primary_mass)
            /
            self._norm
        )


    def logpdf(self, primary_mass, secondary_mass):
        """Natural log of the joint PDF."""

        if(
                secondary_mass > primary_mass
                or
                secondary_mass < self.mass_range[0]
                or
                primary_mass > self.mass_range[1]
        ):
            return -numpy.inf
        result = (
            self._joint_likelihood(secondary_mass, primary_mass, True)
            -
            numpy.log(self._norm)
        )
        return result


    def primary_mass_pdf(self, primary_mass):
        """The PDF(m1) marginalized over m2."""

        return self._primary_mass_likelihood(primary_mass) / self._norm


    def primary_mass_cdf(self, primary_mass):
        """The CDF(m1) marginalized over m2."""

        return self._cumulative_m1_likelihood(primary_mass)[0] / self._norm


    def primary_mass_ppf(self, quantile):
        """Find the primary mass for which the CDF matches given quantile."""

        def equation(primary_mass):
            return self.primary_mass_cdf(primary_mass) - quantile

        return optimize.root_scalar(equation, bracket=self.mass_range).root


    def get_conditional_secondary_mass_distribution(self, primary_mass):
        """Return scipy style RV for the secondary mass given primary mass."""

        return ConditionalSecondaryMassDistribution(self, primary_mass)

