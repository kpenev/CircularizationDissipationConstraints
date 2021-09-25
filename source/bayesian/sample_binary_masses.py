"""Implement sampling of binary star masses."""

from abc import abstractmethod
import logging

import numpy
from scipy import integrate, optimize, stats

from conditional_secondary_mass_distribution import\
    ConditionalSecondaryMassDistribution
from picklable import Picklable
from basic_util import compare_frozen_distributions

class SampleBinaryMasses(Picklable):
    """Allow sampling from joint distribution of the two masses in a binary."""

    _logger = logging.getLogger(__name__)

    @abstractmethod
    def joint_likelihood(self, secondary_mass, primary_mass, return_log=False):
        """The likelihood to sample from (proportional to joint PDF)."""


    @abstractmethod
    def secondary_mass_quad_points(self, primary_mass):
        """Points arg to quad required for reliable marginalization oven m2."""


    def primary_mass_likelihood(self, primary_mass, *_):
        """The derivative of cumulative m1 likelihood."""

        return integrate.quad(
            self.joint_likelihood,
            self.mass_range[0],
            primary_mass,
            args=(primary_mass,),
            points=self.secondary_mass_quad_points(primary_mass),
            **self._m2_integration_config
        )[0]


    def get_cumulative_m1_likelihood(self):
        """Return scaled version of CDF(m1) marginalized over m2."""

        result = integrate.solve_ivp(
            self.primary_mass_likelihood,
            self.mass_range,
            numpy.array([0.0]),
            dense_output=True,
            **self._m1_solve_ivp_config
        )
        assert result.success
        return result.sol


    def _check_pickle(self):
        """Return None if given pickle does not match otherwise unpickle."""

        for entry in self._likelihood_pickle_entires:
            self._logger.debug('Comparing %s to pickled', repr(entry))
            if isinstance(entry, type(stats.uniform())):
                if not compare_frozen_distributions(entry,
                                                    self._load_pickle_object()):
                    self._logger.debug('Distribution mismatch')
                    return None
                self._logger.debug('Distribution match')
            elif entry != self._load_pickle_object():
                self._logger.debug('Simple mismatch')
                return None
            else:
                self._logger.debug('Simple match')

        for attr_name in ['mass_range',
                          '_m1_solve_ivp_config',
                          '_m2_integration_config']:
            if getattr(self, attr_name) != self._load_pickle_object():
                return None

        return self._load_pickle_object()


    def __eq__(self, other):
        """Return True iff self and other would sample fro identical distros."""

        for attr_name in ['_m1_solve_ivp_config',
                          '_m2_integration_config',
                          '_likelihood_pickle_entires']:
            if getattr(self, attr_name) != getattr(other, attr_name):
                return False
        return True


    def __init__(self,
                 *,
                 mass_range,
                 m2_integration_config,
                 m1_solve_ivp_config,
                 likelihood_pickle_entries,
                 pickle_fname):
        """Set-up the constraint."""


        self._m1_solve_ivp_config = dict(m1_solve_ivp_config or dict())
        self._m2_integration_config = m2_integration_config
        self.mass_range = mass_range

        for kwarg, default in [('max_step', 1e-2),
                               ('method', 'LSODA')]:
            if kwarg not in self._m1_solve_ivp_config:
                self._m1_solve_ivp_config[kwarg] = default


        for kwarg, default in [('limit', 200)]:
            if kwarg not in self._m2_integration_config:
                self._m2_integration_config[kwarg] = default

        self._likelihood_pickle_entires = likelihood_pickle_entries
        super().__init__(len(likelihood_pickle_entries) + 4)

        self._cumulative_m1_likelihood = self.check_for_pickled(pickle_fname)
        if self._cumulative_m1_likelihood is None:
            self._cumulative_m1_likelihood = (
                self.get_cumulative_m1_likelihood()
            )
            self.add_to_pickle_file(pickle_fname,
                                    *likelihood_pickle_entries,
                                    self.mass_range,
                                    self._m1_solve_ivp_config,
                                    self._m2_integration_config,
                                    self._cumulative_m1_likelihood)

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
            self.joint_likelihood(secondary_mass, primary_mass)
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
            self.joint_likelihood(secondary_mass, primary_mass, True)
            -
            numpy.log(self._norm)
        )
        return result


    def primary_mass_pdf(self, primary_mass):
        """The PDF(m1) marginalized over m2."""

        return self.primary_mass_likelihood(primary_mass) / self._norm


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

    def __call__(self, m1_cdf_value, m2_cdf_value):
        """
        Transform 2 U(0, 1) idrv to primray & secondary mass per constraints.

        Args:
            m1_cdf_value(float):    The primary mass return is such that the CDF
                of the CDF of the primary mass, marginalized over the secondary
                mass, has this value.

            m2_cdf_value (float):    The secondary mass returned is such that
                the conditional CDF of the secondary mass given the chosen value
                of the primary mass above has this value.

        Returns:
            float, float:
                The two masses that correspond to the given U(0, 1) random
                variates given.
        """

        primary_mass = self.primary_mass_ppf(m1_cdf_value)
        secondary_mass = self.get_conditional_secondary_mass_distribution(
            primary_mass
        ).ppf(
            m2_cdf_value
        )

        return primary_mass, secondary_mass
