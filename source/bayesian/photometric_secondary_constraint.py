"""Define the secondary mass distribution from photometry given primary mass."""

from scipy import integrate, optimize
from scipy.stats import rv_continuous
import numpy

class PhotometricSecondaryConstraint(rv_continuous):
    """
    Distribution of M2 given M1 from photometric measuremnents.

    Partially implements the interface of `scipy.stats.rv_frozen`.

    Pre-computes whatever possible to speed-up the provided methods."
    """

    #base class adapats to signature
    #pylint: disable=arguments-differ
    def _pdf(self, secondary_mass):
        """PDF(m2 | m1)."""

        return (
            self._joint_constraint(self._primary_mass, secondary_mass)
            /
            self._norm
        )

    def _logpdf(self, secondary_mass):
        """Natural log of PDF(m2 | m1)."""

        return (
            self._joint_constraint(self._primary_mass, secondary_mass)
            -
            numpy.log(self._norm)
        )

    def _cdf(self, secondary_mass):
        """CDF(m2 | m1)."""

        return self._cumulative_likelihood(secondary_mass) / self._norm
    #pylint: enable=arguments-differ

    def _cumulative_ode(self, secondary_mass, _):
        """Suitable function for `solve_ivp` to find cumulative likelihood."""

        return self._joint_constraint.pdf(self._primary_mass, secondary_mass)

    def __init__(self, joint_constraint, primary_mass):
        """Set-up the conditinoal secondary mass distrib. given a joint one."""

        super().__init__(self, a=joint_constraint.mass_range[0], b=primary_mass)

        self._joint_constraint = joint_constraint
        self._primary_mass = primary_mass
        minimization = optimize.minimize_scalar(
            lambda secondary_mass: -joint_constraint.logpdf(primary_mass,
                                                            secondary_mass),
            bounds=self.support(),
            method='bounded'
        )
        assert minimization.success
        solved_ode = integrate.solve_ivp(
            self._cumulative_ode,
            self.support(),
            numpy.array([0.0]),
            dense_output=True,
            max_step=1e-2,
            rtol=1e-6,
            atol=1e-9 * numpy.exp(minimization.fun),
            method='LSODA'
        )
        assert solved_ode.success
        self._cumulative_likelihood = solved_ode.sol
        self._norm = self._cumulative_likelihood(primary_mass)
