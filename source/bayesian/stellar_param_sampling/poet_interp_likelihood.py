"""Define [Fe/H] conditional likelihood based on POET stellar evolution."""

import numpy

from feh_conditional_likelihood_base import FeHConditionalLikelihoodBase
from stellar_evolution.change_variables import QuantityEvaluator

class POETInterpLikelihood(FeHConditionalLikelihoodBase):
    """The likelihood of stellar measurements other than [Fe/H]."""

    _stellar_quantities = ('logg', 'teff', 'rho')

    def _age_cdf_integrand(self, age, _, mass, feh):

        result = 1.0
        for quantity, measurement in self._measurements.items():
            predicted = getattr(self._evaluate, quantity)(mass, age, feh)
            if numpy.isnan(predicted):
                return 0.0
            result *= measurement.pdf(predicted)
        return result

    def __getstate__(self):
        """Add distribution to parent's pickled state."""

        return super().__getstate__(), self._measurements

    def __setstate__(self, state):
        """Set the distribution and pass to parent to finish unpickling."""

        super().__setstate__(state[0])
        self._measurements = state[1]
        self._evaluate = QuantityEvaluator(self.interpolator)

    def __eq__(self, other):

        if not isinstance(other, type(self)):
            return False
        if self._measurements.keys() != other._measurements.keys():
            return False
        for quantity in self._measurements:
            if (
                    self._measurements[quantity].kwds
                    !=
                    other._measurements[quantity].kwds
            ):
                return False
        return super().__eq__(other)

    def __init__(self, **kwargs):
        """
        Set likelihood based on measured stellar properties.

        Args:
            kwargs:    The collection of measurements available for this
                star as well as any keyword arguments to pass to
                :func:`solve_ivp` when calculating the integral of the age PDF.
                stellar parameters allowed are:

                    * `'logg'`: log10(surface gravity of the star in cgs)

                    * `'teff'`: The effective temperature of the star in Kelvin

                    * `'rho'`: The mean density of the star in g/cm3

                Each stellar parameter should be specified as a numpy 1-D
                probability distribution.

                All other keys are passed directly to parent`s :func:`__init__`.
        """

        self._measurements = dict()
        for quantity in self._stellar_quantities:
            if quantity in kwargs:
                self._measurements[quantity] = kwargs[quantity]
                del kwargs[quantity]

        super().__init__(**kwargs)

        self._evaluate = QuantityEvaluator(self.interpolator)
