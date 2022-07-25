"""Define a log-likelihood class for cluster SB1 systems."""

import numpy
from astropy import units

from general_purpose_python_modules.binary_utils import rv_semi_amplitude_scale

from bayesian.log_likelihood_binary_stars import LogLikelihoodBinaryStars


#Intended to function just as callable, so no need for other public methods
#pylint: disable=too-few-public-methods
class LogLikelihoodSB1(LogLikelihoodBinaryStars):
    """Log likelihood appropriate for cluster SB1 biaries."""

    def __init__(self,
                 *parent_args,
                 rv_likelihood,
                 **parent_kwargs):

        self._rv_likelihood = rv_likelihood

        super().__init__(
            *parent_args,
            envelope_eccentricity=rv_likelihood.envelope_eccentricity,
            **parent_kwargs
        )

    def calculate_log_likelihood(self, parameters, **_):
        """Evaluate the log-likelihood at the given model parameters."""

        final_eccentricity = super().calculate_final_eccentricity(parameters)

        if (
                final_eccentricity is None
                or
                not (final_eccentricity <= self.envelope_eccentricity)
        ):
            return -numpy.inf

        rvk_scale = rv_semi_amplitude_scale(
            self.get_parameter_value(parameters, 'primary_mass'),
            self.get_parameter_value(parameters, 'secondary_mass'),
            self.get_parameter_value(parameters, 'orbital_period'),
        ).to_value(units.m / units.s)


        return (
            numpy.log(
                self._rv_likelihood(final_eccentricity, rvk_scale)
            )
            -
            numpy.log(
                self._rv_likelihood(self.envelope_eccentricity, rvk_scale)
            )
        )

#pylint: enable=too-few-public-methods
