"""Define a base log-likelihood class that assumes constant phase lag."""

import logging

import numpy
from astropy import units

from orbital_evolution.transformations import phase_lag
from binary_utils import rv_semi_amplitude_scale

from bayesian.log_likelihood_base import LogLikelihoodBase

#Intended to function just as callable, so no need for other public methods
#pylint: disable=too-few-public-methods
class LogLikelihoodSB1(LogLikelihoodBase):
    """
    Base class for log-likelihood assuming powerlaw phase lag with saturation.
    """

    def _get_dissipation(self, parameters):

        star_dissipation = dict(
            spin_frequency_breaks=None,
            spin_frequency_powers=numpy.array([0.0]),
            reference_phase_lag=phase_lag(
                self.get_parameter_value(parameters, 'lgQ_min')
            )
        )
        if 'lgQ_break_period' in self.parameter_indices:
            break_frequency = (2.0 * numpy.pi
                               /
                               self.get_parameter_value(
                                   parameters,
                                   'lgQ_break_period'
                               ).to_value(units.day))
            powerlaw = self.get_parameter_value(parameters, 'lgQ_powerlaw')
            if powerlaw > 0:
                star_dissipation['tidal_frequency_powers'] = numpy.array([
                    powerlaw,
                    0.0
                ])
                star_dissipation['tidal_frequency_breaks'] = numpy.array([
                    break_frequency
                ])
            else:
                star_dissipation['tidal_frequency_powers'] = numpy.array([
                    1.0,
                    0.0,
                    powerlaw
                ])
                star_dissipation['tidal_frequency_breaks'] = numpy.array([
                    2.0 * numpy.pi / 50.0,
                    break_frequency
                ])
        else:
            star_dissipation['tidal_frequency_breaks'] = None
            star_dissipation['tidal_frequency_powers'] = numpy.array([0.0])

        dissipation = dict()
        #Avoiding lgQ actually decreases readability
        #pylint: disable=invalid-name
        for component in ['primary', 'secondary']:
            if (
                    self.get_parameter_value(parameters, component + '_mass')
                    >
                    self.max_dissipative_mstar
            ):
                dissipation[component] = None
            else:
                dissipation[component] = dict(star_dissipation)
        #pylint: enable=invalid-name

        logging.getLogger(__name__).debug('Dissipation: %s', repr(dissipation))

        return dissipation

    def __init__(self,
                 *parent_args,
                 rv_likelihood,
                 powerlaw_dissipation,
                 max_dissipative_mstar=1.2 * units.M_sun,
                 **parent_kwargs):

        self.max_dissipative_mstar = max_dissipative_mstar
        self._rv_likelihood = rv_likelihood

        dissipation_parameters = [
            ('lgQ_min', units.dimensionless_unscaled),
            ('lgQ_inertial_boost', units.dimensionless_unscaled),
        ]
        if powerlaw_dissipation:
            dissipation_parameters.extend([
                ('lgQ_break_period', units.day),
                ('lgQ_powerlaw', units.dimensionless_unscaled)
            ])
        super().__init__(
            *parent_args,
            envelope_eccentricity=rv_likelihood.envelope_eccentricity,
            secondary_is_star=True,
            dissipation_parameters=dissipation_parameters,
            **parent_kwargs
        )

    def calculate_log_likelihood(self, parameters):
        """Evaluate the log-likelihood at the given model parameters."""

        final_eccentricity = super().calculate_final_eccentricity(parameters)

        if (
                final_eccentricity is None
                or
                final_eccentricity > self.envelope_eccentricity
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
