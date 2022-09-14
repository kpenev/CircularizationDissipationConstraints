"""Define a base class for the log-likelihoods of binary stars systems."""

import logging
from abc import ABCMeta

import numpy
from astropy import units

from orbital_evolution.transformations import phase_lag

from bayesian.log_likelihood_base import LogLikelihoodBase

class LogLikelihoodBinaryStars(LogLikelihoodBase, metaclass=ABCMeta):
    """Base class for log-likelihood functions for binary star analysis."""

    def _get_dissipation(self, parameters):

        star_dissipation = dict(
            spin_frequency_breaks=None,
            spin_frequency_powers=numpy.array([0.0]),
            reference_phase_lag=phase_lag(
                self.get_parameter_value(parameters, 'lgQ_min')
                +
                self.get_parameter_value(parameters, 'lgQ_inertial_boost')
            ),
            inertial_mode_enhancement=10.0**(
                self.get_parameter_value(parameters, 'lgQ_inertial_boost')
            ),
            inertial_mode_sharpness=self.get_parameter_value(
                parameters,
                'lgQ_inertial_sharpness'
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
                star_dissipation['tidal_frequency_breaks'] = numpy.array([
                    2.0 * numpy.pi / 50.0,
                    break_frequency
                ])
                star_dissipation['tidal_frequency_powers'] = numpy.array([
                    0.0,
                    powerlaw,
                    0.0
                ])
                star_dissipation['reference_phase_lag'] *= numpy.power(
                    star_dissipation['tidal_frequency_breaks'][0]
                    /
                    star_dissipation['tidal_frequency_breaks'][1],
                    powerlaw
                )
            else:
                star_dissipation['tidal_frequency_breaks'] = numpy.array([
                    break_frequency
                ])
                star_dissipation['tidal_frequency_powers'] = numpy.array([
                    0.0,
                    powerlaw
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
                 powerlaw_dissipation,
                 max_dissipative_mstar=1.2 * units.M_sun,
                 **parent_kwargs):

        self.max_dissipative_mstar = max_dissipative_mstar
        dissipation_parameters = [
            ('lgQ_min', units.dimensionless_unscaled),
            ('lgQ_inertial_boost', units.dimensionless_unscaled),
            ('lgQ_inertial_sharpness', units.dimensionless_unscaled),
        ]
        if powerlaw_dissipation:
            dissipation_parameters.extend([
                ('lgQ_break_period', units.day),
                ('lgQ_powerlaw', units.dimensionless_unscaled)
            ])

        super().__init__(
            *parent_args,
            secondary_is_star=True,
            dissipation_parameters=dissipation_parameters,
            **parent_kwargs
        )
