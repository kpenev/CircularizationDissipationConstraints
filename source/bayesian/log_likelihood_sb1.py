"""Define a base log-likelihood class that assumes constant phase lag."""

import numpy
from astropy import units

from orbital_evolution.transformations import phase_lag

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
        if 'lgQ_break_period' in parameters:
            star_dissipation['tidal_frequency_breaks'] = numpy.array([
                2.0 * numpy.pi
                /
                self.get_parameter_value(
                    parameters,
                    'lgQ_break_period'
                ).to_value(units.day)
            ])
            powerlaw = self.get_parameter_value(parameters, 'lgQ_powerlaw')
            if powerlaw > 0:
                star_dissipation['tidal_frequency_powers'] = numpy.array([
                    powerlaw,
                    0.0
                ])
            else:
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
                dissipation[component] = dict(dissipation)
        #pylint: enable=invalid-name

        return dissipation

    def __init__(self,
                 *parent_args,
                 rv_semiamplitude_constraint,
                 max_dissipative_mstar=1.2 * units.M_sun,
                 **parent_kwargs):

        self.max_dissipative_mstar = max_dissipative_mstar
        self._rv_semiamplitude_constraint = rv_semiamplitude_constraint

        super().__init__(*parent_args,
                         dissipation_parameters=[
                             ('lgQ_min', units.dimensionless_unscaled),
                             ('lgQ_break_period', units.day),
                             ('lgQ_powerlaw', units.dimensionless_unscaled)
                         ],
                         secondary_is_star=True,
                         **parent_kwargs)

    def __call__(self, parameters):
        """Evaluate the log-likelihood at the given model parameters."""

        circularization_likelihood = super().__call__(parameters)
        mass_kwargs = dict(
            primary_mass=self.get_parameter_value(parameters,
                                                  'primary_mass'),
            secondary_mass=self.get_parameter_value(parameters,
                                                    'secondary_mass'),
        )
        return circularization_likelihood * (
            self._rv_semiamplitude_constraint.rv_semi_amplitude_pdf(
                eccentricity=self.final_eccentricity,
                **mass_kwargs
            )
            /
            self._rv_semiamplitude_constraint.rv_semi_amplitude_pdf(
                eccentricity=self.initial_eccentricity,
                **mass_kwargs
            )
        )

#pylint: enable=too-few-public-methods
