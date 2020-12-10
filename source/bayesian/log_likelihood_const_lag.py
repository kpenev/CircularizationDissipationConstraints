"""Define a base log-likelihood class that assumes constant phase lag."""

import numpy
from astropy import units

from orbital_evolution.transformations import phase_lag

from log_likelihood_base import LogLikelihoodBase

class LogLikelihoodConstLag(LogLikelihoodBase):
    """
    Base class for log-likelihood assuming constant phase lag dissipation.
    """

    def _get_dissipation(self, parameters):

        default_dissipation = dict(
            tidal_frequency_breaks=None,
            spin_frequency_breaks=None,
            tidal_frequency_powers=numpy.array([0.0]),
            spin_frequency_powers=numpy.array([0.0])
        )

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
                dissipation[component] = dict(
                    default_dissipation,
                    reference_phase_lag=phase_lag(
                        self.get_parameter_value(parameters,
                                                 'lgQ_' + component)
                    )
                )
        #pylint: enable=invalid-name

        return dissipation

    def __init__(self,
                 *parent_args,
                 max_dissipative_mstar=1.2 * units.M_sun,
                 **parent_kwargs):

        self.parameter_names_units['dissipation'] = [('lgQ_primary', 1),
                                                     ('lgQ_secondary', 1)]

        self.max_dissipative_mstar = max_dissipative_mstar

        super().__init__(*parent_args, **parent_kwargs)
