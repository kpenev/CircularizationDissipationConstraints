#!/usr/bin/env python3

"""Use dynesty to sample SB1 binary star system."""

from eccentricity_pdf import EccentricityPDFNormal
from prior_transform_sb1 import PriorTransformSB1
from log_likelihood_base import LogLikelihoodBase

class LogLikelihoodConstQ(LogLikelihoodBase):
    """SB1 binary log-likelihood assuming Q*'=const and same for both stars."""

    def _get_dissipation(self, parameters):
        """Return the dissipation argument for `find_evolution`."""

        dissipation = default_dissipation = dict(
            tidal_frequency_breaks=None,
            spin_frequency_breaks=None,
            tidal_frequency_powers=numpy.array([0.0]),
            spin_frequency_powers=numpy.array([0.0])
            reference_phase_lag=phase_lag(
                self.get_parameter_value(parameters, 'lgQ')
            )
        )

        return dict(primary=dissipation,
                    secondary=dict(dissipation))

    def __init__(self, system_data, config):
        """Set-up the log-likelihood for the given sysem with given config."""

        self.interpolator = StellarEvolutionManager(
            config.stellar_evolution_interpolator_dir
        ).get_interpolator_by_name(
            'default'
        )

        super().__init__(interpolator=self.interpolator,
                         eccentricity_pdf=<++>,
                         secondary_is_star=True,
                         initial_eccentricity=config.initial_eccentricity)

if __name__ == '__main__':
