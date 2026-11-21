"""Gaia observation adapter; inherit the existing orbital likelihood algorithm."""
import math
import numpy as np
from astropy import units

from bayesian.log_likelihood_windemuth_et_al import LogLikelihoodWindemuth
from bayesian.gaia_observations import conditional_eccentricity, eccentricity_distribution


class GaiaLogLikelihood(LogLikelihoodWindemuth):
    def __init__(self, *, observation, **kwargs):
        self.gaia_observation=observation
        super().__init__(**kwargs)

    def calculate_log_likelihood(self, encoded_parameters, **other_args):
        period=self.get_parameter_value(encoded_parameters,'orbital_period').to_value(units.day)
        radii=[self.get_parameter_value(encoded_parameters,'cmd_'+c+'_radius').to_value(units.R_sun)
               for c in ('primary','secondary')]
        if not math.isfinite(period) or period<=0 or not all(math.isfinite(r) and r>0 for r in radii):
            return -np.inf
        mean,sigma=conditional_eccentricity(self.gaia_observation,period)
        try:
            self._de=eccentricity_distribution(mean,sigma)
        except ValueError:
            return -np.inf
        self._observed_eccentricity_distro=self._de
        self.envelope_weights=self._de.cdf(self.envelope_eccentricity)
        self.system_eccentricity=float(self._de.ppf(.5))
        return super().calculate_log_likelihood(encoded_parameters,**other_args)
