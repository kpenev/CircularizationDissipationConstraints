"""Implement the Meibom & Matthieu 2005 model for P-e envelope."""

import numpy

from .envelope_residual_base import EnvelopeResidualBase

class MeibomMathieuEnvelopeResidual(EnvelopeResidualBase):
    """Use Eq. 1 from Meibom & Matthieu 2005 as the envelope model."""

    def max_eccentricity(self, orbital_period, model_parameters, sim_config):

        circularization_period, alpha, beta, gamma = model_parameters

        return numpy.maximum(
            0.0,
            alpha * (
                1.0
                -
                numpy.exp(beta * (circularization_period - orbital_period))
            )
        )**gamma
