#!/usr/bin/env python3

"""Find best fit parametric model for the P-e envelope given simulations."""

from abc import ABC, abstractmethod

from .unpickle_data import unpickle_data

class EnvelopeModelBase(ABC):
    """Base class for models of the period-eccentricity envelope."""

    def __init__(self, simulations, max_reliable_eccentricity):
        """
        Prepare the envelope model for fitting the given simulation data.

        Args:
            simulations(list):    The simulated envelope data to fit. Usually
                list of unpickle_data results.

            max_reliable_eccentricity(float):    When the simulated envelope
                goes above this value it is no longer used in the fit to allow
                for the effect that we cannot calculate evolutions with
                arbitrarily large initial eccentricity.
        """

        self.simulations = simulations
        self.max_reliable_eccentricity = max_reliable_eccentricity

    @abstractmethod
    def max_eccentricity(self, orbital_period, model_parameters):
        """Return the envelope eccentricity given period and model params."""

    def sum_square_difference(self, model_parameters):
        """Return sum square of difference between model and simulations."""

def meibom_matthieu_2005_model(params):
    """Model from Meibom % Matthieu 2005, param order P', alpha, beta."""

    pcirc, alpha, beta = params


