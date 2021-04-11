#!/usr/bin/env python3

"""Find best fit parametric model for the P-e envelope given simulations."""

from abc import ABC, abstractmethod

import numpy

class EnvelopeResidualBase(ABC):
    """Callable to fit a model of the period-eccentricity envelope."""

    def __init__(self, simulations, fit_eccentricity_range, age_index=None):
        """
        Prepare the envelope model for fitting the given simulation data.

        Args:
            simulations(list):    The simulated envelope data to fit. Usually
                list of unpickle_data results.

            fit_eccentricity_range(float, float):    When the simulated envelope
                goes beyond this range it is no longer used in the fit to allow
                for the effect that we cannot calculate evolutions with
                arbitrarily large initial eccentricity and that some envelope
                models assume a fixed eccentricity at short periods.
        """


        self._simulations = simulations
        envelope_initial_eccentricity = simulations[0][0].eccentricity_grid[-1]
        simulated_ages = simulations[0][0].plot_ages
        for sim_config, _ in simulations:
            assert (sim_config.eccentricity_grid[-1]
                    ==
                    envelope_initial_eccentricity)
            assert (sim_config.plot_ages == simulated_ages).all()

        self.fit_eccentricity_range = fit_eccentricity_range
        self._age_index = age_index

    def get_simulated_ages(self):
        """Return the ages at which simulations have tabulated the evolution."""

        return self._simulations[0][0].plot_ages

    def set_age_index(self, age_index):
        """Set the index in the simulated ages to calculate residuals for."""

        assert age_index > 0
        assert age_index < self._simulations[0][0].plot_ages.size
        self._age_index = age_index

    @abstractmethod
    def max_eccentricity(self, orbital_period, model_parameters, sim_config):
        """
        Return the envelope eccentricity at the given period(s).

        Args:
            orbital_period(float or array):    The period(s) at which to
                evaluate the envelope.

            model_parameters(array):    Values of the parameters that fully
                specify the model.

            sim_config:    The configuration with which a simulation was
                performed. The model is allowed to depend on any
                parameters of that simulation.

        Returns:
            float or array with the same shape as `orbital_period`.
        """

    def __call__(self, model_parameters):
        """
        Sum square difference between model and sim at a given age.

        Args:
            model_parameters:    See same name argument to
                :meth:`max_eccentricity`.

        Returns:
            float:
                The sum of the square of the differences between the model
                envelope evaluated for each simulation at the age identified by
                age_index and the simulated eccentricities at the largest
                simulated initial eccentricity.
        """

        result = numpy.nan

        for sim_config, sim_data in self._simulations:
            simulated_periods, simulated_eccentricities = (
                values[self._age_index, :, -1]
                for values in sim_data
            )
            include_in_fit = numpy.logical_and(
                simulated_eccentricities > self.fit_eccentricity_range[0],
                simulated_eccentricities < self.fit_eccentricity_range[1]
            )
            if not include_in_fit.any():
                continue
            if not numpy.isfinite(result):
                result = 0.0
            model_eccentricities = self.max_eccentricity(
                simulated_periods[include_in_fit],
                model_parameters,
                sim_config
            )
            result += numpy.square(
                model_eccentricities
                -
                simulated_eccentricities[include_in_fit]
            ).sum()

        return result
