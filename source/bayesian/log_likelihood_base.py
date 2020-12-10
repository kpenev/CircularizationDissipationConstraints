"""Define the parent class for all log-likelihood functions."""

import logging
from types import SimpleNamespace
from abc import ABCMeta, abstractmethod

from astropy import units
import numpy

from reproduce_system import find_evolution

#The purpose is to create a simple callable object.
#pylint: disable=too-few-public-methods

class LogLikelihoodBase(metaclass=ABCMeta):
    """
    Base class for log of the likelihood function to sample from.

    Children should update :attr:`parameter_names` to handle specific cases in
    addition to providing :meth:`get_dissipation`.

    Attrs:
        parameter_names_units:    The names and units of the collection of
            parameters fully determining the orbital evolution for a system.

        parameter_order:    The order in which parameters appear in the input
            array to :meth:`__call__`.

        interpolator:    Stellar evolution interpolator to use in orbital
            evolution calculations.

        eccentricity_pdf:    Callable that return the probability density
            for the final eccentricity

        initial_eccentricity:    The "high" initial eccentricity the evolution
            will always start with.
    """

    parameter_names_units = dict(
        dissipation=[
        ],
        evolution=[
            ('disk_dissipation_age', units.Gyr),
            ('disk_period', units.day),
            ('primary_wind_strength', 1),
            ('prinmary_wind_saturation', 1),
            ('primary_core_envelope_coupling_timescale', units.Gyr)
        ],
        system=[
            ('age', units.Gyr),
            ('feh', 1),
            ('orbital_period', units.day),
            ('primary_mass', units.M_sun),
            ('secondary_mass', units.M_sun)
        ]
    )

    _logger = logging.getLogger(__name__)

    @abstractmethod
    def _get_dissipation(self, parameters):
        """Return the dissipation argument for `find_evolution`."""

    def _parse_parameters(self, parameters):
        """Return all keyword arguments to pass to `find_evolution`."""

        system = SimpleNamespace(
            **{
                param_name: self.get_parameter_value(parameters, param_name)
                for param_name in self.parameter_names_units['system']
            }
        )

        kwargs = {
            param_name: self.get_parameter_value(parameters, param_name)
            for param_name in self.parameter_names_units['evolution']
        }
        kwargs['dissipation'] = self._get_dissipation(parameters)
        kwargs['interpolator'] = self.interpolator
        kwargs['initial_eccentricity'] = self.initial_eccentricity
        kwargs['max_age'] = system.age
        kwargs['system'] = system
        kwargs['secondary_is_star'] = self.secondary_is_star

        return kwargs

    def _log_parameters(self, message, parameters, level):
        """
        Issue log message along with a description of this step's parameters.

        Args:
            parameters(array):    The parameter array defining the current steps
                being attempted (to be included in the message).

            message(str):    The message to issue before describing the
                parameters.

            level(str):    One of `'debug'`, `'warning'`, `'info'`, `'error'`,
                `'critical'` definng the importance level of the message to
                issue.

        Returns:
            None
        """

        self._logger.log(
            level,
            message + '\n\t%s: %s' * parameters.size(),
            *(
                sub
                for param_name in self.parameter_names_units
                for sub in (param_name, self.get_parameter_value(parameters,
                                                                 param_name))
            )
        )

    def __init__(self,
                 interpolator,
                 eccentricity_pdf,
                 secondary_is_star,
                 initial_eccentricity):
        """
        Set-up the log-likelihood calculator.

        Args:
            interpolator:    See :attr:`interpolator`.

            eccentricity_pdf:    See :attr:`eccentricity_pdf`.

            secondary_is_star(bool):    True iff the secondary in the system is
                an evolving star.

            initial_eccentricity:    See :attr:`initial_eccentricity`.
        """

        self.interpolator = interpolator
        self.eccentricity_pdf = eccentricity_pdf
        self.initial_eccentricity = initial_eccentricity

        if secondary_is_star:
            self.parameter_names_units['evolution'].extend(
                [
                    ('secondary_disk_period', units.day),
                    ('secondary_wind_strength', 1),
                    ('secondary_wind_saturation', 1),
                    ('secondary_core_envelope_coupling_timescale', units.Gyr)
                ]
            )

        self.secondary_is_star = secondary_is_star

        self.parameter_order = (
            self.parameter_names_units['dissipation']
            +
            self.parameter_names_units['evolution']
            +
            self.parameter_names_units['system']
        )
        self.parameter_indices = dict(
            (name, index)
            for index, (name, units) in enumerate(self.parameter_order)
        )

    def get_parameter_index_units(self, parameter_name):
        """Return the index and units of the parameter with the given name."""

        index = self.parameter_indices[parameter_name]
        return index, self.parameter_order[index][1]

    def get_parameter_value(self, parameters, parameter_name):
        """Return the value with units of a parameter by name."""

        index = self.parameter_indices[parameter_name]
        return parameters[index] * self.parameter_order[index][1]

    def __call__(self, parameters):
        """
        Evaluate the log-likelihood at the given model parameters.

        Args:
            parameters:    The parameters to evaluate the log-likelihood at. The
                order of the model parameters is specified by
                :attr:`parameter_order`.

                * 0 - log10(Q' of the primary)

                * 1 - log10(Q' of the secondary)

                * 2 - current system age in Gyrs

                * 3 - system metallicity ([Fe/H])

                * 4 - current orbital period

                * 5 - age at which binary is formed (in Gyrs)

                * Parameters describing each component (all primary entries
                    first, followed by secondary).

                    * 6, 8/11 - component mass (in solar masses)

                    * 7, 9/12 - **CONDITIONAL**: component radius (Solar radii).
                      Only present if the component is a planet.

                    * 7, 10/12 - **CONDITIONAL**: Spin period (in days) at which
                      the surface of the component is held until the binary
                      forms. Only present if the component is a star.

                    * 8, 11/13 - **CONDITIONAL**: Core-envelope coupling
                      timescale of the component in Gyr. Only present if the
                      component is a star.

                    * 9, 12/14 - **CONDITIONAL**: spin-down saturation frequency
                      (in rad/day) of the component. Only present if the
                      component is a star.

                    * 10, 13/15 - **CONDITIONAL**: wind strength in solar units
                      per Gyr of the component. Only present if the component is
                      a star.

        Returns:
            float:
                Uknown constant times the log-PDF of the observational data for
                the system assuming the given model parameters have exactly the
                specified value. This includes the circularization envelope.
        """

        self._logger.update_context(hex(hash(parameters.tostring()))[2:])

        try:
            self._log_parameters('Evaluating log-likelihood for parameters:',
                                 parameters,
                                 logging.INFO)

            try:
                #False positive: dissipation is included find_evolution_kwargs
                #pylint: disable=no-value-for-parameter
                evolution = find_evolution(
                    **self._parse_parameters(parameters)
                )
                #pylint: enable=no-value-for-parameter
            except AssertionError:
                self._logger.excption('Calculating evolution failed.')
                return -numpy.inf

            expected_final_age = self.get_parameter_value(
                parameters,
                'age'
            ).to_value('Gyr')

            #False positive
            #pylint: disable=no-member
            if numpy.allclose(evolution.age[-1],
                              expected_final_age,
                              rtol=1e-10,
                              atol=1e-10):

                final_eccentricity = evolution.eccentricity[-1]
                #pylint: enable=no-member

                self._logger.info(
                    'Successful evolution found: ef = %g',
                    final_eccentricity
                )

                return numpy.log(self.eccentricity_pdf(final_eccentricity))

            self._logger.error(
                'Evolution terminated prematurely at t=%g (< %g) with ef = %g',
                evolution.age[-1],
                expected_final_age,
                final_eccentricity
            )

            return -numpy.inf
        finally:
            self._logger.revert_context()
#pylint: enable=too-few-public-methods
