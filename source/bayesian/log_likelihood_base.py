"""Define the parent class for all log-likelihood functions."""

from abc import ABCMeta, abstractmethod

import logging
from types import SimpleNamespace

import numpy

from reproduce_system import find_evolution
from bayesian.evolution_parameters import EvolutionParameters
from bayesian.custom_logger_adapter import CustomLoggerAdapter

#Intended to function as callable no need for more public methods
#pylint: disable=too-few-public-methods
class LogLikelihoodBase(EvolutionParameters, metaclass=ABCMeta):
    """
    Base class for log of the likelihood function to sample from.

    Children should update :attr:`parameter_names` to handle specific cases in
    addition to providing :meth:`get_dissipation`.

    Attrs:
        interpolator:    Stellar evolution interpolator to use in orbital
            evolution calculations.

        eccentricity_likelihood(callable):    The likelihood functionof the
            final eccentricity.

    """

    @abstractmethod
    def _get_dissipation(self, parameters):
        """Return the dissipation argument for `find_evolution`."""

    def _parse_parameters(self, parameters):
        """Return all keyword arguments to pass to `find_evolution`."""

        system = SimpleNamespace(
            **{
                param_name: self.get_parameter_value(parameters, param_name)
                for param_name, _ in self.parameter_names_units['system']
            }
        )

        kwargs = {
            (
                'disk_period' if param_name == 'primary_disk_lock_period'
                else (
                    'secondary_disk_period'
                    if param_name == 'secondary_disk_lock_period' else
                    param_name
                )
            ): self.get_parameter_value(parameters, param_name)
            for param_name, _ in self.parameter_names_units['evolution']
        }
        kwargs['dissipation'] = self._get_dissipation(parameters)
        kwargs['interpolator'] = self.interpolator
        kwargs['max_age'] = system.age
        kwargs['system'] = system
        kwargs['secondary_is_star'] = self.secondary_is_star

        return kwargs

    def __init__(self,
                 interpolator,
                 eccentricity_likelihood,
                 secondary_is_star,
                 **kwargs):
        """
        Set-up the log-likelihood calculator.

        Args:
            interpolator:    A POET stelar evolution interpolator to use for
                calculating the orbital evolution.

            eccentricity_likelihood:    See :attr:`eccentricity_likelihood`.

            secondary_is_star(bool):    True iff the secondary in the system is
                an evolving star.

            kwargs:    Arguments in addition to `secondary_is_star` required by
                the parent's :meth:`__init__()`.

        Returns:
            None
        """

        self.interpolator = interpolator
        self.eccentricity_likelihood = eccentricity_likelihood
        self.final_eccentricity = None

        self.secondary_is_star = secondary_is_star
        super().__init__(secondary_is_star=secondary_is_star,
                         **kwargs)

    def __call__(self, parameters):
        """
        Evaluate the log-likelihood at the given model parameters.

        Args:
            parameters:    The parameters to evaluate the log-likelihood at. The
                order of the model parameters is specified by
                :attr:`parameter_order`.

        Returns:
            float:
                Uknown constant times the log-PDF of the observational data for
                the system assuming the given model parameters have exactly the
                specified value. This includes the circularization envelope.
        """

        logger = CustomLoggerAdapter(
            logging.getLogger(__name__),
            dict(param_hash=hex(hash(parameters.tostring()))[2:])
        )

        self.log_parameters('Evaluating log-likelihood for parameters:',
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
            logger.exception('Calculating evolution failed.')
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

            self.final_eccentricity = evolution.eccentricity[-1]
            #pylint: enable=no-member

            logger.info(
                'Successful evolution found: ef = %g',
                self.final_eccentricity
            )

            return numpy.log(
                self.eccentricity_likelihood(self.final_eccentricity)
            )

        logger.error(
            'Evolution terminated prematurely at t=%g (< %g) with ef = %g',
            evolution.age[-1],
            expected_final_age,
            (
                numpy.nan if self.final_eccentricity is None
                else self.final_eccentricity
            )
        )

        return -numpy.inf
#pylint: enable=too-few-public-methods
