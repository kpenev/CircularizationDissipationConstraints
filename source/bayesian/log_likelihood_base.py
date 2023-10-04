"""Define the parent class for all log-likelihood functions."""

from abc import ABCMeta, abstractmethod

import logging
from types import SimpleNamespace

import numpy
from astropy import units

from general_purpose_python_modules.reproduce_system import find_evolution
from bayesian.evolution_parameters import EvolutionParameters

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

        envelope_eccentricity(float):    The maximum eccentricity systems with
            the given period are allowed to have no matter their starting
            eccentricity.

    """

    @abstractmethod
    def get_dissipation(self, parameters):
        """Return the dissipation argument for `find_evolution`."""

    def _parse_parameters(self, parameters, logger):
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
        kwargs['dissipation'] = self.get_dissipation(parameters)
        kwargs['max_age'] = system.age
        kwargs['timeout'] = self._evolution_timeout
        kwargs['system'] = system
        kwargs.update(self._find_evolution_kwargs)
        if (
                kwargs['secondary_is_star']
                and
                (
                    system.secondary_mass.to_value(units.M_sun)
                    <
                    kwargs['interpolator'].mass_range()[0]
                )
        ):
            system.secondary_radius = self.get_parameter_value(
                parameters,
                'cmd_secondary_radius'
            )
            logger.warning(
                'Secondary mass %s M_sun is below the stellar evolution '
                'interpolator mass range. Ignoring evolution and spindown, '
                'fixing radius to %s R_sun',
                repr(system.secondary_mass.to_value(units.M_sun)),
                repr(system.secondary_radius.to_value(units.R_sun))
            )
            kwargs['secondary_is_star'] = False

        if (
                system.primary_mass.to_value(units.M_sun)
                >
                kwargs['interpolator'].mass_range()[1]
                and
                system.primary_mass.to_value(units.M_sun)
                <
                kwargs['interpolator'].mass_range()[1] * 1.05
        ):
            logger.error(
                'Primary mass is slightly above upper interpolator range. '
                'Tweaking %s Msun -> %s Msun.',
                repr(system.primary_mass.to_value(units.M_sun)),
                repr(kwargs['interpolator'].mass_range()[1])
            )
            system.primary_mass = (
                kwargs['interpolator'].mass_range()[1]
                *
                units.M_sun
            )

        return kwargs

    def __init__(self,
                 interpolator,
                 envelope_eccentricity,
                 secondary_is_star,
                 *,
                 evolution_timeout,
                 period_search_factor,
                 scaled_period_guess,
                 prior_only=False,
                 **kwargs):
        """
        Set-up the log-likelihood calculator.

        Args:
            interpolator:    A POET stelar evolution interpolator to use for
                calculating the orbital evolution.

            envelope_eccentricity:    See :attr:`envelope_eccentricity`.

            secondary_is_star(bool):    True iff the secondary in the system is
                an evolving star.

            evolution_timeout(float):    Maximum time to allow for a single
                orbital evolution

            period_search_factor(float):    See same name argument to
                :meth:`InitialConditionSolver.__init__`.

            scaled_period_guess(float):    See same name argument to
                :meth:`InitialConditionSolver.__init__`.

            prior_only(bool):    If True, log-likelihood is always zero,
                thus sampling is done only per the priors.

            kwargs:    Arguments in addition to `secondary_is_star` required by
                the parent's :meth:`__init__()`.

        Returns:
            None
        """

        self.envelope_eccentricity = envelope_eccentricity
        self._evolution_timeout = evolution_timeout
        self.final_eccentricity = None

        self._prior_only = prior_only

        self._find_evolution_kwargs = dict(
            interpolator=interpolator,
            secondary_is_star=secondary_is_star,
            period_search_factor=period_search_factor,
            scaled_period_guess=scaled_period_guess
        )
        super().__init__(secondary_is_star=secondary_is_star,
                         **kwargs)
        self._stashed_results = dict()
        self._stash = False

    def calculate_final_eccentricity(self, parameters):
        """
        Return the final eccentricity of the system with the given parameters.

        Args:
            parameters:    The parameters to evaluate the final eccentricity
                for. The order of the model parameters is specified by
                :attr:`parameter_order`.

        Returns:
            float
        """

        logger = logging.getLogger(__name__)

        self.log_parameters('Evaluating log-likelihood for parameters:',
                            parameters,
                            logging.INFO)

        evolve_parameters = self._parse_parameters(parameters, logger)
        failed = True
        logger.debug(
            'Evolve parameters:\n\t%s',
            '\n\t'.join(
                repr(k) + ':' + repr(v) for k, v in evolve_parameters.items()
            )
        )
        if (
                not evolve_parameters['secondary_is_star']
                and
                not numpy.isfinite(evolve_parameters['system'].secondary_radius)
        ):
            logger.error(
                'Non-stellar secondary with radius=%s',
                repr(evolve_parameters['system'].secondary_radius)
            )
            return None

        while failed and evolve_parameters['initial_eccentricity'] > 0.7:
            try:
                #False positive: dissipation is included find_evolution_kwargs
                #pylint: disable=no-value-for-parameter
                evolution = find_evolution(
                    **evolve_parameters
                )
                #pylint: enable=no-value-for-parameter
            except RuntimeError:
                evolve_parameters['initial_eccentricity'] -= 1e-2
                logger.warning('Calculating evolution failed, trying e0 = %g.',
                               evolve_parameters['initial_eccentricity'])
            except ValueError as error:
                logger.error('Invalid parameter values encountered: %s',
                             str(error))
                return None
            else:
                failed = False

        if failed:
            logger.error('Calculating evolution failed!')
            return None

        expected_final_age = self.get_parameter_value(
            parameters,
            'age'
        ).to_value('Gyr')

        #False positive
        #pylint: disable=no-member
        if numpy.allclose(evolution[0].age[-1],
                          expected_final_age,
                          rtol=1e-10,
                          atol=1e-10):

            self.final_eccentricity = evolution[0].eccentricity[-1]
            #pylint: enable=no-member

            logger.info(
                'Successful evolution found: ef = %g',
                evolution[0].eccentricity[-1]
            )

            return evolution[0].eccentricity[-1]

        logger.error(
            'Evolution terminated prematurely at t=%g (< %g) with ef = %g',
            evolution[0].age[-1],
            expected_final_age,
            (
                numpy.nan if self.final_eccentricity is None
                else self.final_eccentricity
            )
        )

        return None

    def start_stashing(self):
        """
        Store :meth:`__call__()` results for re-use if invoked with same params.

        Can be used to re-set stashing, as currently stashed results are
        cleared.
        """

        self._stashed_results = dict()
        self._stash = True

    def stop_stashing(self):
        """Stop stashing future :meth:`__call__` but keep current stash."""

        self._stash = False

    @abstractmethod
    def calculate_log_likelihood(self, parameters, **other_args):
        """
        Evaluate the log-likelihood at the given model parameters.

        Args:
            paramaters(array):    The parameters for which to evaluate the log
                likelihood. Must uniquely define the likelihood.

            other_args:    For some applications, terms the likelihood depends
                on are calculated during the prior transform. This is a
                mechanism to pass those pre-calculated values. These arguments
                must be fully determined by the parameter values.

        Returns:
            float:    The natural logarithm of the posterior probability (not
                normalized).
        """

    def __call__(self, parameters, **other_args):
        """Same as :meth:`calculate_log_likelihood` but handles stashing."""

        if self._prior_only:
            logging.getLogger(__name__).info('Prior only log-likelihood: 0.0')
            return 0.0

        param_hash = hex(hash(parameters.tostring()))[2:]
        if param_hash in self._stashed_results:
            result = self._stashed_results[param_hash]
            logging.getLogger(__name__).info(
                'Reusing stashed log log_likelihood: %s',
                repr(result)
            )
            return result

        result = self.calculate_log_likelihood(parameters, **other_args)
        logging.getLogger(__name__).info(
            'Calculated log_likelihood: %s',
            repr(result)
        )

        if self._stash:
            self._stashed_results[param_hash] = result

        return result
#pylint: enable=too-few-public-methods
