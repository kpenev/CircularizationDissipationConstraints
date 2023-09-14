"""Define a log-likelihood class for Windemuth et. al. (2019) EBs."""

import logging

import numpy
import scipy
from bayesian.log_likelihood_binary_stars import LogLikelihoodBinaryStars
from general_purpose_python_modules.reproduce_system import find_evolution
from bayesian.windemuth_eccentricity_distribution import W19EccentricityDistribution
from functools import partial

class LogLikelihoodWindemuth(LogLikelihoodBinaryStars):
    """The log-likelihood for Windemuth et. al. (2019) EBs."""

    def __init__(self,
                 *parent_args,
                 envelope_eccentricity,
                 observed_eccentricity_distro,
                 de_distro,
                 pe_distro,
                 **parent_kwargs):
        """
        Prepare the log-likelihood function.

        Args:
            parent_args:    Passed directly to parent`s ``__init__``.

        """

        super().__init__(*parent_args,
                         envelope_eccentricity=envelope_eccentricity,
                         **parent_kwargs)
        self.envelope_weights = observed_eccentricity_distro.eval_sample_cdf(
            envelope_eccentricity
        )
        self._observed_eccentricity_distro = observed_eccentricity_distro
        logging.getLogger(__name__).debug(
            'Init: envelope_weights = %s\nSum: %s',
            repr(self.envelope_weights),
            repr(self.envelope_weights.sum())
        )

        self._de = de_distro
        self._pe = pe_distro

    def _choose_solver(self, parameters, solver = '1d', final_e = 0):
        """Handle the parameters passed to the log-likelihood function."""

        if solver == '1d':
            parameters['solve'] = True
            parameters['initial_eccentricity'] = 0.8
        #if 'period' in parameters:
        #    dsd
        #if 'eccentricity' in parameters:
        #    dfdg
        elif solver == '2d':
            parameters['initial_eccentricity'] = 'solve'
            parameters['system'].eccentricity = final_e

        return parameters
    
    def _likelihood_integrand(self,e,ehat_approx):
        ehat_prime = ehat_approx.deriv()

        ehat_approx_copy = ehat_approx.copy()
        ehat_approx_copy.coef[0] -= e
        inverse_ehat_approx = numpy.polynomial.polynomial.Polynomial((ehat_approx_copy)).roots()
        # We need to figure out which of the values roots returns is real and positive
        # Also we should check how many there are because there can be only one
        # We should also check that the value is between 0 and 0.8
        if inverse_ehat_approx.size > 1:
            if inverse_ehat_approx[0] > 0:
                inverse_e = inverse_ehat_approx[0]
            elif inverse_ehat_approx[1] > 0:
                inverse_e = inverse_ehat_approx[1]
            elif inverse_ehat_approx[0] == 0 or inverse_ehat_approx[1] == 0:
                inverse_e = 0
            else:
                raise ValueError('Something is wrong with inverse_ehat_approx: %s',repr(inverse_ehat_approx))
        elif inverse_ehat_approx.size == 1:
            if numpy.imag(inverse_ehat_approx[0]) != 0:
                raise ValueError('Something is wrong with inverse_ehat_approx: %s',repr(inverse_ehat_approx))
            else:
                inverse_e = inverse_ehat_approx[0]
        
        if inverse_e > 0.8 or inverse_e < 0:
            raise ValueError('inverse_e should not be outside range. Something is wrong with inverse_ehat_approx: %s',repr(inverse_ehat_approx))
        
        return self._de(e) * self._pe(ehat_approx(inverse_e)) / ehat_prime(inverse_e)

    def _calculate_likelihood(self, ehat_approx): #  ef_max,ei_med,dehat_med
        specific_integrand = partial(self._likelihood_integrand,ehat_approx)
        I = scipy.integrate.quad(specific_integrand, 0, 0.8)
        return I

    def calculate_log_likelihood(self,
                                 parameters,
                                 **other_args):
        """Evaluate the log-likelihood at the given model parameters."""

        logger = logging.getLogger(__name__)

        assert 'sample_weights_envelope' in other_args #why?

        # BEGIN PSEUDOCODE
        #Work out parameters such that when we pass it to find_evolution, we get the 1D solver in the way we want
        parameters = self._choose_solver(parameters,'1d')
        max_final_eccentricity = find_evolution(parameters)[0][-1]
        negligible = 1e-3
        De_min = self._de.ppf(1e-3)
        De_max = self._de.ppf(1-1e-3)

        # Heaviside
        if (
            (max_final_eccentricity > self.envelope_eccentricity)
            or
            (max_final_eccentricity is None)
            or
            numpy.isnan(max_final_eccentricity)
        ):
                return -numpy.inf
        
        median_e = self._de.ppf(0.5) # TODO: ?

        De_away_from_zero = De_min > 0
        De_at_zero = not De_away_from_zero
        De_from_zero_to_max = self._de.cdf(max_final_eccentricity)

        logger.debug('Various values:')
        logger.debug('median_e = %s',repr(median_e))
        logger.debug('max_final_eccentricity = %s',repr(max_final_eccentricity))
        logger.debug('De range = %s',repr((De_min,De_max)))
        logger.debug('Envelope eccentricity = %s',repr(self.envelope_eccentricity))
        logger.debug('De_away_from_zero = %s',repr(De_away_from_zero))
        logger.debug('De_at_zero = %s',repr(De_at_zero))
        logger.debug('De_from_zero_to_max = %s',repr(De_from_zero_to_max))

        if (
            De_from_zero_to_max <= negligible
            or
            (
                De_away_from_zero
                and
                max_final_eccentricity < De_min
            ) # D_e is negligible near e=0, and e_hat(e_i=0.8) is below D_e.
            or
            (
                De_at_zero
                and
                max_final_eccentricity <= De_max
            ) # D_e(e_final=0) is not negligible, and e_hat(e_i=0.8) is in or below D_e.
        ):
            logger.debug('In a case where we assume ehat is linear from 0 to 0.8')
            ehat_approx = numpy.polynomial.polynomial.Polynomial((0,max_final_eccentricity / 0.8))
        elif (
            De_away_from_zero
            and
            (
                self._de.cdf(self.envelope_eccentricity)
                -
                De_from_zero_to_max
            ) <= negligible
        ): # If D_e is clearly away from e_final=0 and e_hat(e_i=0.8) is below the envelope but above D_e
            logger.debug('In a case where we assume ehat is linear in the range where D_e is non-negligible')
            parameters = self._handle_parameters(parameters,'second',median_e)
            result = find_evolution(parameters)
            init_e,ehat_prime = result[1][1],result[1][0]
            yintercept = median_e - ehat_prime*init_e
            ehat_approx = numpy.polynomial.polynomial.Polynomial((yintercept,ehat_prime))
        elif (
            (
                De_away_from_zero
                and
                (
                    De_min <= max_final_eccentricity <= De_max
                )
            ) # If D_e is negligible near e=0, but e_hat(e_i=0.8) is sowhere where D_e is not negligible
            or
            (
                De_at_zero
                and
                max_final_eccentricity > De_max
            ) # D_e(e_final=0) is not negligible, and e_hat(e_i=0.8) is above D_e
        ):
            logger.debug('In a case where we want to use quadratic approximation.')
            e_to_match,e_max_initial,e_max_final = (De_min,0.8,max_final_eccentricity) if De_away_from_zero else (De_max,0,0)
            logger.debug('Values if De_away_from_zero: e_to_match = %s, e_max_initial = %s, e_max_final = %s',repr(De_min),repr(0.8),repr(max_final_eccentricity))
            logger.debug('Values if De_at_zero: e_to_match = %s, e_max_initial = %s, e_max_final = %s',repr(De_max),repr(0),repr(0))
            logger.debug('e_to_match = %s',repr(e_to_match))
            logger.debug('e_max_initial = %s',repr(e_max_initial))
            logger.debug('e_max_final = %s',repr(e_max_final))
            parameters = self._handle_parameters(parameters,'second',e_to_match)
            result = find_evolution(parameters)
            init_e,ehat_prime = result[1][1],result[1][0]
            A = numpy.matrix([
                                [init_e**2,        init_e,        1],
                                [e_max_initial**2, e_max_initial, 1],
                                [2*init_e,         1,             0]
                            ])
            B = numpy.matrix([e_to_match,e_max_final,ehat_prime])
            fit = scipy.linalg.lstsq(A,B.T)[0]
            ehat_approx = numpy.polynomial.polynomial.Polynomial((fit[2],fit[1],fit[0]))
        else:
            logger.error('Something went wrong in the Windemuth likelihood function. Please check the code.')
            raise ValueError('Something went wrong in the Windemuth likelihood function. Please check the code.')
        
        ehat_prime = ehat_approx.deriv() #TODO (after debugging): remove, already being handled by _calculate_likelihood

        logger.debug('ehat_approx = %s',repr(ehat_approx))
        logger.debug('ehat_prime = %s',repr(ehat_prime))
        
        likelihood = self._calculate_likelihood(ehat_approx)
        assert likelihood >= 0
        if likelihood == 0:
            return -numpy.inf
        
        logger.debug('likelihood = %s',repr(likelihood))
        logger.debug('log(likelihood) = %s',repr(numpy.log(likelihood)))

        return numpy.log(likelihood)
        ########################################### END PSEUDOCODE

        efinal_cdfs = self._observed_eccentricity_distro.eval_sample_cdf(
            final_eccentricity
        )
        numerator = (
            efinal_cdfs
            /
            self.envelope_weights
            *
            other_args['sample_weights_envelope']
        ).sum()

        denominator = other_args['sample_weights_envelope'].sum()

        logger.debug('Observed e CDF(%s) = %s\nSum: %s',
                     repr(final_eccentricity),
                     repr(efinal_cdfs),
                     repr(efinal_cdfs.sum()))
        logger.debug('envelope_weights = %s\nSum: %s',
                     repr(self.envelope_weights),
                     repr(self.envelope_weights.sum()))

        logger.debug('W19 likelihood = %s / %s',
                     repr(numerator),
                     repr(denominator))

        assert numerator >= 0
        if numerator == 0:
            return -numpy.inf

        return numpy.log(numerator) - numpy.log(denominator)
