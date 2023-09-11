"""Define a log-likelihood class for Windemuth et. al. (2019) EBs."""

import logging

import numpy
import scipy
from bayesian.log_likelihood_binary_stars import LogLikelihoodBinaryStars
from general_purpose_python_modules.reproduce_system import find_evolution

class LogLikelihoodWindemuth(LogLikelihoodBinaryStars):
    """The log-likelihood for Windemuth et. al. (2019) EBs."""

    def __init__(self,
                 *parent_args,
                 envelope_eccentricity,
                 observed_eccentricity_distro,
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

    def _calculate_likelihood(self, ef_max,ei_med,dehat_med):
        # TODO
        return 1

    def _load_de(self, e):
        # TODO
        return 1 #TODO: return a class? That I can just integrate rather than calling this all the time? And like has attributes?
    
    def _load_prior_e(self, ehat_approx):
        # TODO
        return 1

    def calculate_log_likelihood(self,
                                 parameters,
                                 **other_args):
        """Evaluate the log-likelihood at the given model parameters."""

        logger = logging.getLogger(__name__)

        assert 'sample_weights_envelope' in other_args #why?

        #final_eccentricity = self.calculate_final_eccentricity(parameters)
        #
        #if (
        #        final_eccentricity is None
        #        or
        #        not (final_eccentricity <= self.envelope_eccentricity)
        #):
        #    return -numpy.inf

        # BEGIN PSEUDOCODE
        #Work out parameters such that when we pass it to find_evolution, we get the 1D solver in the way we want
        parameters = self._choose_solver(parameters,'1d')
        max_final_eccentricity = find_evolution(parameters)[0][-1]
        negligible = 1e-3
        De = self._load_de()
        De_min,De_max = De.min,De.max #TODO: define the range as PPF of 1e-3. scipy.stats, etc.

        # Placeholder for Heaviside. Can remove when I'm done with pseudocode.
        H = 1
        if (
            (max_final_eccentricity > self.envelope_eccentricity)
            or
            (max_final_eccentricity is None)
            or
            numpy.isnan(max_final_eccentricity)
        ):
                return -numpy.inf
        
        # Okay, we're done with that, we're definitely doing this now
        median_e = pull_from_aether()

        logger.debug('max_final_eccentricity = %s',repr(max_final_eccentricity))
        logger.debug('De range = %s',repr((De_min,De_max)))
        logger.debug('Envelope eccentricity = %s',repr(self.envelope_eccentricity))

        De_away_from_zero = De_min > 0
        De_at_zero = not De_away_from_zero
        De_from_zero_to_max = scipy.integrate.quad(De.func, 0, max_final_eccentricity)

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
            ehat_approx = numpy.polynomial.polynomial.Polynomial((0,max_final_eccentricity / 0.8))
        elif (
            De_away_from_zero
            and
            (
                scipy.integrate.quad(De.func, max_final_eccentricity, self.envelope_eccentricity)
                / #TODO: do this with CDf (CDF of envelope - CDF of max_final_eccentricity)
                De_from_zero_to_max
            ) <= negligible
        ): # If D_e is clearly away from e_final=0 and e_hat(e_i=0.8) is below the envelope but above D_e
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
            e_to_match,e_initial,e_max_final = (De_min,0.8,max_final_eccentricity) if De_away_from_zero else (De_max,0,0)
            parameters = self._handle_parameters(parameters,'second',e_to_match)
            result = find_evolution(parameters)
            init_e,ehat_prime = result[1][1],result[1][0]
            A = numpy.matrix([
                                [init_e**2,    init_e,    1],
                                [e_initial**2, e_initial, 1],
                                [2*init_e,     1,         0]
                            ])
            B = numpy.matrix([e_to_match,e_max_final,ehat_prime])
            fit = scipy.linalg.lstsq(A,B.T)[0]
            ehat_approx = numpy.polynomial.polynomial.Polynomial((fit[2],fit[1],fit[0]))
        else:
            logger.error('Something went wrong in the Windemuth likelihood function. Please check the code.')
            raise ValueError('Something went wrong in the Windemuth likelihood function. Please check the code.')
        
        ehat_prime = ehat_approx.deriv()
        Prior_e = self._load_prior_e(ehat_approx)
        
        #then I plug those various values into here and call the private function I'm going to make
        I = scipy.integrate.quad(De.func * Prior_e.func / ehat_prime, 0, 0.8)
        L = D_theta * H * Pi_theta * Pi_Q * I
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
