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

    def _handle_parameters(self, parameters, mode, median_e = 0):
        """Handle the parameters passed to the log-likelihood function."""

        if mode == 'first':
            parameters['solve'] = True
            parameters['initial_eccentricity'] = 0.8
        #if 'period' in parameters:
        #    dsd
        #if 'eccentricity' in parameters:
        #    dfdg
        elif mode == 'second':
            parameters['initial_eccentricity'] = 'solve'
            parameters['system'].eccentricity = median_e

        return parameters

    def _calculate_likelihood(self, ef_max,ei_med,dehat_med):
        # TODO
        return 1

    def _load_de(self, e):
        # TODO
        return 1
    
    def _load_prior_e(self, e):
        # TODO
        return 1

    def calculate_log_likelihood(self,
                                 parameters,
                                 **other_args):
        """Evaluate the log-likelihood at the given model parameters."""

        assert 'sample_weights_envelope' in other_args

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
        parameters = self._handle_parameters(parameters,'first')
        max_final_eccentricity = find_evolution(parameters)[0][-1]#(finalperiod=parameters['period'],initialeccentricity=0.8)[0][-1]
        negligible = 1e-3

        # Placeholder for Heaviside. Can remove when I'm done with pseudocode.
        H = 1
        if (max_final_eccentricity > self.envelope_eccentricity) or (max_final_eccentricity is None) or numpy.isnan(max_final_eccentricity):
                return -numpy.inf
        elif integral(self._load_de(e), 0, max_final_eccentricity) <= negligible: # Not part of Heaviside but since we're talking about early exits
            estimate likelihood by reverting back to the old assumption that ehat is a linear function between zero and max_final_eccentricity
            return whatever that likelihood is
        
        median_e = pull_from_aether()
        parameters = self._handle_parameters(parameters,'second',median_e)
        ehat_prime = find_evolution(parameters)[1][0]

        # ?
        if ( integral(self._load_de(e), max_final_eccentricity, self.envelope_eccentricity) / integral(self._load_de(e), 0, max_final_eccentricity) ) <= negligible:
             run a 2-D solver finding what initial eccentricity and period will result in final eccentricity near the (median or mean or max likelihood) final eccentricity and the value of e_hatprime at that point
             use the approximation above to calculate the likelihood
             get the various values

        #then I plug those various values into here and call the private function I'm going to make
        I = integral(self._load_de(e) * self._load_prior_e(e) / ehat_prime, 0, 0.8) #ehat prime is a function of e, now that I think about it
        L = D_theta * H * Pi_theta * Pi_Q * I
        ########################################### END PSEUDOCODE


        logger = logging.getLogger(__name__)
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
