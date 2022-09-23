"""Define function that actually carries out the sampling."""

import logging

import dynesty

#Fixed module search paths, not intended to provide anything.
#pylint: disable=unused-import
import update_search_paths
#pylint: enable=unused-import

from bayesian.sampling import setup_process
from bayesian import mcmc_sampling

def sample(log_likelihood, prior_transform, config):
    """
    Perform the sampling of the given log-likelihood/prior given configuration.

    Args:
        log_likelihood(LogLikelihoodBase):    The log-likelihood to sample.

        prior_transform(PriorTransformBase):    Transformation that converts a
            set of iidrv to the arguments required by the log-likelihood.

        config:    Usually the parsed command line.

    Returns:
        None
    """

    #set_start_method('forkserver')
    setup_process(config, 'manage')

    num_params = prior_transform.count_sampled_parameters()

    logging.info(
        'Starting %s sampling of binary %s with %d free parameters.',
        config.sampling,
        config.system,
        num_params
    )

    if config.sampling.lower() == 'nested':
        sampler = dynesty.NestedSampler(
            log_likelihood,
            prior_transform,
            ndim=len(log_likelihood.parameter_order),
            npdim=num_params,
            nlive=1
        )
        sampler.run_nested()
    else:
        mcmc_sampling.run(config,
                          log_likelihood,
                          prior_transform,
                          num_params)
