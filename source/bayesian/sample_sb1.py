#!/usr/bin/env python3

"""Sample SB1 binary star system."""

import logging
import traceback
#from multiprocessing import set_start_method

from astropy import units
from scipy import stats
import dynesty

from stellar_evolution.manager import StellarEvolutionManager
from orbital_evolution.evolve_interface import library as\
    orbital_evolution_library

#Fixed module search paths, not intended to provide anything.
#pylint: disable=unused-import
import update_search_paths
#pylint: disable=wrong-import-order
#pylint: enable=unused-import
#False positive
#pylint: disable=import-error

#False positive due to unusual access.
#pylint: disable=unused-import
import ngc188_util
import ngc6819_util
import m35_util
import hyadespraesepe_util
#pylint: enable=unused-import
from cluster_util import get_rv_likelihood
#pylint: enable=import-error
from bayesian.binary_utils import \
    get_common_binary_star_priors,\
    prepare_sampling_common
from bayesian.sampling import setup_process
from bayesian.prior_transform_sb1 import PriorTransformSB1
from bayesian.sample_sb1_masses import SampleSB1Masses
from bayesian.log_likelihood_sb1 import LogLikelihoodSB1
from bayesian.parse_command_line import parse_command_line
from bayesian import mcmc_sampling
#pylint: enable=wrong-import-order

def get_independent_priors(config, observed_orbit, custom_util):
    """
    Return the independent parameters for the prior transform.

    Args:
        config:    The configuration with which sampling was invoked.

        observed_orbit(pandas.Series):    The entry for the binary to
            sample containing all system parameters.

        custom_util(module):    The <cluster name>_util module corresponding to
            the cluster the sampling binary is a member of.

    Returns:
        [(str, distrbibution, units), ...]:
            The independent parameters that will be sampled for this binary.
    """

    return (
        get_common_binary_star_priors(config)
        +
        [
            (
                'orbital_period',
                stats.norm(float(observed_orbit['Per']),
                           float(observed_orbit['e_Per'])),
                units.day
            ),
            (
                'age',
                custom_util.cluster_age_distribution,
                units.Gyr
            ),
            (
                'feh',
                custom_util.cluster_feh_distribution,
                units.dimensionless_unscaled
            )
        ]
    )

def prepare_sampling(config):
    """Return log-likelihood & prior transform for sampling the selected SB1."""

    interpolator = prepare_sampling_common(config)
    pickle_substitutions = dict(system=config.system,
                                sampling=config.sampling)
    for cluster in ['NGC188', 'NGC6819', 'M35', 'HyadesPraesepe']:
        if config.system.startswith(cluster + '_'):
            binary_id = int(config.system[len(cluster) + 1:])
            custom_util = globals()[cluster.lower() + '_util']
            binary_orbit = custom_util.get_observed_orbit(binary_id)

            photometric_constraint = custom_util.get_photometric_constraint(
                binary_id,
                pickle_fname=(config.photometric_constraint_pickle_fname
                              %
                              pickle_substitutions)

            )
            rv_likelihood = get_rv_likelihood(
                observed_orbit=binary_orbit,
                eccentricity_envelope=custom_util.eccentricity_envelope,
                num_parallel_processes=config.num_parallel_processes,
                interpolation_accuracy=config.rvk_interpolation_accuracy,
                mismatch_plot=(
                    (
                        'RV_likelihood_refinement_'
                        '%(title)s_%(grid_refinement_i)d.png'
                    )
                    if config.rvk_show_interpolation else
                    None
                ),
                pickle_fname=(config.rv_likelihood_pickle_fname
                              %
                              pickle_substitutions)
            )
            log_likelihood = LogLikelihoodSB1(
                interpolator=interpolator,
                rv_likelihood=rv_likelihood,
                powerlaw_dissipation=(
                    config.lgQ_break_period is not None
                    and
                    config.lgQ_powerlaw is not None
                ),
                evolution_timeout=config.evolution_timeout,
                period_search_factor=config.initial_period_search_factor,
                scaled_period_guess=config.initial_period_scaled_guess,
                prior_only=(config.sampling == 'prior')
            )
            prior_transform = PriorTransformSB1(
                sample_binary_masses=SampleSB1Masses(
                    rv_likelihood=rv_likelihood,
                    photometric_constraint=photometric_constraint,
                    orbital_period=float(binary_orbit['Per']),
                    pickle_fname=(config.mass_sampling_pickle_fname
                                  %
                                  pickle_substitutions)
                ),
                independent_parameter_distributions=get_independent_priors(
                    config,
                    binary_orbit,
                    custom_util
                ),
                model_parameter_order=log_likelihood.parameter_order
            )

            return log_likelihood, prior_transform

    raise RuntimeError('Target binary %s belongs to an unsupported cluster.'
                       %
                       config.system)

def main(config):
    """Avoid polluting global namespace."""

    setup_process(config)

#    set_start_method('forkserver')

    log_likelihood, prior_transform = prepare_sampling(config)

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

if __name__ == '__main__':
    try:
        main(
            parse_command_line(
                'Perform Bayesian sampling of a single SB1 binary system.',
                'sb1_sampling.cfg',
                dissipation=True,
                cluster=True,
                primary_properties=('feh', 'logg', 'Teff', 'rho'),
                choose_binary='cluster',
                spindown=2
            )
        )
    except SystemExit:
        pass
    #Meant to simply report exception to log
    #pylint: disable=bare-except
    except:
        logging.critical(traceback.format_exc())
    #pylint: enable=bare-except
    else:
        logging.info('SB1 sampling completed successfully.')
