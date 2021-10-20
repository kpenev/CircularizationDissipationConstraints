#!/usr/bin/env python3

"""Use dynesty to sample SB1 binary star system."""

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
#pylint: enable=unused-import
from cluster_util import get_rv_likelihood
#pylint: enable=import-error
from bayesian.sampling import setup_process
from bayesian.prior_transform_binary_stars import PriorTransformBinaryStars
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

    def get_uniform_distribution(parameter):
        """
        Return a uniform distribution with correct support for given parameter.

        Args:
            parameter(str):    The name of the parameter to get the entry for.

        Returns:
            tuple:
                The entry in independent_parameter_distributions argument to
                :meth:`PriorTransformClusterSB1.__init__()` that correctly
                specifies the given model parameter's prior.
        """

        min_value, max_value = getattr(config, parameter)
        return (
            min_value if min_value == max_value
            else stats.uniform(min_value, max_value - min_value)
        )

    def get_dissipation_parameters():
        """Return list of parameters to add parametrizing tidal dissipation."""

        result = [
            (
                param_name,
                get_uniform_distribution(param_name),
                units.dimensionless_unscaled
            )
            for param_name in ['lgQ_min', 'lgQ_inertial_boost']
        ]
        if (
                config.lgQ_break_period is not None
                and
                config.lgQ_powerlaw is not None
        ):
            result.extend([
                (
                    'lgQ_break_period',
                    get_uniform_distribution('lgQ_break_period'),
                    units.day
                ),
                (
                    'lgQ_powerlaw',
                    get_uniform_distribution('lgQ_powerlaw'),
                    units.dimensionless_unscaled
                )
            ])

        return result


    return (
        get_dissipation_parameters()
        +
        [
            (
                component + '_' + param_name,
                get_uniform_distribution(component + '_' + param_name),
                param_units
            )
            for component in ['primary', 'secondary']
            for param_name, param_units in [
                ('disk_lock_period', units.day),
                ('wind_strength', units.dimensionless_unscaled),
                ('wind_saturation', units.dimensionless_unscaled),
                ('core_envelope_coupling_timescale', units.Myr)
            ]
        ]
        +
        [
            (
                'disk_dissipation_age',
                get_uniform_distribution('disk_dissipation_age'),
                units.Myr
            ),
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
            ),
            (
                'initial_eccentricity',
                get_uniform_distribution('initial_eccentricity'),
                units.dimensionless_unscaled
            )
        ]
    )

def prepare_sampling(config):
    """Return log-likelihood & prior transform for sampling the selected SB1."""

    interpolator = StellarEvolutionManager(
        config.stellar_evolution_interpolator_dir
    ).get_interpolator_by_name(
        'default'
    )
    orbital_evolution_library.read_eccentricity_expansion_coefficients(
        config.eccentricity_expansion_coefficients.encode('ascii')
    )


    for cluster in ['NGC188', 'NGC6819', 'M35']:
        if config.system.startswith(cluster + '_'):
            binary_id = int(config.system[len(cluster) + 1:])
            custom_util = globals()[cluster.lower() + '_util']
            binary_orbit = custom_util.get_observed_orbit(binary_id)

            photometric_constraint = custom_util.get_photometric_constraint(
                binary_id
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
                )
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
                scaled_period_guess=config.initial_period_scaled_guess
            )
            prior_transform = PriorTransformBinaryStars(
                sample_binary_masses=SampleSB1Masses(
                    rv_likelihood=rv_likelihood,
                    photometric_constraint=photometric_constraint,
                    orbital_period=float(binary_orbit['Per']),
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
                choose_binary=True,
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
