#!/usr/bin/env python3

"""Use dynesty to sample SB1 binary star system."""

import logging
import traceback
import functools
import os.path
from multiprocessing import set_start_method, Pool

from astropy import units
from scipy import stats
import numpy
import dynesty
import emcee

from stellar_evolution.manager import StellarEvolutionManager
from orbital_evolution.evolve_interface import library as\
    orbital_evolution_library

#Fixed module search paths, not intended to provide anything.
#pylint: disable=unused-import
import update_search_paths
#pylint: enable=unused-import
import ngc188_util
from bayesian.prior_transform_cluster_sb1 import PriorTransformClusterSB1
from bayesian.log_likelihood_sb1 import LogLikelihoodSB1
from parse_command_line import parse_command_line

def get_independent_priors(config, observed_orbit):
    """Return the independent parameters for the prior transform."""

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
                    param,
                    get_uniform_distribution(param),
                    units.dimensionless_unscaled
                )
                for param in ['lgQ_break_period', 'lgQ_powerlaw']
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
                ngc188_util.cluster_age_distribution,
                units.Gyr
            ),
            (
                'feh',
                ngc188_util.cluster_feh_distribution,
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


    if config.system.startswith('NGC188_'):
        binary_pkm_id = int(config.system[len('NGC188_'):])
        binary_orbit = ngc188_util.get_observed_orbit(binary_pkm_id)

        photometric_constraint = ngc188_util.get_photometric_constraint(
            binary_pkm_id
        )
        rvk_constraint = ngc188_util.get_rvk_constraint(binary_orbit)
        log_likelihood = LogLikelihoodSB1(
            powerlaw_dissipation=(
                config.lgQ_break_period is not None
                and
                config.lgQ_powerlaw is not None
            ),
            rv_semiamplitude_constraint=rvk_constraint,
            interpolator=interpolator,
            eccentricity_likelihood=(
                ngc188_util.get_final_eccentricity_likelihood(binary_orbit)
            )
        )

        prior_transform = PriorTransformClusterSB1(
            photometric_mass_constraint=photometric_constraint,
            rv_semi_amplitude_constraint=rvk_constraint,
            independent_parameter_distributions=get_independent_priors(
                config,
                binary_orbit
            ),
            model_parameter_order=log_likelihood.parameter_order
        )

    return log_likelihood, prior_transform

def mcmc_log_probability(unit_cube_values, prior_transform, log_likelihood):
    """The posterior for MCMC, will track actual params & likelihood."""

    if unit_cube_values.min() < 0 or unit_cube_values.max() > 1:
        return tuple(
            -numpy.inf if i == 0 else numpy.nan
            for i in range(len(log_likelihood.parameter_order) + 1)
        )

    parameters = prior_transform(unit_cube_values)
    log_likelihood_value = log_likelihood(parameters)
    return (
        (log_likelihood_value,)
        +
        tuple(parameters)
    )

def run_mcmc(config, log_likelihood, prior_transform, num_params):
    """Sample the selected system using MCMC."""

    samples_fname = (
        config.samples_fname
        %
        dict(system=config.system,
             sampling=config.sampling)
        +
        '.h5'
    )
    backend = emcee.backends.HDFBackend(samples_fname)
    if not os.path.exists(samples_fname):
        backend.reset(config.mcmc_nwalkers, num_params)
    if backend.iteration > 0:
        logging.info("Extending chanin in '%s', containing %d samples",
                     samples_fname,
                     backend.iteration)

    with Pool(config.num_parallel_processes) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers=config.mcmc_nwalkers,
            ndim=num_params,
            log_prob_fn=functools.partial(mcmc_log_probability,
                                          prior_transform=prior_transform,
                                          log_likelihood=log_likelihood),
            blobs_dtype=[(name, float)
                         for name, _ in log_likelihood.parameter_order],
            backend=backend,
            pool=pool
        )
        sampler.run_mcmc(
            numpy.random.rand(config.mcmc_nwalkers, num_params),
            config.mcmc_nsteps
        )

def main(config):
    """Avoid polluting global namespace."""

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
        run_mcmc(config, log_likelihood, prior_transform, num_params)

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    set_start_method('forkserver')

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
