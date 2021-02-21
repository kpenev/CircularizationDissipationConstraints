#!/usr/bin/env python3

"""Use dynesty to sample SB1 binary star system."""

from collections import namedtuple
import logging
import traceback
from multiprocessing import set_start_method

from astropy import units
import numpy
from scipy import stats

from stellar_evolution.manager import StellarEvolutionManager
from orbital_evolution.transformations import phase_lag
from split_normal_distribution import split_normal

#Fixed module search paths, not intended to provide anything.
#pylint: disable=unused-import
import update_search_paths
#pylint: enable=unused-import
import ngc188_util
from eccentricity_pdf import EccentricityPDF
from prior_transform_cluster_sb1 import PriorTransformClusterSB1
#False positive
#pylint: disable=import-error
from log_likelihood_base import LogLikelihoodBase
from stellar_param_sampling import\
    StarSampler,\
    add_star_sampler_config_args,\
    POETInterpLikelihood
#pylint: enable=import-error
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
                'lgQ_min',
                get_uniform_distribution('lgQ_min'),
                units.dimensionless_unscaled
            )
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

        if config.lgQ_inertial_boost:
            raise RuntimeError('Inertial mode range dissipation boost is not '
                               'yet implemented')

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
                'orbical_period',
                7.0,
                units.day
            ),
            (
                'rv_semi_amplitude',
                stats.norm(16.46, 0.16),
                units.km / units.s
            ),
            (
                'cos_inclination',
                stats.uniform(-1.0, 2.0),
                units.dimensionless_unscaled
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
            )
        ]
    )

def prepare_sampling(config):
    """Return log-likelihood & prior transform for sampling the selected SB1."""

    if cnofig.system.startswith('NGC188_'):
        binary_pkm_id = int(config.system[len('NGC188_'):])
        photometric_constraint = ngc188_util.get_photometric_constraint(
            binary_pkm_id
        )
        rvk_constraint = ngc188_util.get_rvk_constraint(binary_pkm_id)


def main(config):
    """Avoid polluting global namespace."""

    assert config.lgQ_inertial_boost_range is None
    prior_transform = get_prior_transform(config)
    system_data = get_system_data(config)
    log_likelihood = LogLikelihoodConstQ(system_data, config)
    config.feh = system_data.feh
    primary_sampler = StarSampler(
        POETInterpLikelihood(
            logg=system_data.logg,
            teff=system_data.Teff,
            rho=system_data.mean_density,
            rtol=config.time_ode_rtol,
            atol=config.time_ode_atol,
            max_step=config.time_ode_max_step
        ),
        config
    )
    prior_transform = PriorTransformClusterSB1(
        primary_sampler,
        independent_parameter_distributions=(
            [
                get_uniform_parameter(component + '_' + param_name)
                for component in ['primary', 'secondary']
                for param_name in ['disk_lock_period',
                                   'wind_strength',
                                   'wind_saturation',
                                   'core_envelope_coupling_timescale']
            ]
            +
            [
                get_uniform_parameter('disk_dissipation_age'),
                ('orbical_period', 7.0, units.day),
                (
                    'rv_semi_amplitude',
                    stats.norm(16.46, 0.16),
                    units.km / units.s
                ),
                (
                    'cos_inclination',
                    stats.uniform(-1.0, 2.0),
                    units.dimensionless_unscaled
                ),
            ]
        ),
        model_parameter_order=log_likelihood.parameter_order
    )

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
