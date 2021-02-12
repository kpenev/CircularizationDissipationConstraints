#!/usr/bin/env python3

"""Use dynesty to sample SB1 binary star system."""

from collections import namedtuple
import logging
import traceback

from astropy import units
import numpy
import scipy.stats

from stellar_evolution.manager import StellarEvolutionManager
from orbital_evolution.transformations import phase_lag
from split_normal_distribution import split_normal

#Fixed module search paths, not intended to provide anything.
#pylint: disable=unused-import
import update_search_paths
from io_utilities import read_geller_et_al_2009_binaries
#pylint: enable=unused-import
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

class LogLikelihoodConstQ(LogLikelihoodBase):
    """SB1 binary log-likelihood assuming Q*'=const and same for both stars."""

    def _get_dissipation(self, parameters):
        """Return the dissipation argument for `find_evolution`."""

        dissipation = dict(
            tidal_frequency_breaks=None,
            spin_frequency_breaks=None,
            tidal_frequency_powers=numpy.array([0.0]),
            spin_frequency_powers=numpy.array([0.0]),
            reference_phase_lag=phase_lag(
                self.get_parameter_value(parameters, 'lgQ')
            )
        )

        return dict(primary=dissipation,
                    secondary=dict(dissipation))

    def __init__(self, system_data, config):
        """Set-up the log-likelihood for the given sysem with given config."""

        self.interpolator = StellarEvolutionManager(
            config.stellar_evolution_interpolator_dir
        ).get_interpolator_by_name(
            'default'
        )

        super().__init__(
            interpolator=self.interpolator,
            eccentricity_pdf=EccentricityPDF(
                system_data.current_eccentricity,
                system_data.envelope_eccentricity,
                integration_options=config.e_pdf_integration_opts,
                pickle_fname=config.e_pdf_pickle_fname,
                num_parallel_processes=config.num_parallel_processes
            ),
            secondary_is_star=True,
            initial_eccentricity=config.initial_eccentricity,
            dissipation_parameters=[('lgQ', units.dimensionless_unscaled)]
        )

SystemData = namedtuple(
    'SystemData',
    [
        'current_eccentricity',
        'envelope_eccentricity',
        'feh',
        'logg',
        'Teff',
        'mean_density'
    ]
)


def get_system_data(config):
    """Return the system to sample (HATS-18 + vB62)."""

    if config.system.startswith('NGC188'):
        sb1_systems, sb2_systems, age, feh = read_geller_et_al_2009_binaries(
            raw_data=True
        )
        selected = sb1_systems['PKM'] == int(config.system[len('NGC188_'):])
        if not selected.any():
            raise RuntimeError('No SB1 system matching: ' + repr(config.system))
        selected = sb1_systems[selected]
        print(repr(selected.T))
        print('K: ' + repr(float(selected['K'])))

        return SystemData(
            current_eccentricity=scipy.stats.rice(0.233/0.008, scale=0.008),
            envelope_eccentricity=0.4,
            feh=scipy.stats.norm(loc=0.28, scale=0.08),
            logg=scipy.stats.norm(loc=4.436, scale=0.034),
            Teff=scipy.stats.norm(loc=5600.0, scale=120.0),
            mean_density=split_normal.freeze_error_bar(mode=1.37,
                                                       abs_plus_error=0.12,
                                                       abs_minus_error=0.23)
        )

def main(config):
    """Avoid polluting global namespace."""

    def get_uniform_parameter(parameter):
        """
        Return the entry for parameter in independent_parameter_distributions.

        Args:
            parameter(str):    The name of the parameter to get the entry for.

        Returns:
            tuple:
                The entry in independent_parameter_distributions argument to
                :meth:`PriorTransformClusterSB1.__init__()` that correctly
                specifies the given model parameter's prior.
        """

        param_units = dict(
            disk_dissipation_age=units.Myr,
        )
        for component in ['primary', 'secondary']:
            param_units[component + '_disk_lock_period'] = units.day
            param_units[component + '_wind_strength'] = (
                units.dimensionless_unscaled
            )
            param_units[component + '_wind_saturation'] = (
                units.dimensionless_unscaled
            )
            param_units[component + '_core_envelope_coupling_timescale'] = (
                units.Myr
            )

        min_value, max_value = getattr(config, parameter)
        if min_value == max_value:
            return (parameter, min_value, param_units[parameter])

        return (parameter,
                scipy.stats.uniform(min_value, max_value),
                param_units[parameter])

    assert config.lgQ_inertial_boost_range is None
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
                    scipy.stats.norm(16.46, 0.16),
                    units.km / units.s
                ),
                (
                    'cos_inclination',
                    scipy.stats.uniform(-1.0, 2.0),
                    units.dimensionless_unscaled
                ),
            ]
        ),
        model_parameter_order=log_likelihood.parameter_order
    )

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
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
