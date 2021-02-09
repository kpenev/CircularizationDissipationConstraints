#!/usr/bin/env python3

"""Use dynesty to sample SB1 binary star system."""

from collections import namedtuple

from astropy import units
import numpy
import scipy.stats

from stellar_evolution.manager import StellarEvolutionManager
from orbital_evolution.transformations import phase_lag
from split_normal_distribution import split_normal

from eccentricity_pdf import EccentricityPDF
from prior_transform_cluster_sb1 import PriorTransformClusterSB1
from log_likelihood_base import LogLikelihoodBase
from stellar_param_sampling import\
    StarSampler,\
    add_star_sampler_config_args,\
    POETInterpLikelihood
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


def get_system_data():
    """Return the system to sample (HATS-18 + vB62)."""

    return SystemData(
        current_eccentricity=scipy.stats.rice(0.233/0.008, scale=0.008),
        envelope_eccentricity=0.4,
        feh=scipy.stats.norm(loc=0.28, scale=0.08),
        logg=scipy.stats.norm(loc=4.436, scale=0.034),
        Teff=scipy.stats.norm(loc=5600.0, scale=120.0),
        mean_density=split_normal(mode=1.37,
                                  abs_plus_error=0.12,
                                  abs_minus_error=0.23),
    )

def main(config):
    """Avoid polluting global namespace."""

    system_data = get_system_data()
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
    prior_transform = PriorTransformSB1(
        primary_sampler,
        independent_parameter_distributions=[
            ('disk_dissipation_age', 5.0, units.Myr),
            ('disk_period', 5.0, units.day),
            ('primary_wind_strength', 0.17, units.dimensionless_unscaled),
            ('secondary_wind_strength', 0.17, units.dimensionless_unscaled),
            ('primary_wind_saturation', 2.45, units.dimensionless_unscaled),
            ('secondary_wind_saturation', 2.45, units.dimensionless_unscaled),
            ('primary_core_envelope_coupling_timescale', 10.0, units.Myr),
            ('secondary_core_envelope_coupling_timescale', 10.0, units.Myr),
            ('orbical_period', 7.0, units.day),
            ('rv_semi_amplitude', scipy.stats.norm(16.46, 0.16), u.km / u.s),
            (
                'cos_inclination',
                scipy.stats.uniform(-1.0, 2.0),
                units.dimensionless_unscaled
            ),
        ],
        model_parameter_order=log_likelihood.parameter_order
    )

if __name__ == '__main__':
    main(
        parse_command_line(
            'Perform Bayesian sampling of a single SB1 binary system.',
            'sb1_sampling.cfg',
            dissipation=True,
            cluster=True,
            primary_properties=('feh', 'logg', 'Teff', 'rho'),
            choose_binary=True
        )
    )
