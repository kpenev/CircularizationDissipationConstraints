"""Common set-up for sampling binary stars."""

from scipy import stats
from astropy import units

from orbital_evolution.evolve_interface import library as\
    orbital_evolution_library

from stellar_evolution.manager import StellarEvolutionManager
from stellar_evolution.library_interface import MESAInterpolator

import logging

def get_common_binary_star_priors(config):
    """
    Return independent parameters for prior trans shared by all binary datasets.

    Args:
        config:    The configuration with which sampling was invoked.

    Returns:
        [(str, distrbibution, units), ...]:
            Some of the independent parameters that will be sampled for this
            binary.

    """

    def get_distribution(parameter, distro_name='uniform'):
        """
        Return a uniform distribution with correct support for given parameter.

        Args:
            parameter(str):    The name of the parameter to get the entry for.

            distribution(str):    The name of the distribution to set for the
                given parameter from the scipy.stats module.

        Returns:
            tuple:
                The entry in independent_parameter_distributions argument to
                :meth:`PriorTransformClusterSB1.__init__()` that correctly
                specifies the given model parameter's prior.
        """

        min_value, max_value = getattr(config, parameter)
        return (
            min_value if min_value == max_value
            else getattr(stats, distro_name)(min_value, max_value - min_value)
        )

    def get_dissipation_parameters():
        """Return list of parameters to add parametrizing tidal dissipation."""

        result = [
            (
                param_name,
                get_distribution(param_name),
                units.dimensionless_unscaled
            )
            for param_name in ['lgQ_min',
                               'lgQ_inertial_boost',
                               'lgQ_inertial_sharpness']
        ]
        if (
                config.lgQ_break_period is not None
                and
                config.lgQ_powerlaw is not None
        ):
            result.extend([
                (
                    'lgQ_break_period',
                    get_distribution('lgQ_break_period', 'loguniform'),
                    units.day
                ),
                (
                    'lgQ_powerlaw',
                    get_distribution('lgQ_powerlaw'),
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
                get_distribution(component + '_' + param_name),
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
                get_distribution('disk_dissipation_age'),
                units.Myr
            ),
            (
                'initial_eccentricity',
                get_distribution('initial_eccentricity'),
                units.dimensionless_unscaled
            )
        ]
    )


def prepare_sampling_common(config):
    """Common sampling initialization steps for all binary star datasets."""

    MESAInterpolator.set_quantity_lower_limit('iconv', 1e-3)
    print(config.stellar_evolution_interpolator_dir)
    interpolator = StellarEvolutionManager(
        config.stellar_evolution_interpolator_dir
    ).get_interpolator_by_name(
        'default'
    )

    logger = logging.getLogger(__name__)
    logger.debug('Now loading: eccentricity expansion coefficients.')
    orbital_evolution_library.prepare_eccentricity_expansion(
        config.eccentricity_expansion_coefficients.encode('ascii'),
        1e-4,
        True,
        True
    )
    logger.debug('Eccentricity expansion coefficients loaded.')

    return interpolator
