"""Define base class for transforming the unit cube to evolution parameters."""

import scipy
from scipy.stats import norm

from binary_utils import calculate_secondary_mass
from evolution_parameters import EvolutionParameters

#The foal is to define simple callable.
#pylint: disable=too-few-public-methods
class PriorTransformBase(EvolutionParameters):
    """
    Base class for transforming the unit cube to evolution parameters.

    Attrs:
        direct_observables:    See same name argument to :meth:`__init__`.
    """

    _priors_order = ['dissipation', 'evolution', 'system']

    def _get_direct_observables(self, unit_cube_values):
        """Turn the unit-cube values to values for the direct observables."""

        result = dict()
        cube_index = 0
        for name, measurement in self.direct_observable_distributions:
            if measurement is None:
                result[name] = unit_cube_values[cube_index]
                cube_index += 1
            elif getattr(measurement, 'plus_error', 0) == 0:
                assert getattr(measurement, 'minus_error', 0) == 0
                result[name] = measurement
            else:
                cube_value = unit_cube_values[cube_index]
                cube_index += 1

                result[name] = (
                    measurement
                    +
                    norm.ppf(cube_value) * (
                        measurement.minus_error if cube_value < 0.5
                        else measurement.plus_error
                    ) * measurement.unit
                )

        return result

    #Define the intefrace that can be overwritten
    #pylint: disable=no-self-use
    #pylint: disable=unused-argument
    def _get_coupled_parameters(self, direct_observables):
        """Return evolution params which cannot be calculated independantly."""

        return dict()
    #pylint: enable=no-self-use
    #pylint: enable=unused-argument

    @staticmethod
    def get_secondary_mass(direct_observables):
        """
        Return the secondary mass per given direct observables.

        Args:
            direct_observables(dict):     Specific values (with units) to assume
                for the directly observable system parameters. It must define at
                least: `primary_mass`, `orbital_period`, `rv_semi_amplitude`,
                `eccentricity`, and `inclination`.
        """

        return calculate_secondary_mass(
            primary_mass=direct_observables['primary_mass'],
            orbital_period=direct_observables['orbital_period'],
            rv_semi_amplitude=direct_observables['rv_semi_amplitude'],
            eccentricity=direct_observables['eccentricity'],
            inclination=direct_observables['inclination']
        )

    def __init__(self,
                 direct_observable_distributions,
                 secondary_is_star):
        """
        Set-up the prior transform per the given system and evolution params.

        Args:
            direct_observable_distributions(2-tuple):     The names and
                distributions of the directly observable parameters we will
                sample directly from. The distributions are assumed i.i.d.
                following a normal distribution with possibly different positive
                and negative standard deviations given by the `plus_error` and
                `minus_error` attributes to each value respectively. Values must
                have units if not unitless. If errors are zero or unspecified,
                the corresponding parameter is assumed to have a fixed values
                and does not consume a unit cube entry. If an observable has a
                value of None, it is set directly equal to a unique unit cube
                variable.

        Returns:
            None
        """

        super().__init__(secondary_is_star)
        self.direct_observable_distributions = direct_observable_distributions

    def __call__(self, unit_cube_values):
        """Return an array of the parameter values for evolving the system."""

        transformed_values = scipy.empty(shape=(len(self.parameter_order),),
                                         fill_value=scipy.nan,
                                         dtype=float)

        direct_observables = self._get_direct_observables(unit_cube_values)

        for param_index, (param_name, param_units) in enumerate(
                self.parameter_order
        ):
            if param_name in direct_observables:
                transformed_values[param_index] = direct_observables[param_name]
            elif hasattr(self, 'get_' + param_name):
                transformed_values[param_index] = getattr(
                    self,
                    'get_' + param_name
                )(
                    direct_observables
                ).to_value(
                    param_units
                )

        for param_name, param_value in self._get_coupled_parameters(
                direct_observables
        ):
            (
                transformed_index,
                transformed_units
            ) = self.get_parameter_index_units(param_name)
            transformed_values[transformed_index] = param_value.to_value(
                transformed_units
            )

        return transformed_values
#pylint: enable=too-few-public-methods
