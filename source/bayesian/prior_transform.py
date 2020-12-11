"""Define base class for transforming the unit cube to evolution parameters."""

import scipy
from scipy.stats import norm

#The foal is to define simple callable.
#pylint: disable=too-few-public-methods
class PriorTransformBase:
    """
    Base class for transforming the unit cube to evolution parameters.

    Attrs:
        prior_info:    dictionary containing:

            * system: the value of the `system_parameters` argument to
              :meth:`__init__`.

            * dissipation: the value of the `dissipation_parameters` argument to
              :meth:`__init__`.

            * evolution: the value of the `evolution_parameters` argument to
              :meth:`__init__`.

        parameter_names_units:    See same name argument to :meth:`__init__`.
    """

    _parameter_class_order = ['dissipation', 'evolution', 'system']

    def __init__(self,
                 *,
                 parameter_names_units,
                 **prior_info):
        """
        Set-up the prior transform per the given system and evolution params.

        Args:
            system_parameters:    Object with attributes giving the directly
                observed properties of the system (e.g. RV semi-amplitude
                instead of secondary mass), complete with units and error bars.

            evolution_parameters:     All non-system specific parameters
                required to fully specify the evolution (see
                :attr:`LogLikelihoodBase.parameter_names_units`), complete with
                units and error bars.

            dissipation_parameters:    All parameters specifying the dissipation
                (see :attr:`LogLikelihoodBase.parameter_names_units`), complete
                with units and error bars.

            parameter_names_units:    See
                :attr:`LogLikelihoodBase.parameter_names_units`

        Returns:
            None
        """

        self.prior_info = prior_info
        self.parameter_names_units = parameter_names_units

    def __call__(self, unit_cube_values):
        """
        Return an array of the parameter values for evolving the system.

        The order and units of the parameters should be exactly as specified by
        :attr:`parameter_names_units`.
        """

        parameter_values = scipy.empty(shape=(len(self.parameter_names_units),),
                                       dtype=float)

        start_param_index = 0
        cube_index = 0
        for parameter_class in self._parameter_class_order:

            for param_index, (param_name, units) in enumerate(
                    self.parameter_names_units[parameter_class],
                    start_param_index
            ):
                measurement = getattr(self.prior_info[parameter_class],
                                      param_name)
                if getattr(measurement, 'plus_error', 0) == 0:
                    assert getattr(measurement, 'minus_error', 0) == 0
                    parameter_values[param_index] = measurement.to_value(units)
                else:
                    cube_value = unit_cube_values[cube_index]
                    cube_index += 1

                    parameter_values[param_index] = (
                        measurement
                        +
                        norm.ppf(cube_value) * (
                            measurement.minus_error if cube_value < 0.5
                            else measurement.plus_error
                        )
                    ).to_value(units)

            start_param_index += len(
                self.parameter_names_units[parameter_class]
            )

        return parameter_values
#pylint: enable=too-few-public-methods
