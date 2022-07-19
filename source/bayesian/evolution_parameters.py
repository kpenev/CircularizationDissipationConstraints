"""Define a bookkeeping class for the parameters used in the evolution."""

import logging

from astropy import units

class EvolutionParameters:
    """
    Define interface for working with parameters required to find the evolution.

    Attrs:
        parameter_names_units:    The names and units of the collection of
            parameters fully determining the orbital evolution for a system.

        parameter_order:    The order in which parameters appear in the input
            array to :meth:`__call__`.
    """

    def log_parameters(self, message, parameters, level):
        """
        Issue log message along with a description of this step's parameters.

        Args:
            parameters(array):    The parameter array defining the current steps
                being attempted (to be included in the message).

            message(str):    The message to issue before describing the
                parameters.

            level(str):    One of `'debug'`, `'warning'`, `'info'`, `'error'`,
                `'critical'` definng the importance level of the message to
                issue.

        Returns:
            None
        """

        logging.getLogger(__name__).log(
            level,
            message + '\n\t%s: %s %s' * parameters.size,
            *(
                sub
                for (param_name, param_units), value in zip(
                    self.parameter_order,
                    parameters
                )
                for sub in (param_name, repr(value), param_units)
            )
        )

    def __init__(self,
                 secondary_is_star,
                 dissipation_parameters,
                 extra_parameters=None):
        """
        Prepare to manage the parameters for a given system.

        Args:
             secondary_is_star(bool):    True iff the secondary in the system is
                an evolving star.

        Returns:
            None
        """

        #TODO: fix units of wind parameters
        self.parameter_names_units = dict(
            dissipation=dissipation_parameters,
            evolution=[
                ('disk_dissipation_age', units.Gyr),
                ('primary_disk_lock_period', units.day),
                ('primary_wind_strength', units.dimensionless_unscaled),
                ('primary_wind_saturation', units.dimensionless_unscaled),
                ('primary_core_envelope_coupling_timescale', units.Gyr),
                ('initial_eccentricity', units.dimensionless_unscaled)
            ],
            system=[
                ('age', units.Gyr),
                ('feh', units.dimensionless_unscaled),
                ('orbital_period', units.day),
                ('primary_mass', units.M_sun),
                ('secondary_mass', units.M_sun),
                ('cmd_primary_radius', units.R_sun),
                ('cmd_secondary_radius', units.R_sun)
            ]
        )

        if secondary_is_star:
            self.parameter_names_units['evolution'].extend(
                [
                    ('secondary_disk_lock_period', units.day),
                    ('secondary_wind_strength', units.dimensionless_unscaled),
                    ('secondary_wind_saturation', units.dimensionless_unscaled),
                    ('secondary_core_envelope_coupling_timescale', units.Gyr)
                ]
            )
        else:
            self.parameter_names_units['system'].append(
                ('secondary_radius', units.R_sun)
            )

        self.parameter_order = (
            self.parameter_names_units['dissipation']
            +
            self.parameter_names_units['evolution']
            +
            self.parameter_names_units['system']
        )
        if extra_parameters is not None:
            self.parameter_names_units['extra'] = extra_parameters
            self.parameter_order += extra_parameters

        self.parameter_indices = dict(
            (name, index)
            for index, (name, units) in enumerate(self.parameter_order)
        )

    def get_parameter_index_units(self, parameter_name):
        """Return the index and units of the parameter with the given name."""

        index = self.parameter_indices[parameter_name]
        return index, self.parameter_order[index][1]

    def get_parameter_value(self, parameters, parameter_name):
        """Return the value with units of a parameter by name."""

        index = self.parameter_indices[parameter_name]
        return parameters[index] * self.parameter_order[index][1]
