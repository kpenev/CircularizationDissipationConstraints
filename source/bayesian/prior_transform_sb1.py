"""Define prior transformations for SB1 and exoplanet systems."""

from astropy import units

from bayesian.prior_transform_base import PriorTransformBase

#Intended to function as callable no need for more public methods
#pylint: disable=too-few-public-methods
class PriorTransformSB1(PriorTransformBase):
    """Prior transfromations for SB1 binary star and exoplanet systems."""

    def _fill_coupled_parameters(self,
                                 unit_cube_iter,
                                 model_parameters):
        """Fills primary and secondary mass."""

        if model_parameters is None:
            next(unit_cube_iter)
            next(unit_cube_iter)
            return

        primary_mass, secondary_mass = self._sample_binary_masses(
            next(unit_cube_iter),
            next(unit_cube_iter)
        )

        model_parameters['primary_mass'] = primary_mass * units.M_sun
        model_parameters['secondary_mass'] = secondary_mass * units.M_sun

        for component in ['primary', 'secondary']:
            model_parameters[
                'cmd_%s_radius' % component
            ] = (
                self
                .
                _sample_binary_masses
                .
                photometric_constraint
                .
                get_component_radius
            )(
                model_parameters[component + '_mass'].to_value(units.M_sun)
            )[0] * units.R_sun

    def __init__(self,
                 sample_binary_masses,
                 **kwargs):
        """
        Create a transform using the given sampler for primary star properties.

        Args:
            sample_binary_masses(SampleBinaryMasses):    Define the distribution
                of the masses of the system components. Must have
                `photometric_mass_constraint` attribute, used to define fallback
                radius for non-evolving stars.

            kwargs:    Passed directly to parent's `__init__()`.

        Returns:
            None
        """

        super().__init__(**kwargs)
        self._sample_binary_masses = sample_binary_masses
#pylint: enable=too-few-public-methods
