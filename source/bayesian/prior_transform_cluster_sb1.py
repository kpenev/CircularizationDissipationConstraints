"""Define prior transformations for SB1 and exoplanet systems."""

from astropy import units

from bayesian.prior_transform_base import PriorTransformBase

#Intended to function as callable no need for more public methods
#pylint: disable=too-few-public-methods
class PriorTransformClusterSB1(PriorTransformBase):
    """Prior transfromations for SB1 binary star and exoplanet systems."""

    def _fill_coupled_parameters(self,
                                 unit_cube_iter,
                                 model_parameters):
        """Fills [Fe/H], current system age, and primary mass."""

        if model_parameters is None:
            next(unit_cube_iter)
            next(unit_cube_iter)
            return

        primary_mass = self._photometric_mass_constraint.primary_mass_ppf(
            next(unit_cube_iter)
        )
        model_parameters['primary_mass'] = primary_mass * units.M_sun

        self._rv_semiamplitude_constraint.prepare_secondary_sampling(
            primary_mass=model_parameters['primary_mass'],
            eccentricity=model_parameters['initial_eccentricity'],
            orbital_period=model_parameters['orbital_period'],
            secondary_mass_prior=(
                self._photometric_mass_constraint
            ).get_conditional_secondary_mass_distribution(
                primary_mass
            )
        )

        model_parameters['secondary_mass'] = (
            self._rv_semiamplitude_constraint
        ).secondary_mass_ppf(
            next(unit_cube_iter)
        )
        for component in ['primary', 'secondary']:
            model_parameters[
                'cmd_%s_radius' % component
            ] = self._photometric_mass_constraint.get_component_radius(
                model_parameters[component + '_mass'].to_value(units.M_sun)
            )

    def __init__(self,
                 photometric_mass_constraint,
                 rv_semi_amplitude_constraint,
                 **kwargs):
        """
        Create a transform using the given sampler for primary star properties.

        Args:
            photometric_mass_constraint(PhotometricConstraint):    The
                constraint on the component masses based on photometric
                measurements.

            rv_semi_amplitude_constraint(RVSemiAmplitudeConstraint):
                Interface to the RV semi-amplitude distribution marginalized
                over inclination.

            kwargs:    Passed directly to parent's `__init__()`.

        Returns:
            None
        """

        super().__init__(**kwargs)
        self._photometric_mass_constraint = photometric_mass_constraint
        self._rv_semiamplitude_constraint = rv_semi_amplitude_constraint
#pylint: enable=too-few-public-methods
