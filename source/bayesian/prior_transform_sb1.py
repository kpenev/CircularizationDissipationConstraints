"""Define prior transformations for SB1 and exoplanet systems."""

from astropy.units import dimensionless_unscaled as dimensionless

from prior_transform_base import PriorTransformBase
from binary_utils import calculate_secondary_mass

class PriorTransformSB1(PriorTransformBase):
    """Prior transfromations for SB1 binary star and exoplanet systems."""

    def _fill_coupled_parameters(self, unit_cube_iter, model_parameters):
        """Fills [Fe/H], current system age, and primary mass."""

        feh, mass, age = self.primary_star_sampler(
            numpy.array([next(unit_cube_iter),
                         next(unit_cube_iter),
                         next(unit_cube_iter)])
        )

        model_parameters['feh'] = feh * dimensionless
        model_parameters['primary_mass'] = mass * units.M_sun
        model_parameters['age'] = age * units.Gyr

    @staticmethod
    def _calculate_secondary_mass(model_parameters):
        """
        Calculate the secondary mass given model parameters.

        Args:
            model_parameters(dict):     Specific values (with units) to assume
                for the system parameters. It must define at least:
                `primary_mass`, `orbital_period`, `rv_semi_amplitude`,
                `eccentricity`, and `inclination`.
        """

        return calculate_secondary_mass(
            primary_mass=model_parameters['primary_mass'],
            orbital_period=model_parameters['orbital_period'],
            rv_semi_amplitude=model_parameters['rv_semi_amplitude'],
            eccentricity=model_parameters['eccentricity'],
            inclination=model_parameters['inclination']
        )

    def __init__(self, primary_star_sampler, **kwargs):
        """
        Create a transform using the given sampler for primary star properties.

        Args:
            primary_star_sampler(StarSampler):    Callable trat will transform 3
                U(0,1) random values to [Fe/H], mass, and age of a star with
                distributions determined by observational constraints.

            kwargs:    Passed directly to parent's `__init__()`.

        Returns:
            None
        """

        super().__init__(**kwargs)
        self.primary_star_sampler = primary_star_sampler
