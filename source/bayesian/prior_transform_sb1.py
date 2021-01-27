"""Define prior transformations for SB1 and exoplanet systems."""

from astropy.units import dimensionless_unscaled as dimensionless
from abc import abstractmethod

from prior_transform_base import PriorTransformBase
from binary_utils import calculate_secondary_mass

import numpy

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

    def __init__(self,
                 primary_star_sampler,
                 rv_semi_amplitude_distribution,
                 **kwargs):
        """
        Create a transform using the given sampler for primary star properties.

        Args:
            primary_star_sampler(StarSampler):    Callable trat will transform 3
                U(0,1) random values to [Fe/H], mass, and age of a star with
                distributions determined by observational constraints.

            rv_semi_amplitude_distribution(MarginalizedRVKDistribution):
                Interface to the RV semi-amplitude distribution marginalized
                over inclination.

            kwargs:    Passed directly to parent's `__init__()`.

        Returns:
            None
        """

        super().__init__(**kwargs)
        self._primary_star_sampler = primary_star_sampler
        self._rv_semiamplitude_distribution = rv_semi_amplitude_distribution

    @abstractmethod
    def max_secondary_mass(self, feh, metallicity, primary_mass):
        """
