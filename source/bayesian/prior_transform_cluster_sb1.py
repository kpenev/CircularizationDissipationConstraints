"""Define prior transformations for SB1 and exoplanet systems."""

from astropy.units import dimensionless_unscaled as dimensionless
from abc import abstractmethod

from prior_transform_base import PriorTransformBase
from binary_utils import calculate_secondary_mass

import numpy

class PriorTransformClusterSB1(PriorTransformBase):
    """Prior transfromations for SB1 binary star and exoplanet systems."""

    def _fill_coupled_parameters(self, unit_cube_iter, model_parameters):
        """Fills [Fe/H], current system age, and primary mass."""

        primary_mass = self._photometric_mass_constraint.primary_mass_ppf(
            next(unit_cube_iter)
        )
        secondary_mass = (
            self._photometric_mass_constraint
        ).get_conditional_secondary_mass_distribution(
            primary_mass
        ).ppf(
            next(unit_cube_iter)
        )
        model_parameters['primary_mass'] = primary_mass * units.M_sun
        model_parameters['secondary_mass'] = secondary_mass * units.M_sun

    def __init__(self,
                 photometric_mass_constraint,
                 rv_semi_amplitude_distribution,
                 **kwargs):
        """
        Create a transform using the given sampler for primary star properties.

        Args:
            photometric_mass_constraint(PhotometricConstraint):    The
                constraint on the component masses based on photometric
                measurements.

            rv_semi_amplitude_distribution(MarginalizedRVKDistribution):
                Interface to the RV semi-amplitude distribution marginalized
                over inclination.

            kwargs:    Passed directly to parent's `__init__()`.

        Returns:
            None
        """

        super().__init__(**kwargs)
        self._photometric_mass_constraint = photometric_mass_constraint
        self._rv_semiamplitude_distribution = rv_semi_amplitude_distribution

def main(config):
    """Avoid polluting global namespace."""
