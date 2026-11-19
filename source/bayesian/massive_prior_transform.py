"""Derive the release/formation age from the primary rather than sampling it."""
from astropy import units

from bayesian.prior_transform_windemuth_et_al import PriorTransformWindemuth
from bayesian.massive_star_policy import select_cache_root, validate_selection, zams_grid


class MassivePriorTransformWindemuth(PriorTransformWindemuth):
    def _calculate_disk_dissipation_age(self, parameters):
        m1 = parameters['primary_mass'].to_value(units.M_sun)
        m2 = parameters['secondary_mass'].to_value(units.M_sun)
        feh = parameters['feh'].to_value(units.dimensionless_unscaled)
        try:
            validate_selection(m1,m2,feh)
            age = zams_grid(str(select_cache_root())).evaluate(m1,feh)['starting_age_gyr']
        except ValueError:
            # Invalid prior draws still encode finite parameters; the likelihood
            # rejects their selection before any native evolution is attempted.
            age = 0.
        return age*units.Gyr
