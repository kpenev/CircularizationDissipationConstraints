"""Gaussian Gaia stellar inputs and radii derived from the massive-star grid."""
import math
import numpy as np
from astropy import units
from scipy import stats

from bayesian.prior_transform_base import PriorTransformBase
from bayesian.massive_star_policy import select_cache_root, validate_selection, zams_grid


class GaiaPriorTransform(PriorTransformBase):
    coupled_names = ('orbital_period','primary_mass','secondary_mass','feh','age')
    observed_names = ('period','mass1','mass2','feh','age')

    def __init__(self, observation, *, interpolator, **kwargs):
        self.observation = observation
        self.interpolator = interpolator
        self.grid_root = str(select_cache_root())
        self.distributions = {name: stats.norm(observation['means'][name], observation['sigmas'][name])
                              for name in self.observed_names}
        super().__init__(**kwargs)

    def _fill_coupled_parameters(self, unit_cube_iter, model_parameters, identify=False):
        for model_name, observed_name, unit in zip(
                self.coupled_names,self.observed_names,
                (units.day,units.M_sun,units.M_sun,units.dimensionless_unscaled,units.Gyr)):
            u = next(unit_cube_iter)
            if model_parameters is not None:
                model_parameters[model_name] = (u if identify else
                    self.distributions[observed_name].ppf(np.clip(u,np.nextafter(0.,1.),np.nextafter(1.,0.)))*unit)
        if model_parameters is not None and not identify:
            # The two catalog component Gaussians are drawn first; labels then
            # follow the requested convention that the primary is more massive.
            m1,m2=model_parameters['primary_mass'],model_parameters['secondary_mass']
            model_parameters['primary_mass'],model_parameters['secondary_mass']=max(m1,m2),min(m1,m2)

    def _stellar_inputs(self, parameters):
        m1=parameters['primary_mass'].to_value(units.M_sun)
        m2=parameters['secondary_mass'].to_value(units.M_sun)
        feh=parameters['feh'].to_value(units.dimensionless_unscaled)
        validate_selection(m1,m2,feh)
        return m1,m2,feh

    def _calculate_disk_dissipation_age(self, parameters):
        try:
            m1,m2,feh=self._stellar_inputs(parameters)
            age=zams_grid(self.grid_root).evaluate(m1,feh)['starting_age_gyr']
        except ValueError:
            age=0. # Finite placeholder; the inherited likelihood rejects this draw.
        return age*units.Gyr

    def _radius(self, parameters, component):
        try:
            m1,m2,feh=self._stellar_inputs(parameters)
            mass=m1 if component=='primary' else m2
            age=parameters['age'].to_value(units.Gyr)
            if not math.isfinite(age) or age<=0:
                return 0.*units.R_sun
            q=self.interpolator('RADIUS',mass,feh)
            try:
                if not q.min_age<=age<=q.max_age:
                    return 0.*units.R_sun
                value=float(q(age))
                return (value if math.isfinite(value) and value>0 else 0.)*units.R_sun
            finally:
                q.delete()
        except ValueError:
            return 0.*units.R_sun

    def _calculate_cmd_primary_radius(self, parameters):
        return self._radius(parameters,'primary')

    def _calculate_cmd_secondary_radius(self, parameters):
        return self._radius(parameters,'secondary')

    def __call__(self, unit_cube_values):
        result=super().__call__(unit_cube_values)
        result['sample_weights_envelope']=np.ones(1)
        return result
