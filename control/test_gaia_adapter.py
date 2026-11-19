"""Input and distribution checks only; never evolve an orbit or run sampling."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT/'source'),str(ROOT/'control')]
import numpy as np
from astropy import units
from bayesian.gaia_observations import load_systems,conditional_eccentricity,eccentricity_distribution
from bayesian.gaia_prior_transform import GaiaPriorTransform


class GaiaInputTests(unittest.TestCase):
    def test_selected_ids_and_positive_errors(self):
        systems=load_systems()
        self.assertEqual(len(systems),7)
        self.assertNotIn('1857469799576340864',systems)
        self.assertIn('5413997561575480832',systems)
        for key,obs in systems.items():
            self.assertIsInstance(key,str)
            self.assertTrue(all(x>0 for x in obs['sigmas'].values()))

    def test_asymmetric_sigma_and_linear_age(self):
        obs=load_systems()['5413997561575480832']
        self.assertAlmostEqual(obs['sigmas']['mass1'],(.1144621+.144332)/2)
        self.assertAlmostEqual(obs['sigmas']['age'],(.04190202799584+.05155075647698)/2)
        self.assertAlmostEqual(obs['means']['age'],.2238721138568)

    def test_joint_gaussian_covariance(self):
        obs=load_systems()['459349814627528832']
        rng=np.random.default_rng(371)
        z=rng.normal(size=(2,200000))
        periods=obs['means']['period']+obs['sigmas']['period']*z[0]
        means,sigma=conditional_eccentricity(obs,periods)
        eccentricities=means+sigma*z[1]
        np.testing.assert_allclose(np.corrcoef(periods,eccentricities)[0,1],obs['rho'],atol=.008)
        np.testing.assert_allclose(np.std(eccentricities),obs['sigmas']['eccentricity'],rtol=.01)

    def test_eccentricity_support_and_normalization(self):
        for obs in load_systems().values():
            mean,sigma=conditional_eccentricity(obs,obs['means']['period'])
            d=eccentricity_distribution(mean,sigma)
            lo,hi=d.support()
            self.assertGreaterEqual(lo,0)
            self.assertLessEqual(hi,1)
            self.assertAlmostEqual(d.cdf(hi)-d.cdf(lo),1)
            self.assertTrue(np.isfinite(d.pdf(d.ppf(.5))))

    def test_gaussian_ordering_and_linear_age_units(self):
        obs=load_systems()['5413997561575480832']
        p=GaiaPriorTransform.__new__(GaiaPriorTransform)
        from scipy import stats
        p.distributions={key:stats.norm(obs['means'][key],obs['sigmas'][key]) for key in p.observed_names}
        model={}
        p._fill_coupled_parameters(iter([.5,.01,.99,.5,.5]),model)
        self.assertGreaterEqual(model['primary_mass'],model['secondary_mass'])
        self.assertAlmostEqual(model['age'].to_value(units.Gyr),obs['means']['age'])
        self.assertAlmostEqual(model['orbital_period'].to_value(units.day),obs['means']['period'])
        p.independent_parameter_distributions=[]
        self.assertEqual(p.count_sampled_parameters(),5)

    def test_radius_units_and_native_object_cleanup(self):
        class Quantity:
            min_age=.001
            max_age=.5
            def __call__(self,age):return 2.4
            def delete(self):self.deleted=True
        q=Quantity()
        p=GaiaPriorTransform.__new__(GaiaPriorTransform)
        p.interpolator=lambda name,mass,feh:q
        parameters=dict(primary_mass=2.1*units.M_sun,secondary_mass=1.9*units.M_sun,
                        feh=.1*units.dimensionless_unscaled,age=.2*units.Gyr)
        self.assertEqual(p._calculate_cmd_primary_radius(parameters).to_value(units.R_sun),2.4)
        self.assertTrue(q.deleted)
        q.deleted=False
        parameters['age']=.7*units.Gyr
        self.assertEqual(p._calculate_cmd_primary_radius(parameters).value,0)
        self.assertTrue(q.deleted)


if __name__=='__main__':
    unittest.main()
