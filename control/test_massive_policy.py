"""Scientific selection boundaries and both-star coverage without native code."""
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT/'source'),str(ROOT/'control')]
from bayesian.massive_star_policy import validate_selection,prepare_binary_start,zams_grid,FLOORS
from zams_age import CONFIGURATION,SCHEMA


class FakeQuantity:
    def __init__(self, owner, name, mass):
        self.owner,self.name,self.mass=owner,name,mass
        self.min_age=.0001
        self.max_age=.1 if owner.short_secondary and mass==1.67 and name=='IRAD' else 1.
    def __call__(self,age):return float('nan') if self.owner.nonfinite else 1.
    def deriv(self,age):return (1.,0.,0.)
    def delete(self):self.owner.deleted+=1


class FakeInterpolator:
    quantity_list=('RADIUS','LUM','ICONV','IRAD','MRAD','RRAD')
    def __init__(self):self.created=self.deleted=0;self.short_secondary=self.nonfinite=False
    def mass_range(self):return (1.5,3.)
    def feh_range(self):return (-1.1,.7)
    def __call__(self,name,mass,feh):
        self.created+=1
        return FakeQuantity(self,name,mass)


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        masses=[i/100 for i in range(150,301,5)]
        fehs=[i/100 for i in range(-110,71,5)]
        document=dict(schema=SCHEMA,eep=202,source_configuration=CONFIGURATION,
            interpolation='bilinear in mass and FeH; natural logarithm of age in years',
            masses_msun=masses,fehs=fehs,
            zams_age_yr=[[math.exp(17-.6*m+.3*f+.1*m*f) for f in fehs] for m in masses])
        (self.root/'zams_age_grid.json').write_text(json.dumps(document))
        self.interp=FakeInterpolator()
    def tearDown(self):zams_grid.cache_clear();self.temp.cleanup()
    def start(self,**changes):
        params=dict(primary_mass=2.08,secondary_mass=1.67,feh=.01,final_age_gyr=.5)
        params.update(changes)
        return prepare_binary_start(self.interp,grid_root=self.root,**params)
    def test_mass_limits_and_no_extra_low_mass_metallicity_cut(self):
        for m in (1.5,1.65,3.0):
            for pair in ((m,2.),(2.,m)):
                for feh in (-1.2,-.01,0.,.01,.7):
                    validate_selection(*pair,feh)
        for args in ((1.49,2.,-.1),(2.,3.01,-.1),(2.,2.,-1.21),(2.,2.,.71),(2.,2.,float('nan'))):
            with self.assertRaises(ValueError):validate_selection(*args)
    def test_primary_zams_shared_with_secondary(self):
        r=self.start()
        expected=zams_grid(str(self.root)).evaluate(2.08,.01)['starting_age_gyr']
        self.assertEqual(r['starting_age_gyr'],expected)
        self.assertTrue(r['stellar_support_checked'])
        self.assertEqual(self.interp.created,self.interp.deleted)
    def test_secondary_core_support_is_required_even_if_radius_is_covered(self):
        self.interp.short_secondary=True
        with self.assertRaisesRegex(ValueError,'secondary stellar support'):self.start()
        self.assertEqual(self.interp.created,self.interp.deleted)
    def test_target_age_is_rejected_instead_of_clipped(self):
        with self.assertRaises(ValueError):self.start(final_age_gyr=2.)
        with self.assertRaises(ValueError):self.start(final_age_gyr=.0002)
    def test_pending_metallicity_extension_rejected_before_native_queries(self):
        with self.assertRaisesRegex(ValueError,'currently loaded'):self.start(feh=-1.15)
        self.assertEqual(self.interp.created,0)
    def test_nonfinite_native_value_rejected_and_quantity_released(self):
        self.interp.nonfinite=True
        with self.assertRaises(ValueError):self.start()
        self.assertEqual(self.interp.created,self.interp.deleted)
    def test_floors_preserved(self):
        self.assertEqual(FLOORS,dict(ICONV=1e-3,IRAD=1e-4,MRAD=1e-3,RRAD=0.))


if __name__=='__main__':unittest.main()
