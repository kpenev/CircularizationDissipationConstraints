"""Bounded native smoke checks for direct ZAMS spins; no posterior sampling."""
import ctypes
import json
import logging
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path.cwd()
ctypes.CDLL('libgslcblas.so',mode=ctypes.RTLD_GLOBAL)
import numpy as np
from astropy import units
from scipy import stats
import bayesian.sample_windemuth_et_al
import bayesian.log_likelihood_windemuth_et_al as lm
from bayesian.binary_utils import prepare_sampling_common,get_common_binary_star_priors
from bayesian.massive_star_policy import select_cache_root,prepare_binary_start
from general_purpose_python_modules.solve_for_initial_values import InitialValueFinder
def encoded_values(likelihood, **changes):
    values=dict(lgQ_min=8.,lgQ_inertial_boost=0.,lgQ_inertial_sharpness=10.,
                lgQ_break_period=5.,lgQ_powerlaw=0.,disk_dissipation_age=.02,
                primary_disk_lock_period=5.,secondary_disk_lock_period=5.,
                primary_wind_strength=.17,secondary_wind_strength=.17,
                primary_wind_saturation=2.45,secondary_wind_saturation=2.45,
                primary_core_envelope_coupling_timescale=.01,
                secondary_core_envelope_coupling_timescale=.01,
                initial_eccentricity=.3,age=.02,feh=.01,orbital_period=10.,
                primary_mass=2.08,secondary_mass=1.67,
                cmd_primary_radius=2.,cmd_secondary_radius=2.)
    values.update(changes)
    return np.array([values[name] for name,_ in likelihood.parameter_order])


def main():
    logging.basicConfig(level=logging.CRITICAL)
    report=dict(package=str(ROOT),checks=[],native_cases=[])
    out=ROOT/'ZAMS_SPIN_SMOKE.json'
    try:
        cache=select_cache_root()
        interp=prepare_sampling_common(SimpleNamespace(
            stellar_evolution_interpolator_dir=str(cache/'interpolators/rc_env'),
            eccentricity_expansion_coefficients='/work/dal205149/poet/eccentricity_expansion_coef_O30_from_O400.sqlite'))
        cfgname='Gaia_massive_rc_env.cfg' if ROOT.name.startswith('gaia_') else 'W19_massive_rc_env.cfg'
        from bayesian.parse_command_line import parse_command_line
        systemid='5413997561575480832' if ROOT.name.startswith('gaia_') else '6363494'
        sys.argv=[sys.argv[0],systemid,'-c',str(ROOT/'source/sampling_config'/cfgname)]
        config=parse_command_line('Spin smoke',cfgname,dissipation=True,cluster=False,
                                  choose_binary='gaia' if ROOT.name.startswith('gaia_') else 'w19',spindown=2)
        priors=get_common_binary_star_priors(config)
        for component in ('primary','secondary'):
            dist=next(p[1] for p in priors if p[0]==component+'_disk_lock_period')
            np.testing.assert_allclose(dist.ppf([0,.25,.5,.75,1]),[.5,12.875,25.25,37.625,50.])
        report['checks'].append('both configured priors uniform in period 0.5-50 days')
        ll=lm.LogLikelihoodWindemuth(observed_eccentricity_distro=stats.uniform(0,.8),
            interpolator=interp,envelope_eccentricity=.8,powerlaw_dissipation=True,
            evolution_timeout=30,period_search_factor=2,scaled_period_guess=1,
            de_distro=stats.uniform(0,.8),pe_distro=stats.uniform(0,.8),
            system_eccentricity=.3,system_name='ZAMS_SPIN_SMOKE',warm_start_enabled=False,use_legacy_pnn=False)
        captured=[]
        def spy(**kwargs):
            captured.append(kwargs)
            return SimpleNamespace(eccentricity=np.array([np.nan])),[np.nan,np.nan]
        with patch.object(lm,'find_evolution',side_effect=spy):
            ll(encoded_values(ll,primary_disk_lock_period=3.7,secondary_disk_lock_period=17.3),sample_weights_envelope=np.ones(1))
        assert len(captured)==1
        parsed=captured[0]
        assert parsed['disk_period'].to_value(units.day)==3.7
        assert parsed['secondary_disk_period'].to_value(units.day)==17.3
        report['checks'].append('distinct primary and secondary periods reach solver')
        start=parsed['binary_start_age'].to_value(units.Gyr)
        # Approximately 100 yr, with weak tides and an asynchronous orbit.
        end=start+1e-7
        for p1,p2 in [(3.7,17.3),(.5,50.),(50.,.5)]:
            system=SimpleNamespace(primary_mass=2.08*units.M_sun,secondary_mass=1.67*units.M_sun,
                feh=.01,age=end*units.Gyr,orbital_period=10.*units.day)
            finder=InitialValueFinder(system,interp,current_age=system.age,
                disk_period=p1*units.day,secondary_disk_period=p2*units.day,
                initial_obliquity=0.,disk_dissipation_age=start*units.Gyr,binary_start_age=start*units.Gyr,
                dissipation=parsed['dissipation'],primary_wind_strength=.17,secondary_wind_strength=.17,
                primary_wind_saturation=2.45,secondary_wind_saturation=2.45,
                primary_core_envelope_coupling_timescale=.01*units.Gyr,
                secondary_core_envelope_coupling_timescale=.01*units.Gyr,
                secondary_star=True,max_time_step=1e-8,precision=1e-7)
            secondary_l=finder.get_secondary_initial_angmom()
            primary=finder._create_primary();secondary=finder._create_secondary()
            binary=None
            try:
                binary=finder._create_system(primary,secondary,porb_initial=10.,initial_eccentricity=.3,
                    initial_obliquity=0.,initial_secondary_angmom=secondary_l)
                # Verify all four initialized angular momenta using native state.
                binary.evolve(end,1e-8,1e-7,None,timeout=30)
                ev=binary.get_evolution()
                names=['primary_envelope_angmom','primary_core_angmom','secondary_envelope_angmom','secondary_core_angmom']
                actual=np.array([getattr(ev,n)[0] for n in names])
                inertia=np.array([primary.envelope_inertia(start),primary.core_inertia(start),
                                  secondary.envelope_inertia(start),secondary.core_inertia(start)])
                np.testing.assert_allclose(ev.age[0],start,rtol=0,atol=1e-12)
                np.testing.assert_allclose(actual,inertia*2*np.pi/np.array([p1,p1,p2,p2]),rtol=1e-8,atol=1e-12)
                final=binary.final_state()
                assert np.isfinite(final.semimajor) and final.semimajor>0
                assert np.isfinite(final.eccentricity) and 0<=final.eccentricity<1
                assert abs(final.age-end)<1e-12
                report['native_cases'].append(dict(periods_days=[p1,p2],start_age_gyr=start,
                    final_age_gyr=float(final.age),initial_periods_days=(2*np.pi*inertia/actual).tolist(),
                    final_eccentricity=float(final.eccentricity)))
            finally:
                if binary is not None:binary.delete()
                primary.delete();secondary.delete()
        interp.delete()
        report['passed']=True
    except BaseException as error:
        report.update(passed=False,error=repr(error))
        raise
    finally:
        out.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)


if __name__=='__main__':main()
