"""Exercise the copied profile's rejection, ZAMS wiring and orbital solver."""
import ctypes
import hashlib
import json
import logging
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path('/work/dal205149/massive_star_circularization')
sys.path[:0]=[str(ROOT/'source'),str(ROOT/'control'),str(ROOT/'general_purpose_python_modules'),str(ROOT)]
ctypes.CDLL('libgslcblas.so',mode=ctypes.RTLD_GLOBAL)
import numpy as np
from astropy import units
from scipy import stats
# Preserve the production entry point's SciPy compatibility initialization.
import bayesian.sample_windemuth_et_al
import bayesian.log_likelihood_windemuth_et_al as likelihood_module
from bayesian.binary_utils import prepare_sampling_common
from bayesian.massive_prior_transform import MassivePriorTransformWindemuth
from bayesian.massive_star_policy import FLOORS,prepare_binary_start,select_cache_root,zams_grid
from general_purpose_python_modules.solve_for_initial_values import InitialValueFinder
from general_purpose_python_modules.reproduce_system import find_evolution


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
    root=select_cache_root();os.environ['MASSIVE_STAR_GRID_ROOT']=str(root)
    config=SimpleNamespace(stellar_evolution_interpolator_dir=str(root/'interpolators/rc_env'),
                           eccentricity_expansion_coefficients='/work/dal205149/poet/eccentricity_expansion_coef_O30_from_O400.sqlite')
    interp=prepare_sampling_common(config)
    summary=dict(profile='massive-rc-env-primary-zams-v1',grid_root=str(root))
    try:
        likelihood=likelihood_module.LogLikelihoodWindemuth(
            observed_eccentricity_distro=stats.uniform(0,.8),
            interpolator=interp,envelope_eccentricity=.8,powerlaw_dissipation=True,
            evolution_timeout=120,period_search_factor=2,scaled_period_guess=1,
            de_distro=stats.uniform(0,.8),pe_distro=stats.uniform(0,.8),
            system_eccentricity=.3,system_name='SYNTHETIC_PROFILE_CHECK',
            warm_start_enabled=False,use_legacy_pnn=False)
        calls=[]
        original_find=likelihood_module.find_evolution
        def spy(**kwargs):
            calls.append(kwargs)
            return SimpleNamespace(eccentricity=np.array([np.nan])),[np.nan,np.nan]
        likelihood_module.find_evolution=spy
        rejected=[]
        try:
            invalid=[dict(primary_mass=1.49),dict(secondary_mass=3.01),
                     dict(primary_mass=1.65,feh=0.),dict(secondary_mass=1.5,feh=.01),
                     dict(age=.0001),dict(age=10.)]
            for proposal in invalid:
                before=len(calls)
                assert likelihood(encoded_values(likelihood,**proposal),sample_weights_envelope=np.ones(1))==-np.inf
                assert len(calls)==before
                rejected.append(proposal)
            # The current cache may still predate the two lower-metallicity rows.
            if interp.feh_range()[0]>-1.2:
                before=len(calls)
                assert likelihood(encoded_values(likelihood,feh=-1.2),sample_weights_envelope=np.ones(1))==-np.inf
                assert len(calls)==before
            result=likelihood(encoded_values(likelihood),sample_weights_envelope=np.ones(1))
            assert result==-np.inf  # The spy deliberately returns the normal failure sentinel.
            assert len(calls)==1
            parsed=calls[0]
            start=prepare_binary_start(interp,2.08,1.67,.01,.02)
            expected=start['starting_age_gyr']
            assert parsed['binary_start_age'].to_value(units.Gyr)==expected
            assert parsed['disk_dissipation_age'].to_value(units.Gyr)==expected
            assert 'secondary_disk_period' not in parsed
            assert all(parsed['dissipation'][component] is not None for component in ('primary','secondary'))
            summary.update(rejected_before_evolution=rejected,valid_proposal_start_gyr=expected,
                           original_secondary_spin_fallback_preserved=True,both_massive_stars_receive_tides=True)
        finally:likelihood_module.find_evolution=original_find

        # The encoded prior's derived age uses the same ZAMS function and units.
        prior=MassivePriorTransformWindemuth.__new__(MassivePriorTransformWindemuth)
        derived=prior._calculate_disk_dissipation_age(dict(primary_mass=2.08*units.M_sun,
            secondary_mass=1.67*units.M_sun,feh=.01*units.dimensionless_unscaled))
        assert derived.to_value(units.Gyr)==expected
        summary['derived_prior_age_matches']=True
        floor_values={}
        for name,floor in FLOORS.items():
            quantity=interp(name,1.5,-.11)
            try:
                first=max(quantity.min_age*1.001,1e-6)
                ages=np.geomspace(first,zams_grid(str(root)).evaluate(1.5,-.11)['starting_age_gyr'],50)
                actual=np.asarray(quantity(np.ascontiguousarray(ages)))
                assert np.isfinite(actual).all() and (actual>=floor-1e-14).all()
                floor_values[name]=float(actual.min())
            finally:quantity.delete()
        summary['minimum_sampled_quantities']=floor_values

        # A short real unequal-mass evolution, followed by recovery of its known
        # initial orbital period using the unchanged production inverse solver.
        system=SimpleNamespace(primary_mass=2.08*units.M_sun,secondary_mass=1.67*units.M_sun,
            feh=.01,age=.02*units.Gyr,orbital_period=10.*units.day,eccentricity=.3,
            Rprimary=2.*units.R_sun,Rsecondary=2.*units.R_sun)
        options=dict(interpolator=interp,dissipation=parsed['dissipation'],
            disk_period=5.*units.day,
            disk_dissipation_age=.02*units.Gyr,binary_start_age=expected*units.Gyr,
            primary_wind_strength=.17,secondary_wind_strength=.17,
            primary_wind_saturation=2.45,secondary_wind_saturation=2.45,
            primary_core_envelope_coupling_timescale=.01*units.Gyr,
            secondary_core_envelope_coupling_timescale=.01*units.Gyr,
            secondary_is_star=True,max_time_step=1e-3,precision=1e-5)
        forward=find_evolution(system,**options,initial_porb=10.*units.day,
                               initial_eccentricity=.3,solve=False)[0]
        assert np.isfinite(forward.orbital_period[-1]) and np.isfinite(forward.eccentricity[-1])
        assert forward.binary_formation_age==expected
        system.orbital_period=float(forward.orbital_period[-1])*units.day
        recovered=find_evolution(system,**options,initial_eccentricity=.3,solve=True)[0]
        assert np.isfinite(recovered.orbital_period[-1])
        residual=abs(float(recovered.orbital_period[-1])-system.orbital_period.to_value(units.day))
        assert residual<=1e-6
        assert abs(recovered.initial_orbital_period-10.)<1e-4
        assert recovered.binary_formation_age==expected
        summary['orbital_validation']=dict(primary_mass=2.08,secondary_mass=1.67,feh=.01,
            start_age_gyr=expected,final_age_gyr=.02,true_initial_period_days=10.,
            final_period_days=system.orbital_period.to_value(units.day),
            recovered_initial_period_days=float(recovered.initial_orbital_period),
            final_period_residual_days=residual)
    finally:interp.delete()
    manifest=json.loads((ROOT/'SOURCE_COPY_MANIFEST.json').read_text())
    original=Path(manifest['original_root'])
    for name,digest in manifest['source_files'].items():
        assert hashlib.sha256((original/name).read_bytes()).hexdigest()==digest,name
    summary['original_source_unchanged']=True
    (ROOT/'VALIDATION.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('PROFILE_VALIDATED',json.dumps(summary),flush=True)


if __name__=='__main__':main()
