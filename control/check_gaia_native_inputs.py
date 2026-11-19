"""Check all nine Gaussian transforms and native radii; no orbital evolution."""
import ctypes
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT/'source'),str(ROOT/'control'),str(ROOT/'general_purpose_python_modules'),str(ROOT)]
ctypes.CDLL('libgslcblas.so',mode=ctypes.RTLD_GLOBAL)
import numpy as np
from astropy import units
from bayesian.sample_gaia import prepare_sampling
from bayesian.parse_command_line import parse_command_line
from bayesian.gaia_observations import load_systems,conditional_eccentricity,eccentricity_distribution
from bayesian.massive_star_policy import prepare_binary_start,select_cache_root
from bayesian.log_likelihood_windemuth_et_al import LogLikelihoodWindemuth


def main():
    systems=load_systems()
    cache=select_cache_root()
    sys.argv=[sys.argv[0],next(iter(systems)),'-c',str(ROOT/'source/sampling_config/Gaia_massive_rc_env.cfg'),
        '--stellar-evolution-interpolator-dir',str(cache/'interpolators/rc_env'),
        '--eccentricity-expansion-coefficients','/work/dal205149/poet/eccentricity_expansion_coef_O30_from_O400.sqlite',
        '--nn-data-dir',str(ROOT/'solver_data'),'--disable-warm-start']
    config=parse_command_line('Gaia native input checks only','Gaia_massive_rc_env.cfg',
                              dissipation=True,cluster=False,choose_binary='gaia',spindown=2)
    first_likelihood,first_prior=prepare_sampling(config)
    interp=first_prior.interpolator
    records=[]
    with patch('bayesian.sample_gaia.prepare_sampling_common',return_value=interp):
        for source_id,obs in systems.items():
            config.system=source_id
            likelihood,prior=prepare_sampling(config)
            n=prior.count_sampled_parameters()
            transformed=prior(np.full(n,.5))
            encoded=transformed['parameters']
            assert np.isfinite(encoded).all()
            values={name:float(value) for (name,_),value in zip(prior.parameter_order,encoded)}
            m1,m2=sorted((obs['means']['mass1'],obs['means']['mass2']),reverse=True)
            assert values['primary_mass']==m1 and values['secondary_mass']==m2
            assert values['age']==obs['means']['age']
            start=prepare_binary_start(interp,m1,m2,obs['means']['feh'],obs['means']['age'])
            assert values['disk_dissipation_age']==start['starting_age_gyr']
            for component,mass in (('primary',m1),('secondary',m2)):
                q=interp('RADIUS',mass,obs['means']['feh'])
                try:
                    radius=float(q(obs['means']['age']))
                    assert radius>0
                    assert values[f'cmd_{component}_radius']==radius
                finally:q.delete()
            # A spy stops before the inherited orbital algorithm is entered.
            with patch.object(LogLikelihoodWindemuth,'calculate_log_likelihood',return_value=-123.) as spy:
                assert likelihood.calculate_log_likelihood(encoded,**{k:v for k,v in transformed.items() if k!='parameters'})==-123.
                assert spy.call_count==1
            mean,sigma=conditional_eccentricity(obs,values['orbital_period'])
            np.testing.assert_allclose(likelihood._de.kwds['loc'],mean)
            np.testing.assert_allclose(likelihood._de.kwds['scale'],sigma)
            bad=encoded.copy()
            bad[likelihood.parameter_indices['cmd_primary_radius']]=0
            with patch.object(LogLikelihoodWindemuth,'calculate_log_likelihood',side_effect=AssertionError('orbital path called')):
                assert likelihood.calculate_log_likelihood(bad,sample_weights_envelope=np.ones(1))==-np.inf
            conditional=eccentricity_distribution(mean,sigma)
            envelope=float(likelihood.envelope_eccentricity)
            records.append(dict(source_id=source_id,host=obs['host'],sampled_dimensions=n,
                primary_radius_rsun=values['cmd_primary_radius'],secondary_radius_rsun=values['cmd_secondary_radius'],
                primary_zams_gyr=start['starting_age_gyr'],envelope_eccentricity=envelope,
                observed_eccentricity=obs['means']['eccentricity'],
                observed_e_above_pseudo_envelope=obs['means']['eccentricity']>envelope,
                conditional_e_support_min=float(conditional.support()[0]),
                conditional_e_probability_below_pseudo_envelope=float(conditional.cdf(envelope))))
    record=dict(native_input_checks_passed=True,systems_checked=len(records),systems=records,
        grid_root=str(cache),orbital_evolutions_run=0,end_to_end_test_run=False,production_sampling_started=False)
    (ROOT/'GAIA_INPUT_VALIDATION.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':
    main()
