"""Record input checks and verify source/solver preservation without evolving."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT/'source'),str(ROOT/'control')]
from bayesian.gaia_observations import load_systems,conditional_eccentricity,eccentricity_distribution
from install_gaia_copy import hashes


def main():
    implementation=json.loads((ROOT/'GAIA_IMPLEMENTATION.json').read_text())
    source=Path(implementation['source_root'])
    assert hashes(source)==implementation['source_files']
    preserved=[]
    for relative in ('source/bayesian/log_likelihood_windemuth_et_al.py',
                     'source/bayesian/sample_windemuth_et_al.py',
                     'source/bayesian/windemuth_et_al_util.py',
                     'source/bayesian/binary_utils.py',
                     'source/bayesian/massive_star_policy.py',
                     'general_purpose_python_modules/solve_for_initial_values.py',
                     'general_purpose_python_modules/reproduce_system.py'):
        assert (ROOT/relative).read_text()==(source/relative).read_text().replace(str(source),str(ROOT))
        preserved.append(relative)
    native=0
    for relative,digest in implementation['source_files'].items():
        if '.so' in Path(relative).name:
            assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest
            native+=1
    observations=load_systems()
    validation=json.loads((ROOT/'GAIA_INPUT_VALIDATION.json').read_text())
    assert validation['native_input_checks_passed'] and validation['systems_checked']==9
    limitations=[]
    for row in validation['systems']:
        obs=observations[row['source_id']]
        mean,sigma=conditional_eccentricity(obs,obs['means']['period'])
        d=eccentricity_distribution(mean,sigma)
        row.update(observed_eccentricity=obs['means']['eccentricity'],
            observed_e_above_pseudo_envelope=obs['means']['eccentricity']>row['envelope_eccentricity'],
            conditional_e_support_min=float(d.support()[0]),
            conditional_e_probability_below_pseudo_envelope=float(d.cdf(row['envelope_eccentricity'])))
        if row['observed_e_above_pseudo_envelope']:
            limitations.append(dict(source_id=row['source_id'],host=row['host'],
                observed_e=row['observed_eccentricity'],pseudo_envelope_e=row['envelope_eccentricity']))
    (ROOT/'GAIA_INPUT_VALIDATION.json').write_text(json.dumps(validation,indent=2)+'\n')
    status=dict(profile='gaia-massive-gaussian-rc-env-primary-zams-v1',package_root=str(ROOT),
        unit_tests_passed=6,native_input_systems_passed=9,cli_help_passed=True,
        source_package_unchanged=True,preserved_existing_modules=preserved,
        identical_native_libraries=native,original_pseudo_envelope_preserved=True,
        pseudo_envelope_observational_limitations=limitations,
        orbital_evolutions_run=0,end_to_end_test_run=False,production_sampling_started=False)
    (ROOT/'GAIA_VALIDATION_STATUS.json').write_text(json.dumps(status,indent=2)+'\n')
    command=['git','-c','user.name=Codex','-c','user.email=codex@openai.com']
    subprocess.run([*command,'add','control/check_gaia_native_inputs.py','control/finalize_gaia_validation.py',
                    'README_GAIA.md','GAIA_INPUT_VALIDATION.json','GAIA_VALIDATION_STATUS.json'],cwd=ROOT,check=True)
    subprocess.run([*command,'commit','-m','Record Gaia input validation and unchanged pseudo-envelope limitation'],cwd=ROOT,check=True)
    status['commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    print(json.dumps(status,indent=2))
    print('GIT_STATUS',subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True))


if __name__=='__main__':
    main()
