"""Apply the massive-star profile only to the independent /work package copy."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path('/work/dal205149/gaia_massive_star_circularization')
ORIGINAL = Path('/work/dal205149/nesty_sampling_hpc/production_results/w19_dlock5_dyn_w16_code_cdf_20261002')


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Expected exactly one patch anchor: '+old[:100])
    return text.replace(old,new,1)


def save(relative, text):
    ast.parse(text,filename=relative)
    path=ROOT/relative
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(text);temporary.replace(path)


def main():
    manifest=json.loads((ROOT/'SOURCE_COPY_MANIFEST.json').read_text())
    changed=[]

    name='source/bayesian/binary_utils.py'
    text=(ORIGINAL/name).read_text()
    text=replace_once(text,'import logging\n','import logging\nfrom bayesian.massive_star_policy import FLOORS, setup_interpolator_config\n')
    text=replace_once(text,"    MESAInterpolator.set_quantity_lower_limit('iconv', 1e-3)",
                      '    setup_interpolator_config(config)\n    for name, floor in FLOORS.items():\n        MESAInterpolator.set_quantity_lower_limit(name, floor)')
    save(name,text);changed.append(name)

    name='source/bayesian/log_likelihood_binary_stars.py'
    text=(ORIGINAL/name).read_text()
    text=replace_once(text,"                    >\n                    self.max_dissipative_mstar",
                      "                    < self.min_dissipative_mstar\n                    or\n                    self.get_parameter_value(parameters, component + '_mass') > self.max_dissipative_mstar")
    text=replace_once(text,'                 max_dissipative_mstar=1.2 * units.M_sun,',
                      '                 min_dissipative_mstar=1.5 * units.M_sun,\n                 max_dissipative_mstar=3.0 * units.M_sun,')
    text=replace_once(text,'        self.max_dissipative_mstar = max_dissipative_mstar',
                      '        self.min_dissipative_mstar = min_dissipative_mstar\n        self.max_dissipative_mstar = max_dissipative_mstar')
    save(name,text);changed.append(name)

    name='source/bayesian/log_likelihood_windemuth_et_al.py'
    text=(ORIGINAL/name).read_text()
    text=replace_once(text,'from types import SimpleNamespace\nfrom multiprocessing import Manager',
                      'from types import SimpleNamespace\nfrom bayesian.massive_star_policy import prepare_binary_start\nfrom multiprocessing import Manager')
    text=replace_once(text,"        parameters['dissipation'] = self.get_dissipation(encoded_parameters)\n",'')
    start=text.index("        primary_mass = parameters['system'].primary_mass.to(units.M_sun).value")
    end=text.index('        e_to_be_near=0.5',start)
    text=text[:start]+'''        primary_mass = parameters['system'].primary_mass.to_value(units.M_sun)
        secondary_mass = parameters['system'].secondary_mass.to_value(units.M_sun)
        feh = float(parameters['system'].feh)
        try:
            start = prepare_binary_start(
                interpolator, primary_mass, secondary_mass, feh,
                parameters['system'].age.to_value(units.Gyr))
        except ValueError as error:
            logger.info('Unsupported massive-star proposal; zero weight: %s', error)
            return -numpy.inf
        self.starting_age_gyr = start['starting_age_gyr']
        self.starting_age_record = start
        parameters['binary_start_age'] = self.starting_age_gyr * units.Gyr
        parameters['disk_dissipation_age'] = parameters['binary_start_age']
        parameters['carepackage']['starting_age_gyr'] = self.starting_age_gyr
        parameters['dissipation'] = self.get_dissipation(encoded_parameters)
        parameters['system'].feh = feh

'''+text[end:]
    save(name,text);changed.append(name)

    name='source/bayesian/sample_windemuth_et_al.py'
    text=(ORIGINAL/name).read_text()
    text=replace_once(text,'from bayesian.prior_transform_windemuth_et_al import PriorTransformWindemuth',
                      'from bayesian.massive_prior_transform import MassivePriorTransformWindemuth as PriorTransformWindemuth')
    text=replace_once(text,"    for index, component in enumerate(['primary', 'secondary']):",
                      "    independent_parameter_distributions = [entry for entry in independent_parameter_distributions\n                                           if entry[0] != 'disk_dissipation_age']\n    for index, component in enumerate(['primary', 'secondary']):")
    save(name,text);changed.append(name)

    name='general_purpose_python_modules/reproduce_system.py'
    text=(ORIGINAL/name).read_text()
    text=replace_once(text,'                   secondary_disk_period=None,',
                      '                   secondary_disk_period=None,\n                   binary_start_age=None,')
    text=replace_once(text,'        current_age=system.age,','        current_age=system.age,\n        binary_start_age=binary_start_age,')
    save(name,text);changed.append(name)

    name='general_purpose_python_modules/solve_for_initial_values.py'
    text=(ORIGINAL/name).read_text()
    text=replace_once(text,'                 secondary_disk_period=None,',
                      '                 secondary_disk_period=None,\n                 binary_start_age=None,')
    text=replace_once(text,"        porb = getattr(system, 'Porb', None)",'''        self.explicit_binary_start = binary_start_age is not None
        if self.explicit_binary_start:
            disk_dissipation_age = binary_start_age.to(units.Gyr)
            requested = disk_dissipation_age.to_value(units.Gyr)
            if not numpy.isfinite(requested) or not 0 < requested < current_age.to_value(units.Gyr):
                raise ValueError('Invalid primary-ZAMS binary starting age')
        porb = getattr(system, 'Porb', None)''')
    text=replace_once(text,"        requested_age = self.configuration['requested_disk_dissipation_age']",'''        requested_age = self.configuration['requested_disk_dissipation_age']
        if self.explicit_binary_start:
            self.configuration['disk_dissipation_age'] = requested_age
            self.target_state.planet_formation_age = requested_age
            self._formation_age_resolved = True
            return requested_age''')
    text=replace_once(text,"filename = '/home/cloudynestshyshy/Downloads/research_UTD_Tidal_Dissipation/failed'",
                      "filename = os.environ.get('FAILED_SOLUTIONS_DIR', '/work/dal205149/gaia_massive_star_circularization/failed_solutions')")
    save(name,text);changed.append(name)

    # Retain every unrelated catalog filter, changing only the authorized
    # mass/metallicity acceptance and actual stellar coverage checks.
    name='source/bayesian/windemuth_et_al_util.py'
    text=(ORIGINAL/name).read_text()
    text='from bayesian.massive_star_policy import validate_selection, prepare_binary_start\n'+text
    for component in (1,2):
        old=f"        data['posterior_m{component}(msun)'] + data['posterior_m{component}+sigma'] < 1.2"
        new=f"        (data['posterior_m{component}(msun)'] <= 3.0) & (data['posterior_m{component}(msun)'] >= 1.5)"
        text=replace_once(text,old,new)
    anchor="    if interpolator is not None:\n        valid = numpy.logical_and("
    insertion='''    selection = []
    for _, row in data.iterrows():
        try:
            validate_selection(float(row['posterior_m1(msun)']), float(row['posterior_m2(msun)']),
                               float(stellar_evol_lib.feh_from_z(float(row['posterior_z']))))
            selection.append(True)
        except ValueError:
            selection.append(False)
    valid = numpy.logical_and(valid, selection)

'''
    text=replace_once(text,anchor,insertion+anchor)
    begin=text.index('            valid[i] = check_iconv(')
    end=text.index("        _logger.info('Iconv cut",begin)
    text=text[:begin]+'''            try:
                prepare_binary_start(interpolator,
                    float(info['posterior_m1(msun)'].iloc[0]),
                    float(info['posterior_m2(msun)'].iloc[0]),
                    float(stellar_evol_lib.feh_from_z(float(info['posterior_z'].iloc[0]))),
                    10.0**(float(info['posterior_tau(log10yr)'].iloc[0]) +
                           float(info['posterior_tau+sigma'].iloc[0])-9.0))
            except ValueError:
                valid[i] = False
'''+text[end:]
    save(name,text);changed.append(name)

    # Confirm the original numerical search and failure paths were not rewritten.
    def nested_functions(path):
        tree=ast.parse(path.read_text())
        return {node.name:ast.dump(node) for node in ast.walk(tree) if isinstance(node,ast.FunctionDef)}
    old=nested_functions(ORIGINAL/'general_purpose_python_modules/reproduce_system.py')
    new=nested_functions(ROOT/'general_purpose_python_modules/reproduce_system.py')
    for name in ('get_period_range','errfunc','solve_for_point'):
        assert old[name]==new[name],name
    for relative,digest in manifest['source_files'].items():
        assert hashlib.sha256((ORIGINAL/relative).read_bytes()).hexdigest()==digest,relative
    record=dict(profile='massive-rc-env-primary-zams-v1',original_root=str(ORIGINAL),copy_root=str(ROOT),
                changed_existing_files=changed,original_source_unchanged=True,
                numerical_period_search_and_failure_handling_unchanged=True,
                mass_range=[1.5,3.],requested_feh_range=[-1.2,.7],
                low_mass_rule='If either component is in [1.50,1.65] Msun, require FeH < 0',
                spin_initialization='Existing disk-lock prescription, released at primary bilinear ZAMS',
                patched_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in changed})
    (ROOT/'IMPLEMENTATION.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
