"""Create a separate baseline copy, then apply only the authorized Gaia adapter."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

SOURCE=Path('/work/dal205149/massive_star_circularization')
ROOT=Path('/work/dal205149/gaia_massive_star_circularization')
OVERLAY=Path('/work/dal205149/gaia_sampling_overlay.tar.gz')


def hashes(root):
    result={}
    for folder in ('source','general_purpose_python_modules','poet','control'):
        for path in sorted((root/folder).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                result[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def git(*args):
    return subprocess.check_output(['git','-c','user.name=Codex','-c','user.email=codex@openai.com',*args],cwd=ROOT,text=True).strip()


def main():
    before=hashes(SOURCE)
    if ROOT.exists():
        raise RuntimeError(f'Refusing to overwrite an existing package: {ROOT}')
    ROOT.mkdir()
    ignored=shutil.ignore_patterns('__pycache__','*.pyc','.git')
    for folder in ('source','general_purpose_python_modules','poet','data','control'):
        shutil.copytree(SOURCE/folder,ROOT/folder,ignore=ignored)
    for path in SOURCE.iterdir():
        if path.is_file() and path.suffix in ('.py','.sh','.cfg','.md'):
            shutil.copy2(path,ROOT/path.name)
    for folder in ('cache','tmp','logs','results','failed_solutions','solver_data'):
        (ROOT/folder).mkdir(exist_ok=True)
    # All runtime paths in this independent copy must write to its own /work root.
    relocated=[]
    for path in sorted(ROOT.rglob('*')):
        if path.is_file() and path.suffix in ('.py','.cfg','.sh'):
            text=path.read_text()
            if str(SOURCE) in text:
                path.write_text(text.replace(str(SOURCE),str(ROOT)))
                relocated.append(str(path.relative_to(ROOT)))
    (ROOT/'.gitignore').write_text('data/*\n!data/gaia_systems.csv\n!data/selected_systems.csv\npoet/\n__pycache__/\n*.pyc\ncache/\ntmp/\nlogs/\nresults/\nsolver_data/\nfailed_solutions/\n*.npz\n*.pkl\n*.pickle\n*.joblib\n*.so\n*.pdf\n*.png\n')
    git('init')
    git('add','.')
    git('commit','-m','Copy massive-star sampling baseline with isolated work paths')
    baseline=git('rev-parse','HEAD')
    with tarfile.open(OVERLAY,'r:gz') as archive:
        for member in archive.getmembers():
            if member.name.startswith('/') or '..' in Path(member.name).parts or member.issym() or member.islnk():
                raise ValueError(f'Unsafe overlay member {member.name}')
        archive.extractall(ROOT)
    name=ROOT/'source/bayesian/parse_command_line.py'
    text=name.read_text()
    anchor="    if collection == 'cluster':\n"
    assert text.count(anchor)==1
    text=text.replace(anchor,"    if collection == 'gaia':\n        from bayesian.gaia_observations import load_systems\n        choices = list(load_systems())\n    elif collection == 'cluster':\n",1)
    name.write_text(text)
    unchanged=hashes(SOURCE)==before
    assert unchanged,'Source package unexpectedly changed'
    record=dict(source_root=str(SOURCE),package_root=str(ROOT),source_files=before,
        source_unchanged=True,baseline_commit=baseline,relocated_paths=relocated,
        envelope='original pseudo-envelope unchanged',end_to_end_test_run=False,production_sampling_started=False)
    (ROOT/'GAIA_IMPLEMENTATION.json').write_text(json.dumps(record,indent=2)+'\n')
    git('add','source','control','run_gaia_sampling.sh','README_GAIA.md','GAIA_IMPLEMENTATION.json',
        'data/gaia_systems.csv','data/selected_systems.csv')
    git('commit','-m','Add Gaia Gaussian observations, conditional eccentricity and interpolated radii')
    print(json.dumps(dict(package_root=str(ROOT),source_unchanged=unchanged,
                          baseline_commit=baseline,gaia_commit=git('rev-parse','HEAD')),indent=2))


if __name__=='__main__':
    main()
