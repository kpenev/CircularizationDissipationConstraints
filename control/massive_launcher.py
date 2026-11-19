"""Select a validated cache and launch only the isolated massive-star package."""
import ctypes
import json
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT/'source'), str(ROOT/'control')]
from bayesian.massive_star_policy import PROFILE, FLOORS, select_cache_root


def main():
    cache_root = select_cache_root()
    if sys.argv[1:] == ['--show-profile']:
        print(json.dumps(dict(profile=PROFILE,package_root=str(ROOT),grid_root=str(cache_root),
                              native_interpolator=str(cache_root/'interpolators/rc_env'),
                              zams_grid=str(cache_root/'zams_age_grid.json'),
                              floors=FLOORS),indent=2))
        return
    os.environ['MASSIVE_STAR_GRID_ROOT'] = str(cache_root)
    ctypes.CDLL('libgslcblas.so',mode=ctypes.RTLD_GLOBAL)
    supplied = sys.argv[1:]
    defaults = ['--sampling','nested',
                '--stellar-evolution-interpolator-dir',str(cache_root/'interpolators/rc_env'),
                '--eccentricity-expansion-coefficients','/work/dal205149/poet/eccentricity_expansion_coef_O30_from_O400.sqlite',
                '--nn-data-dir',str(ROOT/'solver_data'),
                '--disable-warm-start','-c',str(ROOT/'source/sampling_config/W19_massive_rc_env.cfg')]
    sys.argv = [str(ROOT/'cdf_grid100_launcher.py'),str(ROOT/'source/bayesian/sample_windemuth_et_al.py'),*defaults,*supplied]
    print('Massive-star profile:',PROFILE,'cache:',cache_root,flush=True)
    runpy.run_path(ROOT/'cdf_grid100_launcher.py',run_name='__main__')


if __name__=='__main__':main()
