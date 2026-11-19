"""Launch the separate Gaia package with the validated massive-star cache."""
import ctypes
import json
import os
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parent.parent
sys.path[:0]=[str(ROOT/'source'),str(ROOT/'control')]
from bayesian.massive_star_policy import FLOORS,select_cache_root
from bayesian.gaia_observations import load_systems


def main():
    cache=select_cache_root()
    if sys.argv[1:]==['--show-profile']:
        print(json.dumps(dict(profile='gaia-massive-gaussian-rc-env-primary-zams-v1',
            package_root=str(ROOT),grid_root=str(cache),floors=FLOORS,
            gaia_system_ids=list(load_systems()),envelope='unchanged original pseudo-envelope',
            radius='native bicubic RADIUS at sampled mass, FeH and age',
            sigma='mean of supplied lower and upper errors',
            correlation='Gaussian P marginal times conditional Gaussian e|P',
            end_to_end_test_run=False,production_sampling_started=False),indent=2))
        return
    os.environ['MASSIVE_STAR_GRID_ROOT']=str(cache)
    ctypes.CDLL('libgslcblas.so',mode=ctypes.RTLD_GLOBAL)
    defaults=['--sampling','nested','--stellar-evolution-interpolator-dir',str(cache/'interpolators/rc_env'),
        '--eccentricity-expansion-coefficients','/work/dal205149/poet/eccentricity_expansion_coef_O30_from_O400.sqlite',
        '--nn-data-dir',str(ROOT/'solver_data'),'--disable-warm-start',
        '-c',str(ROOT/'source/sampling_config/Gaia_massive_rc_env.cfg')]
    sys.argv=[str(ROOT/'cdf_grid100_launcher.py'),str(ROOT/'source/bayesian/sample_gaia.py'),*defaults,*sys.argv[1:]]
    runpy.run_path(ROOT/'cdf_grid100_launcher.py',run_name='__main__')


if __name__=='__main__':
    main()
