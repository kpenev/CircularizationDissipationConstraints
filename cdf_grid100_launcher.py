"""Install paper-defined tidal-Q CDF stopping before importing the sampler.

Kept outside source/ to preserve worker-search recovery fingerprints.
"""
import sys
import os
import runpy
from baseline_dlogz_policy import install as install_baseline_dlogz
install_baseline_dlogz()
from bayesian import posterior_cdf_uncertainty as cdf
from cdf_paper_range_policy import install

install(cdf)
print('CDF stopping policy: 100 logarithmic tidal periods, 1-50 days; '
      'range from q97.7(P) <= min(q97.7)+0.7 dex; '
      f'max CDF std <= {os.environ.get("NESTY_POSTERIOR_CDF_TOL", "0.02")} at q=2.275,15.866,84.134,97.725% within range; '
      'physical-parameter CDFs diagnostic only', flush=True)
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name='__main__')
