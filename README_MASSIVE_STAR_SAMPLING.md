# More massive Gaia binaries with a Gaussian-distribution builder

This branch records the source of the revised Gaia package, exported from
`/work/dal205149/gaia_massive_star_circularization_upper3sigma_20261118`.
No running source or sampler was changed by publication.

## Entry points and Gaussian builder

- `control/gaia_launcher.py`: production stellar-profile launcher.
- `source/bayesian/gaia_observations.py`: CSV loading, error conversion,
  covariance validation and conditional observed-e distributions.
- `source/bayesian/gaia_prior_transform.py`: Gaussian stellar/orbital priors,
  component ordering and derived native radii.
- `source/bayesian/gaia_log_likelihood.py`: measured-e likelihood adapter.
- `source/bayesian/sample_gaia.py`: likelihood/prior assembly.
- `source/bayesian/nesty_sampling.py`: dynamic sampling and checkpoints.

Provide the observation CSV and `data/selected_systems.csv` externally.
The selection requires `catalog` and `system_id`; `GAIA_SYSTEMS_CSV` may point
to the observation file. Required observation columns and checks are defined
in `gaia_observations.py`. No observation table or target list is added here.
Use `control/gaia_launcher.py SYSTEM_ID` in the configured environment.

Mass/age/metallicity standard deviations are the mean of the positive lower
and upper error offsets. Age is Gaussian in linear Gyr, not log age. Stellar
variables are independent where no covariance is supplied. Component draws
are sorted so the primary is more massive, and radii are derived from the
native stellar interpolator at each proposed mass, metallicity and age.
Period–eccentricity dependence is retained through a Gaussian P marginal and
conditional e|P:

```text
mean(e | P) = mean(e) + rho * sigma_e / sigma_P * (P - mean(P))
sigma(e | P) = sigma_e * sqrt(1 - rho**2)
```

The loader validates covariance consistency. Eccentricity support intersects
0 ≤ e ≤ 1 and ±8 conditional standard deviations to give finite endpoints to the
inherited likelihood. The population envelope does not replace or inflate
measurement errors or add an outlier bypass.

## Model versions and tests

The default source uses the revised+3sigma envelope. The exact original
module is retained at
`reproducibility/legacy_gaia/paper_eccentricity_envelope.py` (slope2.154176170062391,
plateau from15.438917056042001days). The running deployment has two versions:
Stock 2 uses the revised curve; six other Gaia systems still use the original
source. HSC 2251's cap would change .755888963295→.8 under the revised curve
and requires a fresh posterior for a common model. Other five caps are
unchanged. Do not mix old likelihoods with a revised envelope.

This Gaia profile retains the additional rule that a component ≤1.65 solar
masses requires [Fe/H]<0. The Kepler profile removed that rule for its
explicit selection. The two branches are not identical stellar policies.

`python control/test_massive_policy.py` uses mocked stellar quantities.
`python control/test_gaia_adapter.py` is an integration test requiring the
separately supplied observation/selection files. Its target-count/covariance
fixtures were updated for the current selection; no runtime code changed.

## Shared physical and numerical settings

Both stars have supported masses in1.5–3 solar masses and metallicity in the
requested [Fe/H] domain−1.2…+0.7. Actual native all-quantity age support is
checked; unsupported draws are rejected rather than clipped to grid edges.
The generated MIST2 `rc_env` profile uses a radiative-plus-surface-convective
outer zone and central convective core. Effective floors are ICONV 1e−3,
IRAD 1e−4,MRAD 1e−3,RRAD 0 in native units.

Starting spin periods are independent, uniform **in period from 0.5 to 50 days**.
Both stars start solid-body at the common primary-ZAMS epoch; the secondary
can still be pre-MS. This is not its spin at its own later ZAMS. Legacy
parameter names, winds, saturation and 10 Myr coupling are retained.

The default envelope, for orbital period P in days, is

```text
e_env(P) = min(0.8, 0.06 + 2.55812396483873 * log10(max(P, 7) / 7))
```

It passes through Stock 2 central+3sigma, e=.46662709057158114 at
P=10.093766032005036days, with plateau .8 from13.62612999816196days.
The measured-e likelihood and hard population-envelope gate are retained;
there is no outlier bypass.

Production uses dynesty dynamic nested sampling, `bound=multi`,64 initial
and batch live points,64 workers, **dlogz_init=.05**, and **posterior CDF
uncertainty tolerance=.02** with128 realizations and the existing100-period
1–50 day informative-range policy. Physical-parameter CDFs are diagnostic
only; the maximum dynamic batch count is 200.

## Setup, provenance and limitations

This is **package code and documentation only**. No observational tables,
target lists, raw joint-prior chains, posterior samples, checkpoints,
worker journals, run records, caches or compiled binaries are added.
Existing upstream repository data are unchanged.

Provision research inputs separately, plus the validated generated grid,
ZAMS JSON and `rc_env` native interpolator; compatible POET shared libraries
in `poet/lib/`; the O30-from-O400 eccentricity-coefficient SQLite table;
GSL 2.8 and the production Boost runtime. Deployed paths remain in the source
and config and must be adapted for another installation. Never update a
running deployment in place.

Vendored Python dependencies and their licenses are included. Exact included
source hashes and required external library hashes are in
`reproducibility/DEPLOYED_SOURCE_MANIFEST.json`. Observed Python 3.12 package
versions are recorded separately; this is not a complete portable environment
lock. `PUBLICATION_ADJUSTMENTS.json` records test-fixture-only updates.
The active production model code is unchanged from its deployed snapshot.

`examples/slurm/` contains generic scripts without target lists. Provide
`PACKAGE_ROOT`, `RUN_ROOT`, `SYSTEMS_FILE` and an appropriate node allocation.
The examples set short TMPDIR, isolated outputs and CDF=.02 and do not
submit automatic continuation jobs.

Production preflight/native-support and short spin smoke checks are not
completed posterior-convergence tests or guarantees against native solver
failures. Legacy fallback and solver limitations remain. Initialization can
be slow and has no complete baseline checkpoint until the live-point stage
finishes. A running Slurm step is not evidence of convergence.

Never restore checkpoints or worker journals under changed likelihoods,
priors, envelopes or source fingerprints. `PYTHON_SYNTAX_AUDIT.json` records
an incomplete historical test excluded from this publication; run the explicit profile tests,
not indiscriminate legacy test discovery. The original introduction remains
in `README.rst`; dependency license notices are retained separately.
