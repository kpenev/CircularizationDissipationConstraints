# Dynamic sampling for more massive Kepler binaries

This branch records the source used by the current Kepler package, exported
from `/work/dal205149/kepler_massive_star_circularization_20261118` without
modifying any running job.

## Entry points and external observations

- `control/massive_launcher.py`: production stellar-profile launcher.
- `source/bayesian/sample_windemuth_et_al.py`: Kepler input/likelihood setup.
- `source/bayesian/nesty_sampling.py`: dynamic sampling and checkpoints.
- `source/bayesian/dynesty_worker_journal.py`: worker-batch recovery.
- `baseline_dlogz_policy.py`, `cdf_paper_range_policy.py`, and
  `cdf_grid100_launcher.py`: baseline and posterior CDF stopping policies.

Supply the W19 per-system `thinned_chain` NPZ files and three paper summary
tables externally in the format expected by `windemuth_et_al_util.py`.
These are the Kepler observational inputs; this branch does not use the
Gaia Gaussian-distribution builder. The conditional KDE prior preserves the
joint P/Mtot/Mratio/z/tau dependence;
it does not substitute independent Gaussians. Original component labels,
including q>1, remain paired with their radius summaries. The native
`feh_from_z` convention differs from the plotting proxy log10(Z/.0142).

The production CLI reads an external explicit ID file. Supply that file or
adapt its path in `source/bayesian/parse_command_line.py` for a new deployment.
No target list or joint chain is added by this branch. Use
`control/massive_launcher.py SYSTEM_ID` in the configured environment.

This Kepler profile removes the obsolete low-mass/metal-rich veto so it
honors the final explicitly chosen targets. Its policy unit test was updated
to that deployed behavior without changing runtime code. Run it with
`python control/test_massive_policy.py`; it uses mocked stellar quantities.

Set a per-system `W19_ECCENTRICITY_CACHE` as in the Slurm example. An absent
or empty cache is rebuilt by the existing W19 routine from external inputs.
Do not share a writable eccentricity cache across simultaneous systems.

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

The shared envelope is calibrated to the Stock 2 binary's central eccentricity
plus 3 sigma: e = 0.46662709057158114 at P = 10.093766032005036 days.
Stock 2 is a Gaia-sample calibration reference for this common curve, not
a Kepler observational input or a request to use the Gaia input builder.
The envelope reaches its 0.8 plateau at P = 13.62612999816196 days.
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
not indiscriminate legacy test discovery. `README.rst` is the Kepler branch
landing page; the original general-project introduction is available in
Git history and upstream `master`. Dependency license notices are retained
separately.
