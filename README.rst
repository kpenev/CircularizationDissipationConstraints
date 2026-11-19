Dynamic sampling for more massive Kepler binaries
================================================

This branch provides the package used to constrain tidal dissipation in
Kepler binaries with stellar masses in the supported 1.5–3 solar-mass range.
It uses Windemuth et al. (2019; W19) joint-prior inputs and dynesty dynamic
nested sampling.

See `README_MASSIVE_STAR_SAMPLING.md <README_MASSIVE_STAR_SAMPLING.md>`_
for the physical model, input format, setup, sampling settings, provenance
and validation limits. Generic allocation scripts are in ``examples/slurm/``.

Kepler inputs and package scope
------------------------------

Provide the W19 per-system joint chains, summary tables and an explicit
system-ID list externally. This branch uses the correlated W19 inputs;
the Gaia Gaussian-distribution builder belongs to the separate
``DynamicSamplingForMoreMassiveStarsForGaiaWithGuassianDistributionBuilder``
branch.

This branch adds package code, configuration, dependency source/licenses
and documentation only. It adds no observational samples, target lists,
posterior samples, checkpoints or run outputs. Existing upstream repository
data are unchanged. Native libraries and stellar-grid assets must also be
provisioned separately as described in the detailed guide.

Shared eccentricity envelope
----------------------------

The Kepler model uses the shared population eccentricity envelope. Its
rising segment is calibrated to the Stock 2 binary's measured eccentricity
plus three standard deviations. Stock 2 is an envelope-calibration reference,
not a Kepler observational input. The exact curve and its hard likelihood
gate are documented in the detailed guide.

The original general-project introduction and its cluster/exoplanet article
list remain available in the repository history and on the upstream
``master`` branch.
