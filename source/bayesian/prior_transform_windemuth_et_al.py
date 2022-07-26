"""Define prior transformations based on Windemuth et. al. (2019) samples."""

from astropy import units
import numpy

from stellar_evolution.library_interface import \
    library as poet_stellar_evolution
from general_purpose_python_modules.kde import KDEDistribution

from bayesian.prior_transform_base import PriorTransformBase

class PriorTransformWindemuth(PriorTransformBase):
    """Prior transform based on MCMC samples from Windemuth et. al. (2019)."""

    def _fill_coupled_parameters(self, unit_cube_iter, model_parameters):
        """Fills the primar & secondary mass, age, and [Fe/H]."""

        if model_parameters is None:
            for _ in self._quantities:
                next(unit_cube_iter)
            return


        sampled = dict()
        weights = self.initial_sample_weights
        for quantity in self._quantities:
            self._distributions[quantity].set_weights(weights)
            sampled[quantity] = self._distributions[quantity].ppf(
                next(unit_cube_iter)
            )
            weights *= self._distributions[quantity].eval_sample_pdf(
                sampled[quantity]
            )

        model_parameters['cmd_primary_radius'] = numpy.nan * units.R_sun
        model_parameters['cmd_secondary_radius'] = numpy.nan * units.R_sun

        model_parameters['orbital_period'] = sampled['P'] * units.day
        model_parameters['primary_mass'] = (
            sampled['Mtot']
            /
            (1.0 + sampled['Mratio'])
            *
            units.M_sun
        )
        model_parameters['secondary_mass'] = (
            model_parameters['primary_mass']
            *
            sampled['Mratio']
        )
        model_parameters['feh'] = poet_stellar_evolution.feh_from_z(
            sampled['z']
        ) * units.dimensionless_unscaled
        model_parameters['age'] = 10.0**(sampled['tau'] - 9.0) * units.Gyr
        assert self._sample_weights_envelope is None
        self._sample_weights_envelope = weights


    def __init__(self,
                 samples,
                 initial_sample_weights,
                 kernels=None,
                 **parent_kwargs):
        """
        Create a prior transform to sample from the given samples.

        Args:
            samples(pandas.DataFrame):    The samples for a given Kepler EB from
                Windemuth et. al. (2019).

            initial_sample_weights(array):    Weight to apply to each of the
                samples when combining to get the distribution to sample from.

            kernels(dict):    Kernel functions to convolve the samples with in
                order to get a continuous distribution. Kernels should be
                scipy.stats.rv_continuous instances (or provide equivalent
                functionality). Each kernel is assumed to apply to only one
                variable (i.e. the combined kernel has separation of variables)
                and should already have the correct bandwidth set.

            parent_kwargs:    Passed directly to parent`s ``__init__()``.
        """

        if initial_sample_weights.shape != (samples.shape[0],):
            raise ValueError(
                'Samples and weights PriorTransformWindemuth is being '
                'initialized with have different shapes: '
                f'({samples.shape[0]:d},) and '
                f'{initial_sample_weights.shape!r} respectively!'
            )

        self._quantities = ['P', 'Mtot', 'Mratio', 'z', 'tau']
        if kernels is None:
            min_kernel_width = dict(
                Mtot=0.05,
                Mratio=0.05,
                z=0.001,
                tau=0.1,
                P=3e-6
            )
            kernels = {
                quantity: (
                    'rdist',
                    (),
                    dict(
                        c=4,
                        scale=max(
                            (numpy.std(samples[quantity])
                             *
                             samples[quantity].size**(-0.2)),
                            min_kernel_width[quantity]
                        )
                    )
                )
                for quantity in self._quantities
            }
        self._distributions = {
            quantity: KDEDistribution(samples[quantity], kernels[quantity])
            for quantity in self._quantities
        }
        self.initial_sample_weights = initial_sample_weights

        self._sample_weights_envelope = None

        super().__init__(**parent_kwargs)


    def __call__(self, unit_cube_values):
        """Add sample weights per the envelope eccentricity to return value."""

        self._sample_weights_envelope = None
        result = super().__call__(unit_cube_values)
        assert self._sample_weights_envelope is not None
        result['sample_weights_envelope'] = self._sample_weights_envelope
        return result
