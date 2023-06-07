"""Define prior transformations based on Windemuth et. al. (2019) samples."""

import logging
import corner

from astropy import units
import numpy
from KDEpy import NaiveKDE

from stellar_evolution.library_interface import \
    library as poet_stellar_evolution
from general_purpose_python_modules.kde import KDEDistribution

from bayesian.prior_transform_base import PriorTransformBase

import bayesian.windemuth_et_al_util as weau
from general_purpose_python_modules.multiprocessing_util import setup_process

class PriorTransformWindemuth(PriorTransformBase):
    """Prior transform based on MCMC samples from Windemuth et. al. (2019)."""

    def _fill_coupled_parameters(self,
                                 unit_cube_iter,
                                 model_parameters,
                                 identify=False):
        """Fills the primary & secondary mass, age, and [Fe/H]."""

        if model_parameters is None:
            for _ in self._quantities:
                next(unit_cube_iter)
            return

        logger = logging.getLogger(__name__)

        sampled = dict()
        weights = numpy.copy(self.initial_sample_weights)
        for quantity in self._quantities:
            if identify:
                sampled[quantity] = next(unit_cube_iter)
            else:
                assert (weights >= 0).all()
                assert weights.sum() > 0
                self._distributions[quantity].set_weights(weights)
                sampled[quantity] = self._distributions[quantity].ppf(
                    next(unit_cube_iter)
                )
                weight_scale = self._distributions[quantity].eval_sample_pdf(
                    sampled[quantity]
                )
                weights *= weight_scale
                if weights.sum() == 0:
                    logger.error(
                        'All weights zero, should be impossible: x %s (sum=%s)',
                        repr(weight_scale),
                        repr(weight_scale.sum())
                    )

        if not identify:
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
        if 'e' in self._quantities:
            self.eccentricity = sampled['e']


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
                If None, the prior transform is set up to sample final
                eccentricity as well and not apply initial weights (useful for
                testing and tuning).

            kernels(dict):    Kernel functions to convolve the samples with in
                order to get a continuous distribution. Kernels should be
                scipy.stats.rv_continuous instances (or provide equivalent
                functionality). Each kernel is assumed to apply to only one
                variable (i.e. the combined kernel has separation of variables)
                and should already have the correct bandwidth set.

            parent_kwargs:    Passed directly to parent`s ``__init__()``.
        """

        if (
                initial_sample_weights is not None
                and
                initial_sample_weights.shape != (samples.shape[0],)
        ):
            raise ValueError(
                'Samples and weights PriorTransformWindemuth is being '
                'initialized with have different shapes: '
                f'({samples.shape[0]:d},) and '
                f'{initial_sample_weights.shape!r} respectively!'
            )

        self._quantities = ['P', 'Mtot', 'Mratio', 'z', 'tau']
        if initial_sample_weights is None:
            self._quantities.append('e')
            self.eccentricity = numpy.nan
            self.initial_sample_weights = numpy.ones(samples['P'].size,
                                                     dtype=float)
        else:
            self.initial_sample_weights = initial_sample_weights
        if kernels is None:
            min_kernel_width = dict(
                Mtot=0.01,
                Mratio=0.01,
                z=0.001,
                tau=0.1,
                P=1e-7,
                e=1e-4
            )
            kernels = {
                quantity: (
                    'rdist',
                    (),
                    dict(
                        c=4,
                        scale=max(
                            min_kernel_width[quantity],
                            NaiveKDE(
                                kernel='epa',
                                bw='ISJ'\
                            ).fit(
                                samples[quantity].to_numpy()
                            ).bw
                        )
                    )
                )
                for quantity in self._quantities
            }
        self._distributions = {
            quantity: KDEDistribution(samples[quantity], kernels[quantity])
            for quantity in self._quantities
        }

        self._sample_weights_envelope = None

        super().__init__(**parent_kwargs)


    def __call__(self, unit_cube_values):
        """Add sample weights per the envelope eccentricity to return value."""

        self._sample_weights_envelope = None
        result = super().__call__(unit_cube_values)
        assert self._sample_weights_envelope is not None
        result['sample_weights_envelope'] = self._sample_weights_envelope
        return result

if __name__ == '__main__':
    # We're going to generate two corner plots, BUDDY

    #setup_process()

    id_list_a = {
        10268903: 1.5e-4,
        6962018: 8e-6,
        11616200: 2e-5,
        4380283: 1e-5,
        9110346: 2e-5,
        7732791: 5e-5,
        5039441: 1e-4,
        9656543: 1e-4,
        3834364: 1.2e-3,
        11228612: 2e-4,
        10960995: 3e-6,
        3241344: 2e-4,
        5022440: 3e-6,
        5802470: 2e-6,
        4815612: 1e-6,
        7377033: 1e-3,
        11867071: 3e-3,
        3427776: 2e-3,
        10935310: 1.5e-3,
        10031409: 5e-6,
        9532421: 1.5e-3,
        3973504: 1.5e-3,
        8957954: 2e-6,
        6521542: 2e-4,
        11252617: 1e-3,
        4285087: 1e-5,
        7025851: 4e-5,
        4346875: 1.2e-3,
        7691527: 2e-4,
        6227560: 1e-4,
        8302455: 1e-5,
        12004679: 3e-4,
        7369523: 1e-4,
        9971475: 1.5e-5,
        7129465: 2e-6,
        5181455: 5e-6,
        8381592: 1e-3,
        7376500: 2e-4,
        8618226: 2e-5,
        9649222: 5e-6,
        6546508: 8e-4,
        10385682: 1e-4,
        8460600: 1.5e-3,
        7125636: 7e-4,
        8580438: 5e-5,
        5597970: 5e-6,
        8746310: 1e-5,
        7362852: 7e-5,
        12557713: 2e-3,
        4753988: 3e-4,
        10923260: 1e-4,
        3003991: 2.0e-2,
        6927629: 3e-6,
        8364119: 3e-4,
        6949550: 3e-6,
        9532123: 3e-4,
        9892471: 3e-5,
        2445134: 1.5e-4,
        4948863: 1.5e-4,
        9775253: 5e-7,
        4839180: 4e-5,
        5652260: 1e-4,
        6707942: 1.7e-3,
        7597703: 3e-6,
        11232745: 2e-5,
        8984706: 1.3e-4,
        11409698: 1e-4,
        9353182: 3e-5,
        6594972: 1e-4,
        9025914: 1e-4,
        9665503: 1e-4,
        6185717: 2e-4,
        8414159: 2.5e-5,
        6301030: 1e-5,
        11499757: 5e-5,
        11704044: 3e-6,
        6359798: 2e-5,
        7118545: 3e-5,
        12251779: 8e-5,
        4678171: 7e-5,
        8111622: 2e-5,
        5622250: 5e-5,
        8879427: 3e-4,
        5979863: 1e-3,
        9001468: 2e-4,
        6522750: 2e-4,
        6131659: 5e-5,
        12316447: 3e-5,
        7624297: 2e-4,
        10992733: 8e-5,
        7021177: 1.5e-4,
        10753734: 3e-5,
        10711913: 1.5e-3,
        10518735: 1e-4,
        9016295: 6e-4,
        10258558: 2.5e-4,
        4252226: 1e-4,
        4633434: 2e-3,
        12017140: 1e-4,
        9838060: 2e-3,
        6672229: 2.5e-4,
        7821010: 2e-4,
        10849244: 5e-5,
        10215422: 1e-6,
        5983348: 6e-4,
        7767733: 1.5e-3,
        10651945: 1.5e-4,
        4773155: 8e-5,
        5553624: 2e-4,
        12356914: 2e-3,
        8572936: 2e-5,
        8973000: 1e-4,
        2998124: 1e-4,
        6431670: 1e-5,
        4847832: 1e-5,
        8183389: 1e-3,
        5003117: 1.2e-3,
        12644769: 1e-4,
        8553907: 1e-4,
        12217907: 2e-4,
        7541502: 3e-5,
        10420279: 7e-5,
        4247023: 1.5e-4,
        9164836: 1e-4,
        8610483: 1.5e-4,
        9714123: 1e-3,
        9837544: 1e-5,
        8560285: 1.5e-3,
        8044608: 5e-5,
        10292238: 1.5e-4,
        4824268: 3e-4,
        8760135: 5e-5,
        9839062: 1e-3
    }
    # take the above dict and extract the keys
    new_list = []
    for i in id_list_a:
        new_list.append(i)

    # Stuff used for both parts
    labels = ['P', 'Mtot', 'Mratio', 'z', 'tau']

    # First, we'll generate a corner plot of samples from Windemuth et. al.
    # Just feed corner the Windemuth data and it'll do like all of them
    windemuth_data = numpy.array([])
    for i in new_list:
        i_data = weau.get_samples(i)
        segment = numpy.array([i_data['P'], i_data['Mtot'], i_data['Mratio'], i_data['z'], i_data['tau']])
        windemuth_data=numpy.append(windemuth_data,segment)
        break
    figure1 = corner.corner(segment.T, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 12})
    print(i_data)
    
    # Next, we'll run something from here to generate samples and then make a corner plot of that
    # so we have to make a PriorTransformWindemuth object first
    initial_sample_weights = numpy.ones(i_data['P'].size)
    #print(segment.T)
    #print(segment.T.shape)
    independent_parameter_distributions = list(zip(labels,
                                                   [KDEDistribution(i_data['P'], ('rdist', (), dict(c=4, scale=1e-7))),
                                                    KDEDistribution(i_data['Mtot'], ('rdist', (), dict(c=4, scale=0.01))),
                                                    KDEDistribution(i_data['Mratio'], ('rdist', (), dict(c=4, scale=0.01))),
                                                    KDEDistribution(i_data['z'], ('rdist', (), dict(c=4, scale=0.001))),
                                                    KDEDistribution(i_data['tau'], ('rdist', (), dict(c=4, scale=0.1)))
                                                    ],
                                                   [units.day, units.M_sun, units.dimensionless_unscaled, units.dimensionless_unscaled, units.Gyr])
                                                   )
    model_parameter_order = list(zip(labels, [units.day, units.M_sun, units.dimensionless_unscaled, units.dimensionless_unscaled, units.Gyr]))
    model_parameter_order.append(('orbital_period', units.day))
    #### ^^ I have to include ALL of the parameters in this one, indy and no, so it's not the same as above
    prior_transform = PriorTransformWindemuth(
        samples = i_data,
        initial_sample_weights = initial_sample_weights,
        kernels=None,
        independent_parameter_distributions = independent_parameter_distributions,
        model_parameter_order = model_parameter_order)
    unit_cube_values = numpy.random.rand(10)
    pt_test_data = numpy.array([prior_transform(unit_cube_values)['parameters']])
    print(pt_test_data)
    for i in range(3):#i_data['P'].size - 1):
        unit_cube_values = numpy.random.rand(10)
        arg=numpy.array([prior_transform(unit_cube_values)['parameters']])
        pt_test_data = numpy.concatenate((pt_test_data,arg))
    figure2 = corner.corner(pt_test_data, labels=labels, quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 12})

    # And then we'll make it have there be the corner plots I can see them
    figure1.savefig('corner_plot_1.png')
    figure2.savefig('corner_plot_2.png')