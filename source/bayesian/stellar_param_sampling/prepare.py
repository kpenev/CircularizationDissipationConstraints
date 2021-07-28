#!/usr/bin/env python3
"""Pickle approximations to marginalized CDFs of stellar parameters."""

import multiprocessing as mp

import matplotlib
from matplotlib import pyplot
from configargparse import ArgumentParser, DefaultsFormatter
import numpy
from numpy.random import rand
import logging

from split_normal_distribution import split_normal
from stellar_evolution.manager import StellarEvolutionManager
from stellar_evolution.change_variables import QuantityEvaluator
from random import random
import sys
import corner
#print(sys.path)

sys.path.append('/home/mmmahmud/CircularizationDissipationConstraints/source')
sys.path.append('/home/mmmahmud/general_purpose_python_modules')
sys.path.append('/home/mmmahmud/CircularizationDissipationConstraints/data')


from bayesian.stellar_param_sampling.config_util import\
    add_star_sampler_config_args
#False positive
#pylint: disable=import-error
from bayesian.parse_command_line import parse_quantity_with_errors
#pylint: enable=import-error

from bayesian.stellar_param_sampling.marginalized_parameter_distribution import\
    MarginalizedParamterDistribution
from bayesian.stellar_param_sampling.poet_interp_likelihood import\
    POETInterpLikelihood
from bayesian.stellar_param_sampling.feh_conditional_likelihood_base import\
    FeHConditionalLikelihoodBase
from bayesian.stellar_param_sampling.star_sampler import StarSampler
from bayesian.stellar_param_sampling.gaussian_likelihood import\
    GaussianLikelihood

def parse_configuration():
    """Return the configuration to use per the command line."""

    parser = ArgumentParser(
        description=__doc__,
        default_config_files=[],
        formatter_class=DefaultsFormatter,
        ignore_unknown_config_file_keys=False
    )

    parser.add_argument(
        '--feh',
        type=parse_quantity_with_errors,
        help='The measured [Fe/H] for the star as well as its estimated '
        'standard deviation(s), possibly asymmetric.'
    )
    parser.add_argument(
        '--logg',
        type=parse_quantity_with_errors,
        help='If known, the masured value of log10(g) at the surface of the '
        'star as well as its estimated standard deviation(s), possibly '
        'asymmetric.'
    )
    parser.add_argument(
        '--Teff',
        type=parse_quantity_with_errors,
        help='If known, the masured value of the effective temperature of the '
        'star, in Kelvin, as well as its estimated standard deviation(s), '
        'possibly asymmetric.'
    )
    parser.add_argument(
        '--mean-density', '--density', '--rho',
        type=parse_quantity_with_errors,
        help='If known, the mesured mean stellar density in g/cm3, as well as '
        'its estimated standard deviation(s), possibly asymmetric.'
    )

    parser.add_argument(
        '--lum', '--stellar-luminosity',
        type=parse_quantity_with_errors,
        help='If known, the mesured luminosity, as well as '
             'its estimated standard deviation(s).'
    )


    add_star_sampler_config_args(parser)




    return parser.parse_args()

def marginalized_plots(config,
                       star_sampler,
                       interpolator,
                       marginalized_distribution=None,
                       fast=True):
    """Make fig showing marginaziled PDF and CDF of sampled stellar params."""

    star_sampler.likelihood.disable_caching()
    unit_cube_samples = rand((100 if fast else 10000), 3)
    with mp.Pool(
            config.num_parallel_processes,
            initializer=FeHConditionalLikelihoodBase.set_interpolator,
            initargs=(config.stellar_evolution_interpolator_dir,)
    ) as workers:
        samples = numpy.array(
            workers.map(star_sampler, unit_cube_samples)
        )

    star_sampler.likelihood.enable_caching()

    cdf_x = numpy.empty((2 * samples.shape[0], ),
                        dtype=samples.dtype)

    cdf_y = numpy.empty(cdf_x.shape, dtype=samples.dtype)
    cdf_y[::2] = (numpy.arange(samples.shape[0], dtype=numpy.float64)
                  /
                  samples.shape[0])
    cdf_y[1::2] = cdf_y[::2] + 1.0 / samples.shape[0]

    for var_index, variable in enumerate(['feh', 'mass', 'age']):
        cdf_x[::2] = numpy.sort(samples[:, var_index])
        cdf_x[1::2] = cdf_x[::2]

        pyplot.subplot(2, 3, var_index + 1)
        pyplot.plot(cdf_x, cdf_y, '-k')
        pyplot.title(['[Fe/H]', r'$M_\star\ [M_\odot]$', 't [Gyr]'][var_index])

        pyplot.subplot(2, 3, var_index + 4)
        pyplot.hist(samples[:, var_index],
                    bins=samples.shape[0] // 100,
                    density=True)

        if marginalized_distribution is not None:
            marginalized_distribution.variable = variable
            if variable == 'age':
                marginalized_x = numpy.linspace(0, 14, (10 if fast else 300))
            else:
                marginalized_x = numpy.linspace(
                    *getattr(interpolator, variable + '_range')(),
                    (10 if fast else 300)
                )

            with mp.Pool(
                    config.num_parallel_processes,
                    initializer=GaussianLikelihood.set_interpolator,
                    initargs=(config.stellar_evolution_interpolator_dir,)
            ) as workers:
                marginalized_y = numpy.array(
                    workers.map(
                        marginalized_distribution.pdf,
                        marginalized_x
                    )
                )
            pyplot.plot(marginalized_x,
                        marginalized_y,
                        '-k')
    pyplot.savefig('marginalized_test_samples.eps')



def test_marginalized_pdfs(config, interpolator):
    """Create plots for visually confirming sampler works for known PDF."""

    likelihood = GaussianLikelihood(
        mean=numpy.array([5.0, 1.0, 0.0]),
        covariance=numpy.array(
            [
                [+0.200, +0.010, -0.007],
                [+0.010, +0.040, +0.030],
                [-0.007, +0.030, +0.200]
            ]
        ),
        rtol=config.time_ode_rtol,
        atol=config.time_ode_atol,
        max_step=config.time_ode_max_step
    )

    limits = dict(
        feh=interpolator.feh_range(),
        mass=interpolator.mass_range(),
        age=(
            likelihood.get_min_age,
            likelihood.get_max_age
        )
    )

    marginalized_distribution = MarginalizedParamterDistribution(
        direct_metallicity_distribution=config.feh,
        conditional_mass_age_distribution=likelihood.distribution.pdf,
        variable='feh',
        limits=limits,
        epsrel=1e-5
    )

    star_sampler = StarSampler(likelihood, config)

    marginalized_plots(config,
                       star_sampler,
                       interpolator,
                       marginalized_distribution)
def serialize_poet_likelihood(config, interpolator):
    """Create and pickle a sampler for POET based likelihood."""

    constraints = dict()
    if config.logg is not None:
        constraints['logg'] = config.logg
    if config.Teff is not None:
        constraints['teff'] = config.Teff
    if config.mean_density is not None:
        constraints['rho'] = config.mean_density
    if config.lum is not None:
        constraints['lum'] = config.lum

    likelihood = POETInterpLikelihood(
        **constraints,
        rtol=config.time_ode_rtol,
        atol=config.time_ode_atol,
        max_step=config.time_ode_max_step
    )
    star_sampler = StarSampler(likelihood, config)

    marginalized_plots(config, star_sampler, interpolator, fast=True)
    _corner_plot_of_mass_feh_age(number_of_samples=1000, star_sampler=star_sampler, interpolator=interpolator, config=config)

def _corner_plot_of_mass_feh_age(number_of_samples,
                                 star_sampler,
                                 interpolator,
                                 config):
    unit_cube = numpy.array([random(), random(), random()])
    _feh, _mass, _age = star_sampler.__call__(unit_cube)
    quantity_evaluator_object = QuantityEvaluator(interpolator=interpolator,
                                                  feh=_feh)#,
                                                  #teff = config.Teff,
                                                  #logg = config.logg,
                                                  #lum = config.lum,
                                                  #rho = config.mean_density)
    _teff = quantity_evaluator_object.teff(_mass, _age)
    _logg = quantity_evaluator_object.logg(_mass, _age)
    _lum = quantity_evaluator_object.lum(_mass, _age)
    _rho = quantity_evaluator_object.rho(_mass, _age)
    print('feh ', _feh, ' mass ', _mass, ' age ', _age, ' teff ', _teff, ' logg ', _logg, ' lum ', _lum, ' rho ', _rho)
    #samples = numpy.array([_feh, _mass, _age, _teff, _logg, _lum, _rho])
    samples1 = numpy.array([_feh, _mass, _age])
    samples2 = numpy.array([_feh, _teff, _logg, _lum, _rho])


    for i in range(1, number_of_samples):
        unit_cube = numpy.array([random(), random(), random()])
        _feh, _mass, _age = star_sampler.__call__(unit_cube)
        quantity_evaluator_object = QuantityEvaluator(interpolator=interpolator,
                                                      feh=_feh)#,
                                                      #teff=config.Teff,
                                                      #logg=config.logg,
                                                      #lum=config.lum,
                                                      #rho=config.rho)
        _teff = quantity_evaluator_object.teff(_mass, _age)
        _logg = quantity_evaluator_object.logg(_mass, _age)
        _lum = quantity_evaluator_object.lum(_mass, _age)
        _rho = quantity_evaluator_object.rho(_mass, _age)
        print('feh ', _feh, ' mass ', _mass, ' age ', _age, ' teff ', _teff, ' logg ', _logg, ' lum ', _lum, ' rho ',
              _rho)
        #new_sample = numpy.array([_feh, _mass, _age, _teff, _logg, _lum, _rho])
        new_sample1 = numpy.array([_feh, _mass, _age])
        new_sample2 = numpy.array([_feh, _teff, _logg, _lum, _rho])
        #samples = numpy.vstack((samples, new_sample))
        samples1 = numpy.vstack((samples1, new_sample1))
        samples2 = numpy.vstack((samples2, new_sample2))
        print('samples = ', samples1, samples2)

    #figure = corner.corner(samples1, labels=[r"$feh$", r"$mass$", r"$age$", r"$teff$", r"$logg$", r"$lum$", r"$rho$"],
                           #quantiles=[0.16, 0.5, 0.84],
                           #show_titles=True, title_kwargs={"fontsize": 4})

    figure = corner.corner(samples1, labels=[r"$feh$", r"$mass$", r"$age$"],
                            quantiles=[0.16, 0.5, 0.84],
                            show_titles=True, title_kwargs={"fontsize": 4})
    pyplot.show()
    figure = corner.corner(samples2, labels=[r"$feh$", r"$teff$", r"$logg$", r"$lum$", r"$rho$"],
                           quantiles=[0.16, 0.5, 0.84],
                           show_titles=True, title_kwargs={"fontsize": 4})
    pyplot.show()
    return







def main(config):
    """Avoid polluting the global namespace."""

    mp.set_start_method('forkserver')

    interpolator = StellarEvolutionManager(
        config.stellar_evolution_interpolator_dir
    ).get_interpolator_by_name(
        'default'
    )

    matplotlib.rcParams['figure.dpi'] = config.debug_plot_dpi
    matplotlib.rcParams['figure.autolayout'] = True

    FeHConditionalLikelihoodBase.set_interpolator(interpolator)

    serialize_poet_likelihood(config, interpolator)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(parse_configuration())
