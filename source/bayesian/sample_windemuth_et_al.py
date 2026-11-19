#!/usr/bin/env python3

"""Sample Windemuth et. al. (2019) Kepler eclipsing binaries."""

import logging
import sys
import traceback
from os import path

import numpy
import scipy
from astropy import units
from types import SimpleNamespace
import math

for _name in (
        'array', 'exp', 'inf', 'isfinite', 'isnan', 'logical_and',
        'logical_or', 'nan', 'ones', 'pi', 'zeros'
):
    if not hasattr(scipy, _name):
        setattr(scipy, _name, getattr(numpy, _name))

from general_purpose_python_modules import KDEDistribution
from general_purpose_python_modules import split_normal

from multiprocessing_util import setup_process
from bayesian.parse_command_line import parse_command_line
from bayesian.binary_utils import \
    get_common_binary_star_priors,\
    prepare_sampling_common
from bayesian.windemuth_et_al_util import \
    get_samples,\
    eccentricity_envelope,\
    get_summary_data
from bayesian.massive_prior_transform import MassivePriorTransformWindemuth as PriorTransformWindemuth
from bayesian.log_likelihood_windemuth_et_al import LogLikelihoodWindemuth
from bayesian.sample import sample

from bayesian.windemuth_eccentricity_distribution import W19EccentricityDistribution

def prepare_sampling(config):
    """Return log-likelihood & prior transform for sampling selected binary."""

    interpolator = prepare_sampling_common(config)
    nn_path = config.nn_data_dir
    samples = get_samples(config.system)
    summary_info = get_summary_data(config.system)
    envelope_eccentricity = eccentricity_envelope(numpy.median(samples['P']))
    samples.insert(
        loc=samples.shape[1],
        column='e',
        value=numpy.sqrt(
            samples['esinw']**2
            +
            samples['ecosw']**2
        )
    )
    eccentricity_kernel = (
        'rdist',
        (),
        dict(
            c=4,
            scale=max(
                (
                    numpy.std(samples['e'])
                    *
                    samples['e'].size**(-0.2)
                ),
                0.0001
            )
        )
    )

    observed_eccentricity_distro = KDEDistribution(samples['e'],
                                                   eccentricity_kernel)

    # Get the system
    c=numpy.median(samples['esinw'])
    d=numpy.median(samples['ecosw'])
    e=c/math.sin(math.atan(c/d))

    distro_pickle = path.join(
        path.dirname(path.dirname(path.dirname(path.abspath(__file__)))),
        'data', 'windemuth_eccentricity_distros_blah.pkl'
    )
    log_likelihood = LogLikelihoodWindemuth(
        observed_eccentricity_distro=observed_eccentricity_distro,
        interpolator=interpolator,
        envelope_eccentricity=envelope_eccentricity,
        powerlaw_dissipation=(
            config.lgQ_break_period is not None
            and
            config.lgQ_powerlaw is not None
        ),
        evolution_timeout=config.evolution_timeout,
        period_search_factor=config.initial_period_search_factor,
        scaled_period_guess=config.initial_period_scaled_guess,
        prior_only=(config.sampling == 'prior'),
        de_distro=W19EccentricityDistribution(config.system, pickle_fname=distro_pickle),
        pe_distro=scipy.stats.uniform(loc=0,scale=0.8),
        system_eccentricity=e,
        system_name = config.system,
        nn_path = nn_path,
        warm_start_enabled=(not config.disable_warm_start),
        warm_start_period_gate=config.warm_start_period_gate,
        warm_start_eccentricity_gate=(
            config.warm_start_eccentricity_gate
        ),
        use_legacy_pnn=config.use_legacy_pnn
    )
    independent_parameter_distributions = get_common_binary_star_priors(
        config,
    )
    independent_parameter_distributions = [entry for entry in independent_parameter_distributions
                                           if entry[0] != 'disk_dissipation_age']
    for index, component in enumerate(['primary', 'secondary']):
        independent_parameter_distributions.append(
            (
                'cmd_' + component + '_radius',
                split_normal.freeze_error_bar(
                    mode=summary_info.loc['posterior_r%d(rsun)' % (index + 1)],
                    abs_plus_error=summary_info.loc['posterior_r%d+sigma'
                                                    %
                                                    (index + 1)],
                    abs_minus_error=numpy.abs(
                        summary_info.loc['posterior_r%d-sigma'
                                         %
                                         (index + 1)]
                    )
                ),
                units.R_sun
            )
        )

    prior_transform = PriorTransformWindemuth(
        samples,
        initial_sample_weights=numpy.ones(samples['P'].size, dtype=float),
        independent_parameter_distributions=independent_parameter_distributions,
        model_parameter_order=log_likelihood.parameter_order
    )
    return log_likelihood, prior_transform


def main(config):
    """Avoid polluting global namespace."""
    setup_process(
                    fname_datetime_format=config.fname_datetime_format,
                    system=config.system,
                    sampling=config.sampling,
                    std_out_err_fname=config.std_out_err_fname,
                    logging_fname=config.logging_fname,
                    logging_verbosity=config.logging_verbosity,
                    logging_message_format=config.logging_message_format,
                    logging_datetime_format=config.logging_datetime_format
                  )
    log_likelihood, prior_transform = prepare_sampling(config)
    sample(log_likelihood, prior_transform, config)


if __name__ == '__main__':
    try:
        main(
            parse_command_line(
                'Perform Bayesian sampling of a single SB1 binary system.',
                'sb1_sampling.cfg',
                dissipation=True,
                cluster=False,
                choose_binary='w19',
                spindown=2
            )
        )
    except SystemExit as exit_error:
        sys.exit(exit_error.code)
    #Meant to simply report exception to log
    #pylint: disable=bare-except
    except:
        logging.critical(traceback.format_exc())
        sys.exit(1)
    #pylint: enable=bare-except
    else:
        logging.info('SB1 sampling completed successfully.')
