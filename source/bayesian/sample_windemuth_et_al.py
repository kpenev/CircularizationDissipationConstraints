#!/usr/bin/env python3

"""Sample Windemuth et. al. (2019) Kepler eclipsing binaries."""

import logging
import traceback

import numpy
from astropy import units

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
from bayesian.prior_transform_windemuth_et_al import PriorTransformWindemuth
from bayesian.log_likelihood_windemuth_et_al import LogLikelihoodWindemuth
from bayesian.sample import sample

def prepare_sampling(config):
    """Return log-likelihood & prior transform for sampling selected binary."""

    interpolator = prepare_sampling_common(config)
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
        prior_only=(config.sampling == 'prior')
    )
    independent_parameter_distributions = get_common_binary_star_priors(
        config,
    )
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
        initial_sample_weights=(
            None
            if config.sampling == 'prior' else
            log_likelihood.envelope_weights
        ),
        independent_parameter_distributions=independent_parameter_distributions,
        model_parameter_order=log_likelihood.parameter_order
    )
    return log_likelihood, prior_transform


def main(config):
    """Avoid polluting global namespace."""
    setup_process(
                    fname_datetime_format=config.fname_datetime_format,
                    system=config.system,
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
    except SystemExit:
        pass
    #Meant to simply report exception to log
    #pylint: disable=bare-except
    except:
        logging.critical(traceback.format_exc())
    #pylint: enable=bare-except
    else:
        logging.info('SB1 sampling completed successfully.')
