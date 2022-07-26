#!/usr/bin/env python3

"""Sample Windemuth et. al. (2019) Kepler eclipsing binaries."""

import logging
import traceback

import numpy
from scipy.stats import rdist

from general_purpose_python_modules.kde import KDEDistribution

from bayesian.parse_command_line import parse_command_line
from bayesian.binary_utils import \
    get_common_binary_star_priors,\
    prepare_sampling_common
from bayesian.windemuth_et_al_util import get_samples, eccentricity_envelope
from bayesian.prior_transform_windemuth_et_al import PriorTransformWindemuth
from bayesian.log_likelihood_windemuth_et_al import LogLikelihoodWindemuth
from bayesian.sample import sample

def prepare_sampling(config):
    """Return log-likelihood & prior transform for sampling selected binary."""

    interpolator = prepare_sampling_common(config)
    samples = get_samples(config.system)
    envelope_eccentricity = eccentricity_envelope(numpy.median(samples['P']))
    eccentricity_samples = numpy.sqrt(
        samples['esinw']**2
        +
        samples['ecosw']**2
    )
    eccentricity_kernel = (
        'rdist',
        (),
        dict(
            c=4,
            scale=max(
                (
                    numpy.std(eccentricity_samples)
                    *
                    eccentricity_samples.size**(-0.2)
                ),
                0.001
            )
        )
    )

    observed_eccentricity_distro = KDEDistribution(eccentricity_samples,
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
    prior_transform = PriorTransformWindemuth(
        samples,
        initial_sample_weights=log_likelihood.envelope_weights,
        independent_parameter_distributions=get_common_binary_star_priors(
            config,
        ),
        model_parameter_order=log_likelihood.parameter_order
    )
    return log_likelihood, prior_transform


def main(config):
    """Avoid polluting global namespace."""

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
