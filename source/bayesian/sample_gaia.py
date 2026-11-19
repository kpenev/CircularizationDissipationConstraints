"""Prepare Gaia sampling while retaining the existing pseudo-envelope and solver."""
import sys
from pathlib import Path
from scipy import stats

# Keep the production entry point's SciPy compatibility and multiprocessing setup.
from bayesian.sample_windemuth_et_al import main, parse_command_line
from bayesian.binary_utils import get_common_binary_star_priors,prepare_sampling_common
from bayesian.gaia_observations import get_system,eccentricity_distribution
from bayesian.gaia_prior_transform import GaiaPriorTransform
from bayesian.gaia_log_likelihood import GaiaLogLikelihood
from bayesian.windemuth_et_al_util import eccentricity_envelope


def prepare_sampling(config):
    observation=get_system(config.system)
    interp=prepare_sampling_common(config)
    m,s=observation['means'],observation['sigmas']
    distro=eccentricity_distribution(m['eccentricity'],s['eccentricity'])
    likelihood=GaiaLogLikelihood(
        observation=observation, observed_eccentricity_distro=distro,
        interpolator=interp,envelope_eccentricity=eccentricity_envelope(m['period']),
        powerlaw_dissipation=(config.lgQ_break_period is not None and config.lgQ_powerlaw is not None),
        evolution_timeout=config.evolution_timeout,period_search_factor=config.initial_period_search_factor,
        scaled_period_guess=config.initial_period_scaled_guess,prior_only=(config.sampling=='prior'),
        de_distro=distro,pe_distro=stats.uniform(0,.8),system_eccentricity=m['eccentricity'],
        system_name=str(config.system),nn_path=config.nn_data_dir,
        warm_start_enabled=(not config.disable_warm_start),warm_start_period_gate=config.warm_start_period_gate,
        warm_start_eccentricity_gate=config.warm_start_eccentricity_gate,use_legacy_pnn=config.use_legacy_pnn)
    priors=[p for p in get_common_binary_star_priors(config) if p[0]!='disk_dissipation_age']
    prior=GaiaPriorTransform(observation,interpolator=interp,
                             independent_parameter_distributions=priors,
                             model_parameter_order=likelihood.parameter_order)
    return likelihood,prior


def run():
    # Reuse the production main's setup and sampling dispatch, changing only
    # its data-preparation callback within this Gaia process.
    import bayesian.sample_windemuth_et_al as entry
    entry.prepare_sampling=prepare_sampling
    config=parse_command_line('Sample a Gaia binary with Gaussian observations.',
        str(Path(__file__).resolve().parents[1]/'sampling_config/Gaia_massive_rc_env.cfg'),
        dissipation=True,cluster=False,choose_binary='gaia',spindown=2)
    main(config)


if __name__=='__main__':
    run()
