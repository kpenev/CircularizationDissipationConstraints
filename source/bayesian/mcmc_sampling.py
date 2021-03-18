"""Functions used to perform MCMC sampling."""

import logging
import functools
from multiprocessing import Pool
import os.path

import numpy
import emcee
import h5py
from astropy import units

from bayesian.sampling import get_code_version_str, setup_process

_mutable_config_params = set(['mcmc_nsteps',
                              'num_parallel_processes',
                              'samples_fname',
                              'rvk_show_interpolation',
                              'eccentricity_expansion_coefficients',
                              'stellar_evolution_interpolator_dir',
                              'fname_datetime_format',
                              'std_out_err_fname',
                              'logging_fname',
                              'logging_verbosity',
                              'logging_datetime_format',
                              'logging_message_format'])

_logger = logging.getLogger(__name__)

def log_probability(unit_cube_values, prior_transform, log_likelihood):
    """The posterior for MCMC, will track actual params & likelihood."""

    if unit_cube_values.min() < 0 or unit_cube_values.max() > 1:
        return tuple(
            -numpy.inf if i == 0 else numpy.nan
            for i in range(len(log_likelihood.parameter_order) + 1)
        )

    parameters = prior_transform(unit_cube_values)
    log_likelihood_value = log_likelihood(parameters)
    return (
        (log_likelihood_value,)
        +
        tuple(parameters)
    )

def distribution_to_attribute(distribution):
    """Convert the given frozen distribution (with units?) to h5py attribute."""

    dist_units = units.dimensionless_unscaled
    if isinstance(distribution, tuple):
        assert len(distribution) == 2
        assert hasattr(distribution[0], 'ppf')
        assert isinstance(distribution[1], units.UnitBase)
        distribution, dist_units = distribution

    return numpy.array(
        [
            str(dist_units),
            type(distribution.dist).__name__
        ]
        +
        [
            repr(entry)
            for key_value in dict(a=1, b=2).items()
            for entry in key_value
        ]
    )

def is_distribution(config_value):
    """Return True iff the given config attribute defines a distribution."""

    return (
        hasattr(config_value, 'ppf')
        or
        (
            isinstance(config_value, tuple)
            and
            hasattr(config_value[0], 'ppf')
        )
    )

def compare_chain_configuration(config, chain_group):
    """Return True iff MCMC chan_group can be extended with current config."""

    config_set = set(vars(config).keys())
    config_set -= set(chain_group.attrs.keys())
    config_set -= _mutable_config_params
    if config_set:
        _logger.debug('Configuration parameters %s not saved in chain %s',
                      config_set,
                      chain_group.name)
        return False
    for param, config_value in vars(config).items():
        if param in _mutable_config_params:
            continue
        saved_value = chain_group.attrs[param]
        if is_distribution(config_value):
            if (
                    saved_value
                    !=
                    distribution_to_attribute(config_value)
            ).any():
                _logger.debug(
                    'Distribution for %s (%s) in chain %s does not match %s',
                    param,
                    repr(saved_value),
                    chain_group.name,
                    repr(distribution_to_attribute(config_value))
                )
                return False
        else:
            try:
                len(config_value)
                if (numpy.array(config_value) != saved_value).any():
                    _logger.debug(
                        'Parameter %s (%s) in chain %s does not match %s',
                        param,
                        repr(saved_value),
                        chain_group.name,
                        repr(config_value)
                    )
                    return False
            except TypeError:
                if saved_value != config_value:
                    _logger.debug(
                        'Parameter %s (%s) in chain %s does not match %s',
                        param,
                        repr(saved_value),
                        chain_group.name,
                        repr(config_value)
                    )
                    return False

    return True

def add_configuration_attributes(config, destination):
    """Record the current configuration as attributes of the given group."""

    for parameter, config_value in vars(config).items():
        if parameter in _mutable_config_params:
            continue
        _logger.debug('Adding attribute for %s = %s',
                      parameter,
                      repr(config_value))
        assert parameter not in destination
        if is_distribution(config_value):
            destination[parameter] = distribution_to_attribute(
                config_value
            )
        else:
            try:
                len(config_value)
                destination[parameter] = numpy.array(config_value)
            except TypeError:
                destination[parameter] = (
                    numpy.empty(0) if config_value is None
                    else config_value
                )

def prepare_backend(config, num_params, code_version_str):
    """Return properly configured backend for storing new MCMC samples."""

    def init_chain(chain_group):
        """Prepare newly created group for holding MCMC chains."""

        add_configuration_attributes(
            config,
            chain_group.attrs
        )
        version_group = chain_group.create_group('code_version')
        version_dset = version_group.create_dataset('version',
                                                    (1,),
                                                    dtype=h5py.string_dtype(),
                                                    maxshape=(None,))
        first_iter_dset = version_group.create_dataset('first_iteration',
                                                       (1,),
                                                       dtype=int,
                                                       maxshape=(None,))
        version_dset[0] = code_version_str
        first_iter_dset[0] = 0

    samples_fname = (
        config.samples_fname
        %
        dict(system=config.system,
             sampling=config.sampling)
        +
        '.h5'
    )

    selected_chain_name = None
    next_chain_index = 0
    if os.path.exists(samples_fname):
        with h5py.File(samples_fname, 'r') as samples_file:
            for chain_name, chain_group in samples_file.items():
                if not isinstance(chain_group, h5py.Group):
                    continue
                if compare_chain_configuration(config, chain_group):
                    selected_chain_name = chain_name
                    break
                next_chain_index += 1
    if selected_chain_name is None:
        selected_chain_name = ('chain%05d' % next_chain_index)
        emcee.backends.HDFBackend(
            samples_fname,
            name=selected_chain_name
        ).reset(
            config.mcmc_nwalkers,
            num_params
        )
        with h5py.File(samples_fname, 'a') as samples_file:
            init_chain(samples_file[selected_chain_name])
        _logger.info("Creating new chain in '%s': %s",
                     samples_fname,
                     selected_chain_name)
    else:
        with h5py.File(samples_fname, 'a') as samples_file:
            chain_group = samples_file[selected_chain_name]
            version_group = chain_group['code_version']
            version_dset = version_group['version']
            first_iter_dset = version_group['first_iteration']
            assert version_dset.shape == first_iter_dset.shape
            version_dset.resize((version_dset.size + 1,))
            first_iter_dset.resize((first_iter_dset.size + 1,))
            version_dset[-1] = code_version_str
            first_iter_dset[-1] = chain_group.attrs['iteration']

    backend = emcee.backends.HDFBackend(samples_fname,
                                        selected_chain_name)

    if backend.iteration > 0:
        _logger.info("Extending %s chain in '%s', containing %d samples",
                     selected_chain_name,
                     samples_fname,
                     backend.iteration)

    return backend

#Work around emcee limitation
#pylint: disable=too-few-public-methods
class UnchunkedPool:
    """Disable chunking in Pool.map."""

    def __init__(self, pool):
        """Wrap around the given pool's map."""

        self._pool = pool

    def map(self, *args, **kwargs):
        """Delegate everything to parent, but set chunksize to 1."""

        self._pool.map(*args, **kwargs, chunksize=1)
#pylint: enable=too-few-public-methods

def run(config,
        log_likelihood,
        prior_transform,
        num_params):
    """Sample the selected system using MCMC."""

    sampler_kwargs = dict(
        nwalkers=config.mcmc_nwalkers,
        ndim=num_params,
        log_prob_fn=functools.partial(log_probability,
                                      prior_transform=prior_transform,
                                      log_likelihood=log_likelihood),
        blobs_dtype=[(name, float)
                     for name, _ in log_likelihood.parameter_order],
        backend=prepare_backend(config, num_params, get_code_version_str())
    )

    if config.num_parallel_processes > 1:
        with Pool(
                config.num_parallel_processes,
                initializer=setup_process,
                initargs=[config],
                maxtasksperchild=1
        ) as workers:
            sampler = emcee.EnsembleSampler(**sampler_kwargs,
                                            pool=UnchunkedPool(workers))
            sampler.run_mcmc(
                numpy.random.rand(config.mcmc_nwalkers, num_params),
                config.mcmc_nsteps
            )
    else:
        sampler = emcee.EnsembleSampler(**sampler_kwargs)
        sampler.run_mcmc(
            numpy.random.rand(config.mcmc_nwalkers, num_params),
            config.mcmc_nsteps
        )
