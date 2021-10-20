"""Functions used to perform MCMC sampling."""

import logging
import functools
from multiprocessing import Pool, Process, Queue
import os.path
from glob import glob
import re

import numpy
import emcee
import h5py
from astropy import units
from asteval import Interpreter

from bayesian.sampling import get_code_version_str, setup_process
from bayesian.hacked_emcee_hdf5_backend import HDFBackend

_mutable_config_params = set(['mcmc_nsteps',
                              'num_parallel_processes',
                              'mcmc_recover_initial_conditions',
                              'mcmc_starting_positions_only',
                              'samples_fname',
                              'rvk_show_interpolation',
                              'eccentricity_expansion_coefficients',
                              'stellar_evolution_interpolator_dir',
                              'fname_datetime_format',
                              'std_out_err_fname',
                              'logging_fname',
                              'logging_verbosity',
                              'logging_datetime_format',
                              'logging_message_format',
                              'config_file'])

_logger = logging.getLogger(__name__)

def log_probability(unit_cube_values,
                    prior_transform,
                    log_likelihood,
                    track_final_eccentricity,
                    exclude_from_blob=()):
    """The posterior for MCMC, will track actual params & likelihood."""

    if unit_cube_values.min() < 0 or unit_cube_values.max() > 1:
        _logger.warning(
            'At least one proposed unit cube value is outside the range(0, 1): '
            '%s',
            repr(unit_cube_values)
        )
        return tuple(
            -numpy.inf if i == 0 else numpy.nan
            for i in range(
                len(log_likelihood.parameter_order)
                -
                len(exclude_from_blob)
                +
                (2 if track_final_eccentricity else 1)
            )
        )

    parameters = prior_transform(unit_cube_values)
    log_likelihood_value = log_likelihood(parameters)
    if exclude_from_blob:
        parameters = tuple(
            value
            for value, (param_name, _) in zip(parameters,
                                              prior_transform.parameter_order)
            if param_name not in exclude_from_blob
        )
        _logger.debug('Created blob with %d parameters, excluding %s.',
                      len(parameters),
                      repr(exclude_from_blob))
    else:
        parameters = tuple(parameters)
        _logger.debug('Created blob containing all %d parameters.',
                      len(parameters))
    if track_final_eccentricity:
        parameters += (log_likelihood.final_eccentricity,)
    return (
        (log_likelihood_value,)
        +
        parameters
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

    config_param_defaults = {
        'track_final_eccentricity': False
    }

    config_set = set(vars(config).keys())
    config_set -= set(chain_group.attrs.keys())
    config_set -= _mutable_config_params
    config_set -= set(config_param_defaults.keys())
    if config_set:
        _logger.debug('Configuration parameters %s not saved in chain %s',
                      config_set,
                      chain_group.name)
        return False
    for param, config_value in vars(config).items():
        if param in _mutable_config_params:
            continue
        try:
            saved_value = chain_group.attrs[param]
        except KeyError:
            saved_value = config_param_defaults[param]
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
        elif param == 'evolution_timeout':
            if config_value < saved_value:
                _logger.debug(
                    'Parameter %s (%s) in chain %s does not match %s',
                    param,
                    repr(saved_value),
                    chain_group.name,
                    repr(config_value)
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

def prepare_backend(config, num_params):
    """Return properly configured backend for storing new MCMC samples."""

    def init_chain(chain_group, code_version_str):
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
    code_version_str = get_code_version_str()

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
        HDFBackend(
            samples_fname,
            name=selected_chain_name
        ).reset(
            config.mcmc_nwalkers,
            num_params
        )
        with h5py.File(samples_fname, 'a') as samples_file:
            init_chain(samples_file[selected_chain_name],
                       code_version_str)
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

    backend = HDFBackend(samples_fname, selected_chain_name)

    if backend.iteration > 0:
        _logger.info("Extending %s chain in '%s', containing %d samples",
                     selected_chain_name,
                     samples_fname,
                     backend.iteration)

    return backend, samples_fname, selected_chain_name

#Work around emcee limitation
#pylint: disable=too-few-public-methods
class UnchunkedPool:
    """Disable chunking in Pool.map."""

    def __init__(self, pool):
        """Wrap around the given pool's map."""

        self._pool = pool

    def map(self, *args, **kwargs):
        """Delegate everything to parent, but set chunksize to 1."""

        return self._pool.map(*args, **kwargs, chunksize=1)
#pylint: enable=too-few-public-methods

def evaluate_walker_positions(position_queue,
                              result_queue,
                              log_prob_fn,
                              config):
    """Pick a random positions for MCMC walker with non zero probability."""

    setup_process(config)
    _logger.info('Evaluating starting positions.')
    for position in iter(position_queue.get, 'STOP'):
        log_prob_result = log_prob_fn(position)
        result_queue.put((position, log_prob_result))

def save_initial_position(position,
                          log_prob_result,
                          samples_fname,
                          chain_name,
                          nwalkers=1):
    """Add a good starting position found for the given chain."""

    with h5py.File(samples_fname, 'a') as samples_file:
        if 'starting_positions' not in samples_file[chain_name]:
            samples_file[chain_name].create_group('starting_positions')
        destination = samples_file[chain_name]['starting_positions']
        if 'num_positions_found' not in destination.attrs:
            destination.attrs['num_positions_found'] = 0
            assert 'unit_cube_values' not in destination
            destination.create_dataset(
                'unit_cube_values',
                (nwalkers,) + position.shape,
                maxshape=(None, len(position)),
                dtype=numpy.float64
            )
            assert 'log_prob_results' not in destination
            destination.create_dataset(
                'log_prob_results',
                (nwalkers, len(log_prob_result)),
                maxshape=(None, len(log_prob_result)),
                dtype=numpy.float64
            )

        current_positions = destination.attrs['num_positions_found']

        assert (destination['unit_cube_values'].shape[0]
                ==
                destination['log_prob_results'].shape[0])
        assert destination['unit_cube_values'].shape[1] == len(position)
        assert destination['log_prob_results'].shape[1] == len(log_prob_result)

        assert destination['unit_cube_values'].shape[0] >= current_positions
        if destination['unit_cube_values'].shape[0] == current_positions:
            for dset_name in ['unit_cube_values', 'log_prob_results']:
                destination[dset_name].resize(current_positions + nwalkers,
                                              axis=0)
        destination['unit_cube_values'][current_positions, :] = position
        destination['log_prob_results'][current_positions, :] = log_prob_result
        destination.attrs['num_positions_found'] = current_positions + 1

def load_initial_positions(samples_fname,
                           chain_name,
                           nwalkers,
                           num_params,
                           blobs_dtype):
    """Load previously saved initial positions."""

    starting_positions = numpy.empty((nwalkers, num_params),
                                     dtype=float)
    starting_log_prob = numpy.empty(nwalkers, dtype=float)
    starting_blobs = numpy.empty(nwalkers, dtype=blobs_dtype)

    with h5py.File(samples_fname, 'r') as samples_file:
        if 'starting_positions' in samples_file[chain_name]:
            position_group = samples_file[chain_name]['starting_positions']

            positions_found = position_group.attrs['num_positions_found']

            starting_positions[
                :positions_found,
                :
            ] = position_group[
                'unit_cube_values'
            ][
                :positions_found,
                :
            ]

            starting_log_prob[
                :positions_found
            ] = position_group[
                'log_prob_results'
            ][
                :positions_found,
                0
            ]
            starting_blobs[
                :positions_found
            ] = [
                tuple(row)
                for row in
                position_group[
                    'log_prob_results'
                ][
                    :positions_found,
                    1:
                ]
            ]
        else:
            positions_found = 0

    _logger.info('Loaded %d/%d starting positions from a previous run.',
                 positions_found,
                 nwalkers)
    return (
        starting_positions,
        starting_log_prob,
        starting_blobs,
        positions_found
    )

def recover_initial_positions(config, num_params):
    """Return list of good unit-cube positions from logs of interrupted run."""

    result = numpy.empty((config.mcmc_nwalkers, num_params), dtype=float)
    parse_line = dict(
        unit_cube_start=re.compile(
            r'DEBUG .* bayesian.prior_transform_base: Prior transform: '
            r'U\((?P<values>array\(\[.*)'
        ),
        unit_cube_end=re.compile(
            r'(?P<values>.*\]\))\) -> Parameters:.*'
        ),
        log_probability=re.compile(
            r'INFO .* bayesian.log_likelihood_base: Calculated log_likelihood: '
            r'(?P<log_prob>[^ ]*) \|.*'
        )
    )
    aeval = Interpreter()
    looking_for = 'unit_cube_start'
    total_num_found = 0
    for log_fname in glob(
            config.logging_fname.replace('%(pid)d', '*')
            %
            dict(system=config.system, now='*')
    ):
        num_found = 0
        with open(log_fname) as logf:
            for line in logf:
                match = parse_line[looking_for].fullmatch(line.strip())
                if match:
                    if looking_for == 'unit_cube_start':
                        values = match['values']
                        looking_for = 'unit_cube_end'
                    elif looking_for == 'unit_cube_end':
                        values += match['values']
                        values = aeval(values)
                        looking_for = 'log_probability'
                    elif looking_for == 'log_probability':
                        if (
                                float(match['log_prob'])
                                >
                                config.mcmc_min_initial_log_probability
                        ):
                            result[total_num_found + num_found] = values
                            num_found += 1
                            looking_for = 'unit_cube_start'
                elif looking_for == 'unit_cube_end':
                    values += line

        _logger.debug('Recovered %d potential starting positions from %s',
                      num_found,
                      log_fname)
        total_num_found += num_found
    _logger.info('Recovered %d potential starting positions from previous run.',
                 total_num_found)

    return result[:total_num_found]


#No clean way to simplify
#pylint: disable=too-many-locals
def get_initial_state(*,
                      num_params,
                      blobs_dtype,
                      log_prob_fn,
                      config,
                      samples_fname,
                      chain_name):
    """Pick initial positions for walkers avoiding zero probability spots."""

    _logger.info('Looking for %d suitable walker starting positions',
                 config.mcmc_nwalkers)

    (
        starting_positions,
        starting_log_prob,
        starting_blobs,
        positions_found
    ) = load_initial_positions(samples_fname,
                               chain_name,
                               config.mcmc_nwalkers,
                               num_params,
                               blobs_dtype)

    position_queue = Queue()
    result_queue = Queue()

    if config.mcmc_recover_initial_conditions:
        recovered_positions = recover_initial_positions(config, num_params)
        for position in recovered_positions:
            position_queue.put(position)
    else:
        recovered_positions = numpy.array([])
    for _ in range(config.mcmc_nwalkers - recovered_positions.shape[0]):
        position_queue.put(numpy.random.rand(num_params))

    workers = [
        Process(
            target=evaluate_walker_positions,
            args=(position_queue, result_queue, log_prob_fn, config)
        )
        for _ in range(config.num_parallel_processes)
    ]
    for process in workers:
        process.start()

    while positions_found < config.mcmc_nwalkers:
        position, log_prob_result = result_queue.get()
        if log_prob_result[0] > config.mcmc_min_initial_log_probability:
            starting_positions[positions_found, :] = position
            starting_log_prob[positions_found] = log_prob_result[0]
            starting_blobs[positions_found] = log_prob_result[1:]
            save_initial_position(position,
                                  log_prob_result,
                                  samples_fname,
                                  chain_name,
                                  config.mcmc_nwalkers)
            positions_found += 1
            _logger.debug('%d/%d starting positions found',
                          positions_found,
                          config.mcmc_nwalkers)
        position_queue.put(numpy.random.rand(num_params))

    for process in workers:
        process.terminate()

    return emcee.State(starting_positions, starting_log_prob, starting_blobs)
#pylint: enable=too-many-locals

def get_sampler_config_and_initial_state(config,
                                         log_likelihood,
                                         prior_transform,
                                         num_params):
    """Return configuration and initial state to start or resume sampling."""

    blobs_dtype = [(name, float)
                   for name, _ in log_likelihood.parameter_order]

    if config.track_final_eccentricity:
        blobs_dtype.append(('e_f', float))

    blobs_dtype = numpy.dtype(blobs_dtype)

    backend, samples_fname, chain_name = prepare_backend(config, num_params)

    log_prob_kwargs = dict(
        prior_transform=prior_transform,
        log_likelihood=log_likelihood,
        track_final_eccentricity=config.track_final_eccentricity
    )

    if backend.iteration > 0:
        initial_state = None
        stored_blobs_dtype = backend.get_blobs(flat=True).dtype
        exclude_from_blob = (set(blobs_dtype.names)
                             -
                             set(stored_blobs_dtype.names))
        if exclude_from_blob:
            _logger.warning(
                'Excluding parameters %s from MCMC blobs to allow extending '
                'existing chain',
                repr(exclude_from_blob)
            )

        log_prob_function = functools.partial(
            log_probability,
            exclude_from_blob=exclude_from_blob,
            **log_prob_kwargs
        )
        blobs_dtype = stored_blobs_dtype

    else:
        log_prob_function = functools.partial(
            log_probability,
            **log_prob_kwargs
        )
        initial_state = get_initial_state(num_params=num_params,
                                          blobs_dtype=blobs_dtype,
                                          log_prob_fn=log_prob_function,
                                          config=config,
                                          samples_fname=samples_fname,
                                          chain_name=chain_name)

    return (
        dict(
            nwalkers=config.mcmc_nwalkers,
            ndim=num_params,
            log_prob_fn=log_prob_function,
            blobs_dtype=blobs_dtype,
            backend=backend
        ),
        initial_state
    )


def run(config,
        log_likelihood,
        prior_transform,
        num_params):
    """Sample the selected system using MCMC."""

    sampler_config, initial_state = get_sampler_config_and_initial_state(
        config,
        log_likelihood,
        prior_transform,
        num_params
    )

    if config.mcmc_starting_positions_only:
        _logger.info('Found initial positions for all walkers, exiting!')
        return

    if config.num_parallel_processes > 1:
        with Pool(
                config.num_parallel_processes,
                initializer=setup_process,
                initargs=[config],
                maxtasksperchild=1
        ) as workers:
            sampler = emcee.EnsembleSampler(**sampler_config,
                                            pool=UnchunkedPool(workers))
            sampler.run_mcmc(initial_state, config.mcmc_nsteps)
    else:
        sampler = emcee.EnsembleSampler(**sampler_config)

        sampler.run_mcmc(initial_state, config.mcmc_nsteps)
