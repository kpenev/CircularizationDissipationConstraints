"""Functions used to perform nested sampling with Dynesty."""

import logging
import functools
from contextlib import nullcontext
from multiprocessing import Pool, Process, Queue
import os
import os.path
from glob import glob
import re

import numpy
from scipy.stats import norm
import emcee
import dynesty
import h5py
from astropy import units
from asteval import Interpreter

from multiprocessing_util import get_code_version_str, setup_process, setup_process_map
from bayesian.hacked_emcee_hdf5_backend import HDFBackend
from bayesian.posterior_cdf_uncertainty import estimate_posterior_cdf_uncertainty
from bayesian.dynesty_worker_journal import worker_journal_context, JournalPool

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
                              'config_file',
                              'rv_likelihood_pickle_fname',
                              'mass_sampling_pickle_fname',
                              'photometric_constraint_pickle_fname',
                              'nn_data_dir'])

_logger = logging.getLogger(__name__)

def _weighted_quantile(values, weights, quantiles):
    """Return weighted quantiles for a one-dimensional sample."""

    values = numpy.asarray(values)
    weights = numpy.asarray(weights)
    valid = numpy.logical_and(numpy.isfinite(values),
                              numpy.isfinite(weights))
    valid = numpy.logical_and(valid, weights >= 0.0)
    values = values[valid]
    weights = weights[valid]
    if values.size == 0 or weights.sum() <= 0.0:
        return numpy.full(len(quantiles), numpy.nan)
    weights = weights / weights.sum()
    return numpy.asarray(
        dynesty.utils.quantile(values, quantiles, weights=weights)
    )


def _nested_posterior_summary(results, prior_transform, parameter_names):
    """Summarize the weighted posterior for the requested model parameters."""

    samples = numpy.asarray(results.samples)
    if samples.shape[0] == 0:
        return None
    weights = numpy.asarray(results.importance_weights())
    weights /= weights.sum()
    parameter_order = [
        name
        for name, _ in prior_transform.parameter_order
    ]
    parameter_indices = [
        parameter_order.index(name)
        for name in parameter_names
    ]
    transformed_samples = numpy.asarray([
        prior_transform(sample)['parameters'][parameter_indices]
        for sample in samples
    ])
    quantiles = numpy.asarray([
        _weighted_quantile(transformed_samples[:, param_index],
                           weights,
                           [0.16, 0.5, 0.84])
        for param_index in range(len(parameter_indices))
    ])
    return dict(
        quantiles=quantiles,
        ess=1.0 / numpy.sum(weights * weights),
        nsamples=samples.shape[0]
    )


def _get_results_ncall(results):
    """Return the total number of likelihood calls stored by dynesty."""

    try:
        return int(numpy.sum(results.ncall))
    except (AttributeError, KeyError):
        return ''


def _posterior_weight_summary(results):
    """Return stored posterior sample count and ESS from dynesty weights."""

    samples = numpy.asarray(results.samples)
    weights = numpy.asarray(results.importance_weights())
    weights /= weights.sum()
    return dict(
        nsamples=samples.shape[0],
        ess=1.0 / numpy.sum(weights * weights)
    )


def _get_dynamic_batch_count(results, fallback=''):
    """Return the number of dynamic batches, when dynesty exposes it."""

    for attribute in ('batch_nlive', 'batch_bounds'):
        try:
            return len(getattr(results, attribute))
        except (AttributeError, TypeError):
            pass
    return fallback


def _selected_kl_statistic(stats, statistic):
    """Return the requested aggregate KL statistic."""

    normalized = statistic.lower()
    if normalized in ('mean', 'median', 'std', 'max'):
        return stats['kld_' + normalized]
    if normalized == 'p90':
        return stats['kld_p90']
    if normalized == 'p95':
        return stats['kld_p95']
    if normalized == 'fractional_std':
        return stats['kld_fractional_std']
    raise ValueError(
        'Unsupported posterior KL statistic: '
        f'{statistic!r}; use mean, median, p90, p95, max, std, or fractional_std.'
    )


def _estimate_posterior_kl_uncertainty(results,
                                       method='resample',
                                       nreal=128,
                                       statistic='p90',
                                       seed=None,
                                       approx=False):
    """Estimate dynesty posterior uncertainty from repeated KL realizations."""

    if nreal < 1:
        raise ValueError('POSTERIOR_KL_NREAL must be at least 1')
    random_state = numpy.random.default_rng(seed)
    kld_values = numpy.asarray([
        dynesty.utils.kld_error(
            results,
            error=method,
            rstate=random_state,
            approx=approx
        )[-1]
        for _ in range(nreal)
    ])
    kld_mean = float(numpy.mean(kld_values))
    kld_std = float(numpy.std(kld_values))
    # A nonpositive or nonfinite mean cannot satisfy a fractional criterion.
    kld_fractional_std = (
        kld_std / kld_mean
        if numpy.isfinite(kld_mean) and kld_mean > 0.0
        and numpy.isfinite(kld_std)
        else float('inf')
    )
    stats = dict(
        kld_method=method,
        kld_nreal=nreal,
        kld_statistic=statistic,
        kld_mean=kld_mean,
        kld_median=float(numpy.median(kld_values)),
        kld_std=kld_std,
        kld_fractional_std=kld_fractional_std,
        kld_p90=float(numpy.percentile(kld_values, 90.0)),
        kld_p95=float(numpy.percentile(kld_values, 95.0)),
        kld_max=float(numpy.max(kld_values))
    )
    stats['kld_selected'] = _selected_kl_statistic(stats, statistic)
    return stats, kld_values


def _write_kl_diagnostic_header(diagnostic_file):
    """Write a posterior KL diagnostic table header."""

    diagnostic_file.write(
        'check\titeration\tnsamples\ttotal_likelihood_calls\t'
        'effective_sample_size\tdynamic_batches\tkld_method\t'
        'kld_nreal\tkld_mean\tkld_median\tkld_std\t'
        'kld_fractional_std\tkld_p90\t'
        'kld_p95\tkld_max\tkld_threshold\tselected_kld_statistic\t'
        'selected_kld_value\tkl_converged\n'
    )


def _format_kl_diagnostic_row(check_index,
                              iteration,
                              results,
                              dynamic_batches,
                              kl_stats,
                              kl_threshold,
                              kl_converged):
    """Format one posterior KL convergence diagnostic row."""

    posterior_summary = _posterior_weight_summary(results)
    return (
        f'{check_index:d}\t'
        f'{iteration}\t'
        f'{posterior_summary["nsamples"]:d}\t'
        f'{_get_results_ncall(results)}\t'
        f'{posterior_summary["ess"]:.8g}\t'
        f'{dynamic_batches}\t'
        f'{kl_stats["kld_method"]}\t'
        f'{kl_stats["kld_nreal"]:d}\t'
        f'{kl_stats["kld_mean"]:.8g}\t'
        f'{kl_stats["kld_median"]:.8g}\t'
        f'{kl_stats["kld_std"]:.8g}\t'
        f'{kl_stats["kld_fractional_std"]:.8g}\t'
        f'{kl_stats["kld_p90"]:.8g}\t'
        f'{kl_stats["kld_p95"]:.8g}\t'
        f'{kl_stats["kld_max"]:.8g}\t'
        f'{kl_threshold:.8g}\t'
        f'{kl_stats["kld_statistic"]}\t'
        f'{kl_stats["kld_selected"]:.8g}\t'
        f'{kl_converged}\n'
    )


def _write_weighted_percentile_table(fname,
                                     parameter_names,
                                     samples,
                                     weights):
    """Write p10...p100 weighted percentiles for the given samples."""

    quantile_levels = numpy.arange(0.1, 1.01, 0.1)
    percentile_names = [
        'p10',
        'p20',
        'p30',
        'p40',
        'p50',
        'p60',
        'p70',
        'p80',
        'p90',
        'p100'
    ]
    with open(fname, 'w', encoding='utf-8') as table_file:
        table_file.write(
            'parameter\t'
            +
            '\t'.join(percentile_names)
            +
            '\tmonotonic\tp50_weighted_median_agrees\n'
        )
        for parameter_index, parameter_name in enumerate(parameter_names):
            percentiles = numpy.asarray(
                dynesty.utils.quantile(
                    samples[:, parameter_index],
                    quantile_levels,
                    weights=weights
                )
            )
            weighted_median = _weighted_quantile(
                samples[:, parameter_index],
                weights,
                [0.5]
            )[0]
            table_file.write(
                str(parameter_name)
                +
                '\t'
                +
                '\t'.join(f'{value:.17g}' for value in percentiles)
                +
                f'\t{bool(numpy.all(numpy.diff(percentiles) >= -1e-12))}'
                +
                f'\t{bool(numpy.allclose(percentiles[4], weighted_median))}'
                +
                '\n'
            )


def _write_dynesty_outputs(results,
                           prior_transform,
                           checkpoint_file,
                           stopping_config):
    """Write weighted posterior percentiles and dynesty diagnostics."""

    samples = numpy.asarray(results.samples)
    weights = numpy.asarray(results.importance_weights())
    weights /= weights.sum()
    physical_parameter_names = [
        name
        for name, _ in prior_transform.parameter_order
    ]
    physical_samples = numpy.asarray([
        prior_transform(sample)['parameters']
        for sample in samples
    ])
    _write_weighted_percentile_table(
        checkpoint_file + '.dynesty_unit_parameter_percentiles.tsv',
        [f'unit_cube_{index:d}' for index in range(samples.shape[1])],
        samples,
        weights
    )
    _write_weighted_percentile_table(
        checkpoint_file + '.model_parameter_percentiles.tsv',
        physical_parameter_names,
        physical_samples,
        weights
    )
    diagnostics = dict(
        number_of_stored_nested_samples=samples.shape[0],
        total_likelihood_calls=_get_results_ncall(results),
        posterior_ess=1.0 / numpy.sum(weights * weights),
        final_logz=float(numpy.asarray(results.logz)[-1]),
        final_logz_uncertainty=float(numpy.asarray(results.logzerr)[-1]),
        rejected_proposals_included_as_zero_weight_samples=False
    )
    diagnostics.update(stopping_config)
    with open(checkpoint_file + '.diagnostics.tsv',
              'w',
              encoding='utf-8') as diagnostics_file:
        diagnostics_file.write('quantity\tvalue\n')
        for key, value in diagnostics.items():
            diagnostics_file.write(f'{key}\t{value}\n')


def _posterior_shift_within_tolerance(previous_summary,
                                      current_summary,
                                      tolerance,
                                      minimum_ess=0.0,
                                      width_floor=1e-12):
    """True iff all monitored q16/q50/q84 shifts are within tolerance."""

    previous_quantiles = previous_summary['quantiles']
    current_quantiles = current_summary['quantiles']
    widths = numpy.maximum(
        current_quantiles[:, 2] - current_quantiles[:, 0],
        width_floor
    )
    finite_width = numpy.isfinite(widths)
    shifts = numpy.full(current_quantiles.shape, numpy.inf)
    shifts[finite_width, :] = (
        numpy.abs(current_quantiles[finite_width, :]
                  -
                  previous_quantiles[finite_width, :])
        /
        widths[finite_width, None]
    )
    return (
        shifts,
        (
            numpy.all(shifts <= tolerance)
            and
            current_summary['ess'] >= minimum_ess
        )
    )


def _refresh_restored_sampler_functions(sampler, log_probability_function):
    """Replace checkpointed likelihood objects before dispatching new pool work.

    The physical likelihood contains a native stellar interpolator. A saved
    dynesty queue can be consumed after restore, but workers need the fresh
    likelihood and prior objects built for this process when refilling it.
    Legacy checkpoints discard candidate queues as before. Journal-enabled
    checkpoints preserve completed candidates to reproduce the same search.
    """

    preserve_queue = getattr(sampler, '_worker_journal_version', 0) == 1
    samplers = [sampler]
    if hasattr(sampler, 'sampler') and sampler.sampler is not None:
        samplers.append(sampler.sampler)
    if hasattr(sampler, 'batch_sampler') and sampler.batch_sampler is not None:
        samplers.append(sampler.batch_sampler)
    for active_sampler in samplers:
        active_sampler.loglikelihood.loglikelihood = log_probability_function
        active_sampler.prior_transform = identity_prior_transform
        if hasattr(active_sampler, 'queue') and not preserve_queue:
            active_sampler.queue = []
            active_sampler.nqueue = 0
    _logger.info(
        'Refreshed restored dynesty likelihood/prior; preserve candidate '
        'queues=%s', preserve_queue
    )


def log_probability(unit_cube_values, prior_transform, log_likelihood):
    """Evaluate the likelihood for a Dynesty unit-cube sample.

    >>> log_probability([0.25], lambda values: {'parameters': values},
    ...                 lambda parameters: float(parameters[0]))
    0.25
    """

    unit_cube_values = numpy.asarray(unit_cube_values)
    if unit_cube_values.max() + 1e-10 >= 1:
        _logger.warning('unit_cube_values is too close to 1')
        return -numpy.inf
    try:
        transformed = prior_transform(unit_cube_values)
    except Exception as error:
        if 'weights zero' not in str(error):
            raise
        _logger.warning('ValueError during prior transformation: %s', error)
        return -numpy.inf
    return log_likelihood(**transformed)

def identity_prior_transform(unit_cube_values):
    """Return Dynesty unit-cube values unchanged after local transformation."""

    return unit_cube_values

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

    config_dict = dict(vars(config))
    del config_dict['reseed_random_number_generator']
    config_set = set(config_dict.keys())
    config_set -= set(chain_group.attrs.keys())
    config_set -= _mutable_config_params
    config_set -= set(config_param_defaults.keys())
    if config_set:
        _logger.debug('Configuration parameters %s not saved in chain %s',
                      config_set,
                      chain_group.name)
        return False
    for param, config_value in config_dict.items():
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
                _logger.debug('%s',repr(config_value))
                _logger.debug('%s',repr((numpy.array(config_value) != saved_value)))
                if numpy.array(numpy.array(config_value) != saved_value).any():
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
                if True:#compare_chain_configuration(config, chain_group):
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

    setup_process(
                    fname_datetime_format=config.fname_datetime_format,
                    system=config.system,
                    sampling=config.sampling,
                    std_out_err_fname=config.std_out_err_fname,
                    logging_fname=config.logging_fname,
                    logging_verbosity=config.logging_verbosity,
                    logging_message_format=config.logging_message_format,
                    logging_datetime_format=config.logging_datetime_format,
                    task='init'
                  )
    _logger.info('Evaluating starting positions.')
    for position in iter(position_queue.get, 'STOP'):
        log_prob_result = log_prob_fn(position[0])
        result_queue.put((position[0], log_prob_result))

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
            assert 'independent_normal_values' not in destination
            destination.create_dataset(
                'independent_normal_values',
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

        assert (destination['independent_normal_values'].shape[0]
                ==
                destination['log_prob_results'].shape[0])
        assert destination['independent_normal_values'].shape[1] == (
            len(position)
        )
        assert destination['log_prob_results'].shape[1] == len(log_prob_result)

        assert (
            destination['independent_normal_values'].shape[0]
            >=
            current_positions
        )
        if (
                destination['independent_normal_values'].shape[0]
                ==
                current_positions
        ):
            for dset_name in ['independent_normal_values', 'log_prob_results']:
                destination[dset_name].resize(current_positions + nwalkers,
                                              axis=0)
        destination['independent_normal_values'][current_positions, :] = (
            position
        )
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
                'independent_normal_values'
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

    raise NotImplementedError(
        'Recovering initial positions from logger is currently broken.'
    )
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
                      lgq_min_param_index,
                      blobs_dtype,
                      log_prob_fn,
                      config,
                      samples_fname,
                      chain_name,
                      log_likelihood):
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
            position_queue.put((position,))
    else:
        recovered_positions = numpy.array([])
    for _ in range(config.mcmc_nwalkers - recovered_positions.shape[0]):
        position_queue.put((norm.rvs(size=num_params),))

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
        _logger.debug('Log-likelihood(%s) = %s',
                      repr(position),
                      repr(log_prob_result))
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
            position_queue.put((norm.rvs(size=num_params),))
        else:
            orig_pos_repr = repr(position)
            _logger.debug('Log-likelihood._de.support() = %s',repr(log_likelihood._de.support()))
            if log_prob_result[23] > log_likelihood._de.support()[1]:
                log_likelihood.upper_limit = position[lgq_min_param_index]
                position[lgq_min_param_index] = norm.ppf(
                    (
                        norm.cdf(log_likelihood.upper_limit)
                        -
                        norm.cdf(log_likelihood.lower_limit)
                    )
                    *
                    numpy.random.rand()
                    +
                    norm.cdf(log_likelihood.lower_limit)
                )
                _logger.debug('Tweaking starting position to lower Q: %s -> %s',
                              orig_pos_repr,
                              repr(position))
            elif log_prob_result[23] < log_likelihood._de.support()[0]:
                log_likelihood.lower_limit = position[lgq_min_param_index]
                position[lgq_min_param_index] = norm.ppf(
                    (
                        norm.cdf(log_likelihood.upper_limit)
                        -
                        norm.cdf(log_likelihood.lower_limit)
                    )
                    *
                    numpy.random.rand()
                    +
                    norm.cdf(log_likelihood.lower_limit)
                )
                _logger.debug('Tweaking starting position to raise Q: %s -> %s',
                              orig_pos_repr,
                              repr(position))
            else:
                position = norm.rvs(size=num_params)
                _logger.debug('Declaring starting hopeless: %s -> %s',
                              orig_pos_repr,
                              repr(position))
            position_queue.put((position,))

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

    # add labels for the coeffs
    blobs_dtype.append(('a', float))
    blobs_dtype.append(('b', float))
    blobs_dtype.append(('c', float))
    blobs_dtype.append(('median_e', float))
    blobs_dtype.append(('approximation', float))
    # add label for ehat_prime
    blobs_dtype.append(('ehat_prime', float))

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
        initial_state = get_initial_state(
            num_params=num_params,
            blobs_dtype=blobs_dtype,
            log_prob_fn=log_prob_function,
            lgq_min_param_index=(
                prior_transform.get_unit_cube_indices()['lgQ_min']
            ),
            config=config,
            samples_fname=samples_fname,
            chain_name=chain_name,
            log_likelihood = log_likelihood
        )

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
    """Sample the selected system using Dynesty."""

    _logger.info('Starting sampling with configuration: %r', vars(config))
    nlive = int(getattr(config, 'dynesty_nlive', config.mcmc_nwalkers))
    nprocesses = int(config.num_parallel_processes)
    if nlive < 1 or nprocesses < 1:
        raise ValueError('dynesty-nlive and num-parallel-processes must be positive')
    log_probability_function = functools.partial(
        log_probability,
        prior_transform=prior_transform,
        log_likelihood=log_likelihood
    )
    samples_fname = (
        config.samples_fname
        %
        dict(system=config.system,
             sampling=config.sampling)
    )
    checkpoint_file = samples_fname + '.dynesty.checkpoint'
    os.makedirs(os.path.dirname(checkpoint_file), exist_ok=True)
    resume = os.path.exists(checkpoint_file)
    if os.environ.get("NESTY_REQUIRE_CHECKPOINT") == "1" and not resume:
        raise FileNotFoundError(
            "Continuation requires a Dynesty checkpoint: " + checkpoint_file
        )
    if resume:
        _logger.info('Resuming dynesty from checkpoint: %s', checkpoint_file)
    else:
        _logger.info('Writing dynesty checkpoints to: %s', checkpoint_file)
    stopping_config = dict(
        sampler_type='DynamicNestedSampler'
        if os.environ.get('NESTY_DYNAMIC_NESTED', '0') == '1'
        else
        'NestedSampler',
        nlive=nlive,
        dynesty_checkpoint_file=checkpoint_file,
        dynesty_posterior_source='results.samples with results.importance_weights()',
        posterior_zero_weight_rejected_proposals=False
    )
    pool_context = (
        Pool(
            nprocesses,
            initializer=setup_process_map,
            initargs=[dict(
                fname_datetime_format=config.fname_datetime_format,
                system=config.system,
                sampling=config.sampling,
                std_out_err_fname=config.std_out_err_fname,
                logging_fname=config.logging_fname,
                logging_verbosity=config.logging_verbosity,
                logging_message_format=config.logging_message_format,
                logging_datetime_format=config.logging_datetime_format
            )]
        )
        if nprocesses > 1 else nullcontext()
    )
    with pool_context as raw_workers, worker_journal_context(
            raw_workers, checkpoint_file,
            os.path.dirname(os.path.dirname(__file__))) as workers:
        sampler_kwargs = dict(nlive=nlive)
        if workers is not None:
            sampler_kwargs.update(pool=workers, queue_size=nprocesses)
        if os.environ.get('NESTY_DYNAMIC_NESTED', '0') == '1':
            if resume:
                sampler = dynesty.utils.restore_sampler(
                    checkpoint_file, pool=workers
                )
                _refresh_restored_sampler_functions(
                    sampler, log_probability_function
                )
                _logger.info(
                    "Restored dynamic dynesty sampler at iteration %d",
                    sampler.it - 1
                )
            else:
                sampler = dynesty.DynamicNestedSampler(
                    log_probability_function,
                    identity_prior_transform,
                    ndim=num_params,
                    **sampler_kwargs
                )
            dynamic_kwargs = dict(
                nlive_init=nlive,
                nlive_batch=nlive,
                checkpoint_file=checkpoint_file,
                checkpoint_every=float(os.environ.get("NESTY_CHECKPOINT_EVERY", "0")),
                resume=resume
            )
            stopping_config.update(
                dynamic_nested=True,
                dynamic_nlive_init=nlive,
                dynamic_nlive_batch=nlive,
                dynamic_default_ess_stopping=False,
                dynamic_dlogz='not used for posterior KL stopping'
            )
            if os.environ.get('NESTY_POSTERIOR_CDF_STOP', '0') == '1':
                cdf_tol = float(os.environ.get('NESTY_POSTERIOR_CDF_TOL', '0.01'))
                cdf_nreal = int(os.environ.get('NESTY_POSTERIOR_CDF_NREAL', '128'))
                cdf_confidence = float(
                    os.environ.get('NESTY_POSTERIOR_CDF_CONFIDENCE', '0.95')
                )
                cdf_seed = (
                    int(os.environ['NESTY_POSTERIOR_CDF_SEED'])
                    if 'NESTY_POSTERIOR_CDF_SEED' in os.environ else None
                )
                summary_path = checkpoint_file + '.dynamic_posterior_cdf_std.tsv'
                details_path = checkpoint_file + '.dynamic_posterior_cdf_std_details.tsv'
                summary_columns = (
                    'check', 'nsamples', 'total_likelihood_calls',
                    'effective_sample_size', 'dynamic_batches', 'nreal',
                    'confidence', 'threshold', 'stopping_statistic', 'cdf_std_ddof',
                    'global_sigma_cdf', 'global_p95_absolute_cdf_error',
                    'worst_parameter', 'worst_probability',
                    'worst_tidal_period_days', 'worst_tidal_frequency_rad_per_day',
                    'posterior_cdf_converged', 'simulation_method',
                    'excluded_fixed_parameters'
                )
                detail_columns = (
                    'check', 'parameter', 'observable_kind', 'tidal_period_days',
                    'tidal_frequency_rad_per_day', 'probability',
                    'nominal_physical_quantile', 'nominal_empirical_cdf',
                    'simulated_cdf_mean', 'simulated_cdf_std',
                    'p95_absolute_cdf_error'
                )
                for path, columns in ((summary_path, summary_columns),
                                      (details_path, detail_columns)):
                    with open(path, 'a', encoding='utf-8') as output:
                        if output.tell() == 0:
                            output.write('\t'.join(columns) + '\n')
                with open(summary_path, encoding='utf-8') as output:
                    prior_checks = max(sum(1 for _ in output) - 1, 0)
                stopping_config.update(
                    posterior_stability_stop='weighted_physical_and_tidal_Q_cdf_std',
                    posterior_cdf_stopping_statistic='max_cdf_standard_deviation',
                    posterior_cdf_std_ddof=1,
                    posterior_cdf_p95_used_for_stopping=False,
                    posterior_cdf_threshold=cdf_tol,
                    posterior_cdf_nreal=cdf_nreal,
                    posterior_cdf_confidence=cdf_confidence,
                    posterior_cdf_seed=cdf_seed,
                    posterior_cdf_converged=False,
                    dynamic_dlogz='not used for posterior CDF stopping',
                    dynamic_ess_stop_used=False,
                    dynamic_evidence_uncertainty_stop_used=False
                )

                def dynamic_posterior_cdf_stop(results, stop_kwargs, **_):
                    check_count = stop_kwargs.get('check_count', prior_checks) + 1
                    seed = None if cdf_seed is None else cdf_seed + check_count
                    statistic, details = estimate_posterior_cdf_uncertainty(
                        results, prior_transform,
                        n_realizations=cdf_nreal,
                        confidence=cdf_confidence,
                        tolerance=cdf_tol,
                        seed=seed
                    )
                    stop_kwargs['check_count'] = check_count
                    summary = _posterior_weight_summary(results)
                    row = dict(
                        check=check_count,
                        nsamples=summary['nsamples'],
                        total_likelihood_calls=_get_results_ncall(results),
                        effective_sample_size=summary['ess'],
                        dynamic_batches=_get_dynamic_batch_count(results),
                        nreal=cdf_nreal,
                        confidence=cdf_confidence,
                        threshold=cdf_tol,
                        stopping_statistic=statistic['stopping_statistic'],
                        cdf_std_ddof=statistic['cdf_std_ddof'],
                        global_sigma_cdf=statistic['global_sigma_cdf'],
                        global_p95_absolute_cdf_error=statistic['global_p95_absolute_cdf_error'],
                        worst_tidal_period_days=statistic['worst_tidal_period_days'],
                        worst_tidal_frequency_rad_per_day=statistic['worst_tidal_frequency_rad_per_day'],
                        worst_parameter=statistic['worst_parameter'],
                        worst_probability=statistic['worst_probability'],
                        posterior_cdf_converged=statistic['converged'],
                        simulation_method=statistic['simulation_method'],
                        excluded_fixed_parameters=','.join(
                            statistic['excluded_fixed_parameters']
                        )
                    )
                    with open(summary_path, 'a', encoding='utf-8') as output:
                        output.write('\t'.join(str(row[key]) for key in summary_columns) + '\n')
                    with open(details_path, 'a', encoding='utf-8') as output:
                        for detail in details:
                            detail['check'] = check_count
                            output.write('\t'.join(
                                str(detail[key]) for key in detail_columns
                            ) + '\n')
                    stopping_config.update(
                        posterior_cdf_converged=statistic['converged'],
                        posterior_cdf_final_sigma=statistic['global_sigma_cdf'],
                        posterior_cdf_final_p95_error=statistic['global_p95_absolute_cdf_error'],
                        posterior_cdf_monitored_tidal_frequencies=statistic['monitored_tidal_frequency_count'],
                        posterior_cdf_worst_parameter=statistic['worst_parameter'],
                        posterior_cdf_worst_probability=statistic['worst_probability'],
                        posterior_cdf_check_count=check_count
                    )
                    _logger.info(
                        'Dynamic posterior CDF STD check %d: max_sigma_CDF=%g, '
                        'worst=%s p=%g, threshold=%g, converged=%s',
                        check_count, statistic['global_sigma_cdf'],
                        statistic['worst_parameter'],
                        statistic['worst_probability'], cdf_tol,
                        statistic['converged']
                    )
                    return statistic['converged'], (
                        statistic['global_sigma_cdf'], cdf_tol,
                        statistic['worst_probability']
                    )

                dynamic_kwargs['stop_function'] = dynamic_posterior_cdf_stop
                dynamic_kwargs['stop_kwargs'] = dict()
                dynamic_kwargs['use_stop'] = True
            elif (
                    os.environ.get('NESTY_POSTERIOR_STABILITY_STOP', '0')
                    ==
                    '1'
            ):
                posterior_kl_tol = float(
                    os.environ.get('NESTY_POSTERIOR_KL_TOL', '0.01')
                )
                posterior_kl_nreal = int(
                    os.environ.get('NESTY_POSTERIOR_KL_NREAL', '128')
                )
                posterior_kl_statistic = os.environ.get(
                    'NESTY_POSTERIOR_KL_STATISTIC',
                    'p90'
                )
                posterior_kl_method = os.environ.get(
                    'NESTY_POSTERIOR_KL_METHOD',
                    'resample'
                )
                posterior_kl_approx = (
                    os.environ.get('NESTY_POSTERIOR_KL_APPROX', '0')
                    ==
                    '1'
                )
                posterior_kl_seed = (
                    int(os.environ['NESTY_POSTERIOR_KL_SEED'])
                    if 'NESTY_POSTERIOR_KL_SEED' in os.environ
                    else
                    None
                )
                stopping_config.update(
                    posterior_stability_stop='dynesty_kld_error',
                    posterior_kl_threshold=posterior_kl_tol,
                    posterior_kl_nreal=posterior_kl_nreal,
                    posterior_kl_statistic=posterior_kl_statistic,
                    posterior_kl_method=posterior_kl_method,
                    posterior_kl_approx=posterior_kl_approx,
                    posterior_kl_seed=posterior_kl_seed,
                    dynamic_ess_stop_used=False,
                    dynamic_evidence_uncertainty_stop_used=False
                )
                stability_path = (
                    checkpoint_file
                    +
                    '.dynamic_posterior_kl.tsv'
                )
                with open(stability_path, 'a', encoding='utf-8') as stability_file:
                    if stability_file.tell() == 0:
                        _write_kl_diagnostic_header(stability_file)

                def dynamic_posterior_kl_stop(results,
                                              stop_kwargs,
                                              **_):
                    check_count = stop_kwargs.get('check_count', 0) + 1
                    seed = (
                        None
                        if posterior_kl_seed is None
                        else
                        posterior_kl_seed + check_count
                    )
                    kl_stats, _kl_values = _estimate_posterior_kl_uncertainty(
                        results,
                        method=posterior_kl_method,
                        nreal=posterior_kl_nreal,
                        statistic=posterior_kl_statistic,
                        seed=seed,
                        approx=posterior_kl_approx
                    )
                    kl_converged = (
                        kl_stats['kld_selected']
                        <=
                        posterior_kl_tol
                    )
                    stop_kwargs['check_count'] = check_count
                    dynamic_batches = _get_dynamic_batch_count(
                        results,
                        fallback=check_count
                    )
                    with open(stability_path,
                              'a',
                              encoding='utf-8') as stability_file:
                        stability_file.write(
                            _format_kl_diagnostic_row(
                                check_count,
                                '',
                                results,
                                dynamic_batches,
                                kl_stats,
                                posterior_kl_tol,
                                kl_converged
                            )
                        )
                    _logger.info(
                        'Dynamic posterior KL check %d: method=%s, nreal=%d, '
                        '%s=%g, threshold=%g, converged=%s',
                        check_count,
                        posterior_kl_method,
                        posterior_kl_nreal,
                        posterior_kl_statistic,
                        kl_stats['kld_selected'],
                        posterior_kl_tol,
                        kl_converged
                    )
                    return (
                        kl_converged,
                        (
                            kl_stats['kld_selected'],
                            posterior_kl_tol,
                            kl_stats['kld_mean'],
                            kl_stats['kld_p90'],
                            kl_stats['kld_p95'],
                            kl_stats['kld_max']
                        )
                    )

                dynamic_kwargs['stop_function'] = dynamic_posterior_kl_stop
                dynamic_kwargs['stop_kwargs'] = dict()
                dynamic_kwargs['use_stop'] = True

                # The combined dynamic results are unavailable during an
                # unfinished batch. During the initial base run, the embedded
                # static sampler does have a valid partial results object.
                base_stability_path = (
                    checkpoint_file + '.dynamic_base_iteration_kl.tsv'
                )
                with open(base_stability_path,
                          'a', encoding='utf-8') as base_stability_file:
                    if base_stability_file.tell() == 0:
                        _write_kl_diagnostic_header(base_stability_file)
                last_base_sample_count = [0]

                def dynamic_base_iteration_kl(progress_result,
                                              iteration,
                                              ncall,
                                              **progress_kwargs):
                    dynesty.utils.print_fn(
                        progress_result, iteration, ncall,
                        **progress_kwargs
                    )
                    if (
                            progress_kwargs.get('nbatch') != 0
                            or sampler.internal_state.name != 'INBASE'
                    ):
                        return
                    try:
                        base_results = sampler.sampler.results
                        sample_count = len(base_results.samples)
                        if (
                                sample_count < 2
                                or sample_count <= last_base_sample_count[0]
                        ):
                            return
                        seed = (
                            None if posterior_kl_seed is None else
                            posterior_kl_seed + sample_count
                        )
                        kl_stats, _ = _estimate_posterior_kl_uncertainty(
                            base_results,
                            method=posterior_kl_method,
                            nreal=posterior_kl_nreal,
                            statistic=posterior_kl_statistic,
                            seed=seed,
                            approx=posterior_kl_approx
                        )
                        converged = kl_stats['kld_selected'] <= posterior_kl_tol
                        with open(base_stability_path, 'a',
                                  encoding='utf-8') as base_stability_file:
                            base_stability_file.write(
                                _format_kl_diagnostic_row(
                                    sample_count,
                                    iteration,
                                    base_results,
                                    0,
                                    kl_stats,
                                    posterior_kl_tol,
                                    converged
                                )
                            )
                        last_base_sample_count[0] = sample_count
                        _logger.info(
                            'Dynamic base KL diagnostic: iteration=%d, '
                            'samples=%d, p90=%g (full dynamic stop '
                            'evaluated after each completed batch)',
                            iteration, sample_count, kl_stats['kld_p90']
                        )
                    except Exception:
                        _logger.exception(
                            'Dynamic base iteration KL diagnostic failed '
                            'at iteration %d; sampling continues', iteration
                        )

                dynamic_kwargs['print_func'] = dynamic_base_iteration_kl
                dynamic_kwargs['print_progress'] = True
            elif 'NESTY_DYNAMIC_N_EFFECTIVE' in os.environ:
                dynamic_kwargs['n_effective'] = int(
                    os.environ['NESTY_DYNAMIC_N_EFFECTIVE']
                )
                stopping_config.update(
                    dynamic_n_effective=dynamic_kwargs['n_effective'],
                    dynamic_stop_kwargs='dynesty default ESS stopping'
                )
            if 'NESTY_DYNAMIC_MAXBATCH' in os.environ:
                dynamic_kwargs['maxbatch'] = int(
                    os.environ['NESTY_DYNAMIC_MAXBATCH']
                )
                stopping_config['dynamic_maxbatch'] = dynamic_kwargs['maxbatch']
            if 'NESTY_DYNAMIC_MAXITER' in os.environ:
                dynamic_kwargs['maxiter'] = int(
                    os.environ['NESTY_DYNAMIC_MAXITER']
                )
                stopping_config['dynamic_maxiter'] = dynamic_kwargs['maxiter']
            if 'NESTY_DYNAMIC_MAXCALL' in os.environ:
                dynamic_kwargs['maxcall'] = int(
                    os.environ['NESTY_DYNAMIC_MAXCALL']
                )
                stopping_config['dynamic_maxcall'] = dynamic_kwargs['maxcall']
            _logger.info(
                'Using DynamicNestedSampler with kwargs: %s',
                repr(dynamic_kwargs)
            )
            if isinstance(workers, JournalPool):
                sampler._worker_journal_version = 1
            sampler.run_nested(**dynamic_kwargs)
        else:
            if resume:
                sampler = dynesty.utils.restore_sampler(
                    checkpoint_file, pool=workers
                )
                _refresh_restored_sampler_functions(
                    sampler, log_probability_function
                )
                _logger.info(
                    "Restored static dynesty sampler at iteration %d",
                    sampler.it - 1
                )
            else:
                sampler = dynesty.NestedSampler(
                    log_probability_function,
                    identity_prior_transform,
                    ndim=num_params,
                    **sampler_kwargs
                )
            if isinstance(workers, JournalPool):
                sampler._worker_journal_version = 1
        if (
                os.environ.get('NESTY_DYNAMIC_NESTED', '0') != '1'
                and
                os.environ.get('NESTY_POSTERIOR_STABILITY_STOP', '0') == '1'
        ):
            chunk_iterations = int(
                os.environ.get('NESTY_POSTERIOR_STABILITY_CHUNK', '64')
            )
            if chunk_iterations < 1:
                raise ValueError('NESTY_POSTERIOR_STABILITY_CHUNK must be at least 1')
            posterior_kl_tol = float(
                os.environ.get('NESTY_POSTERIOR_KL_TOL', '0.01')
            )
            posterior_kl_nreal = int(
                os.environ.get('NESTY_POSTERIOR_KL_NREAL', '128')
            )
            posterior_kl_statistic = os.environ.get(
                'NESTY_POSTERIOR_KL_STATISTIC',
                'p90'
            )
            posterior_kl_method = os.environ.get(
                'NESTY_POSTERIOR_KL_METHOD',
                'resample'
            )
            posterior_kl_approx = (
                os.environ.get('NESTY_POSTERIOR_KL_APPROX', '0')
                ==
                '1'
            )
            posterior_kl_seed = (
                int(os.environ['NESTY_POSTERIOR_KL_SEED'])
                if 'NESTY_POSTERIOR_KL_SEED' in os.environ
                else
                None
            )
            max_chunks = int(
                os.environ.get('NESTY_POSTERIOR_STABILITY_MAX_CHUNKS', '200')
            )
            stopping_config.update(
                dynamic_nested=False,
                posterior_stability_stop='dynesty_kld_error',
                posterior_stability_chunk_iterations=chunk_iterations,
                posterior_kl_threshold=posterior_kl_tol,
                posterior_kl_nreal=posterior_kl_nreal,
                posterior_kl_statistic=posterior_kl_statistic,
                posterior_kl_method=posterior_kl_method,
                posterior_kl_approx=posterior_kl_approx,
                posterior_kl_seed=posterior_kl_seed,
                posterior_stability_max_chunks=max_chunks,
                dlogz='None during chunked stability checks',
                final_add_live_dlogz='None'
            )
            _logger.info(
                'Using posterior KL stop: chunk=%d, method=%s, nreal=%d, '
                'statistic=%s, threshold=%g, max_chunks=%d',
                chunk_iterations,
                posterior_kl_method,
                posterior_kl_nreal,
                posterior_kl_statistic,
                posterior_kl_tol,
                max_chunks
            )
            with open(checkpoint_file + '.posterior_kl.tsv',
                      'a+',
                      encoding='utf-8') as stability_file:
                if stability_file.tell() == 0:
                    _write_kl_diagnostic_header(stability_file)
                # Existing rows come from earlier walltime windows.
                # Keep check IDs and random seeds continuous after resume.
                stability_file.seek(0)
                previous_checks = max(0, sum(1 for _ in stability_file) - 1)
                stability_file.seek(0, os.SEEK_END)
                for chunk_index in range(max_chunks):
                    check_number = previous_checks + chunk_index + 1
                    # Dynesty's maxiter is inclusive: maxiter=0 advances one
                    # accepted sample, maxiter=63 advances 64.
                    sampler.run_nested(maxiter=chunk_iterations - 1,
                                       dlogz=None,
                                       add_live=False,
                                       checkpoint_file=checkpoint_file,
                                       checkpoint_every=float(os.environ.get("NESTY_CHECKPOINT_EVERY", "0")),
                                       resume=(resume
                                               and
                                               chunk_index == 0))
                    resume = False
                    seed = (
                        None
                        if posterior_kl_seed is None
                        else
                        posterior_kl_seed + check_number
                    )
                    kl_stats, _kl_values = _estimate_posterior_kl_uncertainty(
                        sampler.results,
                        method=posterior_kl_method,
                        nreal=posterior_kl_nreal,
                        statistic=posterior_kl_statistic,
                        seed=seed,
                        approx=posterior_kl_approx
                    )
                    kl_converged = (
                        kl_stats['kld_selected']
                        <=
                        posterior_kl_tol
                    )
                    stability_file.write(
                        _format_kl_diagnostic_row(
                            check_number,
                            sampler.it,
                            sampler.results,
                            '',
                            kl_stats,
                            posterior_kl_tol,
                            kl_converged
                        )
                    )
                    stability_file.flush()
                    _logger.info(
                        'Posterior KL chunk %d: iteration=%d, method=%s, '
                        'nreal=%d, %s=%g, threshold=%g, converged=%s',
                        check_number,
                        sampler.it,
                        posterior_kl_method,
                        posterior_kl_nreal,
                        posterior_kl_statistic,
                        kl_stats['kld_selected'],
                        posterior_kl_tol,
                        kl_converged
                    )
                    if kl_converged:
                        _logger.info(
                            'Posterior KL uncertainty criterion satisfied; '
                            'adding live points and finishing.'
                        )
                        break
                else:
                    raise RuntimeError(
                        'Posterior KL uncertainty criterion was not satisfied '
                        f'after {max_chunks:d} chunks.'
                    )
            sampler.run_nested(maxiter=-1,
                               dlogz=None,
                               add_live=True,
                               checkpoint_file=checkpoint_file,
                               checkpoint_every=float(os.environ.get("NESTY_CHECKPOINT_EVERY", "0")),
                               resume=False)
        elif os.environ.get('NESTY_DYNAMIC_NESTED', '0') != '1':
            stopping_config.update(
                dynamic_nested=False,
                posterior_stability_stop=False,
                dlogz='dynesty default',
                dlogz_init='not applicable'
            )
            sampler.run_nested(checkpoint_file=checkpoint_file,
                               checkpoint_every=float(os.environ.get("NESTY_CHECKPOINT_EVERY", "0")),
                               resume=resume)
        _write_dynesty_outputs(
            sampler.results,
            prior_transform,
            checkpoint_file,
            stopping_config
        )
        return sampler.results
