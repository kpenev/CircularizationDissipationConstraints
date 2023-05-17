#!/usr/bin/env python3

"""Explore ef(ei) for various systems and dissipation parameters."""

import logging
from multiprocessing import Pool, cpu_count
from functools import partial
from types import SimpleNamespace
from os import path

from matplotlib import pyplot
from matplotlib.backends.backend_pdf import PdfPages
from configargparse import\
    ArgumentParser,\
    DefaultsFormatter,\
    Action as ArgparseAction
import numpy
from scipy.optimize import root_scalar
from astropy import units

from superphot_pipeline import Evaluator

from general_purpose_python_modules import find_evolution, MultiPickle
from orbital_evolution.command_line_util import \
    add_binary_config,\
    add_evolution_config,\
    run_evolution,\
    get_phase_lag_config
from orbital_evolution.evolve_interface import library as\
    orbital_evolution_library
from stellar_evolution.manager import StellarEvolutionManager
from bayesian.sampling import setup_process
from bayesian.basic_util import default_logging_format
from bayesian import windemuth_et_al_util as w19_util

_logger = logging.getLogger(__name__)

_pickle_ignore_config=['config_file',
                       'generate_config_file',
                       'create_c_code',
                       'num_parallel_processes',
                       'fname_datetime_format',
                       'std_out_err_fname',
                       'logging_fname',
                       'logging_verbosity',
                       'logging_datetime_format',
                       'logging_message_format',
                       'pickle_fname',
                       'compare_likelihood_estimates',
                       'per_system_plot_fname',
                       'per_efinal_plot_fname']


#Interface specified by argparse module.
#pylint: disable=too-few-public-methods
class ParseGrid(ArgparseAction):
    """Action for parsing linspace arguments."""

    def __call__(self, parser, namespace, values, option_string=None):
        """Parse a grid option to a numpy.arary()."""

        setattr(namespace,
                self.dest,
                numpy.linspace(float(values[0]),
                               float(values[1]),
                               int(values[2])))
#pylint: enable=too-few-public-methods


def parse_command_line():
    """Return the command line configuration."""

    output_dir = path.join(path.dirname(path.dirname(path.abspath(__file__))),
                           'sampling_output',
                           'eccentricity_map')

    parser = ArgumentParser(
        description=__doc__,
        default_config_files=['ef_vs_ei.cfg'],
        args_for_writing_out_config_file=['--generate-config-file'],
        args_for_setting_config_path=['--config-file', '-c'],
        formatter_class=DefaultsFormatter,
        ignore_unknown_config_file_keys=False
    )
    add_evolution_config(parser)
    add_binary_config(parser, skip=('Porb0', 'e0'))
    parser.add_argument(
        '--initial-eccentricity-grid', '--e0-grid',
        nargs=3,
        default=numpy.linspace(0.0, 0.8, 9),
        action=ParseGrid,
        metavar=('MIN_E', 'MAX_E', 'NUM_E'),
        help='The grid of initial eccentricity values to evolve.'
    )
    parser.add_argument(
        '--max-efinal-grid',
        nargs=3,
        default=numpy.linspace(0.1, 0.6, 6),
        action=ParseGrid,
        metavar=('MIN_E', 'MAX_E', 'NUM_E'),
        help='The initial orbital period is tuned to reproduce each of the '
        'given final eccentricities for the largest starting initial '
        'eccentricity. All other entries in ``--initial-eccentricity-grid`` '
        'will then be tuned to reproduce the same **final** orbital period.'
    )
    parser.add_argument(
        '--max-final-e-tolerance',
        type=float,
        default=1e-3,
        help='The tolerance up to which to reproduce the target max final '
        'eccentricity.'
    )
    parser.add_argument(
        '--final-porb-tolerance',
        type=float,
        default=1e-3,
        help='The tolerance up to which to reproduce the same final orbital '
        'period for all evolutions.'
    )
    parser.add_argument(
        '--period-search-factor',
        type=float,
        default=2.0,
        help='The factor by which to change the initial period guess while '
        'searching for a range surrounding the known present day orbital '
        'period.'
    )
    parser.add_argument(
        '--scaled-period-guess',
        type=float,
        default=1.0,
        help='The search for initial period to bracket the observed final '
        'period will start from this value multiplied by the final orbital '
        'period.'
    )
    parser.add_argument(
        '--num-parallel-processes', '--num-parallel',
        type=int,
        default=cpu_count()//2,
        help='The maximum number of parallel processes to use.'
    )
    parser.add_argument(
        '--fname-datetime-format',
        default='%Y%m%d%H%M%S',
        help='How to format date and time as part of filenames (e.g. when '
        'creating output files for multiprocessing.'
    )
    parser.add_argument(
        '--std-out-err-fname',
        default=path.join(output_dir, '%(task)s_%(now)s_%(pid)d.outerr'),
        help='Filename to redirect worker process stdout and stderr to during '
        'multiprocessing. Should include at least `%%(pid)d` (worker process '
        'id) substitution to avoid mangling, but may also include `%%(system)s`'
        ' (system name) and `%%(now)s` (approximate date and time the process '
        'started).'
    )
    parser.add_argument(
        '--logging-fname',
        default=path.join(output_dir, '%(task)s_%(now)s_%(pid)d.log'),
        help='Filename for log mesasges from sampling. Should include at least '
        '`%%(pid)d` (worker process id) substitution to avoid mangling during '
        'multiprocessing, but may also include `%%(system)s`'
        ' (system name) and `%%(now)s` (approximate date and time the process '
        'started).'
    )
    parser.add_argument(
        '--logging-verbosity', '--verbosity',
        choices=['debug', 'info', 'warning', 'error', 'critical'],
        default='info',
        help='The lowest importance level of logging messages to issue.'
    )
    parser.add_argument(
        '--logging-datetime-format',
        default=None,
        help='How to format date and time as part of filenames (e.g. when '
        'creating output files for multiprocessing.'
    )
    parser.add_argument(
        '--compare-likelihood-estimates',
        choices=['w19'],
        default=None,
        help='If passed selects a collection of systems to compare the '
        'likelihood functions estimated using lineal ef(ei) approximation vs '
        'piecewise linear within the ``--initial-eccentricity-grid`` for each '
        'entry in ``--max-efinal-grid``.'
    )
    parser.add_argument(
        '--per-system-plot-fname',
        default=path.join(output_dir, 'likelihoods_per_system.pdf'),
        help='The filaname under which to save the plot of likelihood '
        'estimates organized by system (vs max final eccentricity).'
    )
    parser.add_argument(
        '--per-efinal-plot-fname',
        default=path.join(output_dir, 'likelihoods_per_system.pdf'),
        help='The filaname under which to save the plot of likelihood '
        'estimates organized by final eccentricity (vs nominal observed '
        'eccentricity).'
    )
    parser.add_argument(
        '--logging-message-format', '--logging-format', '--log-fmt',
        default=default_logging_format,
        help='How to format logging messages. See python logging module '
        'documentation for details.'
    )
    parser.add_argument(
        '--pickle-fname',
        default='final_initial_eccentricity_dependence.pkl',
        help='Filename to save temporary progress in.'
    )

    return parser.parse_args()


class FoundSolution(Exception):
    """Allow immediate termination of a solver when good enough sol is found."""

    def __init__(self, solution):
        """Store the solution found."""

        super().__init__()
        self.solution = solution


def get_final_eccentricity_difference(initial_porb,
                                      max_final_eccentricity,
                                      config,
                                      interpolator):
    """Calc. the difference between calculated and target final e given Porb."""

    config.initial_orbital_period = initial_porb
    config.initial_eccentricity = config.initial_eccentricity_grid.max()
    print('Configuration: ' + repr(config))
    final_state = run_evolution(config,
                                interpolator=interpolator,
                                final_state_only=True)
    _logger.info('Final state: %s', repr(final_state))
    if numpy.isfinite(final_state.eccentricity):
        if (
            numpy.abs(final_state.eccentricity - max_final_eccentricity)
            <=
            config.max_final_e_tolerance
        ):
            raise FoundSolution(initial_porb)
        return final_state.eccentricity - max_final_eccentricity

    return -max_final_eccentricity


def get_final_period(max_final_eccentricity, config, interpolator):
    """Return the final Porb for which max efinal has the given value."""

    porb_min, porb_max = -numpy.inf, numpy.inf
    porb_guess = 15.0
    while not (numpy.isfinite(porb_min) and numpy.isfinite(porb_max)):
        _logger.debug('Search range: %s < Porb < %s, trying: %s',
                      repr(porb_min),
                      repr(porb_max),
                      repr(porb_guess))

        e_diff = get_final_eccentricity_difference(porb_guess,
                                                   max_final_eccentricity,
                                                   config,
                                                   interpolator)
        if e_diff < 0:
            porb_min = max(porb_guess, porb_min)
            porb_guess = 2 * porb_min
        else:
            porb_max = min(porb_guess, porb_max)
            porb_guess = porb_max / 2

    _logger.debug('Searching for initial Porb in range [%s, %s]',
                  repr(porb_min),
                  repr(porb_max))
    try:
        root_scalar(
            get_final_eccentricity_difference,
            args=(max_final_eccentricity, config, interpolator),
            bracket=(porb_min, porb_max),
            xtol=1e-16,
            rtol=1e-12
        )
        raise RuntimeError(
            'Failed to find initial period that reproduces max final e=%s'
            %
            repr(max_final_eccentricity)
        )
    except FoundSolution as porb_found:
        _logger.debug('Initial Porb = %s reproduces max final e = %s',
                      repr(porb_found.solution),
                      repr(max_final_eccentricity))
        config.initial_orbital_period = porb_found.solution
        config.initial_eccentricity = config.initial_eccentricity_grid.max()
        final_state = run_evolution(config,
                                    interpolator=interpolator,
                                    final_state_only=True)
        _logger.debug('Porb final state: %s', repr(final_state))
        assert final_state.age == config.final_age
        assert (
            numpy.abs(final_state.eccentricity - max_final_eccentricity)
            <=
            config.max_final_e_tolerance
        )
        return final_state.orbital_period


def get_final_eccentricity(final_porb_initial_e,
                           find_evolution_kwargs):
    """Return the final eccentricity for given final Porb and initial e."""

    find_evolution_kwargs['system'].orbital_period = (
        float(final_porb_initial_e[0]) * units.day
    )
    evolution = find_evolution(
        initial_eccentricity=float(final_porb_initial_e[1]),
        **find_evolution_kwargs,
    )
    _logger.debug('Evolution for final Porb = %s, initial e = %s: %s',
                  repr(final_porb_initial_e[0]),
                  repr(final_porb_initial_e[1]),
                  repr(evolution))

    if (
        evolution.age[-1] * units.Gyr / find_evolution_kwargs['max_age']
        >=
        (1.0 - 1e-4)
    ):
        _logger.debug(
            'Max evolution age (%s) for final Porb = %s, initial e = %s '
            '(matches expected)',
            repr(evolution.age[-1]),
            repr(final_porb_initial_e[0]),
            repr(final_porb_initial_e[1])
        )

        return evolution.eccentricity[-1]

    _logger.warning(
        'Max evolution age (%s) for final Porb = %s, initial e = %s did not'
        ' reach %s!',
        repr(evolution.age[-1]),
        repr(final_porb_initial_e[0]),
        repr(final_porb_initial_e[1]),
        repr(find_evolution_kwargs['max_age'])
    )
    return numpy.nan


def ensure_ready_to_evolve(config):
    """Make sure the library is ready for calculating evolutions."""

    if not orbital_evolution_library.ready_to_evolve:
        orbital_evolution_library.prepare_eccentricity_expansion(
            config.eccentricity_expansion_fname.encode('ascii'),
            1e-4,
            True,
            True
        )
        orbital_evolution_library.ready_to_evolve = True


def get_porb_final(config, process_config, interpolator):
    """Return the list of final orbital periods that reproduce max final e."""

    pickles = MultiPickle(config.pickle_fname,
                          _pickle_ignore_config + ['initial_eccentricity_grid'])
    config.what = 'porb_final'
    porb_final_list = pickles.check_for_pickled(config)
    if porb_final_list is None:
        ensure_ready_to_evolve(config)
        with Pool(
            config.num_parallel_processes,
            initializer=setup_process,
            initargs=(process_config, 'porb_final'),
            maxtasksperchild=1
        ) as workers:
            porb_final_list = workers.map(
                partial(get_final_period,
                        config=config,
                        interpolator=interpolator),
                config.max_efinal_grid
            )
        pickles.add_result(config, porb_final_list)
    else:
        porb_final_list = porb_final_list[0]
    return porb_final_list


def get_find_evolution_kwargs(config, interpolator):
    """Return the keyword arguments to use for `find_evolution()`."""

    find_evolution_kwargs = dict(
        system=SimpleNamespace(
            primary_mass=config.primary_mass * units.M_sun,
            secondary_mass=config.secondary_mass * units.M_sun,
            feh=config.metallicity,
            age=config.final_age * units.Gyr
        ),
        interpolator=interpolator,
        dissipation=dict(
            primary=get_phase_lag_config(config, True),
            secondary=get_phase_lag_config(config, False)
        ),
        max_age=config.final_age * units.Gyr,
        initial_obliquity=config.initial_obliquity,
        disk_period=(2.0 * numpy.pi / config.disk_lock_frequency) * units.day,
        disk_dissipation_age=config.disk_dissipation_age * units.Gyr,
        primary_wind_saturation=config.primary_wind_saturation_frequency,
        primary_core_envelope_coupling_timescale=(
            config.primary_diff_rot_coupling_timescale * units.Gyr
        ),
        secondary_wind_strength=(
            config.primary_wind_strength
            if config.secondary_wind_strength is None else
            config.secondary_wind_strength
        ),
        secondary_wind_saturation=(
            config.primary_wind_saturation_frequency
            if config.secondary_wind_saturation_frequency is None else
            config.secondary_wind_saturation_frequency
        ),
        secondary_core_envelope_coupling_timescale=(
            config.primary_diff_rot_coupling_timescale
            if config.secondary_diff_rot_coupling_timescale is None else
            config.secondary_diff_rot_coupling_timescale
        ) * units.Gyr,
        orbital_period_tolerance=config.final_porb_tolerance,
        required_ages=None,
        eccentricity_expansion_fname=(
            config.eccentricity_expansion_fname.encode('ascii')
        ),
        timeout=config.max_evolution_runtime,
        learning_features=('final porb', 'e0', 'm1', 'm2', 'lgQ1', 'lgQ2'),
        ic_model_id='final_initial_e_dependence'
    )
    for param in ['initial_obliquity',
                  'primary_wind_strength',
                  'period_search_factor',
                  'scaled_period_guess',
                  'max_time_step',
                  'precision']:
        find_evolution_kwargs[param] = getattr(config, param)
    return find_evolution_kwargs


def calculate_efinal(config, interpolator):
    """Find the final ecc. on a grid of initial ecc. and max final ecc."""

    process_config = SimpleNamespace(
        system='eccentricity_map',
        **{
            cfg: getattr(config, cfg)
            for cfg in ['fname_datetime_format',
                        'std_out_err_fname',
                        'logging_fname',
                        'logging_verbosity',
                        'logging_message_format',
                        'logging_datetime_format']
        }
    )
    setup_process(process_config, task='manage')

    porb_final_list = get_porb_final(config, process_config, interpolator)

    pickles = MultiPickle(config.pickle_fname, _pickle_ignore_config)
    config.what = 'e_final'
    e_final_list = pickles.check_for_pickled(config)
    if e_final_list is None:
        ensure_ready_to_evolve(config)
        _logger.debug('No pickled final eccentricities found! Calculating!')

        find_evolution_kwargs = get_find_evolution_kwargs(config, interpolator)
        tasks = [
            (porb_final, e_initial)
            for porb_final in porb_final_list
            for e_initial in config.initial_eccentricity_grid[:-1]
        ]
        with Pool(
            config.num_parallel_processes,
            initializer=setup_process,
            initargs=(process_config, 'e_final'),
            maxtasksperchild=1
        ) as workers:
            e_final_list = workers.map(
                partial(get_final_eccentricity,
                        find_evolution_kwargs=find_evolution_kwargs),
                tasks
            )

        pickles.add_result(config, e_final_list)
    else:
        assert len(e_final_list) == 1
        e_final_list = e_final_list[0]
    _logger.debug('Final eccentricities: %s', repr(e_final_list))
    return porb_final_list, e_final_list


def estimate_likelihood(e_initial, e_final, observed_e_distro):
    """Calculate the likelihood approximating ef(ei) as piecewise linear."""

    result = 0.0
    e_cdf = observed_e_distro.cdf(e_final)
    _logger.debug(
        'CDF(ef=%s, ei=%s) = %s', repr(e_final),
        repr(e_initial),
        repr(e_cdf)

    )
    for ei_0, ei_1, ef_0, ef_1, ecdf_0, ecdf_1 in zip(e_initial[:-1],
                                                      e_initial[1:],
                                                      e_final[:-1],
                                                      e_final[1:],
                                                      e_cdf[:-1],
                                                      e_cdf[1:]):
        result += (ecdf_1 - ecdf_0) * (ei_1 - ei_0) / (ef_1 - ef_0)
    return result


def plot_likelihoods_per_efinal(
    kic_list,
    likelihoods,
    max_e_final,
    plot_fname,
    x_expression='(maxlike_esinw**2 + maxlike_ecosw**2)**0.5'
):
    """Plot the likelihood estimates and their difference vs summary data."""

    pdf = PdfPages(plot_fname)
    x_values = Evaluator(
        w19_util.get_summary_data()
    )(
        x_expression
    )[kic_list].to_numpy()
    for e_final_index, e_final in enumerate(max_e_final):
        differences = None
        for label, likelihood_values in likelihoods.items():
            pyplot.plot(x_values,
                        likelihood_values[e_final_index],
                        'o',
                        label=label)
            if differences is None:
                differences = numpy.copy(likelihood_values)
            else:
                differences -= likelihood_values
        pyplot.figlegend()
        pyplot.suptitle('$e_{final,max} = %g$' % e_final)
        pdf.savefig()
        pyplot.close()
    pdf.close()


def plot_likelihoods_per_system(
    kic_list,
    likelihoods,
    max_e_final,
    plot_fname
):
    """Plot the likelihood estimates and their difference vs summary data."""

    pdf = PdfPages(plot_fname)
    for kic_index, kic in enumerate(kic_list):
        differences = None
        for label, likelihood_values in likelihoods.items():
            pyplot.plot(max_e_final,
                        [plot_y[kic_index] for plot_y in likelihood_values],
                        'o',
                        label=label)
            if differences is None:
                differences = numpy.copy(likelihood_values)
            else:
                differences -= likelihood_values
        pyplot.figlegend()
        pyplot.suptitle('KIC %d' % kic)
        pdf.savefig()
        pyplot.close()
    pdf.close()



def compare_likelihood_estimates(config, interpolator):
    """Report how good linear ef(ei) approximation is for cmdline scenarios."""

    kic_list = w19_util.get_available_kic(interpolator)
    e_final_list = calculate_efinal(config, interpolator)[1]

    pickles = MultiPickle(config.pickle_fname, _pickle_ignore_config)
    config.what = 'likelihoods'
    likelihoods = pickles.check_for_pickled(config)

    if likelihoods is None:
        likelihoods = dict(linear=[[] for _ in config.max_efinal_grid],
                           piecewise=[[] for _ in config.max_efinal_grid])
        for kic, observed_e_distro in zip(
                kic_list,
                map(w19_util.get_eccentricity_distro, kic_list)
        ):
            _logger.debug('Calculating likelihoods for KIC %d', kic)
            first_e_final = 0
            for e_final_i, max_e_final in enumerate(config.max_efinal_grid):
                last_e_final = (first_e_final
                                +
                                len(config.initial_eccentricity_grid)
                                -
                                1)
                e_final = e_final_list[first_e_final: last_e_final]
                first_e_final = last_e_final

                e_final.append(max_e_final)
                likelihoods['linear'][e_final_i].append(
                    estimate_likelihood(
                        [0, config.initial_eccentricity_grid[-1]],
                        [e_final[0], e_final[-1]],
                        observed_e_distro
                    )
                )
                likelihoods['piecewise'][e_final_i].append(
                    estimate_likelihood(
                        config.initial_eccentricity_grid,
                        e_final,
                        observed_e_distro
                    )
                )
        pickles.add_result(config,
                           likelihoods['linear'],
                           likelihoods['piecewise'])
    else:
        assert len(likelihoods) == 2
        likelihoods = dict(linear=likelihoods[0], piecewise=likelihoods[1])

    plot_likelihoods_per_efinal(kic_list,
                                likelihoods,
                                config.max_efinal_grid,
                                config.per_efinal_plot_fname)
    plot_likelihoods_per_system(kic_list,
                                likelihoods,
                                config.max_efinal_grid,
                                config.per_system_plot_fname)


def main(config):
    """Avoid polluting global namespace."""

    interpolator = StellarEvolutionManager(
        config.stellar_evolution[0]
    ).get_interpolator_by_name(
        config.stellar_evolution[1]
    )
    orbital_evolution_library.ready_to_evolve = False
    compare_likelihood_estimates(config, interpolator)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    main(parse_command_line())
