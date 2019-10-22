#!/usr/bin/env python3

"""A collection of useful plotting functions."""

import pickle
import collections

from matplotlib import pyplot
import numpy
from astropy import units
from configargparse import ArgumentParser, DefaultsFormatter
import asteval

from stellar_evolution.change_variables import QuantityEvaluator
from stellar_evolution.manager import StellarEvolutionManager
from planetary_system_io import read_nasa_planets

from io_utilities import\
    load_progress_pickle,\
    get_nasa_system,\
    read_geller_et_al_2009_binaries
from calculate_e_Q_grid import prepare_nasa_system, fix_semimajor
from process_e_Q_grid import\
    format_eccentricity_vs_lgQ,\
    invert_eccentricity_vs_lgQ,\
    EccentricityEnvelope

def parse_command_line():
    """Parse the command line defining the plots to create."""

    parser = ArgumentParser(
        description=__doc__,
        default_config_files=['plots.cfg'],
        formatter_class=DefaultsFormatter,
        ignore_unknown_config_file_keys=True
    )
    parser.add_argument(
        '--plot',
        action='append',
        choices=['e_vs_P', 'star_solving', 'lgQ_vs_e', 'lgQ_vs'],
        help='Add another type of plot to the list of plots to generate.'
    )
    parser.add_argument(
        '--plot-fname',
        default=None,
        help='A pattern for the filename to save plots under. If empty, plots '
        'are just displayed to the user, but not saved.'
    )
    parser.add_argument(
        '--lgQ-x-axis',
        action='append',
        nargs=2,
        default=[
            (
                '(secondary_radius/semimajor)**5*primary_mass/secondary_mass',
                ''
            )
        ],
        help="Add plots of lg(Q*') vs different quantities. Two entries must be"
        " specified: expression and units. The expression can be any "
        "expression involving system properties that can be converted to the "
        "specified units. Can be passed multiple times resulting in multiple "
        "plots. These arguments are ignored if 'lgQ_vs' plot type is not "
        "included (see --plot)."
    )
    parser.add_argument(
        '--progress-pickle', '--progress',
        default='progress.pickle',
        help='The filename where tha calculated eccentricities were saved.'
    )
    result = parser.parse_args()
    return result

def get_eccentricity_envelope(cmdline_args):
    """Return an EccentricityEnvelope instance set-up per the command line."""

    return EccentricityEnvelope(
        min_period=(3.0 if cmdline_args.nasa_data is None else 0.8),
        max_period=(13.0 if cmdline_args.nasa_data is None else 5.0),
        max_eccentricity=(0.5 if cmdline_args.nasa_data is None else 0.6)
    )

#Simplifies command line arguments.
#pylint: disable=invalid-name
#kwargs only accepted to homogenize call signatures across plotters
#pylint: disable=unused-argument
def plot_e_vs_P(cmdline_args, plot_fname=None, **kwargs):
    """Create a plot of eccentricity vs period."""

    def plot_selection(systems, selected, **kwargs):
        """Plot one selection of points."""

        errors = numpy.dstack(
            (
                numpy.abs(systems.pl_orbeccenerr2),
                numpy.abs(systems.pl_orbeccenerr1)
            )
        )[0].T
        limit_only = systems.pl_orbeccenlim > 0.5
        errors[0][limit_only] = systems.pl_orbeccen[limit_only]
        errors[1][limit_only] = 0.0
        pyplot.errorbar(
            systems.pl_orbper[selected].to_value('day'),
            systems.pl_orbeccen[selected],
            yerr=errors[:, selected],
            **kwargs
        )

    def read_systems():
        """Return the systems to plot as objct with attributes."""

        if cmdline_args.nasa_data is not None:
            return read_nasa_planets(cmdline_args.nasa_data,
                                     eliminate=(),
                                     add_units=True,
                                     need_ages=False)
        if cmdline_args.use_binary_stars:
            systems = read_geller_et_al_2009_binaries()

            field_names = ['pl_orbper',
                           'pl_orbeccen',
                           'pl_orbeccenlim',
                           'pl_orbeccenerr1',
                           'pl_orbeccenerr2',
                           'primary_mass',
                           'secondary_mass']
            field_units = [units.Unit('day'),
                           1,
                           1,
                           1,
                           1,
                           units.M_sun,
                           units.M_sun]
            PlotSystem = collections.namedtuple('PlotSystem',
                                                field_names)
            num_systems = len(systems)

            result = PlotSystem(*(numpy.empty(num_systems, dtype=float) * unit
                                  for unit in field_units))
            for sys_index, system in enumerate(systems):
                result.pl_orbper[sys_index] = system.orbital_period
                result.pl_orbeccen[sys_index] = system.eccentricity
                result.pl_orbeccenlim[sys_index] = False
                result.pl_orbeccenerr1[sys_index] = (
                    system.eccentricity.plus_error
                )
                result.pl_orbeccenerr2[sys_index] = (
                    system.eccentricity.minus_error
                )
                result.primary_mass[sys_index] = system.primary_mass
                result.secondary_mass[sys_index] = system.secondary_mass

            return result

        raise IOError('At least one input source must be specified for e(P) '
                      'plotting')

    def plot_exoplanets(systems):
        """Create e(P) plot for exoplanet systems, i.e. marking density."""

        threshold_density = 2.0 * units.Unit('g/cm3')
        dense = systems.pl_dens > threshold_density
        fluffy = systems.pl_dens < threshold_density
        unknown = numpy.logical_and(numpy.logical_not(dense),
                                    numpy.logical_not(fluffy))

        plot_selection(systems,
                       fluffy,
                       fmt='or',
                       markersize=10,
                       label='fluffy')
        plot_selection(systems,
                       dense,
                       fmt='sb',
                       markersize=15,
                       label='dense')
        plot_selection(systems,
                       unknown,
                       fmt='vg',
                       markersize=15,
                       label='unknown')

        pyplot.plot([0.8, 5.0], [0, 0.6], '-k')
        pyplot.xlim((0.7, 20))

    def plot_binaries(systems):
        """Create e(p) plot for binary stars, i.e. marking primary mass."""

        #False positive
        #pylint: disable=no-member
        hot = systems.primary_mass > 1.4 * units.M_sun
        cool = systems.primary_mass < 1.2 * units.M_sun
        #pylint: enable=no-member
        unknown = numpy.logical_and(numpy.logical_not(cool),
                                    numpy.logical_not(hot))
        plot_selection(systems,
                       cool,
                       fmt='or',
                       markersize=10,
                       label='cool')
        plot_selection(systems,
                       hot,
                       fmt='sb',
                       markersize=15,
                       label='hot')
        plot_selection(systems,
                       unknown,
                       fmt='vg',
                       markersize=15,
                       label='unknown')
        pyplot.plot([3.0, 13.0], [0, 0.5], '-k')
        pyplot.xlim((2.0, 100))


    systems = read_systems()

    pyplot.xscale('log')
    if cmdline_args.nasa_data is not None:
        plot_exoplanets(systems)
    else:
        plot_binaries(systems)
    pyplot.ylim((0, 0.6))
    pyplot.legend()
    if plot_fname is None:
        pyplot.show()
    else:
        pyplot.savefig(plot_fname)
        pyplot.cla()
#pylint: enable=invalid-name
#pylint: enable=unused-argument

def plot_star_solving(interpolator, system, plot_fname=None):
    """Make a plot showing the attempt to solve for stellar mass/age."""

    plot_ages = 10.0**numpy.linspace(-3, 1, 1000)

    for feh, style in [
            #$(system.feh - system.feh.minus_error, ':'),
            (system.feh, '-'),
            #(system.feh + system.feh.plus_error, '--')
    ]:
        evaluator = QuantityEvaluator(interpolator, feh)
        for mass, color in [
                (system.db_star_mass - system.db_star_mass.minus_error, 'r'),
                (system.db_star_mass, 'g'),
                (system.db_star_mass + system.db_star_mass.minus_error, 'b')
        ]:
            rho = numpy.array([evaluator.rho(mass, t) for t in plot_ages])
            teff = numpy.array([evaluator.teff(mass, t) for t in plot_ages])
            pyplot.plot(rho, teff, style + color)
            for age in [0.01, system.db_star_age.to_value('Gyr'), 10.0]:
                pyplot.plot(
                    [evaluator.rho(mass, age)],
                    [evaluator.teff(mass, age)],
                    'o' + color, markersize=10
                )
    pyplot.plot([2.839], [5314], '+')
    if plot_fname is None:
        pyplot.show()
    else:
        pyplot.savefig(plot_fname)
        pyplot.cla()

#Simplifies command line arguments.
#pylint: disable=invalid-name
def plot_single_lgQ_vs_e(system,
                         progress,
                         eccentricity_envelope,
                         plot_fname=None):
    """Show a plot of the final eccentricity vs lgQ for a system."""


    print('System: ' + repr(system))
    plot_data = format_eccentricity_vs_lgQ(progress[system.hostname])
    pyplot.plot(plot_data[:, 0], plot_data[:, 1], '-k')
    pyplot.plot(plot_data[:, 0], plot_data[:, 1], 'ok')

    nominal_eccentricity = system.eccentricity
    low_eccentricity = (system.eccentricity - system.eccentricity.minus_error)
    envelope_eccentricity = eccentricity_envelope(
        system.orbital_period.to_value('day')
    )
    pyplot.axhline(y=nominal_eccentricity, color='r')
    pyplot.axhline(y=low_eccentricity, color='r')
    pyplot.axhline(envelope_eccentricity, color='b')

    nominal_e_lgQ = invert_eccentricity_vs_lgQ(progress[system.hostname],
                                               nominal_eccentricity)
    low_e_lgQ = invert_eccentricity_vs_lgQ(progress[system.hostname],
                                           low_eccentricity)
    envelope_e_lgQ = invert_eccentricity_vs_lgQ(progress[system.hostname],
                                                envelope_eccentricity)
    pyplot.plot([nominal_e_lgQ], [nominal_eccentricity], 'or')
    pyplot.plot([low_e_lgQ], [low_eccentricity], 'or')
    pyplot.plot([envelope_e_lgQ], [envelope_eccentricity], 'ob')

    rplanet = getattr(system, 'db_planet_radius', numpy.nan)
    if numpy.isfinite(rplanet):
        rplanet = rplanet.to_value('R_jup')

    mplanet = getattr(system, 'db_planet_mass', numpy.nan)
    if numpy.isfinite(mplanet):
        mplanet = mplanet.to_value('M_jup')

    pyplot.title(
        str(system.hostname)
        +
        r'($R_p=%g R_j$, $M_p=%g M_j$, $P_{orb}=%g\,d$)'
        %
        (
            rplanet,
            mplanet,
            system.orbital_period.to_value('day')
        )
    )
    pyplot.xlabel(r"$\log_{10}(Q'_\star)$")
    pyplot.ylabel('Eccentricity')
    if plot_fname is None:
        pyplot.show()
    else:
        pyplot.savefig(plot_fname)
        pyplot.cla()
#pylint: enable=invalid-name

def get_system_list(cmdline_args):
    """Return a list of systems for plotting."""

    systems = []
    if cmdline_args.nasa_data:
        nasa_systems = read_nasa_planets(cmdline_args.nasa_data,
                                         eliminate=(),
                                         add_units=True,
                                         need_ages=False)
        systems.extend(
            [
                get_nasa_system(host, nasa_systems)
                for host in sorted(progress.keys())
            ]
        )
    if getattr(cmdline_args, 'use_binary_stars', None):
        systems.extend(read_geller_et_al_2009_binaries())

    return systems

#Simplifies command line arguments.
#pylint: disable=invalid-name
def plot_lgQ_vs_e(progress, cmdline_args, plot_fname=None, **_):
    """Show sequentially plots of lgQ vs e for all systems."""

    systems = get_system_list(cmdline_args)
    eccentricity_envelope = get_eccentricity_envelope(cmdline_args)

    #Only need the keys
    #pylint: disable=consider-iterating-dictionary
    for plot_sys in systems:
    #pylint: enable=consider-iterating-dictionary
        if plot_sys.hostname in progress:
            plot_single_lgQ_vs_e(
                plot_sys,
                progress,
                eccentricity_envelope,
                plot_fname
            )
#pylint: enable=invalid-name

#Simplifies command line arguments.
#pylint: disable=invalid-name
def plot_lgQ_vs(lgQ_x_axes,
                progress,
                cmdline_args,
                interpolator,
                plot_fname=None):
    """Make a plot of the log10(Q*') constraints vs orbital period."""

    def add_stellar_properties(system):
        """Augment the given system with info about its star(s) & orbit."""

        system.primary_radius = interpolator(
            'radius',
            system.primary_mass.to_value('M_sun'),
            system.feh.to_value('')
        )(system.age.to_value('Gyr')) * units.R_sun

        system.secondary_radius = interpolator(
            'radius',
            system.secondary_mass.to_value('M_sun'),
            system.feh.to_value('')
        )(system.age.to_value('Gyr')) * units.R_sun

        fix_semimajor(system)

    def prepare_data():
        """Collect and calculate the data required for the plots."""

        systems = get_system_list(cmdline_args)
        eccentricity_envelope = get_eccentricity_envelope(cmdline_args)

        plot_x = numpy.empty((len(lgQ_x_axes), len(progress)), dtype=float)
        plot_lgQ = numpy.empty(plot_x.shape[1], dtype=[('min', float),
                                                       ('nominal', float),
                                                       ('max', float)])

        limit_flags = numpy.empty(
            plot_x.shape[1],
            dtype=[('two_sided', bool), ('upper', bool), ('lower', bool)]
        )

        x_evaluator = asteval.Interpreter()
        index = 0
        for system in systems:
            if system.hostname not in progress:
                continue
            add_stellar_properties(system)
            lgQ_vs_period = progress[system.hostname]
            print('System: ' + repr(system))

            plot_lgQ['nominal'][index] = invert_eccentricity_vs_lgQ(
                lgQ_vs_period,
                system.eccentricity
            )
            plot_lgQ['min'][index] = invert_eccentricity_vs_lgQ(
                lgQ_vs_period,
                system.eccentricity - system.eccentricity.minus_error
            )
            plot_lgQ['max'][index] = invert_eccentricity_vs_lgQ(
                lgQ_vs_period,
                eccentricity_envelope(system.orbital_period.to_value('day'))
            )

            x_evaluator.symtable = vars(system)
            for x_expression_index, (x_expr, x_units) in enumerate(lgQ_x_axes):
                plot_x[x_expression_index, index] = units.Quantity(
                    x_evaluator(x_expr)
                ).to_value(
                    x_units
                )

            limit_flags['upper'][index] = (
                system.eccentricity_limit
                or
                plot_lgQ['min'][index] == 0
                or
                numpy.isnan(plot_lgQ['min'][index])
            )
            limit_flags['lower'][index] = numpy.isnan(plot_lgQ['max'][index])
            print('lgQ constraints: ' + repr(plot_lgQ[index]))
            print('Flags: ' + repr(limit_flags[index]))
            index += 1

        limit_flags['two_sided'] = numpy.copy(
            numpy.logical_not(
                numpy.logical_or(
                    limit_flags['lower'],
                    limit_flags['upper']
                )
            )
        )
        useful = numpy.copy(
            numpy.logical_not(
                numpy.logical_and(
                    limit_flags['lower'],
                    limit_flags['upper']
                )
            )
        )
        limit_flags['upper'] = numpy.logical_and(
            limit_flags['upper'],
            useful
        )[:]
        limit_flags['lower'] = numpy.logical_and(
            limit_flags['lower'],
            useful
        )[:]

        print(100 * '*')
        print('Lower limit flags:')
        print(repr(useful))
        print(100 * '*')


        return plot_x, plot_lgQ, limit_flags

    plot_x, plot_lgQ, limit_flags = prepare_data()

    print(100 * '*')
    print('Constraints:')
    print(repr(plot_lgQ))
    print(100 * '*')

    lgQ_range = (3, 9)
    for x_index in range(plot_x.shape[0]):
        pyplot.xscale('log')

        pyplot.errorbar(
            x=plot_x[x_index, limit_flags['upper']],
            y=plot_lgQ['max'][limit_flags['upper']],
            yerr=[
                plot_lgQ['max'][limit_flags['upper']] - lgQ_range[0],
                numpy.zeros(limit_flags['upper'].sum())
            ],
            fmt='vr',
            markersize=5,
            linewidth=0.3,
            zorder=0
        )

        lower_errors = plot_lgQ['nominal'] - plot_lgQ['min']
        print('lower_errors: ' + repr(lower_errors[limit_flags['lower']]))
        print('lgQ nominal: ' + repr(plot_lgQ['nominal'][limit_flags['lower']]))
        print('lgQ min: ' + repr(plot_lgQ['min'][limit_flags['lower']]))

        pyplot.errorbar(
            x=plot_x[x_index, limit_flags['lower']],
            y=plot_lgQ['nominal'][limit_flags['lower']],
            yerr=[
                lower_errors[limit_flags['lower']],
                lgQ_range[1] - plot_lgQ['nominal'][limit_flags['lower']]
            ],
            fmt='^b',
            markersize=5,
            linewidth=0.3,
            zorder=1
        )

        pyplot.errorbar(
            x=plot_x[x_index, limit_flags['two_sided']],
            y=plot_lgQ['nominal'][limit_flags['two_sided']],
            yerr=[
                (
                    plot_lgQ['nominal'][limit_flags['two_sided']]
                    -
                    plot_lgQ['min'][limit_flags['two_sided']]
                ),
                (
                    plot_lgQ['max'][limit_flags['two_sided']]
                    -
                    plot_lgQ['nominal'][limit_flags['two_sided']]
                )
            ],
            fmt='og',
            markersize=5,
            linewidth=2,
            zorder=2
        )
        pyplot.xlabel(lgQ_x_axes[x_index][0]
                      +
                      ' [' + lgQ_x_axes[x_index][1] + ']')



        if plot_fname is None:
            pyplot.show()
        else:
            pyplot.savefig(
                plot_fname
                %
                dict(x_label=lgQ_x_axes[x_index][0].replace('/', ':'))
            )
            pyplot.cla()
#pylint: enable=invalid-name

def main():
    """Avoid adding things to global namespace."""

    cmdline_args = parse_command_line()
    progress_pickle = cmdline_args.progress_pickle
    with open(progress_pickle, 'rb') as progress_file:
        pickled_cmdline_args = pickle.load(progress_file)
        progress = load_progress_pickle(progress_file)

    interpolator = StellarEvolutionManager(
        pickled_cmdline_args.stellar_evolution_interpolator_dir
    ).get_interpolator_by_name(
        'default'
    )

    for plot_type in cmdline_args.plot:
        if cmdline_args.plot_fname is None:
            plot_fname = None
        else:
            plot_fname = cmdline_args.plot_fname % dict(
                plot_type=plot_type,
                x_label=('%(x_label)s' if plot_type == 'lgQ_vs' else '')
            )
        globals()['plot_' + plot_type](
            progress=progress,
            lgQ_x_axes=cmdline_args.lgQ_x_axis,
            cmdline_args=pickled_cmdline_args,
            interpolator=interpolator,
            plot_fname=plot_fname
        )

if __name__ == '__main__':
    main()
