#!/usr/bin/env python3

"""A collection of useful plotting functions."""

import pickle
import collections
import os.path

from matplotlib import pyplot, rcParams
import numpy
from astropy import units
from configargparse import ArgumentParser, DefaultsFormatter
import asteval

from kelly_colors import kelly_colors

from stellar_evolution.change_variables import QuantityEvaluator
from stellar_evolution.manager import StellarEvolutionManager
from planetary_system_io import read_nasa_planets

from io_utilities import\
    load_progress_pickle,\
    get_nasa_system,\
    read_geller_et_al_2009_binaries,\
    read_milliman_et_al_2014_binaries
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
        choices=['e_vs_P', 'star_solving', 'lgQ_vs_e', 'lgQ_vs', 'lgQ_change_vs'],
        help='Add another type of plot to the list of plots to generate.'
    )
    parser.add_argument(
        '--plot-fname',
        default=None,
        help='A pattern for the filename to save plots under. It may include a '
        '%%(x_label)s substitution If empty, plots '
        'are just displayed to the user, but not saved.'
    )
    parser.add_argument(
        '--lgQ-x-axis',
        action='append',
        nargs=2,
        default=[],
        help="Add plots of lg(Q*') vs different quantities. Two entries must be"
        " specified: expression and units. The expression can be any "
        "expression involving system properties that can be converted to the "
        "specified units. Can be passed multiple times resulting in multiple "
        "plots. These arguments are ignored if 'lgQ_vs' plot type is not "
        "included (see --plot)."
    )
    parser.add_argument(
        '--progress-pickle', '--progress',
        default=[],
        action='append',
        help='The filename where the calculated eccentricities were saved, or '
        'one of these files for plot type lgQ_change_vs. Default: %(default)s.'
    )
    parser.add_argument(
        '--second-progress-pickle',
        default=None,
        help='Only useful for plot type lgQ_change_vs, and specifies the second'
        ' pickle to compare to.'
    )
    parser.add_argument(
        '--overwrite-interpolator',
        default=None,
        help='If passed, the interpolator directory specified in the unpickled '
        'commandline arguments is ovrewritted with the given value.'
    )
    parser.add_argument(
        '--pickle-systems',
        default='plot_systems.pickle',
        help='A file to store prepared systems for reuse in subsequent '
        'plotting. Default: %(default)s.'
    )
    parser.add_argument(
        '--font-size',
        type=int,
        default=16,
        help='The font size to use for the plots generated. Default: '
        '%(default)s.'
    )
    parser.add_argument(
        '--figure-size',
        nargs=2,
        type=float,
        default=(6.5, 4.0),
        help='the size of the figure in inches. Default: %(default)s'
    )
    parser.add_argument(
        '--axes-vshift',
        type=float,
        default=0.15,
        help='The vertical shift of the figure axes from the bottom of the '
        'figure. Default %(default)s.'
    )
    parser.add_argument(
        '--axes-hshift',
        type=float,
        default=0.1,
        help='The vertical shift of the figure axes from the bottom of the '
        'figure. Default %(default)s.'
    )
    parser.add_argument(
        '--axes-hspan',
        type=float,
        default=0.98,
        help='The horizontal maximum vertical position to which the axis should'
        ' extend. Default %(default)s.'
    )
    parser.add_argument(
        '--axes-vspan',
        type=float,
        default=0.98,
        help='The vertical maximum vertical position to which the axis should '
        'extend. Default %(default)s.'
    )
    parser.add_argument(
        '--pretend-min-eccentricity', '--pretend-emin',
        type=float,
        default=None,
        help='If passed plots are generated assuming all systems minimum '
        'eccentricity has the given value.'
    )
    result = parser.parse_args()
    return result

def get_eccentricity_envelope(cmdline_args):
    """Return an EccentricityEnvelope instance set-up per the command line."""

    return EccentricityEnvelope(
        min_period=(3.0 if cmdline_args.nasa_data is None else 0.8),
        max_period=(20.0 if cmdline_args.nasa_data is None else 5.0),
        max_eccentricity=0.6
    )

#Handling the multiple cases is the whole point
#pylint: disable=too-many-return-statements
#False positive with astropy
#pylint: disable=no-member
def get_unit_label(axis_units):
    """Return the given units properly formatted for plot label."""

    if axis_units == units.day:
        return r'[$days$]'
    if axis_units == units.M_earth:
        return r'[$M_\oplus$]'
    if axis_units == units.R_earth:
        return r'[$R_\oplus$]'
    if axis_units == units.M_jup:
        return r'[$M_j$]'
    if axis_units == units.R_jup:
        return r'[$R_j$]'
    if axis_units == units.M_sun:
        return r'[$M_\odot$]'
    if axis_units == units.R_sun:
        return r'[$R_\odot$]'
    if axis_units == '':
        return ''
    return '[' + str(axis_units) + ']'
#pylint: enable=too-many-return-statements

def get_quantity_label(axis_quantity, planet=False):
    """Format the given quantity for plot label."""

    if axis_quantity == 'lgQ':
        return r"$log_{10}Q'_%s$" % (r'p' if planet else r'\star')
    if axis_quantity == 'orbital_period':
        return r'$P_{orb}$'
    if axis_quantity == 'eccentricity':
        return r'e'
    if axis_quantity == 'secondary_mass':
        return r'$M_p$' if planet else r'$M_2$'
    if axis_quantity == 'secondary_radius':
        return r'$R_p$' if planet else r'$R_2$'
    return str(axis_quantity)
#pylint: enable=no-member

def get_axis_label(axis_quantity, axis_units, planet):
    """Return the axis label for the given quantity and units."""

    return (
        get_quantity_label(axis_quantity, planet)
        +
        ' '
        +
        get_unit_label(axis_units)
    )

#Simplifies command line arguments.
#pylint: disable=invalid-name
#kwargs only accepted to homogenize call signatures across plotters
#pylint: disable=unused-argument
def plot_e_vs_P(cmdline_args, plot_fname=None, save_plot=True, **kwargs):
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
            if (
                    (
                        isinstance(cmdline_args.use_binary_stars, bool)
                        and
                        cmdline_args.use_binary_stars
                    )
                    or
                    cmdline_args.use_binary_stars.upper() == 'NGC188'
            ):
                systems = read_geller_et_al_2009_binaries()
            elif cmdline_args.use_binary_stars.upper() == 'NGC6819':
                systems = read_milliman_et_al_2014_binaries()
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
                           #False positive
                           #pylint: disable=no-member
                           units.M_sun,
                           units.M_sun]
                           #pylint: enable=no-member
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
                       markersize=5,
                       label='fluffy',
                       zorder=10)
        plot_selection(systems,
                       dense,
                       fmt='sb',
                       markersize=5,
                       label='dense',
                       zorder=20)
        plot_selection(systems,
                       unknown,
                       fmt='vg',
                       markersize=5,
                       label='unknown',
                       zorder=30)

        pyplot.plot([0.8, 5.0], [0, 0.6], '-k')
        pyplot.xlim((0.7, 20))
        pyplot.ylim((0, 0.8))

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
        if (
                isinstance(cmdline_args.use_binary_stars, bool)
                and
                cmdline_args.use_binary_stars
        ):
            pyplot.plot([3.0, 20.0], [0, 0.6], '-k', zorder=100)
            pyplot.ylim((0, 0.6))
        else:
            assert cmdline_args.use_binary_stars.upper() == 'NGC6819'
            pyplot.plot([2.1, 18.0], [0, 0.6], '-k')
            pyplot.ylim((0, 0.8))
        pyplot.xlim((2.0, 100))


    systems = read_systems()

    pyplot.xscale('log')
    if cmdline_args.nasa_data is not None:
        plot_exoplanets(systems)
    else:
        plot_binaries(systems)
    pyplot.legend()
    pyplot.xlabel('$P_{orb}$ [days]')
    pyplot.ylabel('eccentricity')
    if hasattr(cmdline_args, 'use_binary_stars'):
        pyplot.title(
            'NGC 188' if (
                isinstance(cmdline_args.use_binary_stars, bool)
                and
                cmdline_args.use_binary_stars
            ) else (
                cmdline_args.use_binary_stars.upper()[:3]
                +
                ' '
                +
                cmdline_args.use_binary_stars.upper()[3:]
            )
        )
    else:
        pyplot.title('Exoplanets')
    if plot_fname is None:
        pyplot.show()
    elif save_plot:
        pyplot.savefig(plot_fname)
        pyplot.cla()
#pylint: enable=invalid-name
#pylint: enable=unused-argument

def plot_star_solving(interpolator,
                      system,
                      plot_fname=None,
                      save_plot=True):
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
    elif save_plot:
        pyplot.savefig(plot_fname)
        pyplot.cla()

#Simplifies command line arguments.
#pylint: disable=invalid-name
def solve_lgQ_limits(lgQ_vs_period,
                     nominal_eccentricity,
                     low_eccentricity,
                     envelope_eccentricity):
    """Return lgQ to reproduce nominal, low and envolpe eccentricity."""

    return (
        invert_eccentricity_vs_lgQ(lgQ_vs_period,
                                   nominal_eccentricity),
        invert_eccentricity_vs_lgQ(lgQ_vs_period,
                                   low_eccentricity),
        invert_eccentricity_vs_lgQ(lgQ_vs_period,
                                   envelope_eccentricity,
                                   default_min=min(lgQ_vs_period.keys()))
    )

def plot_single_lgQ_vs_e(system,
                         progress,
                         eccentricity_envelope,
                         plot_fname=None,
                         save_plot=True):
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

    nominal_e_lgQ, low_e_lgQ, envelope_e_lgQ = solve_lgQ_limits(
        progress[system.hostname],
        nominal_eccentricity,
        low_eccentricity,
        envelope_eccentricity
    )

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
    pyplot.xlabel(get_axis_label('lgQ', '', True))
    pyplot.ylabel(get_axis_label('eccentricity', '', True))
    if plot_fname is None:
        pyplot.show()
    elif save_plot:
        pyplot.savefig(plot_fname)
        pyplot.cla()
#pylint: enable=invalid-name

def get_system_list(cmdline_args, interpolator=None):
    """Return a list of systems for plotting."""

    if (
            cmdline_args.pickle_systems
            and
            os.path.exists(cmdline_args.pickle_systems)
    ):
        with open(cmdline_args.pickle_systems, 'rb') as system_pickle:
            while True:
                try:
                    pickled_cmdline_args = pickle.load(system_pickle)
                except EOFError:
                    break
                systems = pickle.load(system_pickle)
                if vars(pickled_cmdline_args) == vars(cmdline_args):
                    return systems
    systems = []
    #False positive
    #pylint: disable=no-member
    if cmdline_args.nasa_data:
        nasa_systems = read_nasa_planets(cmdline_args.nasa_data,
                                         eliminate=(),
                                         add_units=True,
                                         need_ages=False)
        if interpolator is None:
            systems.extend([
                get_nasa_system(host, nasa_systems)
                for host in sorted(nasa_systems.pl_hostname)
            ])
        else:
            transiting = nasa_systems.pl_discmethod == 'Transit'
            for host in sorted(nasa_systems.pl_hostname[transiting]):
                try:
                    this_system = get_nasa_system(host, nasa_systems)
                    prepare_nasa_system(this_system,
                                        interpolator,
                                        cmdline_args.small_planet_density)
                    systems.append(this_system)
                except AssertionError:
                    pass
    #pylint: enable=no-member

    if getattr(cmdline_args, 'use_binary_stars', None):
        systems.extend(
            (
                read_geller_et_al_2009_binaries()
                if (
                        (
                            isinstance(cmdline_args.use_binary_stars, bool)
                            and
                            cmdline_args.use_binary_stars
                        )
                        or
                        cmdline_args.use_binary_stars.upper() == 'NGC188'
                ) else
                read_milliman_et_al_2014_binaries()
            )
        )

    if cmdline_args.pickle_systems:
        with open(cmdline_args.pickle_systems, 'ab') as system_pickle:
            pickle.dump(cmdline_args, system_pickle)
            pickle.dump(systems, system_pickle)

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

def get_lgQ_constraints(lgQ_x_axes,
                        progress,
                        cmdline_args,
                        interpolator,
                        get_hostnames=False):
    """Collect and calculate the data for lgQ_vs and lgQ_change_vs plots."""

    def add_stellar_properties(system):
        """Augment the given system with info about its star(s) & orbit."""

        print('System: ' + repr(system))
        if not hasattr(system, 'primary_radius'):
            #False positive
            #pylint: disable=no-member
            system.primary_radius = interpolator(
                'radius',
                system.primary_mass.to_value('M_sun'),
                system.feh.to_value('')
            )(system.age.to_value('Gyr')) * units.R_sun
            #pylint: enable=no-member

        if not hasattr(system, 'secondary_radius'):
            #False positive
            #pylint: disable=no-member
            system.secondary_radius = interpolator(
                'radius',
                system.secondary_mass.to_value('M_sun'),
                system.feh.to_value('')
            )(system.age.to_value('Gyr')) * units.R_sun
            #pylint: enable=no-member

        fix_semimajor(system)

    systems = get_system_list(cmdline_args, interpolator)
    eccentricity_envelope = get_eccentricity_envelope(cmdline_args)

    plot_x = numpy.empty((len(lgQ_x_axes), len(progress)), dtype=float)
    plot_lgQ = numpy.empty(plot_x.shape[1], dtype=[('min', float),
                                                   ('nominal', float),
                                                   ('max', float)])

    limit_flags = numpy.empty(
        plot_x.shape[1],
        dtype=[('two_sided', bool), ('upper', bool), ('lower', bool)]
    )
    assumed_default_density = numpy.empty(plot_x.shape[1],
                                          dtype=bool)
    is_giant = numpy.empty(plot_x.shape[1],
                           dtype=bool)

    x_evaluator = asteval.Interpreter()
    index = 0
    hostnames = []
    for system in systems:
        if system.hostname not in progress:
            continue
        hostnames.append(system.hostname)
        add_stellar_properties(system)
        lgQ_vs_period = progress[system.hostname]
        print('System: ' + repr(system))

        assumed_default_density[index] = getattr(system,
                                                 'assumed_default_density',
                                                 False)
        #False positive
        #pylint: disable=no-member
        is_giant[index] = system.secondary_radius > 6 * units.R_earth
        #pylint: enable=no-member

        nominal_eccentricity = system.eccentricity
        min_eccenticity = system.eccentricity - system.eccentricity.minus_error
        if cmdline_args.pretend_min_eccentricity:
            nominal_eccentricity = numpy.nanmin((
                nominal_eccentricity,
                cmdline_args.pretend_min_eccentricity
            ))
            min_eccenticity = numpy.nanmin((
                min_eccenticity,
                cmdline_args.pretend_min_eccentricity
            ))
            system.eccentricity_limit = False

        (
            plot_lgQ['nominal'][index],
            plot_lgQ['min'][index],
            plot_lgQ['max'][index]
        ) = solve_lgQ_limits(
            lgQ_vs_period,
            nominal_eccentricity,
            min_eccenticity,
            max(
                eccentricity_envelope(
                    system.orbital_period.to_value('day')
                ),
                system.eccentricity
            )
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


    result = plot_x, plot_lgQ, limit_flags, assumed_default_density, is_giant
    if get_hostnames:
        return result + (numpy.array(hostnames),)
    return result

def set_x_axis(quantity, planets):
    """Set the x axis appropriately for the givne quantity."""

    if quantity == 'orbital_period':
        if planets:
            pyplot.xscale('linear')
            pyplot.xlim(1, 4.5)
        else:
            pyplot.xscale('linear')
            pyplot.xlim(3, 50)
    else:
        pyplot.xscale('log')
        pyplot.autoscale()

def plot_lgQ_vs(lgQ_x_axes,
                progress,
                cmdline_args,
                interpolator,
                plot_fname=None,
                save_plot=True,
                label=''):
    """Make a plot of the log10(Q*') constraints vs orbital period."""

    if not hasattr(plot_lgQ_vs, "color_index"):
        plot_lgQ_vs.color_index = 2

    def add_points(*,
                   plot_x,
                   plot_y,
                   plot_errors,
                   assumed_default_density,
                   limit=False,
                   distinguish=None):
        """Add a set of points to the plot color-coding for default density."""

        plot_style = dict(
            markersize=5,
            linewidth=0.3
        )
        if limit == 'upper':
            plot_style['fmt'] = 'v'
            plot_style['zorder'] = 0
        elif limit == 'lower':
            plot_style['fmt'] = '^'
            plot_style['zorder'] = 1
        else:
            assert not limit
            plot_style['fmt'] = 'o'
            plot_style['linewidth'] = 2
            plot_style['zorder'] = 2
            plot_style['label'] = label

        for include in [
                numpy.logical_not(assumed_default_density),
                assumed_default_density
        ]:
            plot_style['markeredgecolor'] = kelly_colors[
                plot_lgQ_vs.color_index
            ]
            plot_style['ecolor'] = kelly_colors[
                plot_lgQ_vs.color_index
            ]
            plot_style['markerfacecolor'] = kelly_colors[
                plot_lgQ_vs.color_index
            ]

            if distinguish is None:
                sub_include_list = [include]
            else:
                sub_include_list = [
                    numpy.logical_and(distinguish, include),
                    numpy.logical_and(numpy.logical_not(distinguish), include)
                ]

            for sub_include in sub_include_list:
                pyplot.errorbar(
                    x=plot_x[sub_include],
                    y=plot_y[sub_include],
                    yerr=[err[sub_include] for err in plot_errors],
                    **plot_style
                )
                plot_style['markerfacecolor'] = 'none'
                plot_style['zorder'] -= 10
                if 'label' in plot_style:
                    del plot_style['label']

    (
        plot_x,
        plot_lgQ,
        limit_flags,
        assumed_default_density,
        is_giant
    ) = get_lgQ_constraints(lgQ_x_axes, progress, cmdline_args, interpolator)

    print(100 * '*')
    print('Constraints:')
    print(repr(plot_lgQ))
    print(100 * '*')

    lgQ_range = (3, 9)
    for x_index in range(plot_x.shape[0]):
        print('Plotting lgQ vs ' + repr(lgQ_x_axes[x_index]))
        set_x_axis(lgQ_x_axes[x_index][0], cmdline_args.nasa_data)

        add_points(
            plot_x=plot_x[x_index, limit_flags['upper']],
            plot_y=plot_lgQ['max'][limit_flags['upper']],
            plot_errors=[
                plot_lgQ['max'][limit_flags['upper']] - lgQ_range[0],
                numpy.zeros(limit_flags['upper'].sum())
            ],
            assumed_default_density=assumed_default_density[
                limit_flags['upper']
            ],
            limit='upper',
            distinguish=is_giant[limit_flags['upper']]
        )

        lower_errors = plot_lgQ['nominal'] - plot_lgQ['min']
        print('lower_errors: ' + repr(lower_errors[limit_flags['lower']]))
        print('lgQ nominal: ' + repr(plot_lgQ['nominal'][limit_flags['lower']]))
        print('lgQ min: ' + repr(plot_lgQ['min'][limit_flags['lower']]))

        add_points(
            plot_x=plot_x[x_index, limit_flags['lower']],
            plot_y=plot_lgQ['nominal'][limit_flags['lower']],
            plot_errors=[
                lower_errors[limit_flags['lower']],
                lgQ_range[1] - plot_lgQ['nominal'][limit_flags['lower']]
            ],
            assumed_default_density=assumed_default_density[
                limit_flags['lower']
            ],
            limit='lower',
            distinguish=is_giant[limit_flags['lower']]
        )

        selected = numpy.logical_and(plot_lgQ['nominal'] > 3.5,
                                     limit_flags['two_sided'])
        add_points(
            plot_x=plot_x[x_index, selected],
            plot_y=plot_lgQ['nominal'][selected],
            plot_errors=[
                (
                    plot_lgQ['nominal'][selected]
                    -
                    plot_lgQ['min'][selected]
                ),
                (
                    plot_lgQ['max'][selected]
                    -
                    plot_lgQ['nominal'][selected]
                )
            ],
            assumed_default_density=assumed_default_density[
                selected
            ],
            distinguish=is_giant[selected]
        )

        pyplot.xlabel(
            get_axis_label(*lgQ_x_axes[x_index], cmdline_args.nasa_data)
        )
        pyplot.ylabel(get_axis_label('lgQ', '', cmdline_args.nasa_data))
        if cmdline_args.nasa_data:
            pyplot.axhspan(6, 7, color='lightgrey', zorder=-100)

        if plot_fname is None:
            pyplot.show()
            plot_lgQ_vs.color_index = 2
        elif save_plot:
            print('Saving and clearing: '
                  +
                  plot_fname
                  %
                  dict(x_label=lgQ_x_axes[x_index][0].replace('/', ':')))
            pyplot.legend()
            pyplot.savefig(
                plot_fname
                %
                dict(x_label=lgQ_x_axes[x_index][0].replace('/', ':'))
            )
            pyplot.cla()
            plot_lgQ_vs.color_index = 2
        else:
            plot_lgQ_vs.color_index += 1
            print('Awaiting more points')

def plot_lgQ_change_vs(*,
                       lgQ_x_axes,
                       progress,
                       second_progress,
                       cmdline_args,
                       interpolator,
                       plot_fname=None,
                       save_plot=True):
    """Create a plot of how the lg(Q') constraints change between two runs."""

    def get_matched_constraints():
        """Match the data from the first and second progress pickle."""
        (
            first_plot_x,
            first_lgQ,
            first_limit_flags,
            first_assumed_default_density,
            first_is_giant,
            first_hostnames
        ) = get_lgQ_constraints(lgQ_x_axes,
                                progress,
                                cmdline_args,
                                interpolator,
                                get_hostnames=True)

        (
            second_plot_x,
            second_lgQ,
            second_limit_flags,
            second_assumed_default_density,
            second_is_giant,
            second_hostnames
        ) = get_lgQ_constraints(lgQ_x_axes,
                                second_progress,
                                cmdline_args,
                                interpolator,
                                get_hostnames=True)
        print('1st set of hostnames: ' + repr(first_hostnames))
        print('2nd set of hostnames: ' + repr(second_hostnames))

        first_in_second = numpy.in1d(first_hostnames, second_hostnames)
        second_in_first = numpy.in1d(second_hostnames, first_hostnames)

        plot_x = first_plot_x[:, first_in_second]
        limit_flags = first_limit_flags[first_in_second]
        second_limit_flags = second_limit_flags[second_in_first]
        assumed_default_density = first_assumed_default_density[first_in_second]
        is_giant = first_is_giant[first_in_second]

        limit_flags['two_sided'] = numpy.logical_and(
            limit_flags['two_sided'],
            second_limit_flags['two_sided']
        )
        limit_flags['upper'] = numpy.logical_and(
            numpy.logical_or(
                numpy.logical_or(
                    limit_flags['upper'],
                    limit_flags['two_sided']
                ),
                numpy.logical_or(
                    second_limit_flags['upper'],
                    second_limit_flags['two_sided']
                )
            ),
            numpy.logical_not(limit_flags['two_sided'])
        )
        limit_flags['lower'] = numpy.logical_and(
            numpy.logical_or(
                numpy.logical_or(
                    limit_flags['lower'],
                    limit_flags['two_sided']
                ),
                numpy.logical_or(
                    second_limit_flags['lower'],
                    second_limit_flags['two_sided']
                )
            ),
            numpy.logical_not(limit_flags['two_sided'])
        )

        assert (second_plot_x[:, second_in_first] == plot_x).all()
        assert (second_assumed_default_density[second_in_first]
                ==
                assumed_default_density).all()
        assert (second_is_giant[second_in_first] == is_giant).all()
        assert (first_hostnames[first_in_second]
                ==
                second_hostnames[second_in_first]).all()

        return (plot_x,
                first_lgQ[first_in_second],
                second_lgQ[second_in_first],
                limit_flags,
                assumed_default_density,
                is_giant)

    def add_points(*,
                   plot_x,
                   plot_y1,
                   plot_y2,
                   assumed_default_density,
                   limit=False,
                   distinguish=None):

        plot_style = dict(
            markersize=5,
            linewidth=1.0
        )
        if limit == 'upper':
            plot_style['fmt'] = 'v'
            colors = ['red', 'orange']
            plot_style['zorder'] = 0
        elif limit == 'lower':
            plot_style['fmt'] = '^'
            colors = ['blue', 'cyan']
            plot_style['zorder'] = 1
        else:
            assert not limit
            plot_style['fmt'] = 'o'
            colors = ['green', 'magenta']
            plot_style['linewidth'] = 2
            plot_style['zorder'] = 2

        for color_index, include in enumerate(
                [
                    numpy.logical_not(assumed_default_density),
                    assumed_default_density
                ]
        ):
            plot_style['markeredgecolor'] = colors[color_index]
            plot_style['ecolor'] = colors[color_index]
            plot_style['markerfacecolor'] = colors[color_index]

            if distinguish is None:
                sub_include_list = [include]
            else:
                sub_include_list = [
                    numpy.logical_and(distinguish, include),
                    numpy.logical_and(numpy.logical_not(distinguish), include)
                ]

            for sub_include in sub_include_list:
                plot_include = numpy.logical_and(
                    sub_include,
                    numpy.abs(plot_y2 - plot_y1) < 1
                )
                pyplot.errorbar(
                    x=plot_x[plot_include],
                    y=plot_y1[plot_include],
                    yerr=[
                        plot_y1[plot_include] * 0,
                        numpy.abs(plot_y2[plot_include] - plot_y1[plot_include])
                    ],
                    **plot_style
                )

                plot_style['markerfacecolor'] = 'none'
                plot_style['zorder'] -= 10

    (
        plot_x,
        first_lgQ,
        second_lgQ,
        limit_flags,
        assumed_default_density,
        is_giant
    ) = get_matched_constraints()

    for x_index in range(plot_x.shape[0]):
        set_x_axis(lgQ_x_axes[x_index][0], cmdline_args.nasa_data)

        add_points(
            plot_x=plot_x[x_index, limit_flags['upper']],
            plot_y1=first_lgQ['max'][limit_flags['upper']],
            plot_y2=second_lgQ['max'][limit_flags['upper']],
            assumed_default_density=assumed_default_density[
                limit_flags['upper']
            ],
            limit='upper',
            distinguish=is_giant[limit_flags['upper']]
        )

        add_points(
            plot_x=plot_x[x_index, limit_flags['lower']],
            plot_y1=first_lgQ['nominal'][limit_flags['lower']],
            plot_y2=second_lgQ['nominal'][limit_flags['lower']],
            assumed_default_density=assumed_default_density[
                limit_flags['lower']
            ],
            limit='lower',
            distinguish=is_giant[limit_flags['lower']]
        )

        selected = numpy.logical_and(first_lgQ['nominal'] > 3.5,
                                     limit_flags['two_sided'])

        add_points(
            plot_x=plot_x[x_index, selected],
            plot_y1=first_lgQ['nominal'][selected],
            plot_y2=second_lgQ['nominal'][selected],
            assumed_default_density=assumed_default_density[selected],
            distinguish=is_giant[selected]
        )

        pyplot.xlabel(
            get_axis_label(*lgQ_x_axes[x_index], cmdline_args.nasa_data)
        )
        pyplot.ylabel(get_axis_label('lgQ', '', cmdline_args.nasa_data))

        if plot_fname is None:
            pyplot.show()
        elif save_plot:
            pyplot.savefig(
                plot_fname
                %
                dict(x_label=lgQ_x_axes[x_index][0].replace('/', ':'))
            )
            pyplot.cla()
#pylint: enable=invalid-name

def load_progress(progress_pickle):
    """Return a pickled progress and command line arguments."""

    with open(progress_pickle, 'rb') as progress_file:
        pickled_cmdline_args = pickle.load(progress_file)
        progress = load_progress_pickle(progress_file)
    return pickled_cmdline_args, progress

def dataset_label(cmdline_args):
    """Return a label to use for the dataset from a progress pickle."""

    if cmdline_args.nasa_data is not None:
        assert not cmdline_args.use_binary_stars
        return 'Exoplanets'
    else:
        if (
                (
                    isinstance(cmdline_args.use_binary_stars, bool)
                    and
                    cmdline_args.use_binary_stars
                )
                or
                cmdline_args.use_binary_stars.upper() == 'NGC188'
        ):
            return 'NGC 188'
        elif cmdline_args.use_binary_stars.upper() == 'NGC6819':
            return 'NGC 6819'
        else:
            assert False


def main():
    """Avoid adding things to global namespace."""

    cmdline_args = parse_command_line()

    rcParams['font.size'] = cmdline_args.font_size
    figure = pyplot.figure(figsize=cmdline_args.figure_size)
    #The axes become the default axes so they do get used.
    #pylint: disable=unused-variable
    axes = figure.add_axes([cmdline_args.axes_hshift,
                            cmdline_args.axes_vshift,
                            cmdline_args.axes_hspan - cmdline_args.axes_hshift,
                            cmdline_args.axes_vspan - cmdline_args.axes_vshift])
    #pylint: enable=unused-variable

    pickled_cmdline_args = load_progress(cmdline_args.progress_pickle[0])[0]

    if getattr(cmdline_args, 'second_progress_pickle', None):
        second_progress = load_progress(cmdline_args.second_progress_pickle)[1]

    interpolator = StellarEvolutionManager(
        pickled_cmdline_args.stellar_evolution_interpolator_dir
        if cmdline_args.overwrite_interpolator is None else
        cmdline_args.overwrite_interpolator
    ).get_interpolator_by_name(
        'default'
    )

    print('Processing progress pickles: ' + repr(cmdline_args.progress_pickle))

    for plot_type in cmdline_args.plot:
        if cmdline_args.plot_fname is None:
            plot_fname = None
        else:
            plot_fname = cmdline_args.plot_fname % dict(
                plot_type=plot_type,
                x_label=('%(x_label)s' if plot_type == 'lgQ_vs' else '')
            )
        for plot_pickle in cmdline_args.progress_pickle:
            print('Plotting pickle: ' + repr(plot_pickle))
            plot_cmdline_args, progress = load_progress(plot_pickle)
            plot_cmdline_args.pickle_systems = cmdline_args.pickle_systems
            plot_cmdline_args.pretend_min_eccentricity = (
                cmdline_args.pretend_min_eccentricity
            )


            arguments = dict(progress=progress,
                             lgQ_x_axes=cmdline_args.lgQ_x_axis,
                             cmdline_args=plot_cmdline_args,
                             interpolator=interpolator,
                             plot_fname=plot_fname,
                             label=dataset_label(plot_cmdline_args))
            if plot_type == 'lgQ_change_vs':
                arguments['second_progress'] = second_progress

            globals()['plot_' + plot_type](
                save_plot=(
                    plot_pickle
                    ==
                    cmdline_args.progress_pickle[-1]
                ),
                **arguments
            )

if __name__ == '__main__':
    main()
