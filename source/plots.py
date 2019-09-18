#!/usr/bin/env python3

"""A collection of useful plotting functions."""

from matplotlib import pyplot
import numpy

from astropy import units
from configargparse import ArgumentParser, DefaultsFormatter

from stellar_evolution.change_variables import QuantityEvaluator
from planetary_system_io import read_nasa_planets
from stellar_evolution.manager import StellarEvolutionManager

from io_utilities import load_progress_pickle, get_nasa_system
from calculate_e_Q_grid import prepare_nasa_system
from process_e_Q_grid import\
    format_eccentricity_vs_lgQ,\
    invert_eccentricity_vs_lgQ,\
    EccentricityEnvelope
from command_line_utilities import\
    fix_system_units,\
    add_assumptions_cmdline_args,\
    add_path_cmdline_args

def parse_command_line():
    """Parse the command line defining the plots to create."""

    parser = ArgumentParser(
        description=__doc__,
        default_config_files=['plots.cfg'],
        formatter_class=DefaultsFormatter,
        ignore_unknown_config_file_keys=True
    )
    add_assumptions_cmdline_args(parser)
    add_path_cmdline_args(parser)
    result = parser.parse_args()
    fix_system_units(result)
    return result

def plot_eccentricity_vs_period(systems):
    """Create a plot of eccentricity vs period."""

    def plot_selection(selected, **kwargs):
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

    threshold_density = 2.0 * units.Unit('g/cm3')
    dense = systems.pl_dens > threshold_density
    fluffy = systems.pl_dens < threshold_density
    unknown = numpy.logical_and(numpy.logical_not(dense),
                                numpy.logical_not(fluffy))
    pyplot.xscale('log')
    plot_selection(fluffy, fmt='or', markersize=10, label='fluffy')
    plot_selection(dense, fmt='sb', markersize=15, label='dense')
    plot_selection(unknown, fmt='vg', markersize=15, label='unknown')
    pyplot.plot([0.8, 5.0], [0, 0.6], '-k')
    pyplot.ylim((0, 0.6))
    pyplot.xlim((0.7, 20))
    pyplot.legend()
    pyplot.show()

def plot_star_solving(interpolator, system):
    """Make a plot showing the attempt to solve for stellar mass/age."""

    plot_ages = 10.0**numpy.linspace(-3, 1, 1000)

    for feh, style in [
#            (system.feh - system.feh.minus_error, ':'),
            (system.feh, '-'),
#            (system.feh + system.feh.plus_error, '--')
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
    pyplot.show()

def plot_lgQ_vs_eccentricity(system, progress):
    """Show a plot of the final eccentricity vs lgQ for a system."""


    eccentricity_envelope = EccentricityEnvelope()
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

    pyplot.title(
        system.hostname
        +
        r'($R_p=%g R_j$, $M_p=%g M_j$, $P_{orb}=%g\,d$)'
        %
        (
            system.db_planet_radius.to_value('R_jup'),
            system.db_planet_mass.to_value('M_jup'),
            system.orbital_period.to_value('day')
        )
    )
    pyplot.xlabel(r"$\log_{10}(Q'_\star)$")
    pyplot.ylabel('Eccentricity')
    pyplot.show()

def plot_all_systems_lgQ_vs_eccentricity(progress_pickle, system_data):
    """Show sequentially plots of lgQ vs e for all systems."""

    progress = load_progress_pickle(progress_pickle)
    systems = read_nasa_planets(system_data,
                                eliminate=(),
                                add_units=True,
                                need_ages=False)
    for host in progress.keys():
        plot_lgQ_vs_eccentricity(
            get_nasa_system(host, systems),
            progress
        )

def plot_lgQ_vs_period(progress_pickle,
                       system_data,
                       interpolator,
                       small_planet_density):
    """Make a plot of the log10(Q*') constraints vs orbital period."""

    progress = load_progress_pickle(progress_pickle)
    systems = read_nasa_planets(system_data,
                                eliminate=(),
                                add_units=True,
                                need_ages=False)
    print('Systems semimajor: ' + repr(systems.pl_orbsmax))
    eccentricity_envelope = EccentricityEnvelope()

    plot_x = numpy.empty(len(progress), dtype=float)
    plot_lgQ_min = numpy.empty(plot_x.size, dtype=float)
    plot_lgQ_nominal = numpy.empty(plot_x.size, dtype=float)
    plot_lgQ_max = numpy.empty(plot_x.size, dtype=float)
    lower_limit = numpy.empty(plot_x.size, dtype=bool)
    upper_limit = numpy.empty(plot_x.size, dtype=bool)
    lgQ_range = (3, 9)
    for index, (host, lgQ_vs_period) in enumerate(progress.items()):
        system = get_nasa_system(host, systems)
        prepare_nasa_system(system, interpolator, small_planet_density)
        print('System: ' + repr(system))

        nominal_e_lgQ = invert_eccentricity_vs_lgQ(
            lgQ_vs_period,
            system.eccentricity
        )
        low_e_lgQ = invert_eccentricity_vs_lgQ(
            lgQ_vs_period,
            system.eccentricity - system.eccentricity.minus_error
        )
        envelope_e_lgQ = invert_eccentricity_vs_lgQ(
            lgQ_vs_period,
            eccentricity_envelope(system.orbital_period.to_value('day'))
        )

#        plot_x[index] = system.orbital_period.to_value('day')
        plot_x[index] = (
            (
                system.db_planet_radius
                /
                system.semimajor
            )**5.0
            *
            (
                system.db_star_mass
                /
                system.db_planet_mass
            )**1.0
        ).to_value('')
#        plot_x[index] = (
#            3.0 * system.db_planet_mass
#            /
#            (4.0 * numpy.pi * system.db_planet_radius**3)
#        ).to_value('g/cm3')
        plot_x[index] = system.db_planet_radius.to_value('R_jup')

        plot_lgQ_min[index] = low_e_lgQ
        plot_lgQ_nominal[index] = nominal_e_lgQ
        plot_lgQ_max[index] = envelope_e_lgQ

        upper_limit[index] = (system.eccentricity_limit
                              or
                              (not bool(low_e_lgQ)))
        lower_limit[index] = (envelope_e_lgQ is None)


    not_limit = numpy.logical_not(numpy.logical_or(lower_limit, upper_limit))
    useful = numpy.logical_not(numpy.logical_and(lower_limit, upper_limit))
    upper_limit = numpy.logical_and(upper_limit, useful)
    lower_limit = numpy.logical_and(lower_limit, useful)

#    pyplot.xscale('log')

    pyplot.errorbar(
        x=plot_x[upper_limit],
        y=plot_lgQ_max[upper_limit],
        yerr=[
            plot_lgQ_max[upper_limit] - lgQ_range[0],
            numpy.zeros(upper_limit.sum())
        ],
        fmt='vr',
        markersize=10,
        linewidth=0.3
    )

    lower_errors = plot_lgQ_nominal - plot_lgQ_min
    print('lower_errors: ' + repr(lower_errors[lower_limit]))
    print('lgQ nominal: ' + repr(plot_lgQ_nominal[lower_limit]))
    print('lgQ min: ' + repr(plot_lgQ_min[lower_limit]))

    pyplot.errorbar(
        x=plot_x[lower_limit],
        y=plot_lgQ_nominal[lower_limit],
        yerr=[
            lower_errors[lower_limit],
            lgQ_range[1] - plot_lgQ_nominal[lower_limit]
        ],
        fmt='^b',
        markersize=10,
        linewidth=0.3
    )


    pyplot.errorbar(
        x=plot_x[not_limit],
        y=plot_lgQ_nominal[not_limit],
        yerr=[(plot_lgQ_nominal[not_limit] - plot_lgQ_min[not_limit]),
              (plot_lgQ_max[not_limit] - plot_lgQ_nominal[not_limit])],
        fmt='og',
        markersize=10,
        linewidth=3
    )

    pyplot.show()

if __name__ == '__main__':
    cmdline_args = parse_command_line()
    interpolator = StellarEvolutionManager(
        cmdline_args.stellar_evolution_interpolator_dir
    ).get_interpolator_by_name(
        'default'
    )

    plot_lgQ_vs_period(cmdline_args.progress_pickle,
                       cmdline_args.nasa_data,
                       interpolator,
                       cmdline_args.small_planet_density)
