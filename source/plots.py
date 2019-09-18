#!/usr/bin/env python3

"""A collection of useful plotting functions."""

from matplotlib import pyplot
import numpy
from sys import argv

from astropy import units

from stellar_evolution.change_variables import QuantityEvaluator
from planetary_system_io import read_nasa_planets

from io_utilities import load_progress_pickle, get_nasa_system
from process_e_Q_grid import\
    format_eccentricity_vs_lgQ,\
    invert_eccentricity_vs_lgQ,\
    EccentricityEnvelope

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

    progress = load_progress_pickle(argv[1])
    systems = read_nasa_planets(system_data,
                                eliminate=(),
                                add_units=True,
                                need_ages=False)
    for host in progress.keys():
        plot_lgQ_vs_eccentricity(
            get_nasa_system(host, systems),
            progress
        )

def plot_lgQ_vs_period(progress_pickle, system_data):
    """Make a plot of the log10(Q*') constraints vs orbital period."""

    progress = load_progress_pickle(argv[1])
    systems = read_nasa_planets(system_data,
                                eliminate=(),
                                add_units=True,
                                need_ages=False)
    eccentricity_envelope = EccentricityEnvelope()

    plot_porb, plot_lgQ_min, plot_lgQ_nominal, plot_lgQ_max = [], [], [], []
    for host, lgQ_vs_period in progress.items():
        system = get_nasa_system(host, systems)

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

        plot_porb.append(system.orbital_period.to_value('day'))
        plot_lgQ_min.append(low_e_lgQ)
        plot_lgQ_nominal.append(nominal_e_lgQ)
        plot_lgQ_max.append(envelope_e_lgQ)
    pyplot.semilogx(plot_porb, plot_lgQ_min, 'or')
    pyplot.semilogx(plot_porb, plot_lgQ_nominal, 'og')
    pyplot.semilogx(plot_porb, plot_lgQ_max, 'ob')
    pyplot.show()

if __name__ == '__main__':
    plot_lgQ_vs_period('progress.pickle',
                       '../data/planets_2019.09.03_07.12.43.csv')
