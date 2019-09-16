"""A collection of useful plotting functions."""

from matplotlib import pyplot
import numpy

from astropy import units

from stellar_evolution.change_variables import QuantityEvaluator

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
