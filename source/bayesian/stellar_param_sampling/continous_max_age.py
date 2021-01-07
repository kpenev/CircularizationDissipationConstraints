"""Function to find a continuous max age vs mass and [Fe/H] for POET stars."""

from matplotlib import pyplot
import numpy
from scipy.interpolate import RectBivariateSpline

def plot_max_age(interpolator):
    """Display a plot of the max interp age as a function of M and [Fe/H]."""

    plot_masses = numpy.linspace(
        *interpolator.mass_range(),
        1000
    )
    plot_feh = numpy.linspace(
        interpolator.feh_range(),
        1000
    )

    continous_max_age = get_continuous_max_age(interpolator)

    max_age_diff = numpy.empty(shape=(plot_feh.size - 1, plot_masses.size - 1),
                               dtype=float)
    for feh_ind, feh in enumerate(0.5 * (plot_feh[1:]
                                         +
                                         plot_feh[:-1])):
        for mass_ind, mass in enumerate(0.5 * (plot_masses[1:]
                                               +
                                               plot_masses[:-1])):
            max_age_diff[feh_ind, mass_ind] = (
                min(
                    interpolator(
                        'radius',
                        mass,
                        feh
                    ).max_age,
                    13.7
                )
                -
                continous_max_age(mass, feh)
            )

    print('Min age diff: ' + repr(numpy.amin(max_age_diff)))

    pyplot.pcolormesh(
        plot_masses,
        plot_feh,
        max_age_diff,
        edgecolors='none',
        shading='flat'
    )
    pyplot.colorbar()
    pyplot.show()

#TODO: allow using finer grid with smaller offset
def get_continuous_max_age(interpolator):
    """Return continous f(mass, [Fe/H]) giving max age in POET stellar evol."""

    grid_masses = numpy.array([round(float(m), 3)
                               for m in interpolator.track_masses])
    grid_feh = numpy.array([round(float(feh), 3)
                            for feh in interpolator.track_feh])

    tiny = 10.0 * numpy.finfo(numpy.float64).eps

    max_ages = numpy.empty(shape=(grid_masses.size, grid_feh.size),
                           dtype=float)
    for feh_ind, feh in enumerate(grid_feh):
        for mass_ind, mass in enumerate(grid_masses):
            min_max_age = 13.7 + 0.155
            for mass_dir in [-1, 1]:
                if (
                        mass_ind == 0 and mass_dir < 0
                        or
                        mass_ind == grid_masses.size - 1 and mass_dir > 0
                ):
                    continue

                for feh_dir in [-1, 1]:
                    if (
                            feh_ind == 0 and feh_dir < 0
                            or
                            feh_ind == grid_feh.size - 1 and feh_dir > 0
                    ):
                        continue
                    print(
                        'Checking M=%s, [Fe/H]=%s'
                        %
                        (
                            repr(mass + mass_dir * tiny),
                            repr(feh + feh_dir * tiny),
                        )
                    )

                    min_max_age = min(
                        interpolator(
                            'radius',
                            mass + mass_dir * tiny,
                            feh + feh_dir * tiny
                        ).max_age,
                        min_max_age
                    )
            max_ages[mass_ind, feh_ind] = min_max_age - 0.155

    return RectBivariateSpline(grid_masses, grid_feh, max_ages, kx=1, ky=1)
