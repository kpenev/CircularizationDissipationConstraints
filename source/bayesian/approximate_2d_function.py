"""Implement fast approximate evaluation of expensive 2D functions."""

from pickle import Pickler, Unpickler
from collections import namedtuple

from matplotlib import pyplot
from scipy.interpolate import RectBivariateSpline
import numpy

ApproximationConfig = namedtuple(
    'ApproximationConfig',
    [
        'support',
        'min_grid_points',
        'tolerance',
        'spline_options'
    ]
)

def _interpolate(x_grid, y_grid, values, target_x, target_y, **spline_options):
    """
    Evaluate an interpolation on given grids and values at given locations

    Args:
        x_grid(1-D array):    The x values of the grid over which
            interpolation is to be defined.

        y_grid(1-D array):    The y values of the grid over which
            interpolation is to be defined.

        values(2-D array):     The known values of the function to
            interpolate at each [Fe/H], mass combination from the above
            grids. The first index should correspond to feh_grid and the
            second to mass_grid.

        target_x(1-D array):    New x values where the interpolation is to
            be evaluated.

        target_y(1-D array):    New y values where the interpolation is to
            be evaluated.

        spline_options:    Passed directly to `RectBivariateSpline.__init__`.

    Returns:
        2-D array:
            The values predicted by interpolation of all possible
            combinations of target_x and target_y entries. First
            index is alonge tirget_x and the second is along
            target_y.
    """

    return RectBivariateSpline(
        x_grid,
        y_grid,
        values,
        **spline_options
    )(
        target_x,
        target_y,
        grid=True
    )

def _select_mismatches_all(calculated_values, interpolated_values, tolerance):
    """Return x and y grid indices to refine."""

    return numpy.nonzero(
        numpy.absolute(calculated_values - interpolated_values)
        >
        tolerance
    )

def _select_mismatches_worst(calculated_values, interpolated_values, tolerance):
    """Return y and y index of worst mismatch as 1-element arrays."""

    abs_difference = numpy.absolute(calculated_values - interpolated_values)
    worst_index = numpy.unravel_index(
        numpy.argmax(
            abs_difference,
            axis=None
        ),
        calculated_values.shape
    )

    if abs_difference[worst_index] <= tolerance:
        return numpy.array([], dtype=int), numpy.array([], dtype=int)

    return tuple(numpy.array([ind]) for ind in worst_index)


def _get_plot_grid(grid):
    """Return a new grid with point 1/2 between input grid."""

    result = numpy.empty(shape=(grid.size + 1), dtype=grid.dtype)
    result[1:-1] = 0.5 * (grid[1:] + grid[:-1])
    result[0] = 2.0 * grid[0] - result[1]
    result[-1] = 2.0 * grid[-1] - result[-2]
    return result


def _plot_interp_details(*,
                         difference,
                         interp_data,
                         x_grid,
                         y_grid,
                         max_discrepancy_ind,
                         direction,
                         **spline_options):
    """Show the interpolation in mass."""

    assert direction in 'xy'

    interp_grid = locals()[direction + '_grid']
    interp_x = numpy.linspace(interp_grid[0], interp_grid[-1], 100)
    interp_y = _interpolate(
        x_grid[ : : 2],
        y_grid[ : : 2],
        interp_data,
        (
            x_grid[max_discrepancy_ind[0]: max_discrepancy_ind[0] + 1]
            if direction == 'y' else
            interp_x
        ),
        (
            interp_x
            if direction == 'y' else
            y_grid[
                max_discrepancy_ind[1]
                :
                max_discrepancy_ind[1] + 1
            ]
        ),
        **spline_options
    ).flatten()
    main_plot = pyplot.subplot(211)

    plot_slice = (
        numpy.s_[max_discrepancy_ind[0], :]
        if direction == 'mass' else
        numpy.s_[:, max_discrepancy_ind[1]]
    )
    pyplot.plot(interp_x, interp_y, '-r', linewidth=0.25)

    pyplot.plot(
        interp_grid[ : : 2],
        interp_data[plot_slice],
        '.r',
        label='interp input'
    )

    plot_grid = locals()[direction + '_grid']
    pyplot.plot(
        plot_grid,
        calculated_values[plot_slice],
        '.g',
        label='calculated'
    )
    pyplot.plot(
        plot_grid,
        interpolated_values[plot_slice],
        '.b',
        label='interpolated'
    )
    pyplot.legend()

    pyplot.subplot(212, sharex=main_plot)
    pyplot.plot(
        plot_grid,
        difference[plot_slice],
        '.k'
    )
    self._handle_debug_plot(
        title=(
            title
            +
            '_interp_vs_calc_'
            +
            (
                'vM_FeH=%g' % feh_grid[max_discrepancy_ind[0]]
                if direction == 'mass' else
                'vFeH_M=%g' % mass_grid[max_discrepancy_ind[1]]
            )
        )
    )


def _get_mismatch_indices(values,
                          x_grid,
                          y_grid,
                          configuration,
                          debug_title):
    """
    Return indices where interpolating values fails on current grid.

    Args:
        values(2-D array):    The values of the quantity being interpolated
            at the current grid. The first index should iterate over x and
            the second over y.

        x_grid(1-D array):    The current grid of x values at which the
            function values are known.

        y_grid(1-D array):    The current grid of y values at which the
            function values are known.

        debug_title(str):    The title to use for the plot showing the
            current interpolation performance.

    Returns:
        1-D int array:
            Indices within `x_grid` where interpolation using 1/4 of the
            grid is discrepant with the other 3/4 of the values.

        1-D int array:
            Indices within `y_grid` where interpolation fails.
    """

    mismatches = (numpy.array([], dtype=int), numpy.array([], dtype=int))
    for x_offset in [0, 1]:
        for y_offset in [0, 1]:
            if x_offset == y_offset == 0:
                continue

            calculated_values = values[x_offset : : 2, y_offset : : 2]
            interpolated_values = _interpolate(
                x_grid[ : : 2],
                y_grid[ : : 2],
                values[ : : 2, : : 2],
                x_grid[x_offset : : 2],
                y_grid[y_offset : : 2],
                **configuration.spline_options
            )
            _plot_interpolation_performance(
                calculated_values=calculated_values,
                interpolated_values=interpolated_values,
                x_grid=x_grid[x_offset : : 2],
                y_grid=y_grid[y_offset : : 2],
                interp_data=values[ : : 2, : : 2],
                title=(debug_title
                       +
                       ' x_di=%d, y_di=%d' % (x_offset, y_offset))
            )

            new_mismatches = globals()[
                '_select_mismatches_'
                +
                configuration.grid_refine_algorithm
            ](
                calculated_values,
                interpolated_values,
                configuration.tolerance
            )

            mismatches = tuple(
                numpy.unique(
                    numpy.concatenate((
                        old,
                        2 * new + offset
                    ))
                )
                for new, old, offset in zip(new_mismatches,
                                            mismatches,
                                            [x_offset, y_offset])
            )

    return mismatches


def _tune_interpolation(func, configuration):
    """
    Find a interpolation grid to ensure interpolation to specified precision.

    Args:
         func(callable):    The function to approximate.

         config(namedtuple):    The configuration of how the interpolation is
            to be constructed.

    Returns:
        RectBivariateSpline:
            The spline approximation to the given function.
    """

def approximate_function(self,
                         func,
                         support,
                         pickle_fname,
                         *,
                         min_grid_points=(100, 100),
                         tolerance=1e-6,
                         num_parallel_processes=1,
                         show_mismatch_plot=False,
                         **spline_options):
    """
    Return an interpolation that approximates the given function.

    Keeps a library of generated interpolations in the given pikcle file and
    if one with matching configuration is found, it is re-used.

    Args:
        func(callable):    The function to approximate. Should be picklable
            and provide quality comparison (needed to check if existing
            pickle matches)

        support(iterable):    Iterable containing 4 numbers: xmin, xmax,
            ymin, ymax specifying the area over which the function must
            be approximated.

        pickle_fname(str):    The filename to check for previously stored
            approximation and where to store a newly generated one if not
            found.

        min_grid_points(int, int):    Iterable of 2 integers specifying the
            minimun interpolation grid resolution along each argument of the
            function being approximated.

        tolerance(float):    The interpolation grid resolution is increased
            until the inteprolaiton at mid points of grid cells and grid
            walls is within `tolerance` of the calculated value.

        num_parallel_processes(int):    How many simultaneous processes to
            launch when generating a new interpolation. Ignored if existing
            interpolation is found.

        show_mismatch_plot(bool):    If True, a plot showing the
            interpolation performance at each grid refinement step is shown
            (pausing the inteprolaiton until the plot is closed).

        spline_options:    Any additional keyword arguments to pass to
            `scipy.interpolate.RectBivariateSpline` (used to create the
            interpolaiton).

    Returns:
        RectBivariateSpline:
            The approximation of the function satisfying the given
            configuration.
    """

    for order_opt in ['kx', 'ky']:
        if order_opt not in spline_options:
            spline_options[order_opt] = 1

    configuration = ApproximationConfig(support,
                                        min_grid_points,
                                        tolerance,
                                        spline_options)
    result = _check_for_pickled(func, configuration, pickle_fname)

    if result is not None:
        return result

    result = _tune_interpolation(func, configuration)
    _add_to_pickle_file(result, func, configuration, pickle_fname)
    return result
