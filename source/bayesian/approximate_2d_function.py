"""Implement fast approximate evaluation of expensive 2D functions."""

from collections import namedtuple

from scipy.interpolate import RectBivariateSpline
import numpy

from plot_2d_interpolation import Plot2DInterpolation

ApproximationConfig = namedtuple(
    'ApproximationConfig',
    [
        'support',
        'min_grid_points',
        'tolerance',
        'spline_options',
        'grid_refine_algorithm'
    ]
)


class Approximate2DFunction(RectBivariateSpline, Plot2DInterpolation):
    """
    Approximate but fast evalution of expensive 2D functions.

    Attributes:
        configuration(namedtuple):    The configuration with which the
            approximation was constructed.

        func(callable):    The function being approximated.
    """

    #This is sufficiently simple.
    #pylint: disable=too-many-arguments
    def _interpolate(self, x_grid, y_grid, values, target_x, target_y):
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
            **self.configuration.spline_options
        )(
            target_x,
            target_y,
            grid=True
        )
    #pylint: enable=too-many-arguments


    def _select_mismatches_all(self, calculated_values, interpolated_values):
        """Return x and y grid indices to refine."""

        return numpy.nonzero(
            numpy.absolute(calculated_values - interpolated_values)
            >
            self.configuration.tolerance
        )


    def _select_mismatches_worst(self, calculated_values, interpolated_values):
        """Return y and y index of worst mismatch as 1-element arrays."""

        abs_difference = numpy.absolute(calculated_values - interpolated_values)
        worst_index = numpy.unravel_index(
            numpy.argmax(
                abs_difference,
                axis=None
            ),
            calculated_values.shape
        )

        if abs_difference[worst_index] <= self.configuration.tolerance:
            return numpy.array([], dtype=int), numpy.array([], dtype=int)

        return tuple(numpy.array([ind]) for ind in worst_index)


    def _get_mismatch_indices(self, values, debug_title):
        """
        Return indices where interpolating values fails on current grid.

        Args:
            values(2-D array):    The values of the quantity being interpolated
                at the current grid. The first index should iterate over x and
                the second over y.

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
                interpolated_values = self._interpolate(
                    self._x_grid[ : : 2],
                    self._y_grid[ : : 2],
                    values[ : : 2, : : 2],
                    self._x_grid[x_offset : : 2],
                    self._y_grid[y_offset : : 2],
                )
                self._plot_interpolation_performance(
                    calculated_values=calculated_values,
                    interpolated_values=interpolated_values,
                    x_grid=self._x_grid[x_offset : : 2],
                    y_grid=self._y_grid[y_offset : : 2],
                    interp_data=values[ : : 2, : : 2],
                    title=(debug_title
                           +
                           ' x_di=%d, y_di=%d' % (x_offset, y_offset))
                )

                new_mismatches = getattr(
                    self,
                    (
                        '_select_mismatches_'
                        +
                        self.configuration.grid_refine_algorithm
                    )
                )(
                    calculated_values,
                    interpolated_values,
                    self.configuration.tolerance
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


    def __init__(self,
                 func,
                 support,
                 *,
                 min_grid_points=(100, 100),
                 tolerance=1e-6,
                 num_parallel_processes=1,
                 grid_refine_algorithm='worst',
                 debug_plots=None,
                 **spline_options):
        """
        Setup an interpolation that approximates the given function.

        Keeps a library of generated interpolations in the given pikcle file and
        if one with matching configuration is found, it is re-used.

        Args:
            func(callable):    The function to approximate. Should be
                vectorized, picklable, and provide equality comparison (for
                checking if existing pickle matches).

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

            debug_plots(dict or None):    Each key enables another kind of plot
                showing the progress of the tuning, with the corresponding value
                specifying a `%(keyword)` substitution string that exands to the
                filanema under which to save the plot. The substitution keywords
                depend on the type of plot.

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

        self.configuration = ApproximationConfig(support,
                                                 min_grid_points,
                                                 tolerance,
                                                 spline_options,
                                                 grid_refine_algorithm)
        self.func = func
        self.support = support
        self._x_grid = numpy.linspace(support[0],
                                      support[1],
                                      min_grid_points[0])
        self._y_grid = numpy.linspace(support[2],
                                      support[3],
                                      min_grid_points[1])
        self._values = func(*numpy.meshgrid(self._x_grid, self._y_grid)).T


        self._debug_plots = debug_plots or dict()
        self._tune_interpolation(num_parallel_processes)
        super().__init__(self._x_grid,
                         self._y_grid,
                         self._values,
                         **self.configuration.spline_options)
