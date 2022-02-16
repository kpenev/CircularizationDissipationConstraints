#!/usr/bin/env python3
"""Implement fast approximate evaluation of expensive 2D functions."""

from collections import namedtuple
import logging
from itertools import count
from multiprocessing import Pool

from matplotlib import pyplot
from scipy.interpolate import RectBivariateSpline
import numpy

from bayesian.plot_2d_interpolation import Plot2DInterpolation
from bayesian.picklable import Picklable

ApproximationConfig = namedtuple(
    'ApproximationConfig',
    [
        'support',
        'min_grid_points',
        'min_grid_steps',
        'tolerance',
        'spline_options',
        'refine_limit',
        'refine_1d'
    ]
)


#No reasonable way to simplify
#pylint: disable=too-many-instance-attributes
class Approximate2DFunction(RectBivariateSpline,
                            Plot2DInterpolation,
                            Picklable):
    """
    Approximate but fast evalution of expensive 2D functions.

    Attributes:
        configuration(namedtuple):    The configuration with which the
            approximation was constructed.

        func(callable):    The function being approximated.
    """

    _logger = logging.getLogger(__name__)

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
                interpolate at each x, y combination from the above grids. The
                first index should correspond to x and the second to y.

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


    def _select_mismatches(self, calculated_values, interpolated_values):
        """Return x and y grid indices to refine."""

        residuals = numpy.absolute(calculated_values - interpolated_values)
        if self.configuration.refine_limit:
            mean_end = -self.configuration.refine_limit + 1
            if mean_end == 0:
                mean_end = residuals.size
            tolerance = max(
                self.configuration.tolerance,
                numpy.mean(
                    numpy.partition(
                        residuals.flatten(),
                        (
                            -self.configuration.refine_limit - 1,
                            -self.configuration.refine_limit
                        )
                    )[
                        -self.configuration.refine_limit - 1
                        :
                        (-self.configuration.refine_limit + 1) or residuals.size
                    ]
                )
            )
        else:
            tolerance = self.configuration.tolerance

        result = numpy.nonzero(residuals > tolerance)

        self._logger.debug('Selected %d cells to refine', result[0].size)

        return result


    def _get_mismatch_indices(self):
        """
        Return indices where interpolating values fails on current grid.

        Args:
           None

        Returns:
            (1-D int array, 1-D int array):
                Indices within `x_grid`/`y_grid` where interpolation using 1/4
                of the grid is discrepant with the other 3/4 of the values.

            int:
                The direction (0 for x, 1 for y) along which the interpolation
                is worse.
        """

        mismatches = (numpy.array([], dtype=int), numpy.array([], dtype=int))
        max_1d_error = numpy.empty((2, 2), dtype=numpy.float64)
        for x_offset in [0, 1]:
            for y_offset in [0, 1]:
                if x_offset == y_offset == 0:
                    continue

                calculated_values = self._values[x_offset : : 2, y_offset : : 2]
                interpolated_values = self._interpolate(
                    self._x_grid[ : : 2],
                    self._y_grid[ : : 2],
                    self._values[ : : 2, : : 2],
                    self._x_grid[x_offset : : 2],
                    self._y_grid[y_offset : : 2],
                )
                self._plot_interpolation_performance(
                    calculated_values=calculated_values,
                    interpolated_values=interpolated_values,
                    x_grid=self._x_grid[x_offset : : 2],
                    y_grid=self._y_grid[y_offset : : 2],
                    interp_data=self._values[ : : 2, : : 2],
                    x_offset=x_offset,
                    y_offset=y_offset
                )

                new_mismatches = self._select_mismatches(
                    calculated_values,
                    interpolated_values,
                )

                max_1d_error[x_offset, y_offset] = numpy.max(
                    numpy.absolute(calculated_values - interpolated_values)
                )

                self._logger.debug('Max absolute difference: %s',
                                   max_1d_error[x_offset, y_offset])

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

        return mismatches, (1 if max_1d_error[0, 1] > max_1d_error[1, 0] else 0)


    def _get_grid_refinement(self):
        """
        Return new x and y values to add to grid to improve the interpolation.

        Args:
            None

        Returns:
            (1-D float array, 1-D int array):
                Sorted extra x values to add to the grid near places where
                the interpolation precision is insufficient and the number of
                current x grid nodes smaller than the corresponding new
                x value.

            (1-D float array, 1-D int array):
                Same as above but for the y grid.
        """

        def get_new_grid_points(mismatch_indices, current_grid, min_grid_step):
            """Return new values to add per the given mismatch indices."""

            if mismatch_indices.size == 0:
                return None

            below_indices = numpy.unique(
                numpy.concatenate((
                    (
                        mismatch_indices
                        if mismatch_indices[-1] < current_grid.size - 1 else
                        mismatch_indices[:-1]
                    ),
                    (
                        mismatch_indices
                        if mismatch_indices[0] > 0 else
                        mismatch_indices[1:]
                    ) - 1
                ))
            )
            valid = (
                (current_grid[below_indices + 1] - current_grid[below_indices])
                >
                2.0 * min_grid_step
            )
            below_indices = below_indices[valid]
            return (
                0.5 * (current_grid[below_indices]
                       +
                       current_grid[below_indices + 1]),
                below_indices + 1
            )

        mismatch_indices, worse_direction = list(self._get_mismatch_indices())

        self._logger.debug('Mismatch indices: %s', repr(mismatch_indices))
        self._converged = (mismatch_indices[0].size == 0
                           and
                           mismatch_indices[1].size == 0)

        no_refinement = numpy.array([], dtype=float), numpy.array([], dtype=int)

        result = [
            (get_new_grid_points(*args) or no_refinement)
            for args in zip(mismatch_indices,
                            [self._x_grid, self._y_grid],
                            self.configuration.min_grid_steps)
        ]


        if self.configuration.refine_1d == 'fewer_ind':
            if result[0][0].size > result[1][0].size:
                return [no_refinement, result[1]]
            return [result[0], no_refinement]

        if self.configuration.refine_1d == 'worse_direction':
            if worse_direction == 0:
                return [result[0], no_refinement]
            assert worse_direction == 1
            return [no_refinement, result[1]]

        return result


    def _calculate_function_values(self, x_grid, y_grid, workers):
        """
        Calculate function at the given grid points using parallel processing.

        Args:
            x_grid(1-D array):    The x values at which to evaluate the
                function.

            y_grid(1-D array):    The y values at which to evaluate the
                function.

        Returns:
            numpy.array(shape=(x_grid.shape, y_grid.shape)):
                The values of the function at the specified grid.
        """

        eval_y, eval_x = numpy.meshgrid(y_grid, x_grid)

        result = numpy.array(
            list(
                workers.starmap(
                    self.func,
                    zip(eval_x.flatten(), eval_y.flatten())
                )
            )
        ).reshape(x_grid.size, y_grid.size)

        return result


    def _tune_interpolation(self, workers):
        """Find a grid dense enough to achive desired approximation."""

        def insert_entries(current, new, num_before, destination):
            """
            Set destination to new entries inserted among current.

            Args:
                current:    The current array to add entries to. Not modified.

                new:    The new entries to add.

                num_before:    The number of current entries to precede each
                    entry in new.

                destination:    The array to fill. Sholud already be
                    pre-allocated and is completely overwritten.

            Returns:
                None
            """

            current_start = 0
            for new_index, (current_end, new_entry) in enumerate(zip(num_before,
                                                                     new)):
                destination[
                    current_start + new_index
                    :
                    current_end + new_index
                ] = current[current_start : current_end]

                destination[current_end + new_index] = new_entry

                current_start = current_end

            destination[current_start + len(new) : ] = current[current_start : ]

        def add_grid_points(grid, new_points, num_smaller):
            """
            Return new grid combining the current grid with new points.

            Args:
                grid:    The grid to add points to. Not modified.

                new_points:    The new values to add to the grid.

                num_smaller:    The number of current grid points smaller than
                    each entry in new_points.

            Returns:
                1-D array:
                    The new grid.
            """

            result = numpy.empty(shape=(grid.size + new_points.size),
                                 dtype=grid.dtype)
            insert_entries(grid, new_points, num_smaller, result)
            return result

        for self._grid_refinement_iteration in count():
            grid_refinement = self._get_grid_refinement()

            if grid_refinement[0][0].size == grid_refinement[1][0].size == 0:
                return

            new_grids = [
                add_grid_points(args[0], *args[1])
                for args in zip(
                    (self._x_grid, self._y_grid), grid_refinement
                )
            ]
            for grid, label in zip(new_grids, 'xy'):
#                self._logger.debug('New %s grid: %s',
#                                   self._plot_labels[label],
#                                   repr(grid.T))
                self._logger.debug('Min %s grid step: %s',
                                   self._plot_labels[label],
                                   repr((grid[1:] - grid[:-1]).min()))

            if grid_refinement[1][0].size == 0:
                new_y_old_x_function_values = self._values
            else:
                function_values_to_add = self._calculate_function_values(
                    self._x_grid,
                    grid_refinement[1][0],
                    workers
                )
                new_y_old_x_function_values = [[None] * new_grids[1].size
                                               for x in self._x_grid]
                for x_index in range(self._x_grid.size):
                    insert_entries(self._values[x_index],
                                   function_values_to_add[x_index],
                                   grid_refinement[1][1],
                                   new_y_old_x_function_values[x_index])

            if grid_refinement[0][0].size == 0:
                self._values = numpy.copy(new_y_old_x_function_values)
            else:
                function_values_to_add = self._calculate_function_values(
                    grid_refinement[0][0],
                    new_grids[1],
                    workers

                )
                self._values = numpy.empty(
                    (new_grids[0].size, new_grids[1].size),
                    dtype=numpy.float64
                )
                insert_entries(new_y_old_x_function_values,
                               function_values_to_add,
                               grid_refinement[0][1],
                               self._values)

            self._x_grid, self._y_grid = new_grids


    def __eq__(self, other):
        """Return True iff self and other are identical approximations."""

        return (
            self.func == other.func
            and
            self.support == other.support
            and
            self.configuration == other.configuration
            and
            (self._x_grid == other._x_grid).all()
            and
            (self._y_grid == other._y_grid).all()
            and
            (self._values == other._values).all()
        )


    def _check_pickle(self):
        """Check if given file contains a re-usable pickle of desired approx."""

        pickled = self._load_pickle_object()
        if self.func != pickled:
            self._logger.debug('Pickled function %s does not match %s',
                               repr(pickled),
                               repr(self.func))
            return None

        pickled = self._load_pickle_object()
        if self.support != pickled:
            self._logger.debug('Pickled support %s does not match %s',
                               repr(pickled),
                               repr(self.support))
            return None

        pickled = self._load_pickle_object()
        if self.configuration != pickled:
            self._logger.debug('Pickled configuration %s does not match %s',
                               repr(pickled),
                               repr(self.configuration))
            return None

        return (
            self._load_pickle_object(),
            self._load_pickle_object(),
            self._load_pickle_object(),
        )

    def _get_initial_grid(self):
        """Return an initial grid from which to start refining interpolation."""

        return (
            numpy.linspace(self.configuration.support[0],
                           self.configuration.support[1],
                           self.configuration.min_grid_points[0]),
            numpy.linspace(self.configuration.support[2],
                           self.configuration.support[3],
                           self.configuration.min_grid_points[1])
        )

    def __init__(self,
                 func,
                 support,
                 *,
                 min_grid_points=(100, 100),
                 tolerance=1e-6,
                 min_grid_steps=None,
                 num_parallel_processes=1,
                 grid_refine_limit=0,
                 grid_refine_1d=None,
                 pickle_fname='approximate_2d.pkl',
                 debug_plots=None,
                 plot_labels=None,
                 **spline_options):
        """
        Setup an interpolation that approximates the given function.

        Keeps a library of generated interpolations in the given pikcle file and
        if one with matching configuration is found, it is re-used.

        Args:
            func(callable):    The function to approximate.

            support(iterable):    Iterable containing 4 numbers: xmin, xmax,
                ymin, ymax specifying the area over which the function must
                be approximated.

            min_grid_points(int, int):    Iterable of 2 integers specifying the
                minimun interpolation grid resolution along each argument of the
                function being approximated.

            tolerance(float):    The interpolation grid resolution is increased
                until the inteprolaiton at mid points of grid cells and grid
                walls is within `tolerance` of the calculated value.

            min_grid_steps(float, float):    The smallest steps allowed each of
                the grid directions.

            num_parallel_processes(int):    How many simultaneous processes to
                launch when generating a new interpolation. Ignored if existing
                interpolation is found.

            grid_refine_limit(int):    At each step, only the
                `grid_refine_limit` most discrepant cells are sub-divided. A
                value of `0` results in no limit.

            grid_refine_1d(bool):    If `None`, cells chosen for sub-division
                are sub-divided along both x and y. If `'fewer_ind'`, only the
                direction with fewer indices to refine is refined. If
                `'wore_direction'` only  dimension that would improve the 1-D
                interpolation that shows the bigger maximum discepancy is
                refined.

            pickle_fname(str):    A filename which to check for pickle
                approximations and store newly created one if none of those
                stored in the file matches what is being constructed.

            debug_plots:    See same name argument to
                `Plot2DInterpolation.__init__`.

            plot_labels:    See same name argument to
                `Plot2DInterpolation.__init__`.

            spline_options:    Any additional keyword arguments to pass to
                `scipy.interpolate.RectBivariateSpline` (used to create the
                interpolaiton).

        Returns:
            RectBivariateSpline:
                The approximation of the function satisfying the given
                configuration.
        """

        Picklable.__init__(self, 6)

        for order_opt in ['kx', 'ky']:
            if order_opt not in spline_options:
                spline_options[order_opt] = 1

        self.configuration = ApproximationConfig(
            support,
            min_grid_points,
            (
                min_grid_steps
                or
                (
                    1e-12 * (support[1] - support[0]),
                    1e-12 * (support[3] - support[2])
                )
            ),
            tolerance,
            spline_options,
            grid_refine_limit,
            grid_refine_1d
        )
        self.func = func
        self.support = support
        self._logger.debug(
            'Deriving approximation to %s(%s < x < %s, %s < y < %s) with '
            'configuration: %s',
            repr(func),
            *tuple(support),
            repr(self.configuration)
        )

        self._x_grid, self._y_grid, self._values = (
            self.check_for_pickled(pickle_fname)
            or
            (None, None, None)
        )

        if (
                self._x_grid is None
                or
                self._y_grid is None
                or
                self._values is None
        ):
            self._x_grid, self._y_grid = self._get_initial_grid()
            Plot2DInterpolation.__init__(self, debug_plots, plot_labels)

            self._debug_plots = debug_plots or dict()
            self._converged = False
            with Pool(num_parallel_processes) as workers:
                self._values = self._calculate_function_values(self._x_grid,
                                                               self._y_grid,
                                                               workers)
                self._tune_interpolation(workers)

            assert self._converged
            self._logger.info(
                'Approximating %s over %s <= %s <= %s and %s <= %s <= %s '
                'converged using %dx%d grid.',
                self._plot_labels['function'],
                repr(support[0]),
                self._plot_labels['x'],
                repr(support[1]),
                repr(support[2]),
                self._plot_labels['y'],
                repr(support[3]),
                self._x_grid.size,
                self._y_grid.size
            )

            self.add_to_pickle_file(pickle_fname,
                                    self.func,
                                    self.support,
                                    self.configuration,
                                    self._x_grid,
                                    self._y_grid,
                                    self._values)

        else:
            Plot2DInterpolation.__init__(self, debug_plots, plot_labels)
            self._logger.info(
                'Approximation of %s over %s <= %s <= %s and %s <= %s <= %s, '
                'using %dx%d grid, found in pickle file.',
                self._plot_labels['function'],
                repr(support[0]),
                self._plot_labels['x'],
                repr(support[1]),
                repr(support[2]),
                self._plot_labels['y'],
                repr(support[3]),
                self._x_grid.size,
                self._y_grid.size
            )

        RectBivariateSpline.__init__(self,
                                     self._x_grid,
                                     self._y_grid,
                                     self._values,
                                     **self.configuration.spline_options)
#pylint: enable=too-many-instance-attributes

if __name__ == '__main__':

    def fit_func(x, y):
        """Example function to fit."""

        result = numpy.sin(2.0 * x + 1.5) * numpy.cos(3.0 * y - 1.5)
        return result

    logging.basicConfig(level=logging.DEBUG)

    approx = Approximate2DFunction(
        fit_func,
        (0.0, 1.0, 2.0, 3.0),
        #debug_plots=dict(
        #    interpolation_performance='%(title)s.png'
        #),
        min_grid_steps=(1e-6, 1e-6),
        grid_refine_algorithm='all'
    )

    #Debugging purposes
    #pylint: disable=protected-access
    pyplot.plot(numpy.arange(approx._x_grid.size - 1),
                approx._x_grid[1:] - approx._x_grid[:-1],
                'or',
                label='x steps')
    pyplot.twinx()
    pyplot.plot(numpy.arange(approx._y_grid.size - 1),
                approx._y_grid[1:] - approx._y_grid[:-1],
                'og',
                label='y steps')
    #pylint: enable=protected-access
    pyplot.figlegend()
    pyplot.show()
