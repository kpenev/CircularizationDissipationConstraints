"""Define class for fast approximate evaluation of expensive 2D functions."""

from pickle import Pickler, Unpickler
from collections import namedtuple

from matplotlib import pyplot
from scipy.interpolate import RectBivariateSpline

ApproximationConfig = namedtuple(
    'ApproximationConfig',
    [
        'support',
        'min_grid_points',
        'tolerance',
        'spline_options'
    ]
)

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
    Find sufficiently dense interpolation grid to approximate the function.

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
