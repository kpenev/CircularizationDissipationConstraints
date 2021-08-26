"""Plotting for `tuned_2d_interpolation` module."""

import inspect
from abc import ABC, abstractmethod

from matplotlib import pyplot
import numpy

#Intended just as base class. Public methods come from children
#pylint: disable=too-few-public-methods
class Plot2DInterpolation(ABC):
    """Implement plotting of the progress of 2D interpolation tuning."""

    @property
    def _x_grid(self):
        raise NotImplementedError


    @property
    def _y_grid(self):
        raise NotImplementedError


    #Intended to be overwritten by child classes
    #pylint: disable=missing-function-docstring
    @property
    def configuration(self):
        raise NotImplementedError
    #pylint: enable=missing-function-docstring


    #This is sufficiently simple.
    #pylint: disable=too-many-arguments
    @abstractmethod
    def _interpolate(self, x_grid, y_grid, values, target_x, target_y):
        raise NotImplementedError
    #pylint: enable=too-many-arguments

    def _handle_debug_plot(self, **fname_substitutions):
        """Either save of display the currently set-up plot."""

        for caller in inspect.stack()[1:]:
            if caller.function.startswith('_plot_'):
                break

        #This should fail if caller remains undefined
        #pylint: disable=undefined-loop-variable
        caller = caller.function
        assert caller.startswith('_plot_')
        #pylint: enable=undefined-loop-variable

        filename = (
            self._debug_plots[caller[len('_plot_'):]]
            %
            dict(
                fname_substitutions,
                grid_refinement_i=self._grid_refinement_iteration
            )
        )
        if filename:
            pyplot.savefig(filename)
            pyplot.clf()
        else:
            pyplot.show()


    #Sufficient structure provided by sub-functions.
    #pylint: disable=too-many-statements
    def _plot_interpolation_performance(self,
                                        *,
                                        calculated_values,
                                        interpolated_values,
                                        x_grid,
                                        y_grid,
                                        interp_data,
                                        title):
        """Show plot of how the interpolation performs as grid is refined."""

        if 'interpolation_performance' not in self._debug_plots:
            return

        def get_plot_grid(grid):
            """Return a new grid with point 1/2 between input grid."""

            result = numpy.empty(shape=(grid.size + 1), dtype=grid.dtype)
            result[1:-1] = 0.5 * (grid[1:] + grid[:-1])
            result[0] = 2.0 * grid[0] - result[1]
            result[-1] = 2.0 * grid[-1] - result[-2]
            return result


        def plot_interp_details(difference,
                                max_discrepancy_ind,
                                direction):
            """Show the interpolation in mass."""

            assert direction in 'xy'

            interp_grid = getattr(self, '_%s_grid' % direction)
            interp_x = numpy.linspace(interp_grid[0], interp_grid[-1], 100)
            interp_y = self._interpolate(
                self._x_grid[ : : 2],
                self._y_grid[ : : 2],
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
                )
            ).flatten()
            main_plot = pyplot.subplot(211)

            plot_slice = (
                numpy.s_[max_discrepancy_ind[0], :]
                if direction == 'y' else
                numpy.s_[:, max_discrepancy_ind[1]]
            )
            pyplot.plot(interp_x, interp_y, '-r', linewidth=0.25)

            pyplot.plot(
                interp_grid[ : : 2],
                interp_data[plot_slice],
                '.r',
                label='interp input'
            )

            plot_grid = y_grid if direction == 'y' else x_grid
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
                        'vM_FeH=%g' % x_grid[max_discrepancy_ind[0]]
                        if direction == 'y' else
                        'vFeH_M=%g' % y_grid[max_discrepancy_ind[1]]
                    )
                )
            )

        def plot_interp_performance(difference, max_discrepancy_ind):
            """Create multi-panel plot showing the current interpolation."""

            pyplot.subplot(221)
            plot_feh = get_plot_grid(x_grid)
            plot_masses = get_plot_grid(y_grid)
            pyplot.pcolormesh(plot_masses,
                              plot_feh,
                              calculated_values,
                              edgecolors='none')
            pyplot.xlabel(r'$M_\star$ [$M_\odot$]')
            pyplot.ylabel('[Fe/H]')
            pyplot.title('Calculated')
            pyplot.colorbar()

            pyplot.subplot(222)
            pyplot.pcolormesh(plot_masses,
                              plot_feh,
                              interpolated_values,
                              edgecolors='none')
            pyplot.xlabel(r'$M_\star$ [$M_\odot$]')
            pyplot.ylabel('[Fe/H]')
            pyplot.title('Interpolated')
            pyplot.colorbar()

            pyplot.subplot(223)
            pyplot.pcolormesh(plot_masses,
                              plot_feh,
                              difference,
                              edgecolors='none')
            pyplot.xlabel(r'$M_\star$ [$M_\odot$]')
            pyplot.ylabel('[Fe/H]')
            pyplot.title('Calculated - Interpolated')
            pyplot.colorbar()

            pyplot.subplot(224)
            pyplot.plot(y_grid,
                        difference[max_discrepancy_ind[0], : ],
                        '.r')
            pyplot.xlabel(r'$M_\star$ [$M_\odot$]')
            pyplot.ylabel('calc - interp')

            pyplot.twiny()
            pyplot.plot(x_grid,
                        difference[ :, max_discrepancy_ind[1]],
                        '.b')
            pyplot.xlabel('[Fe/H]')

            pyplot.suptitle(title)

            self._handle_debug_plot(title=title)

        difference = calculated_values - interpolated_values
        max_discrepancy_ind = numpy.unravel_index(
            numpy.argmax(
                numpy.absolute(difference),
                axis=None
            ),
            calculated_values.shape
        )
        plot_interp_details(difference, max_discrepancy_ind, 'y')
        plot_interp_details(difference, max_discrepancy_ind, 'x')

        plot_interp_performance(difference, max_discrepancy_ind)
    #pylint: enable=too-many-statements

    def __init__(self, debug_plots):
        """Configure the plots to create."""

        self._debug_plots = debug_plots
        self._grid_refinement_iteration = None
#pylint: enable=too-few-public-methods
