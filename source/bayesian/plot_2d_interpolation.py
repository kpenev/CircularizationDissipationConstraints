"""Plotting for `tuned_2d_interpolation` module."""

from matplotlib import pyplot
import numpy

def plot_interpolation_performance(*,
                                    calculated_values,
                                    interpolated_values,
                                    feh_grid,
                                    mass_grid,
                                    interp_data,
                                    title):
    """Show plot of how the interpolation performs as grid is refined."""

    def plot_interp_performance(difference, max_discrepancy_ind):
        """Create multi-panel plot showing the current interpolation."""

        pyplot.subplot(221)
        plot_feh = get_plot_grid(feh_grid)
        plot_masses = get_plot_grid(mass_grid)
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
        pyplot.plot(mass_grid,
                    difference[max_discrepancy_ind[0], : ],
                    '.r')
        pyplot.xlabel(r'$M_\star$ [$M_\odot$]')
        pyplot.ylabel('calc - interp')

        pyplot.twiny()
        pyplot.plot(feh_grid,
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
    plot_interp_details(difference, max_discrepancy_ind, 'mass')
    plot_interp_details(difference, max_discrepancy_ind, 'feh')

    plot_interp_performance(difference, max_discrepancy_ind)

#pylint: enable=too-many-statements



