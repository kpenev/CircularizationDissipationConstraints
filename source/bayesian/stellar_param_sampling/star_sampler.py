"""The class that implements efficient sampling of stellar properties."""

import inspect
from itertools import count
from pickle import Pickler, Unpickler, UnpicklingError

from matplotlib import pyplot
import numpy
from scipy.stats import norm
from scipy.integrate import cumtrapz
from scipy.interpolate import RectBivariateSpline
from astropy import units as u

class StarSampler:
    """Implenet calculation, saving, and loading of star parameter sampling."""

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

    def _plot_initial_feh_grid(self):
        """Display plots showing initial [Fe/H] grid was correctly generated."""

        if 'initial_feh_grid' not in self._debug_plots:
            return

        pyplot.plot(self._feh_grid, '.')
        x_range = numpy.array([0, self._feh_grid.size - 1])
        pyplot.plot(x_range,
                    x_range * self.config.feh_max_step + self._feh_grid[0],
                    '-')
        pyplot.plot(x_range[::-1],
                    #False positive
                    #pylint: disable=invalid-unary-operand-type
                    -x_range * self.config.feh_max_step + self._feh_grid[-1],
                    #pylint: enable=invalid-unary-operand-type
                    '-')
        pyplot.axhline(self.config.feh.value)
        pyplot.axhline(self.config.feh.value + self.config.feh.plus_error)
        pyplot.axhline(self.config.feh.value - self.config.feh.minus_error)
        pyplot.show()

        scaled_feh_diff = self._feh_grid - self.config.feh.value
        scaled_feh_diff[scaled_feh_diff > 0] /= self.config.feh.plus_error
        scaled_feh_diff[scaled_feh_diff < 0] /= self.config.feh.minus_error
        feh_cdf = norm.cdf(scaled_feh_diff)
        x_med = numpy.argmin(numpy.fabs(self._feh_grid - self.config.feh.value))
        pyplot.plot(feh_cdf, '.')
        pyplot.plot(x_range,
                    0.5 + (x_range - x_med) * self.config.feh_max_cdf_step,
                    '-')

        self._handle_debug_plot()

    #Sufficient structure provided by sub-functions.
    #pylint: disable=too-many-statements
    def _plot_interpolation_performance(self,
                                        *,
                                        calculated_values,
                                        interpolated_values,
                                        feh_grid,
                                        mass_grid,
                                        interp_data,
                                        title):
        """Show plot of how the interpolation performs as grid is refined."""

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

            assert direction in ['mass', 'feh']

            interp_grid = getattr(self, '_%s_grid' % direction)
            interp_x = numpy.linspace(interp_grid[0], interp_grid[-1], 100)
            interp_y = self._interpolate(
                self._feh_grid[ : : 2],
                self._mass_grid[ : : 2],
                interp_data,
                (
                    feh_grid[max_discrepancy_ind[0]: max_discrepancy_ind[0] + 1]
                    if direction == 'mass' else
                    interp_x
                ),
                (
                    interp_x
                    if direction == 'mass' else
                    mass_grid[
                        max_discrepancy_ind[1]
                        :
                        max_discrepancy_ind[1] + 1
                    ]
                )
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

            plot_grid = mass_grid if direction == 'mass' else feh_grid
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

    @classmethod
    def list_debug_plots(cls):
        """List all debug plots this class can generate."""

        return [k[len('_plot_'):]
                for k in vars(cls).keys()
                if k.startswith('_plot_')]

    def _calculate_age_cdfs(self, mass_grid, feh_grid):
        """
        Calculate unnormalized age CDFs at the given grid points.

        Args:
            mass_grid(1-D array):     The masses at which to calculate the CDFs.

            feh_grid(1-D array):    The [Fe/H] values at which to calculate the
                CDFs.

        Returns:
            [[OdeSolution, ...], ...]:
                Unnormalized CDF(age | mass, [Fe/H]) at each mass [Fe/H]
                combination. The first index is over [Fe/H] and the second is
                over mass.
        """

        return [
            [
                self._log_likelihood.age_integral(mass * u.M_sun, feh)
                for mass in mass_grid
            ]
            for feh_index, feh in enumerate(feh_grid)
        ]


    def _update_mass_cdfs(self):
        """
        Update the mass CDFs for a new grid assuming age CDFs have been updated.

        Angs:
            None

        Returns:
            None
        """

        self._mass_cdf = numpy.empty(
            shape=(self._feh_grid.size, self._mass_grid.size),
            dtype=float
        )

        for feh_index, feh_age_cdf in enumerate(self._age_cdf):
            age_cdf_norm = numpy.array([
                float(cdf(cdf.t_max)) for cdf in feh_age_cdf
            ])
            self._mass_cdf[feh_index, :] = cumtrapz(age_cdf_norm,
                                                    self._mass_grid,
                                                    initial=0.0)
            self._mass_cdf[feh_index, :] /= self._mass_cdf[feh_index, -1]

    def _get_initial_feh_grid(self, min_feh, max_feh):
        """
        Create the initial [Fe/H] grid to start deriving the interpolation from.
        """

        def get_half_grid_offsets(side):
            """Return the grid pts on one side (plus or minus) of the median."""

            assert side in ['plus', 'minus']

            distribution = norm(scale=getattr(self.config.feh, side + '_error'))

            offset = distribution.isf(self.config.max_discarded_feh_probability
                                      /
                                      2.0)
            offset = min(
                offset,
                (
                    max_feh - self.config.feh.value if side == 'plus'
                    else self.config.feh.value - min_feh
                )
            )

            result = []
            while offset > 0:
                result.append(offset)
                current_sf = distribution.sf(offset)
                offset = max(
                    offset - self.config.feh_max_step,
                    distribution.isf(current_sf + self.config.feh_max_cdf_step)
                )

            result = numpy.array(result)
            if side == 'minus':
                #False positive
                #pylint: disable=invalid-unary-operand-type
                return -result
                #pylint: enable=invalid-unary-operand-type
            return result[ : : -1]

        return numpy.concatenate(
            (get_half_grid_offsets('minus'),
             [0],
             get_half_grid_offsets('plus'))
        ) + self.config.feh.value

    def _get_initial_mass_grid(self, min_mass, max_mass):
        """
        Create the initial mass grid to start deriving the interpolation from.
        """

        return numpy.linspace(
            min_mass,
            max_mass,
            int(
                numpy.ceil((max_mass - min_mass) / self.config.mass_max_step)
            ) + 1
        )

    @staticmethod
    def _interpolate(feh_grid, mass_grid, values, target_feh, target_masses):
        """
        Evaluate an interpolation on given grids and values at given locations

        Args:
            feh_grid(1-D array):    The [Fe/H] values of the grid over which
                interpolation is to be defined.

            mass_grid(1-D array):     The stellar masses at which the
                interpolation is to be defined.

            values(2-D array):     The known values of the function to
                interpolate at each [Fe/H], mass combination from the above
                grids. The first index should correspond to feh_grid and the
                second to mass_grid.

            target_feh(1-D array):    New [Fe/H] values where the interpolation
                is to be evaluated.

            target_masses(1-D array):    New stellar masses where the
                interpolation is to be evaluated.

        Returns:
            2-D array:
                The values predicted by interpolation of all possible
                combinations of target_feh and target_masses entries. First
                index is alonge tirget_feh and the second is along
                target_massses.
        """

        return RectBivariateSpline(
            feh_grid, mass_grid, values, kx=1, ky=1
        )(
            target_feh,
            target_masses,
            grid=True
        )

    @staticmethod
    def _select_mismatches_all(calculated_values,
                               interpolated_values,
                               tolerance):
        """Return [Fe/H] and mass indices to refine."""

        return numpy.nonzero(
            numpy.absolute(calculated_values - interpolated_values)
            >
            tolerance
        )

    @staticmethod
    def _select_mismatches_worst(calculated_values,
                                 interpolated_values,
                                 tolerance):
        """Return [Fe/H], mass index of worst mismatch as 1-element arrays."""

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

        return tuple(
            numpy.array([ind]) for ind in worst_index
        )

    def _get_mismatch_indices(self, values, tolerance, debug_title):
        """
        Return indices where interpolating values fails on current grid.

        Args:
            values:    The values of the quantity being interpolated at
                the current grid. The first index should iterate over [Fe/H]
                and the second over mass.

        Returns:
            1-D int array:
                Indices within :attr:`self._feh_grid` where interpolation
                using 1/4 of the grid is discrepant with the other 3/4 of
                the values.

            1-D int array:
                Indices within :attr:`self._mass_grid` where interpolation
                fails.
        """

        mismatches = (numpy.array([], dtype=int), numpy.array([], dtype=int))
        for feh_offset in [0, 1]:
            for mass_offset in [0, 1]:
                if feh_offset == mass_offset == 0:
                    continue

                calculated_values = values[feh_offset : : 2, mass_offset : : 2]
                interpolated_values = self._interpolate(
                    self._feh_grid[ : : 2],
                    self._mass_grid[ : : 2],
                    values[ : : 2, : : 2],
                    self._feh_grid[feh_offset : : 2],
                    self._mass_grid[mass_offset : : 2],
                )
                self._plot_interpolation_performance(
                    calculated_values=calculated_values,
                    interpolated_values=interpolated_values,
                    feh_grid=self._feh_grid[feh_offset : : 2],
                    mass_grid=self._mass_grid[mass_offset : : 2],
                    interp_data=values[ : : 2, : : 2],
                    title=(debug_title
                           +
                           ' feh_di=%d, m_di=%d' % (feh_offset, mass_offset))
                )

                new_mismatches = getattr(
                    self,
                    '_select_mismatches_' + self.config.grid_refine_algorithm
                )(calculated_values, interpolated_values, tolerance)

                mismatches = tuple(
                    numpy.unique(
                        numpy.concatenate((
                            old,
                            2 * new + offset
                        ))
                    )
                    for new, old, offset in zip(new_mismatches,
                                                mismatches,
                                                [feh_offset, mass_offset])
                )

        return mismatches

    def _eval_age_cdfs(self, age):
        """Return the same shape as self._age_cdf but evaluated at age."""

        return numpy.array(
            [
                [
                    float(age_cdf(age)) for age_cdf in feh_age_cdf_funcs
                ]
                for feh_age_cdf_funcs in self._age_cdf
            ]
        )


    def _get_grid_refinement(self):
        """
        Return [Fe/H] and masses to add to grid to improve the interpolation.

        Args:
            None

        Returns:
            (1-D float array, 1-D int array):
                Sorted extra [Fe/H] values to add to the grid near places where
                the interpolation precision is insufficient and the number of
                current [Fe/H] grid nodes smaller than the corresponding new
                [Fe/H] value.

            (1-D float array, 1-D int array):
                Same as above but for the mass grid.
        """

        def get_new_grid_points(mismatch_indices, current_grid):
            """Return new values to add per the given mismatch indices."""

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
            return (
                0.5 * (current_grid[below_indices]
                       +
                       current_grid[below_indices + 1]),
                below_indices + 1
            )

        mismatch_indices = list(
            self._get_mismatch_indices(
                self._mass_cdf,
                self.config.mass_cdf_interp_tolerance,
                'CDF(M)'
            )
        )

        for age in 10.0**(self.config.age_cdf_check_log_ages):
            new_mismatches = self._get_mismatch_indices(
                self._eval_age_cdfs(age),
                self.config.age_cdf_interp_tolerance,
                'CDF(t=%g)' % age
            )
            mismatch_indices = [
                numpy.unique(numpy.concatenate((old, new)))
                for old, new in zip(mismatch_indices, new_mismatches)
            ]
            print('Mismatch indices: ' + repr(mismatch_indices))

        return [
            get_new_grid_points(*args)
            for args in zip(mismatch_indices, [self._feh_grid, self._mass_grid])
        ]

    def _tune_grid_resolution(self):
        """Increase grid resolution until interpolation tolerances are met."""

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

            new_feh_grid, new_mass_grid = (
                add_grid_points(args[0], *args[1])
                for args in zip(
                    (self._feh_grid, self._mass_grid), grid_refinement
                )
            )
            print('New [Fe/H] grid: ' + repr(new_feh_grid.T))
            print('New M* grid: ' + repr(new_mass_grid.T))

            age_cdfs_to_add = self._calculate_age_cdfs(grid_refinement[1][0],
                                                       self._feh_grid)
            new_mass_old_feh_age_cdfs = [[None] * new_mass_grid.size
                                         for feh in self._feh_grid]
            for feh_index in range(self._feh_grid.size):
                insert_entries(self._age_cdf[feh_index],
                               age_cdfs_to_add[feh_index],
                               grid_refinement[1][1],
                               new_mass_old_feh_age_cdfs[feh_index])

            age_cdfs_to_add = self._calculate_age_cdfs(new_mass_grid,
                                                       grid_refinement[0][0])
            self._age_cdf = [[None] * new_mass_grid.size
                             for feh in new_feh_grid]
            insert_entries(new_mass_old_feh_age_cdfs,
                           age_cdfs_to_add,
                           grid_refinement[0][1],
                           self._age_cdf)

            self._mass_grid = new_mass_grid
            self._feh_grid = new_feh_grid

            self._update_mass_cdfs()

    def _prepare_new_sampler(self):
        """
        Build new sampler of the given log_likelihood satisfying given config.

        Args:
            See :meth:`__init__`.

        Returns:
            None
        """

        self._debug_plots = dict(self.config.debug_plot)
        self._feh_grid = self._get_initial_feh_grid(
            float(self._log_likelihood.interpolator.track_feh[0]),
            float(self._log_likelihood.interpolator.track_feh[-1])
        )
        self._mass_grid = self._get_initial_mass_grid(
            float(self._log_likelihood.interpolator.track_masses[0]),
            float(self._log_likelihood.interpolator.track_masses[-1])
        )
        self._plot_initial_feh_grid()

        self._age_cdf = self._calculate_age_cdfs(self._mass_grid,
                                                 self._feh_grid)
        self._update_mass_cdfs()

        self._tune_grid_resolution()

    def _check_for_pickled(self):
        """
        Check for a pre-pickled sampler for the given log_likelihood and config.

        Args:
            see :meth:`__init__`

        Returns:
            bool:
                Whether a matching sampler was found.
        """

        def compare_config(pickled_config):
            """Return True iff the pickled config matches input config."""

            ignore_args = ['pickle_fname',
                           'grid_refine_algorithm',
                           'num_parallel_processes',
                           'debug_plot',
                           'debug_plot_dpi',
                           'stellar_evolution_interpolator_dir']

            pickled_cfg_dict = dict(vars(pickled_config))
            input_cfg_dict = dict(vars(self.config))

            for arg in ignore_args:
                if arg in pickled_cfg_dict:
                    del pickled_cfg_dict[arg]
                if arg in input_cfg_dict:
                    del input_cfg_dict[arg]

            return pickled_cfg_dict == input_cfg_dict


        try:
            with open(self.config.pickle_fname, 'rb') as pickle_file:
                unpickler = Unpickler(pickle_file)
                while True:
                    section, nobjects = unpickler.load()
                    assert isinstance(section, str)
                    assert isinstance(nobjects, int)
                    if section == 'StarSampler':
                        assert nobjects == 6
                        nobjects -= 1
                        if not compare_config(unpickler.load()):
                            break
                        nobjects -= 1
                        if self._log_likelihood != unpickler.load():
                            break

                        self._mass_cdf = unpickler.load()
                        self._age_cdf = unpickler.load()
                        self._feh_grid = unpickler.load()
                        self._mass_grid = unpickler.load()
                        return True

                    for _ in range(nobjects):
                        unpickler.load()
        except UnpicklingError:
            return False

    def _add_to_pickle_file(self):
        """Append the current sampler to the pickle file specified in config."""

        with open(self.config.pickle_fname, 'ab') as pickle_file:
            pickler = Pickler(pickle_file)
            pickler.dump(('StarSampler', 6))
            pickler.dump(self.config)
            pickler.dump(self._log_likelihood)
            pickler.dump(self._mass_cdf)
            pickler.dump(self._age_cdf)
            pickler.dump(self._feh_grid)
            pickler.dump(self._mass_grid)

    def __init__(self, log_likelihood, config):
        """
        Find sampler for the given log_likelihood satisfying the given config.

        Args:
            log_likelihood(LogLikelihoodBase):     The log-likelihood function
                to base sampling on. It should not include the direct
                measurement of the metallicity specified through config.

            config:    The configuration specifying what is considered good
                approximation to the true distributoin. See command lines of
                `prepare.py` executable.

        Returns:
            None
        """

        self.config = config
        self._log_likelihood = log_likelihood
        self._mass_cdf = None
        self._age_cdf = None
        self._feh_grid = None
        self._mass_grid = None
        self._grid_refinement_iteration = None

        if not self._check_for_pickled():
            self._prepare_new_sampler()
            self._add_to_pickle_file()
