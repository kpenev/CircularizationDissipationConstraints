"""combine constraints in a moving window."""

import numpy
import scipy.integrate
from scipy.stats import norm

class MovingWindowConstraints:
    """Calcualte combined constraints in a moving window on indep. varable."""

    @staticmethod
    def _combine_constraints(ymin,
                             ymax,
                             ymin_stdev,
                             ymax_stdev,
                             range_quantiles,
                             *,
                             pdf_sample_npoints=1025,
                             range_sigma_factor=3):
        """Combine the given constraints into a (min, max) showing 1-sigma."""

        def window_pdf(y):
            """The combined PDF of given constsraints (unnormalized)."""

            result = numpy.zeros(y.shape, dtype=float)
            for index in range(ymin.size):
                try:
                    min_sigma = float(ymin_stdev)
                except TypeError:
                    min_sigma = ymin_stdev[index]

                try:
                    max_sigma = float(ymax_stdev)
                except TypeError:
                    max_sigma = ymax_stdev[index]

                result -= (
                    numpy.square(
                        numpy.minimum((y - ymin[index]) / min_sigma, 0.0)
                    )
                    +
                    numpy.square(
                        numpy.minimum((ymax[index] - y) / max_sigma, 0.0)
                    )
                )
            print('Log(Window pdf) = ' + repr(result))
            return numpy.exp(result)

        def solve_linear_interp(y, x_below, y_below, x_above, y_above):
            """Find x at which the line through given points has the given y."""

            return (x_below
                    +
                    (y - y_below) * (x_above - x_below) / (y_above - y_below))

        def get_eval_bounds(sigma_factor):
            """Select range over which to evaluate the pdf."""

            eval_bounds = [
                (
                    ymin
                    -
                    sigma_factor * ymin_stdev
                ),
                (
                    ymax
                    +
                    sigma_factor * ymax_stdev
                )
            ]
            print('Bounds data: ' + repr(eval_bounds))
            for bound_ind, bound in enumerate(eval_bounds):
                finite_bound = numpy.isfinite(bound)
                if finite_bound.any():
                    eval_bounds[bound_ind] = bound[finite_bound].min()
                else:
                    eval_bounds[bound_ind] = numpy.nan

            print('First guess bounds: ' + repr(eval_bounds))

            result_type = 'two sided'
            for bound_ind, bound in enumerate(eval_bounds):
                if not numpy.isfinite(bound):
                    result_type = 'lower limit' if bound_ind else 'upper limit'
                    eval_bounds[bound_ind] = eval_bounds[1 - bound_ind]

            print('Final bounds: ' + repr(eval_bounds))

            return eval_bounds, result_type

        print('Selected ymin: ' + repr(ymin))
        print('Selected ymax: ' + repr(ymax))
        print('Selected ymin_stdev: ' + repr(ymin_stdev))
        print('Selected ymax_stdev: ' + repr(ymax_stdev))

        if ymin.size == 0:
            return -numpy.inf, numpy.inf

        eval_bounds, result_type = get_eval_bounds(range_sigma_factor)
        if eval_bounds[0] == eval_bounds[-1]:
            constraint_range = get_eval_bounds(1.0)[0]
        else:
            print('Eval bounds: ' + repr(eval_bounds))
            eval_y = numpy.linspace(*eval_bounds, pdf_sample_npoints)

            print('Eval y: ' + repr(eval_y))

            pdf_values = window_pdf(eval_y)
            print('PDF values: ' + repr(pdf_values))

            cdf_values = scipy.integrate.cumtrapz(pdf_values,
                                                  eval_y,
                                                  initial=0)

            cdf_values /= cdf_values[-1]

            print('CDF values:  ' + repr(cdf_values))
            print('Target quantiles: ' + repr(range_quantiles))

            limit_indices = numpy.searchsorted(cdf_values, range_quantiles)

            assert limit_indices[0] > 0
            assert limit_indices[1] > limit_indices[0]

            constraint_range = tuple(
                solve_linear_interp(range_quantiles[i],
                                    eval_y[limit_indices[i] - 1],
                                    cdf_values[limit_indices[i] - 1],
                                    eval_y[limit_indices[i]],
                                    cdf_values[limit_indices[i]])
                for i in range(2)
            )
        if result_type == 'two sided':
            return constraint_range
        elif result_type == 'lower limit':
            return constraint_range[0], numpy.inf
        else:
            assert result_type == 'upper limit'
            return -numpy.inf, constraint_range[1]


    def __init__(self,
                 x,
                 ymin,
                 ymax,
                 window_width,
                 *,
                 ymin_stdev=0.0,
                 ymax_stdev=0.0,
                 range_quantiles=norm.cdf([-1, 1])):
        """Calculate the moving window constraints and their breaks."""

        def combine_constraints(window_center):
            """Combine the all constraints with x values within a window."""

            selected_points = numpy.logical_and(
                x >= window_center - window_width / 2.0,
                x <= window_center + window_width / 2.0
            )
            selected_points = numpy.logical_and(
                selected_points,
                numpy.logical_or(
                    numpy.isfinite(ymin),
                    numpy.isfinite(ymax)
                )
            )
            try:
                selected_ymin_stdev = float(ymin_stdev)
            except TypeError:
                assert ymin_stdev.shape == x.shape
                selected_ymin_stdev = ymin_stdev[selected_points]

            try:
                selected_ymax_stdev = float(ymax_stdev)
            except TypeError:
                assert ymax_stdev.shape == x.shape
                selected_ymax_stdev = ymax_stdev[selected_points]

            result = self._combine_constraints(
                ymin[selected_points],
                ymax[selected_points],
                selected_ymin_stdev,
                selected_ymax_stdev,
                self.range_quantiles
            )
            print('Constraint for x=%s is: %s'
                  %
                  (repr(window_center), repr(result)))
            return result

        assert x.shape == ymin.shape
        assert x.shape == ymax.shape

        self.range_quantiles = range_quantiles
        self._breaks = numpy.unique(
            numpy.concatenate([x - window_width / 2.0,
                               x + window_width / 2.0])
        )

        self._constraints = numpy.vectorize(
            combine_constraints
        )(
            0.5 * (self._breaks[1:] + self._breaks[:-1])
        )

    def __call__(self, x):
        """Return the contraint interval at the given x value."""

        range_index = numpy.searchsorted(self._break, x) - 1

        assert range_index >= 0

        return (constraint[range_index] for constraint in self._constraints)

    def get_plot_arguments(self, minx=-numpy.inf, maxx=numpy.inf):
        """Return pyplot.fill_betwen x, y1, and y2 args to show constraint."""


        plot_x = numpy.array([self._breaks, self._breaks]).T.flatten()[:-1]
        selected = numpy.logical_and(plot_x > minx, plot_x < maxx)

        return (
            (plot_x[selected],)
            +
            tuple(
                numpy.concatenate([
                    constraint[0:1],
                    numpy.array([constraint, constraint]).T.flatten()
                ])[selected]
                for constraint in self._constraints
            )
        )
