"""Implement PDF of predicted eccentricity per observed and envelope."""

from abc import ABCMeta, abstractmethod
import pickle

import scipy.integrate
from scipy.interpolate import InterpolatedUnivariateSpline
import scipy
import numpy

class EccentricityPDFBase(metaclass=ABCMeta):
    """Final eccentricity PDF agnostic of measured and envelope eccen. PDFs."""

    envelope_eccentricity = 1.0

    def _integrand_envelope_pdf(self, e_observed, e_envelope):
        """Integrand when both observed and envelope ecc. are PDFs."""

        return (
            self.observed_eccentricity(e_observed)
            *
            #Only used if envelope_eccentricity is overwritten by callable
            #pylint: disable=not-callable
            self.envelope_eccentricity(e_envelope)
            #pylint: enable=not-callable
            /
            (e_envelope - e_observed)
        )

    def _integrand_envelope_value(self, e_observed):
        """Integrand when the envelope is only a single value."""

        return (
            self.observed_eccentricity(e_observed)
            /
            (self.envelope_eccentricity - e_observed)
        )

    def _get_required_precision(self):
        """Return the (absolute, relative) precision to calculate the PDF to."""

        default_precision = 1.5e-8

        if self.integration_options:
            return (
                100.0 * self.integration_options.get('epsabs',
                                                     default_precision),
                100.0 * self.integration_options.get('epsrel',
                                                     default_precision)
            )

        return 100.0 * default_precision, 100.0 * default_precision

    def _compute_interpolation(self, max_refinements=10):
        """Find an interpolation of sufficient precision."""

        max_e = (1 if callable(self.envelope_eccentricity)
                 else self.envelope_eccentricity)

        (
            required_abs_precision,
            required_rel_precision
        ) = self._get_required_precision()


        old_grid = scipy.linspace(0, max_e, 50)
        calculate_values = scipy.vectorize(self.__call__)
        old_values = calculate_values(old_grid)
        interpolation = InterpolatedUnivariateSpline(old_grid, old_values)

        for refinement in range(max_refinements):
            grid_size = 2 * old_grid.size - 1
            grid = scipy.linspace(0, max_e, grid_size)

            new_values = calculate_values(grid[1::2])
            interpolated_new_values = interpolation(grid[1::2])

            values = scipy.empty(grid_size)
            values[0::2] = old_values
            values[1::2] = new_values

            interpolation = InterpolatedUnivariateSpline(grid, values)


            print('Refinement %d: max error overshoot: %25.16e'
                  %
                  (
                      refinement,
                      scipy.amax(
                          numpy.abs(new_values - interpolated_new_values)
                          /
                          numpy.maximum(required_abs_precision,
                                        required_rel_precision * new_values)
                      )
                  ))

            if (
                    numpy.abs(new_values - interpolated_new_values)
                    <
                    numpy.maximum(required_abs_precision,
                                  required_rel_precision * new_values)
            ).all():
                return interpolation

            old_grid = grid
            old_values = values

        raise RuntimeError(
            'Failed to find interpolation with sufficient precision.'
        )

    def __init__(self,
                 integration_options=None,
                 load_interp_from=None,
                 save_interp_to=None):
        """
        Prepare the PDF for evaluation.

        Args:
            integration_options:    The `opts` argument to pass to
                `scipy.integrate.nquad`.

            load_interp_from:    If not None, this should be a file open for
                binary reading to load a pickle of a pre-computed interpolation
                from. In this case, :meth:`__call__` will return the value of
                the loaded interpolation instead of computing the required
                integrals.

            save_interp_to:    If not None, this should be a file open for
                binary writing. In this case, the function derives an
                interpolation by calling itself on an adaptively refined grid of
                points and saves that interpolation to the file. The
                :meth:`__call__` will return the value of the calculated
                interpolation instead of computing the required integrals.
        """

        self.integration_options = integration_options

        self._interpolation = None

        assert load_interp_from is None or save_interp_to is None

        if load_interp_from is not None:
            self._interpolation = pickle.load(load_interp_from)

        if save_interp_to is not None:
            self._interpolation = self._compute_interpolation()
            pickle.dump(self._interpolation, save_interp_to)

    @abstractmethod
    def observed_eccentricity(self, e_now):
        """The PDF of the present day eccentricity per observation data."""

    def __call__(self, e_predicted):
        """Return the PDF evaluated at specified present day eccentricity."""

        assert 0 <= e_predicted <= 1

        if (
                not callable(self.envelope_eccentricity)
                and
                e_predicted > self.envelope_eccentricity
        ):
            return 0.0

        if self._interpolation is not None:
            return self._interpolation(e_predicted)

        if callable(self.envelope_eccentricity):

            result, abserr = scipy.integrate.nquad(
                self._integrand_envelope_pdf,
                ((0, e_predicted), (e_predicted, 1)),
                opts=self.integration_options
            )
        else:
            result, abserr = scipy.integrate.quad(
                self._integrand_envelope_value,
                0,
                e_predicted,
                **(self.integration_options or dict())
            )

        (
            required_abs_precision,
            required_rel_precision
        ) = self._get_required_precision()

        max_abserr = max(required_abs_precision,
                         required_rel_precision * result)

        if abserr > max_abserr:
            raise RuntimeError(
                'Integration error too large: f(e = %g) = %g +- %g (> %g).'
                %
                (e_predicted, result, abserr, max_abserr)
            )

        return result
