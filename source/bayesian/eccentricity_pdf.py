#!/usr/bin/env python3
"""Implement PDF of predicted eccentricity per observed and envelope."""

import logging
from pickle import Pickler, Unpickler
import os.path

from matplotlib import pyplot
import scipy.integrate
from scipy.interpolate import InterpolatedUnivariateSpline
import numpy

from split_normal_distribution import split_normal

#Intended to simply provide callable. No need for further public methods.
#pylint: disable=too-few-public-methods
class EccentricityPDF():
    """Final eccentricity PDF agnostic of measured and envelope eccen. PDFs."""

    _logger = logging.getLogger(__name__)

    def _integrand_envelope_pdf(self, e_observed, e_envelope):
        """Integrand when both observed and envelope ecc. are PDFs."""

        return (
            self.observed_eccentricity.pdf(e_observed)
            *
            self.envelope_eccentricity.pdf(e_envelope)
            /
            (e_envelope - e_observed)
        )

    def _integrand_envelope_value(self, e_observed):
        """Integrand when the envelope is only a single value."""

        return (
            self.observed_eccentricity.pdf(e_observed)
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

        max_e = (1 if hasattr(self.envelope_eccentricity, 'pdf')
                 else self.envelope_eccentricity)

        (
            required_abs_precision,
            required_rel_precision
        ) = self._get_required_precision()


        old_grid = numpy.linspace(0, max_e, 50)
        calculate_values = scipy.vectorize(self.__call__)
        old_values = calculate_values(old_grid)
        interpolation = InterpolatedUnivariateSpline(old_grid, old_values)

        for refinement in range(max_refinements):
            grid_size = 2 * old_grid.size - 1
            grid = numpy.linspace(0, max_e, grid_size)

            new_values = calculate_values(grid[1::2])
            interpolated_new_values = interpolation(grid[1::2])

            values = numpy.empty(grid_size)
            values[0::2] = old_values
            values[1::2] = new_values

            interpolation = InterpolatedUnivariateSpline(grid, values)


            self._logger.debug(
                'Refinement %d: max error overshoot: %25.16e',
                refinement,
                numpy.amax(
                    numpy.abs(new_values - interpolated_new_values)
                    /
                    numpy.maximum(required_abs_precision,
                                  required_rel_precision * new_values)
                )
            )

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

    def _check_for_pickled(self, pickle_fname):
        """
        Check if a pre-pickled interpolation exists for the current setup.

        Args:
            None

        Returns:
            bool:
                Whether a matching interpolation was found.
        """

        def compare_next(pickled_observed_e,
                         pickled_envelope_e,
                         pickled_integration_opts):
            """Check if the next pickled eccentricity PDF matches."""

            if pickled_observed_e.kwds != self.observed_eccentricity.kwds:
                return False
            if hasattr(self.envelope_eccentricity, 'pdf'):
                if(
                        getattr(pickled_envelope_e, 'kwds', None)
                        !=
                        self.envelope_eccentricity.kwds
                ):
                    return False
            else:
                if pickled_envelope_e != self.envelope_eccentricity:
                    return False
            return pickled_integration_opts == self.integration_options

        if pickle_fname is None:
            return False

        if not os.path.exists(pickle_fname):
            return False

        try:
            with open(pickle_fname, 'rb') as pickle_file:
                unpickler = Unpickler(pickle_file)
                while True:
                    section, nobjects = unpickler.load()
                    if section == 'EccentricityPDF':
                        assert nobjects == 4
                        nobjects -= 3
                        if compare_next(unpickler.load(),
                                        unpickler.load(),
                                        unpickler.load()):
                            self._logger.info(
                                'Found matching pickled eccentricity PDF.'
                            )
                            self._interpolation = unpickler.load()
                            return True

                    for _ in range(nobjects):
                        unpickler.load()
        except EOFError:
            self._logger.info('No matching pickled eccentricity PDF found.')
            return False

    def _pickle_interpolation(self, pickle_fname):
        """Appends a pickle of the currently set-up interpolation a file."""

        if pickle_fname is None:
            return

        with open(pickle_fname, 'ab') as pickle_file:
            pickler = Pickler(pickle_file)
            pickler.dump(('EccentricityPDF', 4))
            pickler.dump(self.observed_eccentricity)
            pickler.dump(self.envelope_eccentricity)
            pickler.dump(self.integration_options)
            pickler.dump(self._interpolation)

    def __init__(self,
                 observed_eccentricity,
                 envelope_eccentricity=1.0,
                 *,
                 integration_options=None,
                 pickle_fname=None):
        """
        Prepare the PDF for evaluation.

        Args:
            observed_eccentricity:    The distribution of the present day
                eccentricity of the system. Should provide
                `scipy.stats.rv_continuous` interface.

            envelope_eccentricity:    Either a single float or another
                distribution specifying the envelope eccentricity.

            integration_options:    The `opts` argument to pass to
                `scipy.integrate.nquad`.

            pickled_fname:    If not None, this should be the name of a file to
                load/save pre-computed interpolaiton with the given
                configuration from/to.

        Returns:
            None
        """

        self.envelope_eccentricity = envelope_eccentricity
        self.observed_eccentricity = observed_eccentricity

        self.integration_options = integration_options

        self._interpolation = None

        if not self._check_for_pickled(pickle_fname):
            self._interpolation = self._compute_interpolation()
            self._pickle_interpolation(pickle_fname)

    def __call__(self, e_predicted):
        """Return the PDF evaluated at specified present day eccentricity."""

        assert 0 <= e_predicted <= 1

        if (
                not hasattr(self.envelope_eccentricity, 'pdf')
                and
                e_predicted > self.envelope_eccentricity
        ):
            return 0.0

        if self._interpolation is not None:
            return self._interpolation(e_predicted)

        if hasattr(self.envelope_eccentricity, 'pdf'):

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
#pylint: enable=too-few-public-methods

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    plot_e = numpy.linspace(0, 1, 100)

    e_now_distros = [
        split_normal.freeze_error_bar(
            mode=0.2,
            abs_plus_error=0.1,
            abs_minus_error=0.05
        ),
        split_normal.freeze_error_bar(
            mode=0.2,
            abs_plus_error=0.01,
            abs_minus_error=0.008
        ),
        split_normal.freeze_error_bar(
            mode=0.3,
            abs_plus_error=0.1,
            abs_minus_error=0.05
        )
    ]

    e_pdf = numpy.vectorize(
        EccentricityPDF(
            observed_eccentricity=e_now_distros[0],
            pickle_fname='test_eccentricity_pdf.pkl'
        )
    )
    pyplot.plot(plot_e, e_pdf(plot_e), '-r')
    pyplot.show()
    e_pdf = numpy.vectorize(
        EccentricityPDF(
            observed_eccentricity=e_now_distros[1],
            envelope_eccentricity=0.5,
            pickle_fname='test_eccentricity_pdf.pkl'
        )
    )
    pyplot.plot(plot_e, e_pdf(plot_e), '-r')
    pyplot.show()
    e_pdf = numpy.vectorize(
        EccentricityPDF(
            observed_eccentricity=e_now_distros[2],
            envelope_eccentricity=split_normal.freeze_error_bar(
                mode=0.5,
                abs_plus_error=0.1,
                abs_minus_error=0.1
            ),
            pickle_fname='test_eccentricity_pdf.pkl'
        )
    )
    pyplot.plot(plot_e, e_pdf(plot_e), '-r')
    pyplot.show()
