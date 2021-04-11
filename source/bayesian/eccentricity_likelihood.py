#!/usr/bin/env python3
"""Implement likelihood of predicted eccentricity per observed and envelope."""

import logging
from matplotlib import pyplot
import numpy

from split_normal_distribution import split_normal

#Intended to simply provide callable. No need for further public methods.
#pylint: disable=too-few-public-methods
class EccentricityLikelihood():
    """Final eccentricity likelihood given measured and envelope eccen. CDFs."""

    _logger = logging.getLogger(__name__)

    def __init__(self,
                 observed_eccentricity,
                 envelope_eccentricity=1.0):
        """
        Prepare the likelihood for evaluation.

        Args:
            observed_eccentricity:    The distribution of the present day
                eccentricity of the system. Should provide
                `scipy.stats.rv_continuous` interface.

            envelope_eccentricity:    Either a single float or another
                distribution specifying the envelope eccentricity.

        Returns:
            None
        """

        self.envelope_eccentricity = envelope_eccentricity
        self.observed_eccentricity = observed_eccentricity

    def __call__(self, e_predicted):
        """Return the likelihood evaluated at specified present day eccen."""

        assert 0 <= e_predicted <= 1

        if (
                not hasattr(self.envelope_eccentricity, 'pdf')
                and
                e_predicted > self.envelope_eccentricity
        ):
            return 0.0

        result = self.observed_eccentricity.cdf(e_predicted)
        if hasattr(self.envelope_eccentricity, 'sf'):
            result *= self.envelope_eccentricity.sf(e_predicted)

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
        ),
        split_normal.freeze_error_bar(
            mode=0.2,
            abs_plus_error=0.1,
            abs_minus_error=0.05
        )
    ]

    e_pdf = numpy.vectorize(
        EccentricityLikelihood(
            observed_eccentricity=e_now_distros[0],
        )
    )
    pyplot.plot(plot_e, e_pdf(plot_e), '-r')
    pyplot.show()
    e_pdf = numpy.vectorize(
        EccentricityLikelihood(
            observed_eccentricity=e_now_distros[0],
            envelope_eccentricity=0.5
        )
    )
    pyplot.plot(plot_e, e_pdf(plot_e), '-r')
    pyplot.show()

    e_pdf = numpy.vectorize(
        EccentricityLikelihood(
            observed_eccentricity=e_now_distros[1],
            envelope_eccentricity=0.5,
        )
    )
    pyplot.plot(plot_e, e_pdf(plot_e), '-r')
    pyplot.show()
    e_pdf = numpy.vectorize(
        EccentricityLikelihood(
            observed_eccentricity=e_now_distros[2],
            envelope_eccentricity=split_normal.freeze_error_bar(
                mode=0.5,
                abs_plus_error=0.1,
                abs_minus_error=0.1
            )
        )
    )
    pyplot.plot(plot_e, e_pdf(plot_e), '-r')
    pyplot.show()
    e_pdf = numpy.vectorize(
        EccentricityLikelihood(
            observed_eccentricity=e_now_distros[3],
            envelope_eccentricity=split_normal.freeze_error_bar(
                mode=0.5,
                abs_plus_error=0.1,
                abs_minus_error=0.1
            )
        )
    )
    pyplot.plot(plot_e, e_pdf(plot_e), '-r')
    pyplot.show()
