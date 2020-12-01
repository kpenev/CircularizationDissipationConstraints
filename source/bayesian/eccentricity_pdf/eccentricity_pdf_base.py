"""Implement PDF of predicted eccentricity per observed and envelope."""

from abc import ABCMeta, abstractmethod

import scipy.integrate

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

    def __init__(self, integration_options=None):
        """
        Prepare the PDF for evaluation.

        Args:
            integration_options:    The `opts` argument to pass to
                `scipy.integrate.nquad`.
        """

        self.integration_options = integration_options


    @abstractmethod
    def observed_eccentricity(self, e_now):
        """The PDF of the present day eccentricity per observation data."""

    def __call__(self, e_predicted):
        """Return the PDF evaluated at specified present day eccentricity."""

        assert 0 <= e_predicted <= 1

        if callable(self.envelope_eccentricity):
            result, abserr = scipy.integrate.nquad(
                self._integrand_envelope_pdf,
                ((0, e_predicted), (e_predicted, 1)),
                opts=self.integration_options
            )
        else:
            if e_predicted > self.envelope_eccentricity:
                return 0.0
            result, abserr = scipy.integrate.quad(
                self._integrand_envelope_value,
                0,
                e_predicted,
                **(self.integration_options or dict())
            )

        if self.integration_options:
            assert abserr < max(
                self.integration_options.get('epsabs', 1e-7),
                self.integration_options.get('epsrel', 1e-7) * result
            )
        else:
            assert abserr < max(1e-7, 1e-7 * result)

        return result
