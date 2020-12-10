#!/usr/bin/env python3
"""Implement predicted eccen. PDF with normal observed and envelope eccen."""

from functools import partial

from scipy.stats import norm

from .eccentricity_pdf_base import EccentricityPDFBase

if __name__ == '__main__':
    from matplotlib import pyplot
    from numpy import linspace, vectorize

#Public methods defined in base class.
#pylint: disable=too-few-public-methods
class EccentricityPDFNormal(EccentricityPDFBase):
    """Final eccen. PDF assuming normal PDF for observed and envelope eccen."""

    def observed_eccentricity(self, e_now):
        """Evaluate the observed eccentricity PDF."""

        return norm.pdf(
            e_now,
            loc=self.e_observed_mean,
            scale=self.e_observed_stddev[0 if e_now > self.e_observed_stddev
                                         else 1]
        )

    def __init__(self,
                 e_observed_mean,
                 e_observed_stddev,
                 *,
                 e_envelope_mean=1.0,
                 e_envelope_stddev=0.0,
                 **base_args):
        """
        Set-up the PDF per the specified mean and standard deviations.

        Args:
            e_observed_mean:    The mean of the normal distribution giving the
                present day eccentricity PDF.

            e_observed_stddev:    The standard deviation of the normal
                distribution giving the present day eccentricity PDF. It can
                either be a single number in which case the error is assumet
                symmetrical or a 2-iterable of poisitive and negative error (in
                that order).

            e_envelope_mean:    The mean of the normal distribution giving the
                envelope eccentricity PDF.

            e_observed_stddev:    The standard deviation of the normal
                distribution giving the envelope eccentricity PDF. If zero, a
                delta function is assumed.

            base_args:    Passed directly to
                :meth:`EccentricityPDFBase.__init__`.

        Returns:
            None
        """

        self.e_observed_mean = e_observed_mean
        try:
            self.e_observed_stddev = (
                float(e_observed_stddev),
                float(e_observed_stddev)
            )
        except TypeError:
            assert len(e_observed_stddev) == 2
            self.e_observed_stddev = e_observed_stddev

        if e_envelope_stddev == 0:
            self.envelope_eccentricity = e_envelope_mean
        else:
            self.envelope_eccentricity = partial(norm.pdf,
                                                 loc=e_envelope_mean,
                                                 scale=e_envelope_stddev)

        super().__init__(**base_args)
#pylint: enable=too-few-public-methods

if __name__ == '__main__':
    plot_e = linspace(0, 1, 100)

    pdf = vectorize(EccentricityPDFNormal(0.3, 0.1))
    pyplot.plot(plot_e, pdf(plot_e))
    pyplot.show()

    pdf = vectorize(EccentricityPDFNormal(0.3, 0.01, e_envelope_mean=0.5))
    pyplot.plot(plot_e, pdf(plot_e))
    pyplot.show()

    pdf = vectorize(EccentricityPDFNormal(0.3, 0.1, e_envelope_mean=0.5))
    pyplot.plot(plot_e, pdf(plot_e))
    pyplot.show()

    pdf = vectorize(
        EccentricityPDFNormal(0.3,
                              0.1,
                              e_envelope_mean=0.5,
                              e_envelope_stddev=0.1)
    )
    pyplot.plot(plot_e, pdf(plot_e), '-g')

    with open('test_interp.pkl', 'rb') as interp_file:
        pdf = vectorize(
            EccentricityPDFNormal(0.3,
                                  0.1,
                                  e_envelope_mean=0.5,
                                  e_envelope_stddev=0.1,
                                  load_interp_from=interp_file)
        )
        pyplot.plot(plot_e, pdf(plot_e), ':r')
    pyplot.show()
