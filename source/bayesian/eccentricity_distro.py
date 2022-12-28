#!/usr/bin/env python3

from matplotlib import pyplot
from scipy import stats
from scipy.integrate import quad
import numpy

class EccentricityDistroUnnorm_gen(stats.rv_continuous):
    """Not normalized distribution to use for KDE(eccentricity) from samples."""

    def _pdf(self, x, s_to_n):
        return stats.rice.pdf(x, s_to_n) / x

    def _argcheck(self, s_to_n):
        return s_to_n >= 0


EccentricityDistroUnnorm = EccentricityDistroUnnorm_gen(
    name='EccentricityKernelUnnorm',
    a=0.0
)

class EccentricityDistro:

    def __init__(self, center, uncertainty):
        """Define kernel at the given location with given uncertainty."""

        s_to_n = center/uncertainty
        self._distro = EccentricityDistroUnnorm(s_to_n=s_to_n,
                                                scale=uncertainty,
                                                loc=0.0)
        self._zero_val = numpy.exp(-s_to_n**2/2) / uncertainty
        points = numpy.linspace(center - 10.0 * uncertainty,
                                center + 10.0 * uncertainty,
                                10)
        points = numpy.concatenate(([0], points[points>0]))
        self._norm = (
            quad(self._distro.pdf, a=0, b=points[-1], points=points)[0]
            +
            quad(self._distro.pdf, a=points[-1], b=numpy.inf)[0]
        )

    def pdf(self, x):
        """Evaluate the PDF fixing x=0 and x<0 values."""

        x = numpy.atleast_1d(x)
        result = self._distro.pdf(x)
        result[x==0] = self._zero_val
        result[x<0] = 0.0
        result /= self._norm
        return result

    def cdf(self, x):
        """Evaluate the CDF fixing x=0 and x<0 values."""

        x = numpy.atleast_1d(x)
        result = self._distro.cdf(x) / self._norm
        result[x<=0] = 0.0
        return result

if __name__ == '__main__':
    val=0.5
    unc=0.001
    e_distro = EccentricityDistro(val, unc)
    plot_x = numpy.linspace(0.0, val + 5.0 * unc, 1000)#0.9 * loc, 1.1 * loc, 100)
    pyplot.plot(
        plot_x,
        e_distro.pdf(plot_x),
        label='e-distro PDF'
    )
    pyplot.plot(
        plot_x,
        e_distro.cdf(plot_x),
        label='e-distro CDF'
    )
    pyplot.axhline(y=1)

#    pyplot.plot(
#        plot_x,
#        gauss.pdf(plot_x) / loc,
#        label='Gauss'
#    )
    pyplot.legend()
    pyplot.show()
