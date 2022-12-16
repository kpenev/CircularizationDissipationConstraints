"""Define eccentricity distribution from samples of e cos(w) and e sin(w)."""

import numpy
from scipy import stats
from scipy.integrate import quad

class EccentricityDistributionSamples(stats.rv_continuous):
    """Implement KDE(e cos(w), e sin(w)) marginalized over w."""

    @staticmethod
    def _kernel_arg(samples, x):
        """Get difference between samples (inner index) and x (outer index)."""

        rhs, lhs = numpy.meshgrid(samples, x)
        return numpy.squeeze(lhs - rhs)


    def _pdf_integrand(self, periapsis, eccentricity):
        """The function to integrate over periapsis to calculate the PDF."""

        return (
            self._cos_kernel.pdf(
                self._kernel_arg(
                    self._cos_samples,
                    eccentricity * numpy.cos(periapsis)
                )
            )
            *
            self._sin_kernel.pdf(
                self._kernel_arg(
                    self._sin_samples,
                    eccentricity * numpy.sin(periapsis)
                )
            )
        )

    #This is scipy intended usage
    #pylint: disable=arguments-differ
    def _pdf(self, x):
        """Marginalize over periapsis to get pdf(eeccentricity)."""

        return quad(self._pdf_integrand,
                    0,
                    2.0 * numpy.pi,
                    args=(x,))[0]
    #pylint: enable=arguments-differ

    def __init__(self,
                 sin_samples,
                 cos_samples,
                 sin_kernel=None,
                 cos_kernel=None):
        """
        Set the samples and the kernels to convolve them with.

        Args:
            sin_samples:    A numpy array of the samples of ``e sin(w)``.

            cos_samples:    A numpy array of the samples of ``e cos(w)`` should
                match sin_samples.

            sin_kernel:    A continuous PDF to use as the kernel for
                ``e sin(w)``. If left unspecifided, Epanechnikov kernel is used
                with width of std(sample values) * sample_size^0.2 or 1e-4
                (whichever is greater).

            cos_kernel:    Same as sin_kernel but for ``e cos(w)``.

        Returns:
            None
        """

        def get_kernel(specified_kernel, samples):
            """Return the specified kernel or default one if None."""

            return (
                specified_kernel
                or
                stats.rdist(
                    c=4,
                    scale=max(
                        numpy.std(samples) * samples.size**(-0.2),
                        0.0001
                    )
                )
            )

        def update_support2(kernel, samples):
            """Return what to add to the square of each support limit."""

            kernel_min, kernel_max = kernel.support()
            lower, upper = (samples.min() + kernel_min,
                            samples.max() + kernel_max)
            lower2 = lower**2
            upper2 = upper**2

            return (
                min(lower2, upper2) if lower * upper > 0 else 0,
                max(lower2, upper2)
            )

        self._sin_samples = numpy.ravel(sin_samples)
        self._cos_samples = numpy.ravel(cos_samples)

        self._sin_kernel = get_kernel(sin_kernel, self._sin_samples)
        self._cos_kernel = get_kernel(cos_kernel, self._cos_samples)

        support_min2, support_max2 = update_support2(self._sin_kernel,
                                                     self._sin_samples)
        add_support = update_support2(self._sin_kernel, self._sin_samples)
        support_min2 += add_support[0]
        support_max2 += add_support[1]

        super().__init__(a=support_min2**0.5, b=support_max2**0.5)
