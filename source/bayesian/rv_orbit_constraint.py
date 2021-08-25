"""Class that incorporates RV constraints on binary orbits."""

class RVOrbitConstraint:
    """Conveniently and efficiency incorporate RV constraint of SB1 orbits."""

    def _pdf_inclination_integrand(self,
                                   cos_inclination,
                                   max_rv_semiamplitude):
        """
        The integrand for marginalizing the PDF over inclination.

        Args:
            cos_inclination:    The value of cos(i), where i is the angle
                between the orbital angular momentum and the line of sight.

            max_rv_semiamplitude:    The semiamplitude that would be observed
                for the given orbit if it were viewed edge on.

        Returns:
            float:
                PDF of observing RV given the specified orbit.
        """

        return self._observed_rvk.pdf(
            max_rv_semiamplitude
            *
            numpy.sqrt(1.0 - numpy.square(cos_inclination))
        )

    def _cdf_inclination_integrand(self, sin_inclination, max_rv_semiamplitude):
        """The integrand for marginalizing the CDF over inclination."""

        return (self._observed_rvk.cdf(max_rv_semiamplitude * sin_inclination)
                /
                numpy.sqrt(1.0 - numpy.square(sin_inclination)))

    def _sf_inclination_integrand(self, sin_inclination, max_rv_semiamplitude):
        """The integrand for marginalizing survival functn over inclination."""

        return (self._observed_rvk.sf(max_rv_semiamplitude * sin_inclination)
                /
                numpy.sqrt(1.0 - numpy.square(sin_inclination)))

    def _get_inclination_integration_breaks(self,
                                            max_rv_semiamplitude,
                                            cdf_step=0.01):
        """Return array of points integration must hit for accuracy."""

        result = (
            self._observed_rvk.ppf(numpy.arange(cdf_step, 1.0, cdf_step))
            /
            max_rv_semiamplitude
        )
        return result[result < 1.0]

    def _marginalize_inclination(self,
                                 max_rv_semiamplitude,
                                 integrand=None,
                                 value_only=True):
        """Marginalize PDF/CDF/SF for given max RV semi-amplitude over incl."""

        if max_rv_semiamplitude == 0:
            return 0.0

        points = self._get_inclination_integration_breaks(max_rv_semiamplitude)
        split = min(self._observed_rvk.isf(1e-6) / max_rv_semiamplitude, 1.0)
        if integrand is None:
            integrand = self._pdf_inclination_integrand

            split = numpy.sqrt(1.0 - numpy.square(split))
            points = numpy.sqrt(1.0 - numpy.square(points))


        integration_result = integrate.quad(
            integrand,
            0.0,
            split,
            args=(max_rv_semiamplitude,),
            points=points,
            full_output=True,
            **self._integration_options
        )
        result = numpy.array(integration_result[:2])
        if split < 1.0:
            integration_result = integrate.quad(
                integrand,
                split,
                1.0,
                args=(max_rv_semiamplitude,),
                points=points,
                full_output=True,
                **self._integration_options
            )
            result += numpy.array(integration_result[:2])

        result *= 2.0 / numpy.pi

        if value_only:
            return result[0]

        return result

    def __init__(self,
                 *,
                 observed_rvk,
                 observed_eccentricity,
                 max_discarded_probabiity,
                 interpolation_accuracy,
                 num_parallel_processes,
                 pickle_fname,
                 show_mismatch_plot=False,
                 **integration_options):
        """
        Set the distributions of RV semi-amplitude and eccentricity per RV data.

        Args:
            observed_rvk(rv_continuous):    The empirical distribution of the RV
                semi-amplitude. Assumed independent of `observed_eccentricity`.

            observed_eccentricity(rv_continuous):    The empirical distribution
                of the orbital eccentricity. Assumed independent of
                `observed_rvk`.

            max_discarded_probabiity(float):    The tails of the RV
                semi-amplitude distribution with weight less than this are
                truncated to define the range for interpolation.

            interplation_accuracy(float, float):    The maximum error allowed
                in the PDF interpolation as a fraction of the largest PDF value,
                and as the PDF at the inteprolated position. Comparison is to
                directly integrated values.

            integration_options(dict):    Passed directly to
                :func:`integrate.quad()` to control integration. Must not
                include `points`.

        Returns:
            None
        """

        self._interpolation_accuracy = interpolation_accuracy
        self._max_discarded_probability = max_discarded_probabiity

        self._observed_rvk = observed_rvk
        self._observed_eccentricity = observed_eccentricity
        self._integration_options = integration_options

    def
