"""Class that incorporates RV constraints on binary orbits."""

class RVOrbitConstraint:
    """Conveniently and efficiency incorporate RV constraint of SB1 orbits."""

    def _marginalize_eccentricity(self,
                                  rv_semiamplitude_scale,
                                  value_only=True):
        """
        Marginazile the likelihood of RV data over eccentricity and inclination.

        Args:
            rv_semiamplitude_scale(float):    The RV semi-amplitude the orbit
                would have if it were circular and edge-on.

            value_only(bool):    If True only the value of the integration is
                returned. Otherwise, the value and its error estimate are
                returned.

        Returns:
            The product of the eccentricity likelihood function and radial
            velocity semi-amplitude likelihood function marginalized over all
            possible inclinations and eccentricity in the range
        """



    def __init__(self,
                 *,
                 observed_rvk,
                 observed_eccentricity,
                 max_discarded_probabiity,
                 interpolation_accuracy,
                 pickle_fname,
                 num_parallel_processes=1,
                 show_mismatch_plot=False,
                 **integration_options):
        """
        Set the distributions of RV semi-amplitude and eccentricity per RV data.

        Args:
            observed_rvk:    See same name argument to
                `approximate_rv_likelihood.__init__()`.

            observed_eccentricity:    See same name argument to
                `approximate_rv_likelihood.__init__()`.

            max_discarded_probabiity:    See same name argument to
                `approximate_rv_likelihood.__init__()`.

            interplation_accuracy(float, float):    The maximum error allowed
                in the PDF interpolation as a fraction of the largest PDF value,
                and as the PDF at the inteprolated position. Comparison is to
                directly integrated values.

            integration_options:    See same name argument to
                `approximate_rv_likelihood.__init__()`.

        Returns:
            None
        """

        self._interpolation_accuracy = interpolation_accuracy
        self._max_discarded_probability = max_discarded_probabiity

        self._integration_options = integration_options

    def
