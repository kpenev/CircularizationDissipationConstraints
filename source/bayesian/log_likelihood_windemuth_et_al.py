"""Define a log-likelihood class for Windemuth et. al. (2019) EBs."""

from bayesian.log_likelihood_binary_stars import LogLikelihoodBinaryStars

class LogLikelihoodWindemuth(LogLikelihoodBinaryStars):
    """The log-likelihood for Windemuth et. al. (2019) EBs."""

    def __init__(self,
                 *parent_args,
                 envelope_eccentricity,
                 observed_eccentricity_distro,
                 **parent_kwargs):
        """
        Prepare the log-likelihood function.

        Args:
            parent_args:    Passed directly to parent`s ``__init__``.

        """

        super().__init__(*parent_args,
                         envelope_eccentricity=envelope_eccentricity,
                         **parent_kwargs)
        self.envelope_weights = observed_eccentricity_distro.eval_sample_cdf(
            envelope_eccentricity
        )
        self._observed_eccentricity_distro = observed_eccentricity_distro

    def calculate_log_likelihood(self,
                                 parameters,
                                 **other_args):
        """Evaluate the log-likelihood at the given model parameters."""

        assert 'sample_weights_envelope' in other_args

        final_eccentricity = super().calculate_final_eccentricity(parameters)

        numerator = (
            self._observed_eccentricity_distro.eval_sample_cdf(
                final_eccentricity
            )
            /
            self.envelope_weights
            *
            other_args['sample_weights_envelope']
        ).sum()

        return numerator / other_args['sample_weights_envelope'].sum()
