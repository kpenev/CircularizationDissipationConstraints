"""Define 3-D correlated Gaussian log-likelihood for debugging."""

from scipy.stats import multivariate_normal

from log_likelihood_base import LogLikelihoodBase


#Simply specialized abstract class.
#pylint: disable=too-few-public-methods
class GaussianLogLikelihood(LogLikelihoodBase):
    """Log-likelihood class with 3-D correlated Gaussian for debugging."""

    def _age_cdf_integrand(self, age, _, mass, feh):

        return self.distribution.pdf([age, mass, feh])

    def __init__(self, mean, covariance, *args, **kwargs):
        """
        Set up the log-likelihood with the given mean and covariance.

        The order of the parameters are: age, mass, [Fe/H]

        Args:
            mean:    See `scipy.stats.multivariate_normal`.

            covariance:    See `scipy.stats.multivariate_normal`.

            args:    Directly passed to parent's __init__

            kwargs:    Directly passed to parent's __init__

        Returns:
            None
        """

        self.distribution = multivariate_normal(mean, covariance)
        super().__init__(*args, **kwargs)
#pylint: enable=too-few-public-methods
