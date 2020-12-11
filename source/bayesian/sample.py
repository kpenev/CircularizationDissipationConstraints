"""Define ln(circularization PDF) class for a binary star system."""
from log_likelihood_const_lag import LogLikelihoodConstLag
from eccentricity_pdf import EccentricityPDFNormal

def get_log_likelihood(system,
                       e_envelope_mean,
                       e_envelope_stddev,
                       interpolator,
                       initial_eccentricity=0.5):

    """
    Return the log-likelihood function to use for the given system.

    Args:
        system:    Object with attributes containing the system parameters,
            complete with units and error bars.

        e_envelope_mean(float):    The mean of the eccentricity envelope to
            impose on this system.

        e_envelope_stddev(float):    The estimated standard deviation of the
            eccentricity envelope to impose on this system.

    Returns:
        LogLikelihoodConstLag:
            The log-likelihood function to use when constraining the tidal
            dissiipation for the given system.
    """

    return LogLikelihoodConstLag(
        interpolator=interpolator,
        eccentricity_pdf=EccentricityPDFNormal(
            e_observed_mean=system.eccentricity,
            e_observed_stddev=(system.eccentricity.plus_error,
                               system.eccentricity.plus_error),
            e_envelope_mean=e_envelope_mean,
            e_envelope_stddev=e_envelope_stddev
        ),
        secondary_is_star=True,
        initial_eccentricity=initial_eccentricity
    )
