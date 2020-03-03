#!/usr/bin/env python3
#pylint: disable=invalid-name

"""Extract limits on log10(Q*') from a pre-calculated grid of eccentricities."""

import numpy

def format_eccentricity_vs_lgQ(eccentricity_vs_lgQ):
    """
    Return 2-D numpy array with lgQ in [:, 0] and e in [:, 1].

    Args:
        eccentricity_lgQ(dict):    The dictionary of pre-computed
            eccentricity vs log10(Q*') as unpickled from the grid.

    Return:
        numpy.array:
            See doc line.
    """

    data = [(item[0], item[1][1]) for item in eccentricity_vs_lgQ.items()]
    return numpy.array(sorted(data, key=lambda item: item[0]))

def _solve_line(x0, y0, x1, y1, target_y):
    """Return x0 < x < x1 where a line crosses a target y value or None."""

    result = (target_y - y0) * (x0 - x1) / (y0 - y1) + x0
    return result if x0 <= result <= x1 else None

class LinearEccentricityEnvelope:
    """Piecewise-linear model for the eccentricity envelovpe."""

    def _eccentricity_envelope_line(self, orbital_period):
        """Evaluate the straight line portion of the eccentricity envelope."""

        return (
            self.max_eccentricity
            *
            (orbital_period - self.min_period)
            /
            (self.max_period - self.min_period)
        )

    def __init__(self, min_period=0.8, max_period=5.0, max_eccentricity=0.6):
        """Setup envelope going from 0 to max_e in the given period range."""

        self.min_period = min_period
        self.max_period = max_period
        self.max_eccentricity = max_eccentricity

    def __call__(self, orbital_period):
        """Return the eccentricity envelovpe at the given orbital period."""

        try:
            if orbital_period < self.min_period:
                return 0.0
            if orbital_period < self.max_period:
                return self._eccentricity_envelope_line(orbital_period)
            return self.max_eccentricity
        except ValueError:
            result = numpy.zeros(orbital_period.shape, dtype=float)
            result[
                numpy.logical_and(
                    orbital_period > self.min_period,
                    orbital_period < self.max_period
                )
            ] = self._eccentricity_envelope_line(orbital_period)
            result[orbital_period >= self.max_period] = self.max_eccentricity
            return result

    def get_period(self, eccentricity):
        """Return the period where the e-envelope has the given value."""

        if eccentricity < 0 or eccentricity > self.max_eccentricity:
            return numpy.nan

        return (
            eccentricity
            *
            (self.max_period - self.min_period)
            /
            self.max_eccentricity
            +
            self.min_period
        )

class MeibomEccentricityEnvelope:
    """Model for the eccentricity envelope after Meibom & Mathieu 2004."""

    def __init__(self, min_period, alpha, beta, gamma):
        """Set the parameters defining the envelope."""

        self.min_period = min_period
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def __call__(self, orbital_period):
        """Return the envelope at the given orbital period."""

        return self.alpha * (
            1.0
            -
            numpy.exp(
                self.beta
                *
                numpy.minimum(
                    (self.min_period - orbital_period),
                    0.0
                )
            )
        )**self.gamma

    def get_period(self, eccentricity):
        """Return the period where the e-envelope has the given value."""

        return (
            self.min_period
            -
            1.0 / self.beta
            *
            numpy.log(1.0 - (eccentricity / self.alpha)**(1.0 / self.gamma))
        )

def invert_eccentricity_vs_lgQ(eccentricity_vs_lgQ,
                               eccentricity,
                               default_min=None,
                               default_max=None):
    """
    Estimate log10(Q*') which reproduced the given eccentricity for a system.

    Args:
        eccentricity_lgQ(dict):    The dictionary of pre-computer
            eccentricity vs log10(Q*') as unpickled from the grid.

        eccentricity(float):     The eccentricity to reproduce.

    Returns:
        float:
            An estamite of the value of log10(Q*') at which the given
            eccentricity is reproduced for the given system.
    """

    interp_data = format_eccentricity_vs_lgQ(eccentricity_vs_lgQ)

    if (
            not numpy.isfinite(eccentricity)
            or
            not numpy.isfinite(interp_data[:, 1]).any()
    ):
        return None

    if numpy.nanmax(interp_data[:, 1]) < eccentricity:
        return default_max

    if numpy.nanmin(interp_data[:, 1]) > eccentricity:
        return default_min

    for i in range(interp_data.shape[0] - 1):
        result = _solve_line(*interp_data[i: i + 2].flatten(), eccentricity)
        if result is not None:
            return result

    print('Something weird is going on when solvirg for e=%.16f with data: '
          %
          eccentricity
          +
          repr(interp_data))
    assert False
    return None
