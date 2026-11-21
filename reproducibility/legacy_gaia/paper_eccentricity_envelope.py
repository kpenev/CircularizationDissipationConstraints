"""Three-segment envelope with a requested0.06 short-period level."""
import numpy as np

BREAK_PERIOD_DAYS = 7.0
MIN_ECCENTRICITY = 0.06
ANCHOR_PERIOD_DAYS = 10.093766032005036  # Stock 2
ANCHOR_ECCENTRICITY = 0.40241749057158116
SLOPE = (ANCHOR_ECCENTRICITY-MIN_ECCENTRICITY)/np.log10(ANCHOR_PERIOD_DAYS/BREAK_PERIOD_DAYS)
MAX_ECCENTRICITY = 0.8  # inherited maximum initial eccentricity
SATURATION_PERIOD_DAYS = BREAK_PERIOD_DAYS * 10**((MAX_ECCENTRICITY-MIN_ECCENTRICITY)/SLOPE)
MAX_PERIOD_DAYS = 50.0


class PaperEccentricityEnvelope:
    """Flat to7d, rise to e=.8, then flat through50d and beyond."""

    def __call__(self, period_days):
        period = np.asarray(period_days, dtype=float)
        if np.any(~np.isfinite(period)) or np.any(period <= 0):
            raise ValueError('Envelope periods must be finite and positive')
        result = np.minimum(MAX_ECCENTRICITY, MIN_ECCENTRICITY+SLOPE*np.log10(
            np.maximum(period, BREAK_PERIOD_DAYS)/BREAK_PERIOD_DAYS))
        return float(result) if result.ndim == 0 else result
