"""Manual corrections to OC limits to fix modeling problems."""

from numpy import nan

manual_limits = {
    'NGC188': {
        4705: (5.5, 5.5, nan),
        4962: (5.0, 5.0, nan),
        4965: (None, None, 7.0),
        5052: (nan, nan, None), #outlier
        4904: (nan, nan, None)  #outlier
    },
    'NGC6819': {
        33002: (None, None, 7.0),
        49002: (nan, nan, 8.0),
        32006: (nan, nan, nan), #outlier
        23018: (6.0, 6.2, 7.52)
    },
    'PRAESEPE/HYADES': {
        'vB22': (nan, nan, 6.0),
        'J288': (nan, nan, 7.0),
        'vB117': (5.5, 5.5, nan),
        'KW181': (5.8, 5.8, nan),
        'G7-192': (nan, nan, nan) #outlier
    }
}
