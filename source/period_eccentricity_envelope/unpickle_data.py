"""Define a function for reading pickled period-eccentricity evolutions."""

import pickle

import numpy

def unpickle_data(pickle_fname):
    """Return the pickled data from the given file."""

    with open(pickle_fname, 'rb') as pickle_file:
        pickled_config = pickle.load(pickle_file)
        period_evolutions = numpy.full(
            shape=(pickled_config.plot_ages.size,
                   pickled_config.orbital_period_grid.size,
                   pickled_config.eccentricity_grid.size),
            fill_value=numpy.nan,
            dtype=numpy.float64
        )
        eccentricity_evolutions = numpy.full(
            shape=period_evolutions.shape,
            fill_value=numpy.nan,
            dtype=numpy.float64
        )

        try:
            while True:
                period_index = pickled_config.orbital_period_grid.searchsorted(
                    pickle.load(pickle_file)
                )
                ecc_index = pickled_config.eccentricity_grid.searchsorted(
                    pickle.load(pickle_file)
                )
                evolution = pickle.load(pickle_file)
                num_steps = evolution.age.size
                if not (
                        numpy.abs(pickled_config.plot_ages[:num_steps]
                                  -
                                  evolution.age)
                        <
                        1e-10
                ).all():
                    continue
                period_evolutions[:num_steps, period_index, ecc_index] = (
                    evolution.orbital_period
                )
                eccentricity_evolutions[:num_steps, period_index, ecc_index] = (
                    evolution.eccentricity
                )
        except EOFError:
            pass

    return (
        pickled_config,
        (
            period_evolutions,
            eccentricity_evolutions
        )
    )
