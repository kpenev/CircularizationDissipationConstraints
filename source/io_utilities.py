"""Utilities for input and output of exoplanet data."""

import os.path
import pickle
from types import SimpleNamespace

from astropy import units
import numpy

def get_nasa_system(system_id, nasa_systems):
    """
    Return a single system from a parsed NASA exoplanets archive file.

    Args:
        system_id(int or str):    Either the hostname of the parent star in the
            system or an index within `nasa_systems`.

        nasa_systems:    A value returned by read_nasa_planets().

    Returns:
        SimpleNamespace:
            An object with all attributes defined that are required to calculate
            the evolution of the system.
    """

    def get_quantity(system_index, column_name):
        """Return a properly formatted Quantity instance with errors."""

        print('Getting ' + column_name + ': ' + repr(getattr(nasa_systems,
                                                             column_name)[system_index]))
        result = units.Quantity(
            getattr(nasa_systems, column_name)[system_index]
        )
        result.plus_error = getattr(
            nasa_systems,
            column_name+'err1'
        )[system_index]
        result.minus_error = -getattr(
            nasa_systems,
            column_name+'err2'
        )[system_index]
        print('Result: ' + repr(result))
        return result

    print('Getting system identified by: ' +  repr(system_id) + ' type: ' +
          repr(type(system_id)))

    try:
        system_index = int(system_id)
    except ValueError:
        system_index = numpy.where(nasa_systems.pl_hostname == system_id)[0][0]

    result = SimpleNamespace(
        hostname=nasa_systems.pl_hostname[system_index],
        star_density=get_quantity(system_index, 'st_dens'),
        db_star_mass=get_quantity(system_index, 'st_mass'),
        db_star_age=get_quantity(system_index, 'st_age'),
        db_star_radius=get_quantity(system_index, 'st_rad'),
        db_planet_mass=get_quantity(system_index, 'pl_bmassj'),
        db_planet_radius=get_quantity(system_index, 'pl_radj'),
        planet_to_star_radius_ratio=get_quantity(system_index,
                                                 'pl_ratror'),
        teff=get_quantity(system_index, 'st_teff'),
        feh=get_quantity(system_index, 'st_metfe'),
        logg=get_quantity(system_index, 'st_logg'),
        rv_semi_amplitude=get_quantity(system_index, 'pl_rvamp'),
        eccentricity=get_quantity(system_index, 'pl_orbeccen'),
        eccentricity_limit=(nasa_systems.pl_orbeccenlim[system_index] > 0.5),
        semimajor_to_rstar_ratio=get_quantity(system_index, 'pl_ratdor'),
        orbital_period=get_quantity(system_index, 'pl_orbper'),
        semimajor=get_quantity(system_index, 'pl_orbsmax'),
        impact_parameter=get_quantity(system_index, 'pl_imppar'),
        transit_duration=get_quantity(system_index, 'pl_trandur')
    )
    return result

def load_progress_pickle(progress_pickle_fname):
    """
    Return a dictionary containing previously calculated final eccentricities.

    Args:
        progress_pickle_fname(str):    The filename containing pickles of
            previously calculated eccentricities. It should contain sequences of
            pickles containing:
                * system host star name (e.g. 'HATS-18')
                * lgQ (e.g. 6.0)
                * initial eccentricity (e.g. 0.55)
                * final eccentricity (e.g. 0.3214)

    Returns:
        dict:
            * keys: host star name
            * values: dict:
                * keys: lgQ
                * values: (initial eccentricity, final eccentricity)
    """

    result = dict()
    if os.path.exists(progress_pickle_fname):
        with open(progress_pickle_fname, 'rb') as progress_file:
            try:
                while True:
                    hostname = pickle.load(progress_file)
                    assert isinstance(hostname, str)
                    #lgQ is more readable than say lgq or lg_q or ...
                    #pylint: disable=invalid-name
                    lgQ = pickle.load(progress_file)
                    #pylint: enable=invalid-name
                    assert isinstance(lgQ, float)
                    initial_eccentricity = pickle.load(progress_file)
                    assert isinstance(initial_eccentricity, float)
                    final_eccentricity = pickle.load(progress_file)
                    assert isinstance(final_eccentricity, float)
                    if hostname not in result:
                        result[hostname] = dict()
                    assert lgQ not in result[hostname]
                    result[hostname][lgQ] = (initial_eccentricity,
                                             final_eccentricity)
            except EOFError:
                pass

    return result

#lgQ is more readable than say lgq or lg_q or ...
#pylint: disable=invalid-name
def pickle_new_result(hostname,
                      lgQ,
                      initial_eccentricity,
                      final_eccentricity,
                      progress_pickle_fname):
    """Add new result to the progress pickle file (see load_progress_pickle)."""

    with open(progress_pickle_fname, 'ab') as progress_file:
        pickle.dump(hostname, progress_file)
        pickle.dump(lgQ, progress_file)
        pickle.dump(initial_eccentricity, progress_file)
        pickle.dump(final_eccentricity, progress_file)
#pylint: enable=invalid-name
