"""Utilities for input and output of exoplanet data."""

import os.path
import pickle
from types import SimpleNamespace

from astropy import units
import numpy
import pandas

from planetary_system_io import read_cds_pipe_table

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

        print('Getting '
              +
              column_name
              +
              ': '
              +
              repr(getattr(nasa_systems, column_name)[system_index]))
        result = units.Quantity(
            getattr(nasa_systems, column_name)[system_index]
        )
        result.plus_error = units.Quantity(
            getattr(
                nasa_systems,
                column_name+'err1'
            )[system_index],
            unit=result.unit
        )
        result.minus_error = units.Quantity(
            -getattr(
                nasa_systems,
                column_name+'err2'
            )[system_index],
            unit=result.unit
        )
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

def get_quantity(value, plus_error, minus_error, unit):
    """Return a properly formatted Quantity instance with errors."""

    result = units.Quantity(
        value,
        unit=unit
    )
    result.plus_error = units.Quantity(
        plus_error,
        unit=unit
    )
    result.minus_error = units.Quantity(
        minus_error,
        unit=unit
    )
    return result

def read_milliman_et_al_2014_binaries(
        single_lined_orbits_fname=(
            '../data/Milliman_et_al_2014_WIYN_single_lined_orbits.tsv'
        ),
        double_lined_orbits_fname=(
            '../data/Milliman_et_al_2014_WIYN_double_lined_orbits.tsv'
        )
):
    """Read Geller et al 2009 NGC6819 binaries in format like that of exopl."""

    def create_system(record):
        """Return a properly created system from the given binary record."""

        return SimpleNamespace(
            hostname=record['WOCS'],
            age=get_quantity(2.6, 0.25, 0.25, 'Gyr'),
            eccentricity=get_quantity(record['e'],
                                      record['e_e'],
                                      record['e_e'],
                                      ''),
            eccentricity_limit=False,
            feh=get_quantity(0.09, 0.03, 0.03, ''),
            orbital_period=get_quantity(record['Per'],
                                        record['e_Per'],
                                        record['e_Per'],
                                        'day'),
            primary_mass=get_quantity(numpy.nan,
                                      numpy.nan,
                                      numpy.nan,
                                      'M_sun'),
            secondary_mass=get_quantity(numpy.nan,
                                        numpy.nan,
                                        numpy.nan,
                                        'M_sun')
        )

    single_lined_data = read_cds_pipe_table(single_lined_orbits_fname)
    double_lined_data = read_cds_pipe_table(single_lined_orbits_fname)
    print('double_lined_data:' + repr(double_lined_data))

    return (
        [create_system(record) for record in double_lined_data]
        +
        [create_system(record) for record in single_lined_data]
    )

def read_geller_et_al_2009_binaries(
        single_lined_orbits_fname=(
            '../data/Geller_et_al_2009_WIYN_single_lined_orbits.tsv'
        ),
        double_lined_orbits_fname=(
            '../data/Geller_et_al_2009_WIYN_double_lined_orbits.tsv'
        ),
        physical_parameters_fname=(
            '../data/Geller_et_al_2009_WIYN_physical_parameters.tsv'
        )
):
    """Read Geller et al 2009 NGC 188 binaries in format like that of exopl."""

    def create_system(record):
        """Return a properly created system from the given record."""

        m1_plus_error = m1_minus_error = 0.1
        m2_plus_error = m2_minus_error = 0.15
        if record['l_M1']:
            m1_minus_error = record['M1']
            m2_minus_error = record['M2']

        return SimpleNamespace(
            hostname=record['PKM'],
            age=get_quantity(6.3, 0.2, 0.2, 'Gyr'),
            eccentricity=get_quantity(record['e'],
                                      record['e_e'],
                                      record['e_e'],
                                      ''),
            eccentricity_limit=False,
            feh=get_quantity(0.21, 0.03, 0.03, ''),
            orbital_period=get_quantity(record['Per'],
                                        record['e_Per'],
                                        record['e_Per'],
                                        'day'),
            primary_mass=get_quantity(record['M1'],
                                      m1_plus_error,
                                      m1_minus_error,
                                      'M_sun'),
            secondary_mass=get_quantity(record['M2'],
                                        m2_plus_error,
                                        m2_minus_error,
                                        'M_sun')
        )

        #Parameters defined for planetary systems but not here.
        #impact_parameter=<Quantity nan>,
        #logg=<Quantity 4.7>,
        #planet_to_star_radius_ratio=<Quantity 0.02704>,
        #primary_radius=<Quantity 0.57 solRad>,
        #rv_semi_amplitude=<Quantity nan m / s>,
        #secondary_radius=<Quantity 0.15 jupiterRad>,
        #semimajor=<Quantitynan AU>,
        #semimajor_to_rstar_ratio=<Quantity nan>,
        #star_density=<Quantity nan g / cm3>,
        #teff=<Quantity 4128. K>,
        #transit_duration=<Quantity 0.0804 d>

    physical_parameters = pandas.DataFrame(
        read_cds_pipe_table(physical_parameters_fname)
    )

    single_lined_data = pandas.merge(
        physical_parameters,
        pandas.DataFrame(
            read_cds_pipe_table(single_lined_orbits_fname)
        ),
        on='PKM'
    )
    double_lined_data = pandas.merge(
        physical_parameters,
        pandas.DataFrame(
            read_cds_pipe_table(double_lined_orbits_fname)
        ),
        on='PKM'
    )
    print('single lined data: ' + repr(single_lined_data))
    print('double lined data: ' + repr(double_lined_data))

    return (
        [create_system(record[1]) for record in double_lined_data.iterrows()]
        +
        [create_system(record[1]) for record in single_lined_data.iterrows()]
    )

def init_progress_pickle(cmdline_args):
    """
    If a pickle file exists check it matches cmdline_args, otherwise create it.

    Args:
        cmdline_args:    The parsed command line arguments.

    Returns:
        dict:
            The pickled calculated eccentricites contained in the given pickle
            or empty dictionary if the file did not exist.
    """

    if os.path.exists(cmdline_args.progress_pickle):
        with open(cmdline_args.progress_pickle, 'rb') as progress_file:
            pickled_cmdline_args = pickle.load(progress_file)
            pickled_cfg = dict(vars(pickled_cmdline_args))
            cmdline_cfg = dict(vars(cmdline_args))
            if 'initial_eccentricity' not in pickled_cfg:
                pickled_cfg['initial_eccentricity'] = 0.55
            if 'stellar_lgQ' not in pickled_cfg:
                pickled_cfg['stellar_lgQ'] = numpy.inf
            for ignore_arg in ['progress_pickle',
                               'num_parallel_processes',
                               'use_binary_stars']:
                if ignore_arg in pickled_cfg:
                    del pickled_cfg[ignore_arg]
                if ignore_arg in cmdline_cfg:
                    del cmdline_cfg[ignore_arg]
            print('Pickled config: ' + repr(pickled_cfg))
            print('Command line config: ' + repr(cmdline_cfg))
            assert pickled_cfg == cmdline_cfg
            return load_progress_pickle(progress_file)
    else:
        with open(cmdline_args.progress_pickle, 'wb') as progress_file:
            pickle.dump(cmdline_args, progress_file)
        return dict()


def load_progress_pickle(progress_file):
    """
    Return a dictionary containing previously calculated final eccentricities.

    Args:
        progress_file:    An already opened file containing pickles of
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
    try:
        while True:
            hostname = pickle.load(progress_file)
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
