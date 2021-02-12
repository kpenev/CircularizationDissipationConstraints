"""Utilities for input and output of exoplanet data."""

import sys
import os.path
import pickle
import itertools
from types import SimpleNamespace

from astropy import units
import numpy
import pandas

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))

#Need to update search path
#pylint: disable=wrong-import-position
from planetary_system_io import read_cds_pipe_table
from fit_ngc6819_masses import fit_milliman
from command_line_utilities import data_dir
#False positive
#pylint: disable=import-error
from multiple_star_catalogue.multiple_star_catalogue import\
    MultipleStarCatalogue
from sophie_catalogue.sophie_catalogue import SophieCatalogue
#pylint: enable=import-error
#pylint: enable=wrong-import-position

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

    def get_system_quantity(system_index, column_name):
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
        star_density=get_system_quantity(system_index, 'st_dens'),
        db_star_mass=get_system_quantity(system_index, 'st_mass'),
        db_star_age=get_system_quantity(system_index, 'st_age'),
        db_star_radius=get_system_quantity(system_index, 'st_rad'),
        db_planet_mass=get_system_quantity(system_index, 'pl_bmassj'),
        db_planet_radius=get_system_quantity(system_index, 'pl_radj'),
        planet_to_star_radius_ratio=get_system_quantity(system_index,
                                                        'pl_ratror'),
        teff=get_system_quantity(system_index, 'st_teff'),
        feh=get_system_quantity(system_index, 'st_metfe'),
        logg=get_system_quantity(system_index, 'st_logg'),
        rv_semi_amplitude=get_system_quantity(system_index, 'pl_rvamp'),
        eccentricity=get_system_quantity(system_index, 'pl_orbeccen'),
        eccentricity_limit=(nasa_systems.pl_orbeccenlim[system_index] > 0.5),
        semimajor_to_rstar_ratio=get_system_quantity(system_index, 'pl_ratdor'),
        orbital_period=get_system_quantity(system_index, 'pl_orbper'),
        semimajor=get_system_quantity(system_index, 'pl_orbsmax'),
        impact_parameter=get_system_quantity(system_index, 'pl_imppar'),
        transit_duration=get_system_quantity(system_index, 'pl_trandur')
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

def create_binary_star_systems(single_lined_data, double_lined_data, age, feh):
    """Return properly formatted systems from NGC6819 and NGC188 orbit fits."""

    def create_single_system(record):
        """Create a single system from the given record."""

        m1_plus_error = m1_minus_error = 0.1
        m2_plus_error = m2_minus_error = 0.15
        if 'l_M1' in record and record['l_M1']:
            m1_minus_error = record['M1']
            m2_minus_error = record['M2']

        if record['M1'] > 1.2:
            masses = dict(
                primary_mass=get_quantity(record['M2'],
                                          m2_plus_error,
                                          m2_minus_error,
                                          'M_sun'),
                secondary_mass=get_quantity(record['M1'],
                                            m1_plus_error,
                                            m1_minus_error,
                                            'M_sun'),
                secondary_radius=get_quantity(1.0, 0.1, 0.1, 'R_sun')
            )
        else:
            masses = dict(
                primary_mass=get_quantity(record['M1'],
                                          m1_plus_error,
                                          m1_minus_error,
                                          'M_sun'),
                secondary_mass=get_quantity(record['M2'],
                                            m2_plus_error,
                                            m2_minus_error,
                                            'M_sun')
            )

        return SimpleNamespace(
            hostname=(record['PKM'] if 'PKM' in record else record['WOCS']),
            age=age,
            eccentricity=get_quantity(record['e'],
                                      record['e_e'],
                                      record['e_e'],
                                      ''),
            eccentricity_limit=False,
            feh=feh,
            orbital_period=get_quantity(record['Per'],
                                        record['e_Per'],
                                        record['e_Per'],
                                        'day'),
            **masses
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



    return [
        create_single_system(record[1])
        for record in itertools.chain(double_lined_data.iterrows(),
                                      single_lined_data.iterrows())
    ]

def read_milliman_data(
        single_lined_orbits_fname=(
            os.path.join(
                data_dir,
                'Milliman_et_al_14_NGC6819_RV_single_lined_orbits.tsv'
            )
        ),
        double_lined_orbits_fname=(
            os.path.join(
                data_dir,
                'Milliman_et_al_14_NGC6819_RV_double_lined_orbits.tsv'
            )
        ),
        photometry_fname=(
            os.path.join(
                data_dir,
                'Milliman_et_al_14_NGC6819_RV_summary_with_VIphotometry.tsv'
            )
        )
):
    """Read the data from the Milliman et al (2014) tables as pandas frames."""

    photometry = pandas.DataFrame(read_cds_pipe_table(photometry_fname))

    single_lined_data = pandas.merge(
        pandas.DataFrame(
            read_cds_pipe_table(single_lined_orbits_fname)
        ),
        photometry,
        on='WOCS'
    )
    double_lined_data = pandas.merge(
        pandas.DataFrame(
            read_cds_pipe_table(double_lined_orbits_fname)
        ),
        photometry,
        on='WOCS'
    )
    single_lined_data['e_Vmag'] = pandas.Series(0.02, single_lined_data.index)
    single_lined_data['e_V-I'] = pandas.Series(0.03, single_lined_data.index)

    double_lined_data['e_Vmag'] = pandas.Series(0.02, double_lined_data.index)
    double_lined_data['e_V-I'] = pandas.Series(0.03, double_lined_data.index)

    return photometry, single_lined_data, double_lined_data

def read_milliman_et_al_2014_binaries(
        single_lined_orbits_fname=(
            os.path.join(
                data_dir,
                'Milliman_et_al_2014_WIYN_single_lined_orbits.tsv'
            )
        ),
        double_lined_orbits_fname=(
            os.path.join(
                data_dir,
                'Milliman_et_al_2014_WIYN_double_lined_orbits.tsv'
            )
        ),
        photometry_fname=(
            os.path.join(
                data_dir,
                'Milliman_et_al_14_NGC6819_RV_summary_with_VIphotometry.tsv'
            )
        )
):
    """Read Milliman et al 2014 NGC6819 binaries in format like that of exop."""

    ngc6819_age = get_quantity(2.6, 0.25, 0.25, 'Gyr')
    ngc6819_feh = get_quantity(0.09, 0.03, 0.03, '')
    single_lined_data, double_lined_data = read_milliman_data(
        single_lined_orbits_fname,
        double_lined_orbits_fname,
        photometry_fname
    )[1:]
    fit_milliman(single_lined_data, double_lined_data)
    return create_binary_star_systems(
        single_lined_data,
        double_lined_data,
        ngc6819_age,
        ngc6819_feh
    )

def read_geller_et_al_2009_binaries(
        single_lined_orbits_fname=(
            os.path.join(
                data_dir,
                'Geller_et_al_2009_WIYN_single_lined_orbits.tsv'
            )
        ),
        double_lined_orbits_fname=(
            os.path.join(
                data_dir,
                'Geller_et_al_2009_WIYN_double_lined_orbits.tsv'
            )
        ),
        physical_parameters_fname=(
            os.path.join(
                data_dir,
                'Geller_et_al_2009_WIYN_physical_parameters.tsv'
            )
        ),
        raw_data=False
):
    """Read Geller et al 2009 NGC 188 binaries in format like that of exopl."""

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

    ngc188_age = get_quantity(6.3, 0.2, 0.2, 'Gyr')
    ngc188_feh = get_quantity(0.21, 0.03, 0.03, '')

    if raw_data:
        return (
            single_lined_data,
            double_lined_data,
            ngc188_age,
            ngc188_feh
        )
    return create_binary_star_systems(single_lined_data,
                                      double_lined_data,
                                      ngc188_age,
                                      ngc188_feh)

#pylint: disable=too-many-statements
def format_hyades_praesepe_binaries(systems,
                                    age,
                                    feh,
                                    resolve_secondary_mass_range):
    """Format the given systems from Hyades or Praesepe format."""

    def format_system(input_sys, msc, sophie):
        """Re-format the given system to attributes with proper names."""

        def get_parameter(*alias_param_names):
            """Add error attributes to the given quantity."""

            for param_name in alias_param_names:
                quantity = input_sys.get(param_name)
                if quantity is None:
                    continue
                if param_name.startswith('ModelM'):
                    #False positive
                    #pylint: disable=no-member
                    error = 0.0 * units.M_sun
                    #pylint: enable=no-member
                    if param_name == 'ModelM2' and quantity.size == 2:
                        quantity = getattr(
                            numpy,
                            resolve_secondary_mass_range
                        )(
                            quantity
                        )
                else:
                    error = input_sys['err' + param_name]

                if isinstance(error, tuple):
                    plus_error, minus_error = error
                else:
                    plus_error = minus_error = error

                plus_error = plus_error or 0.0
                minus_error = minus_error or 0.0

                if isinstance(quantity, float):
                    result = units.Quantity(quantity, unit='')
                    result.plus_error = units.Quantity(plus_error, unit='')
                    result.minus_error = units.Quantity(minus_error, unit='')
                else:
                    result = units.Quantity(quantity)
                    result.plus_error = plus_error
                    result.minus_error = minus_error
                return result
            raise KeyError("'" + "' | '".join(alias_param_names) + "'")

        #pylint: disable=too-many-branches
        def get_masses():
            """Return the two component masses, combining everything avaible."""

            #False positive
            #pylint: disable=no-member
            primary_mass = input_sys.get(
                'M1',
                input_sys.get(
                    'ModelM1',
                    input_sys.get('ModelM1Inner')
                )
            )
            secondary_mass = input_sys.get(
                'M2',
                input_sys.get(
                    'ModelM2',
                    input_sys.get('ModelM2Inner')
                )
            )
            if secondary_mass is not None and secondary_mass.size == 2:
                secondary_mass = getattr(
                    numpy,
                    resolve_secondary_mass_range
                )(
                    secondary_mass
                )

            if hasattr(input_sys, 'OtherIDs'):
                if primary_mass is None or secondary_mass is None:
                    hdid = str(
                        input_sys['OtherIDs'].get('HD', input_sys.get('HDE'))
                    )
                    sophie_info = sophie(HD=hdid)
                    if sophie_info is not None:
                        print('In SOPHIE catalogue')
                        if primary_mass is None:
                            primary_mass = sophie_info.get(
                                'Mass',
                                sophie_info.get('Mprimary')
                            )[0] * units.M_sun
                            if not numpy.isfinite(primary_mass):
                                primary_mass = None

                        if secondary_mass is None:
                            secondary_mass = (sophie_info['M2sini'][0]
                                              *
                                              units.M_jup)
                            if not numpy.isfinite(secondary_mass):
                                secondary_mass = None

                if primary_mass is None or secondary_mass is None:
                    msc_info = msc(HD=hdid)

                    orbital_period = input_sys.get(
                        'Porb',
                        input_sys.get('PInner')
                    ).to_value('day')

                    if msc_info is not None:
                        print('In MSC catalogue')
                        best_match = numpy.inf
                        for index, msc_period in enumerate(
                                numpy.power(10.0, msc_info['logP'].array)
                        ):
                            if abs(msc_period - orbital_period) < best_match:
                                best_match = abs(msc_period - orbital_period)
                                best_index = index
                        print('Period discrepancy: ' + repr(best_match))
                        if primary_mass is None:
                            primary_mass = (msc_info['Mass1'].array[best_index]
                                            *
                                            units.M_sun)
                        if secondary_mass is None:
                            secondary_mass = (
                                msc_info['Mass1'].array[best_index]
                                *
                                units.M_sun
                            )

            nan_mass = get_quantity(numpy.nan,
                                    numpy.nan,
                                    numpy.nan,
                                    'M_sun')
            if primary_mass is None:
                return dict(
                    primary_mass=nan_mass,
                    secondary_mass=nan_mass,
                )
            primary_mass.plus_error = primary_mass.minus_error = (
                0.0 * units.M_sun
            )
            if secondary_mass is None:
                secondary_mass = nan_mass
            else:
                secondary_mass.plus_error = secondary_mass.minus_error = (
                    0.0 * units.M_sun
                )

            if primary_mass > 1.2 * units.M_sun:
                return dict(
                    primary_mass=secondary_mass,
                    secondary_mass=primary_mass,
                    secondary_radius=get_quantity(1.0, 0.1, 0.1, 'R_sun')
                )
            return dict(
                primary_mass=primary_mass,
                secondary_mass=secondary_mass
            )
            #pylint: disable=no-member
        #pylint: enable=too-many-branches

        print('Formatting system: ' + repr(input_sys))
        return SimpleNamespace(
            hostname=input_sys['ID'],
            age=age,
            feh=get_quantity(feh, 0.004, 0.004, ''),
            eccentricity=get_parameter('Ecc', 'EccInner'),
            eccentricity_limit=False,
            orbital_period=get_parameter('Porb', 'PInner'),
            **get_masses()
        )

    msc = MultipleStarCatalogue()
    sophie = SophieCatalogue()
    return [
        format_system(input_sys, msc, sophie)
        for input_sys in filter(
            lambda s: (
                s['member']
                and
                s['ID'] not in ['J271', 'vB176']
            ),
            systems
        )
    ]
#pylint: enable=too-many-statements

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
                               'use_binary_stars',
                               'fallback_initial_eccentricity',
                               'known_to_fail']:
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
                * keys: (lgQ, initial_eccentricity)
                * values: final eccentricity
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
            attempt_id = (lgQ, initial_eccentricity)
            assert attempt_id not in result[hostname]
            result[hostname][attempt_id] = final_eccentricity
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
