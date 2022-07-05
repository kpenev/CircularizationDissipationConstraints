import logging
import math
import sys
import corner
import time
import h5py
from datetime import datetime

from stellar_evolution.manager import StellarEvolutionManager

import matplotlib
from bayesian.stellar_param_sampling.prepare import serialize_poet_likelihood
from split_normal_distribution import split_normal
from stellar_evolution.change_variables import QuantityEvaluator

print(sys.path)
import planetary_system_io
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as un
from abc import ABCMeta, abstractmethod
import emcee
import sys
from scipy.stats import rice
from scipy.stats import norm, stats
from scipy.special import erf
from scipy.optimize import fsolve
from sympy import *
from scipy.special import i0
from scipy.integrate import nquad
from bayesian.stellar_param_sampling.poet_interp_likelihood import POETInterpLikelihood
from bayesian.stellar_param_sampling.star_sampler import StarSampler
from random import random, randint
import multiprocessing as mp
from multiprocessing import Pool, Queue, Process, Value
from datetime import datetime

from bayesian.stellar_param_sampling.feh_conditional_likelihood_base import \
    FeHConditionalLikelihoodBase
import astropy.constants as const

##########################
from manual_exoplanet_data import data as manual_data
from astropy.units import Unit, Quantity

#######################
sys.path.append('/home/mmmahmud/CircularizationDissipationConstraints/source')
sys.path.append('/home/mmmahmud/general_purpose_python_modules')
sys.path.append('/home/mmmahmud/CircularizationDissipationConstraints/data')
#sys.path.append('/home/mmm161430/CircularizationDissipationConstraints/source')
#sys.path.append('/home/mmm161430/projects/git/general_purpose_python_modules')
#sys.path.append('/home/mmm161430/projects/git/poet')
#sys.path.append('/home/mmm161430/CircularizationDissipationConstraints/data')
#sys.path.append('/home/mmm161430/emcee')
#sys.path.append('/home/mmm161430/lib')

if not sys.warnoptions:
    import warnings

    from sqlalchemy.exc import SAWarning

    warnings.filterwarnings('ignore',
                            r"^Dialect sqlite\+pysqlite does \*not\* support Decimal objects natively\, "
                            "and SQLAlchemy must convert from floating point - rounding errors and other "
                            "issues may occur\. Please consider storing Decimal numbers as strings or "
                            "integers on this platform for lossless storage\.$",
                            SAWarning, r'^sqlalchemy\.sql\.type_api$')

sys.path.append('/home/mmmahmud/poet/PythonPackage')
sys.path.append('../scripts')

from orbital_evolution.evolve_interface import library as \
    orbital_evolution_library

from reproduce_system import *

def getPathOfExoplanetSystemsData():
    return '/home/mmmahmud/CircularizationDissipationConstraints/data/PS_2021.07.13_00.12.38.csv' # '/home/mmm161430/CircularizationDissipationConstraints/data/PS_2021.07.13_00.12.38.csv'
def getStellarEvolutionInterpolatorsDirectory():
    return '/home/mmmahmud/poet/stellar_evolution_interpolators' # '/home/mmm161430/projects/git/poet/stellar_evolution_interpolators'
def getEccentricityExpansionCoefficientsFile():
    return b"/media/mmmahmud/USB/eccentricity_expansion_coef_O400.sqlite" # b"/home/mmm161430/projects/git/poet/eccentricity_expansion_coef_O400.sqlite"


def phi(z):
    return 0.5 * (1 + erf(z / math.sqrt(2)))


def alpha(ci):
    return 1 - ci / 100.0


def constraints(smallest_acceptable_value_of_orbital_period=0,
                largest_acceptable_value_of_orbital_period=10,
                smallest_acceptable_value_of_primary_mass=0.4,
                largest_acceptable_value_of_primary_mass=1.2,
                smallest_acceptable_value_of_secondary_mass=0,
                largest_acceptable_value_of_secondary_mass=25000,  # mass of a brown dwarf
                smallest_acceptable_value_of_stellar_metallicity=-1.014,
                largest_acceptable_value_of_stellar_metallicity=0.537,
                smallest_acceptable_value_of_stellar_age=0,
                largest_acceptable_value_of_stellar_age=10,
                smallest_acceptable_value_of_eccentricity_now=0,
                largest_acceptable_value_of_eccentricity_now=0.45):
    smallest = {'orbital period': smallest_acceptable_value_of_orbital_period,
                'primary mass': smallest_acceptable_value_of_primary_mass,
                'secondary mass': smallest_acceptable_value_of_secondary_mass,
                'stellar metallicity': smallest_acceptable_value_of_stellar_metallicity,
                'stellar age': smallest_acceptable_value_of_stellar_age,
                'present eccentricity': smallest_acceptable_value_of_eccentricity_now}
    largest = {'orbital period': largest_acceptable_value_of_orbital_period,
               'primary mass': largest_acceptable_value_of_primary_mass,
               'secondary mass': largest_acceptable_value_of_secondary_mass,
               'stellar metallicity': largest_acceptable_value_of_stellar_metallicity,
               'stellar age': largest_acceptable_value_of_stellar_age,
               'present eccentricity': largest_acceptable_value_of_eccentricity_now}
    return smallest, largest


def constraints_for_eccentricity_envelope(smallest_acceptable_value_of_secondary_radius=8,
                                          largest_acceptable_value_of_secondary_radius=math.inf,
                                          smallest_acceptable_value_of_planet_mass_sin_i=50,
                                          largest_acceptable_value_of_planet_mass_sin_i=math.inf):
    smallest = {'secondary radius': smallest_acceptable_value_of_secondary_radius,
                'planet mass times sin i': smallest_acceptable_value_of_planet_mass_sin_i}
    largest = {'secondary radius': largest_acceptable_value_of_secondary_radius,
               'planet mass times sin i': largest_acceptable_value_of_planet_mass_sin_i}
    return smallest, largest


def constraints_are_satisfied(orbital_period,
                              primary_mass,
                              secondary_mass,
                              stellar_metallicity,
                              eccentricity_now,
                              stellar_age,
                              constraints=constraints()):
    smallest = constraints[0]
    largest = constraints[1]
    if ((orbital_period <= largest['orbital period'] and orbital_period > smallest['orbital period'])
            and (primary_mass > smallest['primary mass'] and primary_mass < largest['primary mass'])
            and (secondary_mass > smallest['secondary mass'] and secondary_mass < largest['secondary mass'])
            and (stellar_metallicity > smallest['stellar metallicity'] and stellar_metallicity < largest[
                'stellar metallicity'])
            and (eccentricity_now >= smallest['present eccentricity'] and eccentricity_now <= largest[
                'present eccentricity'])
            and (stellar_age >= smallest['stellar age'] and stellar_age <= largest['stellar age'])):
        return True
    return False


def constraints_for_eccentricity_envelope_are_satisfied(secondary_radius,
                                                        planet_mass_sin_i,
                                                        constraints=constraints_for_eccentricity_envelope()):
    smallest = constraints[0]
    largest = constraints[1]
    if secondary_radius <= largest['secondary radius'] and secondary_radius > smallest['secondary radius']:
        return True
    if planet_mass_sin_i <= largest['planet mass times sin i'] and planet_mass_sin_i > smallest[
        'planet mass times sin i']:
        return True

    return False

def setup_process(config):
    """Logging and I/O setup for the current processes."""
    def ensure_directory(fname):
        """Make sure the directory containing the given name exists."""
        dirname = os.path.dirname(fname)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    fname_substitutions = dict(
        now=datetime.now().strftime(config.fname_datetime_format),
        system=config.system,
        pid=os.getpid()
    )
    std_out_err_fname = config.std_out_err_fname % fname_substitutions
    ensure_directory(std_out_err_fname)

    io_destination = os.open(
        std_out_err_fname,
        os.O_WRONLY | os.O_TRUNC | os.O_CREAT | os.O_DSYNC,
        mode=0o666
        )
    os.dup2(io_destination, 1)
    os.dup2(io_destination, 2)

    logging_fname = config.logging_fname % fname_substitutions
    ensure_directory(logging_fname)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        handler.close()
    logging_config = dict(
        filename=logging_fname,
        level=config.logging_level,
        format=config.logging_message_format,
        )
    if config.logging_datetime_format is not None:
        logging_config['datefmt'] = config.logging_datetime_format
        logging.basicConfig(**logging_config)

class SuperEccentricityDistribution(metaclass=ABCMeta):
    @abstractmethod
    def probability_density_of_eccentricity(self, e):
        pass


class EccentricityDistribution(SuperEccentricityDistribution):

    def __init__(self,
                 mean_e_now,
                 e_now_upper_uncertainty,
                 e_now_lower_uncertainty,
                 e_env,
                 percentile_for_e_now_upper_uncertainty=phi(1),  # or sometimes 1 - alpha(68.0)/2
                 percentile_for_e_now_lower_uncertainty=1 - phi(1)  # or sometimes alpha(68.0)/2
                 ):

        self.mean_e_now = mean_e_now
        self.e_now_upper_uncertainty = e_now_upper_uncertainty
        self.e_now_lower_uncertainty = e_now_lower_uncertainty
        self.percentile_for_e_now_upper_uncertainty = percentile_for_e_now_upper_uncertainty
        self.percentile_for_e_now_lower_uncertainty = percentile_for_e_now_lower_uncertainty

        self.rice_parameters_are_found = True
        self.e_env = e_env
        self.b, self.s = self.roots_for_Rice_parameters()
        self.inv_norm = self.cdf(1.0)

    def equations_to_be_solved_for_Rice_distribution_parameters(self, x):
        b = x[0]
        s = x[1]
        first = rice.cdf((self.mean_e_now + self.e_now_upper_uncertainty), b,
                         scale=s) - self.percentile_for_e_now_upper_uncertainty
        second = rice.cdf((self.mean_e_now + self.e_now_lower_uncertainty), b,
                          scale=s) - self.percentile_for_e_now_lower_uncertainty
        if math.isnan(first) or math.isnan(second):
            logging.warning('Iteration does not converge')
            self.rice_parameters_are_found = False
        return [first, second]

    def equation_to_be_solved_for_Rice_distribution_parameter_s_when_b_zero(self, x):
        s = x[0]
        eqn = rice.cdf((self.mean_e_now + self.e_now_upper_uncertainty), 0,
                       scale=s) - self.percentile_for_e_now_upper_uncertainty
        if math.isnan(eqn):
            logging.warning('Iteration does not converge')
            self.rice_parameters_are_found = False
        return [eqn]

    def roots_for_Rice_parameters(self):
        estimated_s = self.e_now_upper_uncertainty
        if self.mean_e_now == 0:
            try:
                s = fsolve(self.equation_to_be_solved_for_Rice_distribution_parameter_s_when_b_zero,
                           np.asarray([estimated_s]))
            except:
                logging.warning('Rice parameters cannot be worked out')
                self.rice_parameters_are_found = False
                return [math.nan, math.nan]
            else:
                self.rice_parameters_are_found = True
                return [0, s[0]]
        estimated_b = self.mean_e_now / (self.e_now_upper_uncertainty)
        roots = [math.nan, math.nan]
        try:
            roots = fsolve(self.equations_to_be_solved_for_Rice_distribution_parameters,
                           np.asarray([estimated_b, estimated_s]))
        except:
            logging.warning('Rice parameters cannot be worked out')
            self.rice_parameters_are_found = False
        else:
            self.rice_parameters_are_found = True
        return roots

    def pdf(self, e):
        val = i0((e / self.s) * self.b)
        if val == math.inf:
            return 0  # Then math.exp(-((e/s)**2+b**2)/2) = 0
        return math.exp(-((e / self.s) ** 2 + self.b ** 2) / 2) * val

    def cdf(self, e_now):
        value = nquad(self.pdf, [[0, e_now]])
        return value[0]

    def cumulative_density_function_of_present_eccentricity(self, e_now):
        if not (math.isnan(self.s) or math.isnan(self.b)):
            if self.inv_norm == 0:
                return math.inf
            value = self.cdf(e_now) / self.inv_norm
            return value
        logging.warning(
            'Cumulative density function of present eccentricity does not exist for the given e_now and its uncertainties')
        return math.nan

    def probability_density_of_eccentricity(self, e):
        if e > 1 or e < 0:
            return 0
        if e <= self.e_env:
            return self.cumulative_density_function_of_present_eccentricity(e)
        return 0

    def plot_probability_density_of_eccentricity_vs_eccentricity_graph(self):
        eccentricity = np.linspace(0, 1, 100)
        probability_density_of_eccentricity = []
        for i in range(0, len(eccentricity)):
            probability_density_of_eccentricity = probability_density_of_eccentricity + [
                self.probability_density_of_eccentricity(eccentricity[i])]
        M_cdf = []
        for i in range(0, len(eccentricity)):
            M_cdf = M_cdf + [self.cumulative_density_function_of_present_eccentricity(eccentricity[i])]
        M_pdf = []
        for i in range(0, len(eccentricity)):
            M_pdf = M_pdf + [self.pdf(eccentricity[i])]
        plt.plot(eccentricity, probability_density_of_eccentricity,
                 label="Probality density of eccentricity (f(e)) vs. eccentricity (e)")
        # naming the x axis
        plt.xlabel('Eccentricity (e)')
        # naming the y axis
        plt.ylabel('probability density of eccentricity (f(e))')
        # giving a title to my graph
        plt.title('Probability density of eccentricity vs eccentricity')
        # function to show the plot
        plt.show()
        plt.plot(eccentricity, M_cdf, label="cdf of M(e) vs. eccentricity (e)")
        # naming the x axis
        plt.xlabel('Eccentricity (e)')
        # naming the y axis
        plt.ylabel('M_cdf ')
        # giving a title to my graph
        plt.title('cdf of M(e) vs eccentricity')
        # function to show the plot
        plt.show()
        plt.plot(eccentricity, M_pdf, label="pdf of M(e) vs. eccentricity (e)")
        # naming the x axis
        plt.xlabel('Eccentricity (e)')
        # naming the y axis
        plt.ylabel('M_pdf ')
        # giving a title to my graph
        plt.title('M_pdf of eccentricity vs eccentricity')
        # function to show the plot
        plt.show()
        return


class System:
    def __init__(self,
                 primary_mass,
                 secondary_mass,
                 secondary_radius,
                 feh,
                 orbital_period,
                 obliquity,
                 age):
        self.primary_mass = primary_mass
        self.Mprimary = primary_mass
        self.secondary_mass = secondary_mass
        self.Msecondary = secondary_mass
        self.secondary_radius = secondary_radius
        self.Rsecondary = secondary_radius
        self.feh = feh
        self.Porb = orbital_period
        self.orbital_period = orbital_period
        self.obliquity = obliquity
        self.age = age

    def printing(self):
        print('Primary mass = ', self.primary_mass, '=', self.primary_mass.to(un.kg))
        print('Secondary mass = ', self.secondary_mass, '=', self.secondary_mass.to(un.kg))
        print('Secondary radius = ', self.secondary_radius, '=', self.secondary_radius.to(un.m))
        print('Stellar metallicity = ', self.feh.to(un.dimensionless_unscaled))
        print('Orbital period = ', self.orbital_period, '=', self.orbital_period.to(un.s))
        print('Obliquity = ', self.obliquity.to(un.deg))
        print('Age = ', self.age, '=', self.age.to(un.s))

class EnvelopeEccentricityDistribution:

    def __init__(self,
                 path = getPathOfExoplanetSystemsData(), #'/home/mmmahmud/CircularizationDissipationConstraints/data/PS_2021.07.13_00.12.38.csv',
                 maximum_number_of_data_points=math.inf,
                 threshold_value_of_envelope_eccentricity=0.001,
                 constraints=constraints_for_eccentricity_envelope(),
                 largest_acceptable_value_of_envelope_eccentricity=0.5
                 ):

        self.path = path

        ##############################################################################
        class Structure:
            """An empty class used only to hold user defined attributes."""

            def __init__(self, **initial_attributes):
                """Create a class with (optionally) initial attributes."""

                for attribute_name, attribute_value in initial_attributes.items():
                    setattr(self, attribute_name, attribute_value)

            def format(self, prefix=''):
                """Generate a tree-like representation of self."""

                result = ''
                for attr_name in dir(self):
                    if attr_name[0] != '_':
                        attribute = getattr(self, attr_name)
                        if isinstance(attribute, Structure):
                            result += (prefix
                                       +
                                       '|-'
                                       +
                                       attr_name
                                       +
                                       '\n'
                                       +
                                       attribute.format(prefix + '| '))
                        else:
                            result += (prefix
                                       +
                                       '|-'
                                       +
                                       attr_name
                                       +
                                       ': '
                                       +
                                       str(attribute)
                                       +
                                       '\n')
                return result
        def convert_nasa_unit_to_astropy(unit_str):
            """Return the astropy unit matching the one specified in input file."""

            print('Unit str: ' + repr(unit_str))
            if unit_str in ['days', 'hrs']:
                unit_str = unit_str[:-1]
            elif unit_str == 'decimal degrees':
                unit_str = 'degree'
            elif (
                    unit_str in ['dex', 'Earth flux', 'sexagesimal']
                    or
                    unit_str.startswith('log10(')
                    or
                    unit_str.startswith('log(')
            ):
                return None
            elif unit_str == 'Solar mass':
                unit_str = 'solMass'
            elif unit_str == 'Solar radii':
                unit_str = 'solRad'
            elif unit_str.endswith(' mass'):
                unit_str = unit_str.split()[0].lower() + 'Mass'
            elif unit_str.endswith(' radii'):
                unit_str = unit_str.split()[0].lower() + 'Rad'
            elif unit_str.startswith('percent'):
                return 0.01

            print('Converted to: ' + repr(unit_str))

            return Unit(unit_str)

        def read_ages(nasa_planets,
                      age_file_standard='inputs/versioned/getages.txt',
                      age_file_manual_density='inputs/versioned/getages_nodensity.txt',
                      manual_densities='inputs/versioned/age_variables_nodensity.txt'):
            """
            Complete the NASA exoplanet archive planets with age information.

            Args:
                - nasa_planets: The planets read from a CSV file downloaded from the
                                NASA exoplanet archive. On output, this gets updated
                                with the information from the various input files.
                - age_file_standard: The name of the file containing the derived
                                     ages.
                - age_file_manual_density: The name of the file with ages derived
                                           from manually extracted densities.
                - manual_densities: The name of the file containing the manually
                                    extracted densities themselves.

            Returns: None
            """

            def read_file(filename, columns):
                """
                Read one of the input files and update nasa_planets.

                Args:
                    - filename: The name of the file to read.
                    - columns: a dictionary of the quantities to read from the file
                               (keys) and the columns that contain them. The quantity
                               pl_hostname must be among the columns.

                Returns: None
                """

                hostname_list = list(nasa_planets.pl_hostname)
                num_systems = len(hostname_list)
                for quantity, column in columns.items():
                    if not hasattr(nasa_planets, quantity):
                        setattr(nasa_planets,
                                quantity,
                                numpy.full((num_systems,), numpy.nan))
                with open(filename, 'r') as input_file:
                    for line in input_file:
                        entries = line.split()
                        host = entries[columns['pl_hostname']]
                        system_index = 0
                        while (
                                system_index < len(hostname_list)
                                and
                                (
                                        not hostname_list[system_index].startswith(host)
                                        or
                                        (
                                                len(hostname_list[system_index]) > len(host)
                                                and
                                                hostname_list[system_index][len(host)] != ' '
                                        )
                                )
                        ):
                            system_index += 1
                        if system_index == len(hostname_list):
                            continue
                        for quantity, column in columns.items():
                            if quantity != 'pl_hostname':
                                try:
                                    entry_val = int(entries[column])
                                except ValueError:
                                    entry_val = float(entries[column])
                                getattr(nasa_planets, quantity)[system_index] = (
                                    entry_val
                                )

            age_file_columns = dict(pl_hostname=0,
                                    st_mass=2,
                                    st_masserr1=3,
                                    st_rad=6,
                                    st_raderr1=7,
                                    st_age=10,
                                    st_ageerr1=11,
                                    st_lum=14,
                                    st_lumerr1=15)

            density_file_columns = dict(pl_hostname=0,
                                        st_dens=10,
                                        st_denserr1=11,
                                        st_denserr2=12)

            read_file(age_file_standard, age_file_columns)
            read_file(age_file_manual_density, age_file_columns)
            read_file(manual_densities, density_file_columns)

        def read_nasa_planets(csv_filename,
                              eliminate=('SWEEPS-11',
                                         'HD 41004 B',
                                         'PSR J1719-1438',
                                         'K2-22',
                                         'WASP-19 B',
                                         'HATS-18'),
                              fill_missing=manual_data,
                              need_ages=True,
                              add_units=False):
            """
            Read a CSV file downloaded from the NASA Exoplanet Archive to a dict.

            Args:
                csv_filename:    The name of the comma separated file downloaded from
                    http://exoplanetarchive.ipac.caltech.edu.

            Returns:
                A structure with the column names as attributes containing the
                corresponding values properly formatted.
            """

            def do_eliminate():
                """Eliminate the systems listed in eliminate."""
                system_names = []
                for i in range(1, len(data)):
                    name = data[i][0]
                    if type(name) is numpy.bytes_:
                        system_names = system_names + [name.decode()]
                # system_names = [name.decode() for name in data[:, 1]]
                # The above for loop is written instead of the code system_names = [name.decode() for name in data[:, 1]]
                # couple of name in data[:, 1] were not decoded by decode() method, since they were not numpy.bytes_
                # object. So, I have included a checking command: if type(name) is numpy.bytes_

                delete_indices = []
                for system in eliminate:
                    if system in system_names:
                        delete_indices.append(system_names.index(system))
                a = numpy.delete(data, delete_indices, 0)
                return a

            def do_fill_missing(result):
                """Add the data from fill_missing to result."""

                system_names = list(
                    getattr(
                        result,
                        (
                            'hostname' if hasattr(result, 'hostname')
                            else 'fpl_hostname'
                        )
                    )
                )
                for fill_system in fill_missing:
                    try:
                        fill_index = system_names.index(fill_system['pl_hostname'])
                    except ValueError:
                        continue
                    for quantity, value in fill_system.items():
                        if hasattr(result, quantity):
                            target_column = getattr(result, quantity)
                            if isinstance(target_column, Quantity):
                                unit = target_column.unit
                            else:
                                unit = 1
                            target_column[fill_index] = (value * unit)




            with open(csv_filename, 'r') as csv_file:
                while csv_file.readline()[0] == '#':
                    data_start = csv_file.tell()
                csv_file.seek(data_start)
                data = numpy.genfromtxt(csv_file,
                                        delimiter=',',
                                        dtype=None,
                                        comments=None)
            data_columns = []
            for i in data[0]:
                if type(i) is numpy.bytes_:
                    data_columns = data_columns + [i.decode()]

            # data_columns = [col.decode() for col in data[0]]
            # the above loop was written instead of data_columns = [col.decode() for col in data[0]]

            if eliminate:
                data = do_eliminate()

            result = Structure()
            string_columns = ['pl_hostname',
                              'hostname',
                              'pl_name',
                              'pl_discmethod',
                              'discoverymethod',
                              'rastr',
                              'decstr',
                              'pl_bmassprov',
                              'st_optband',
                              'rowupdate',
                              'pl_letter',
                              'pl_tsystemref',
                              'pl_locale',
                              'pl_facility',
                              'pl_telescope',
                              'pl_instrument',
                              'pl_publ_date',
                              'hd_name',
                              'hip_name',
                              'st_spstr',
                              'st_metratio',
                              'st_optmagband',
                              'st_nirmagband',
                              'st_spt',
                              'swasp_id']
            column_name_list = []
            with open(csv_filename, 'r') as csv_file:
                for line in csv_file:
                    if line[0] != '#':
                        continue
                    entries = line.strip().rstrip(')').split()
                    if len(entries) < 4 or entries[1] != 'COLUMN':
                        continue
                    column_name = entries[2].strip(':')
                    column_index = data_columns.index(column_name)
                    column_values = data[:, column_index][1:]
                    if add_units:
                        if entries[-1][-1] == ']':
                            column_units = convert_nasa_unit_to_astropy(
                                line.strip().rstrip(')').rsplit('[', 1)[-1][:-1]
                            )
                        else:
                            column_units = None
                    if (
                            column_name in string_columns
                            or
                            column_name[0] == 'f' and column_name[1:] in string_columns
                            or
                            column_name.endswith('_str')
                            or
                            column_name.endswith('link')
                    ):
                        column_values = [v.decode() for v in column_values]
                    else:
                        print('column_name: ' + repr(column_name))
                        column_values = [
                            numpy.nan if v == b'' else float(v)
                            for v in data[:, column_index][1:]
                        ]
                    column_values = numpy.array(column_values)

                    if add_units and column_units is not None:
                        print(column_name + ' units: ' + repr(column_units))
                        column_values *= column_units
                    setattr(
                        result,
                        column_name,
                        column_values
                    )
                    column_name_list.append(column_name)

            if fill_missing:
                do_fill_missing(result)

            if need_ages:
                read_ages(result)

            for column_name in column_name_list:
                if column_name.endswith('err2'):
                    column = getattr(result, column_name)
                    nan_indices = numpy.isnan(column)
                    column[nan_indices] = -getattr(result,
                                                   column_name[:-1] + '1')[nan_indices]

            return result

        readPlanet = read_nasa_planets(self.path,
                                       eliminate=('SWEEPS-11', 'HD 41004 B', 'PSR J1719-1438', 'K2-22', 'HATS-67 b'),
                                       need_ages=False, )
        ###########################################################################################

        #readPlanet = planetary_system_io.read_nasa_planets(self.path, eliminate=('SWEEPS-11','HD 41004 B','PSR J1719-1438','K2-22'), need_ages=False,)

        self.planet_name = readPlanet.pl_name
        self.orbital_period = readPlanet.pl_orbper  # days
        self.orbital_period_upper_uncertainty = readPlanet.pl_orbpererr1
        self.orbital_period_lower_uncertainty = readPlanet.pl_orbpererr2
        self.orbital_period_limit_flag = readPlanet.pl_orbperlim
        self.primary_mass = readPlanet.st_mass  # solar mass
        self.primary_mass_upper_uncertainty = readPlanet.st_masserr1
        self.primary_mass_lower_uncertainty = readPlanet.st_masserr2
        self.primary_mass_limit_flag = readPlanet.st_masslim
        self.secondary_mass = readPlanet.pl_masse  # Earth mass
        self.secondary_mass_upper_uncertainty = readPlanet.pl_masseerr1
        self.secondary_mass_lower_uncertainty = readPlanet.pl_masseerr2
        self.secondary_mass_limit_flag = readPlanet.pl_masselim
        self.stellar_metallicity = readPlanet.st_met  # or readPlanet.st_metfe
        self.stellar_metallicity_upper_uncertainty = readPlanet.st_meterr1  # or readPlanet.st_metfeerr1
        self.stellar_metallicity_lower_uncertainty = readPlanet.st_meterr2  # or readPlanet.st_metfeerr2
        self.stellar_metallicity_limit_flag = readPlanet.st_metlim
        self.semi_major_axis = readPlanet.pl_orbsmax
        self.semi_major_axis_upper_uncertainty = readPlanet.pl_orbsmaxerr1
        self.semi_major_axis_lower_uncertainty = readPlanet.pl_orbsmaxerr2
        self.semi_major_axis_flag_limit = readPlanet.pl_orbsmaxlim
        self.primary_radius = readPlanet.st_rad  # solar radius
        self.primary_radius_upper_uncertainty = readPlanet.st_raderr1
        self.primary_radius_lower_uncertainty = readPlanet.st_raderr2
        self.primary_radius_limit_flag = readPlanet.st_radlim
        self.secondary_radius = readPlanet.pl_rade  # earth radius
        self.secondary_radius_upper_uncertainty = readPlanet.pl_radeerr1
        self.secondary_radius_lower_uncertainty = readPlanet.pl_radeerr2
        self.secondary_radius_limit_flag = readPlanet.pl_radelim
        self.stellar_age = readPlanet.st_age  # GYear
        self.stellar_age_upper_uncertainty = readPlanet.st_ageerr1
        self.stellar_age_lower_uncertainty = readPlanet.st_ageerr2
        self.stellar_age_limit_flag = readPlanet.st_agelim
        self.eccentricity_now = readPlanet.pl_orbeccen
        self.eccentricity_now_upper_uncertainty = readPlanet.pl_orbeccenerr1
        self.eccentricity_now_lower_uncertainty = readPlanet.pl_orbeccenerr2
        self.eccentricity_now_limit_flag = readPlanet.pl_orbeccenlim
        self.obliquity = readPlanet.pl_orbincl  # degrees
        self.obliquity_upper_uncertainty = readPlanet.pl_orbinclerr1
        self.obliquity_lower_uncertainty = readPlanet.pl_orbinclerr2
        self.obliquity_limit_flag = readPlanet.pl_orbincllim
        self.vsini = readPlanet.st_vsin  # or readPlanet.st_vsini in Km/s
        self.vsini_upper_uncertainty = readPlanet.st_vsinerr1  # or readPlanet.st_vsinierr1
        self.vsini_lower_uncertainty = readPlanet.st_vsinerr2  # or readPlanet.st_vsinierr2
        self.vsini_limit_flag = readPlanet.st_vsinlim
        self.planet_mass_sin_i = readPlanet.pl_msinie  # Earth mass
        self.planet_mass_sin_i_upper_uncertainty = readPlanet.pl_msinieerr1
        self.planet_mass_sin_i_lower_uncertainty = readPlanet.pl_msinieerr2
        self.planet_mass_sin_i_limit_flag = readPlanet.pl_msinielim
        self.stellar_log_g = readPlanet.st_logg  # log10(cm/s**2)
        self.stellar_log_g_upper_uncertainty = readPlanet.st_loggerr1
        self.stellar_log_g_lower_uncertainty = readPlanet.st_loggerr2
        self.stellar_log_g_limit_flag = readPlanet.st_logglim
        self.stellar_effective_temperature = readPlanet.st_teff  # Kelvin
        self.stellar_effective_temperature_upper_uncertainty = readPlanet.st_tefferr1
        self.stellar_effective_temperature_lower_uncertainty = readPlanet.st_tefferr2
        self.stellar_effective_temperature_limit_flag = readPlanet.st_tefflim
        self.stellar_density = readPlanet.st_dens  # gm/cm**3
        self.stellar_density_upper_uncertainty = readPlanet.st_denserr1
        self.stellar_density_lower_uncertainty = readPlanet.st_denserr2
        self.stellar_density_limit_flag = readPlanet.st_denslim
        self.stellar_luminosity = readPlanet.st_lum  # log(solar)
        self.stellar_luminosity_upper_uncertainty = readPlanet.st_lumerr1
        self.stellar_luminosity_lower_uncertainty = readPlanet.st_lumerr2
        self.stellar_luminosity_limit_flag = readPlanet.st_lumlim
        self.ratio_of_planet_to_stellar_radius = readPlanet.pl_ratror
        self.ratio_of_planet_to_stellar_radius_upper_uncertainty = readPlanet.pl_ratrorerr1
        self.ratio_of_planet_to_stellar_radius_lower_uncertainty = readPlanet.pl_ratrorerr2
        self.planet_density = readPlanet.pl_dens
        self.planet_density_upper_uncertainty = readPlanet.pl_denserr1
        self.planet_density_lower_uncertainty = readPlanet.pl_denserr2

        self.envelope_eccentricity_function = self.create_envelope_eccentricity_function(maximum_number_of_data_points,
                                                                                         threshold_value_of_envelope_eccentricity,
                                                                                         largest_acceptable_value_of_envelope_eccentricity,
                                                                                         constraints)

    def create_envelope_eccentricity_function(self,
                                              maximum_number_of_data_points,
                                              threshold_value_of_envelope_eccentricity,
                                              largest_acceptable_value_of_envelope_eccentricity,
                                              constraints=constraints_for_eccentricity_envelope(),
                                              x_attribute='log of semi major axis over planetary radius',
                                              # or 'log of orbital period'
                                              manual_envelope=True,
                                              eliminate=('HATS-67 b', 'HATS-69 b', 'HATS-62 b')):

        def find_records_of_the_points_on_envelope(records_of_certain_attribute_and_eccentricity_now, attribute):
            records_of_the_points_on_envelope = []
            maximum_eccentricity_now = threshold_value_of_envelope_eccentricity
            starting_point_of_the_significant_tidal_dissipation_region_found = False
            end_point_of_the_significant_tidal_dissipation_region_found = False
            for i in range(0, len(records_of_certain_attribute_and_eccentricity_now)):
                present_eccentricity_of_ith_element = records_of_certain_attribute_and_eccentricity_now[i][
                    'present eccentricity']
                if present_eccentricity_of_ith_element > maximum_eccentricity_now:
                    if (not starting_point_of_the_significant_tidal_dissipation_region_found) and i > 0:
                        records_of_the_points_on_envelope = records_of_the_points_on_envelope + [
                            {attribute: records_of_certain_attribute_and_eccentricity_now[i - 1][attribute],
                             'envelope eccentricity': threshold_value_of_envelope_eccentricity}]
                        print('Name of the planet belongs to the binary system on the envelope ',
                              records_of_certain_attribute_and_eccentricity_now[i - 1]['planet name'],
                              attribute, ' = ',
                              records_of_certain_attribute_and_eccentricity_now[i - 1][attribute],
                              'envelope eccentricity = ', threshold_value_of_envelope_eccentricity,
                              'orbital eccentricity limit flag = ',
                              records_of_certain_attribute_and_eccentricity_now[i - 1][
                                  'orbital eccentricity limit flag'])
                    if present_eccentricity_of_ith_element <= largest_acceptable_value_of_envelope_eccentricity:
                        maximum_eccentricity_now = present_eccentricity_of_ith_element
                    if present_eccentricity_of_ith_element > largest_acceptable_value_of_envelope_eccentricity:
                        maximum_eccentricity_now = largest_acceptable_value_of_envelope_eccentricity
                        end_point_of_the_significant_tidal_dissipation_region_found = True
                    starting_point_of_the_significant_tidal_dissipation_region_found = True
                    records_of_the_points_on_envelope = records_of_the_points_on_envelope + [{attribute:
                                                                                                  records_of_certain_attribute_and_eccentricity_now[
                                                                                                      i][
                                                                                                      attribute],
                                                                                              'envelope eccentricity': maximum_eccentricity_now}]
                    print('Name of the planet belongs to the binary system on the envelope ',
                          records_of_certain_attribute_and_eccentricity_now[i]['planet name'],
                          attribute, ' = ',
                          records_of_certain_attribute_and_eccentricity_now[i][attribute],
                          'envelope eccentricity = ', maximum_eccentricity_now,
                          'orbital eccentricity limit flag = ',
                          records_of_certain_attribute_and_eccentricity_now[i]['orbital eccentricity limit flag'])
                    if end_point_of_the_significant_tidal_dissipation_region_found:
                        break
            return records_of_the_points_on_envelope

        if x_attribute == 'log of orbital period':
            # For present eccentricity vs. log of orbital period plot:
            records_of_log_orbital_period_and_eccentricity_now = []
            j = -1
            for i in range(0, len(self.orbital_period)):
                if not (math.isnan(self.orbital_period[i])
                        or math.isnan(self.eccentricity_now[i])
                ) and self.eccentricity_now_limit_flag[i] == 0 and not (self.planet_name[i] in eliminate):
                    if (constraints_for_eccentricity_envelope_are_satisfied(
                            secondary_radius=self.secondary_radius[i],
                            planet_mass_sin_i=self.planet_mass_sin_i[i],
                            constraints=constraints)):
                        records_of_log_orbital_period_and_eccentricity_now = (
                                records_of_log_orbital_period_and_eccentricity_now
                                + [{'log of orbital period': math.log(self.orbital_period[i], 10),
                                    'present eccentricity': self.eccentricity_now[i],
                                    'planet name': self.planet_name[i],
                                    'primary mass': self.primary_mass[i],
                                    'orbital eccentricity limit flag': self.eccentricity_now_limit_flag[i]}])
                        j = j + 1
                if j >= maximum_number_of_data_points - 1:
                    break
            # Sorting
            records_of_log_orbital_period_and_eccentricity_now = sorted(
                records_of_log_orbital_period_and_eccentricity_now,
                key=lambda key_attribute_for_sorting: key_attribute_for_sorting['log of orbital period'])
            # Finding points on the envelope
            records_of_the_points_on_envelope = find_records_of_the_points_on_envelope(
                records_of_log_orbital_period_and_eccentricity_now, x_attribute)
        if x_attribute == 'log of semi major axis over planetary radius':
            # For present eccentricity vs. log of semi-major axis over planetary radius plot:
            records_of_log_semi_major_axis_over_planetary_radius_and_eccentricity_now = []
            j = -1
            for i in range(0, len(self.semi_major_axis)):
                if not (math.isnan(self.semi_major_axis[i])
                        or math.isnan(self.secondary_radius[i])
                        or math.isnan(self.eccentricity_now[i])
                ) and self.eccentricity_now_limit_flag[i] == 0 and self.semi_major_axis_flag_limit[i] == 0 and not (
                        self.planet_name[i] in eliminate):
                    if (constraints_for_eccentricity_envelope_are_satisfied(
                            secondary_radius=self.secondary_radius[i],
                            planet_mass_sin_i=self.planet_mass_sin_i[i],
                            constraints=constraints)):
                        records_of_log_semi_major_axis_over_planetary_radius_and_eccentricity_now = (
                                records_of_log_semi_major_axis_over_planetary_radius_and_eccentricity_now
                                + [{'log of semi major axis over planetary radius': (
                            math.log(self.semi_major_axis[i] / self.secondary_radius[i], 10)),
                            'present eccentricity': self.eccentricity_now[i],
                            'planet name': self.planet_name[i],
                            'primary mass': self.primary_mass[i],
                            'orbital eccentricity limit flag': self.eccentricity_now_limit_flag[i]}])
                        j = j + 1

                if j >= maximum_number_of_data_points - 1:
                    break
            # Sorting
            records_of_log_semi_major_axis_over_planetary_radius_and_eccentricity_now = sorted(
                records_of_log_semi_major_axis_over_planetary_radius_and_eccentricity_now,
                key=lambda key_attribute_for_sorting:
                key_attribute_for_sorting['log of semi major axis over planetary radius'])
            # Finding points on the envelope
            records_of_the_points_on_envelope = find_records_of_the_points_on_envelope(
                records_of_log_semi_major_axis_over_planetary_radius_and_eccentricity_now, x_attribute)

        def envelope_eccentricity_function(x):
            log_of_x = math.log(x, 10)
            min_log_of_x = records_of_the_points_on_envelope[0][x_attribute]
            max_log_of_x = records_of_the_points_on_envelope[-1][x_attribute]
            if log_of_x <= min_log_of_x:
                return threshold_value_of_envelope_eccentricity
            if log_of_x >= max_log_of_x:
                return largest_acceptable_value_of_envelope_eccentricity
            for i in range(1, len(records_of_the_points_on_envelope)):
                if log_of_x == records_of_the_points_on_envelope[i][x_attribute]:
                    return records_of_the_points_on_envelope[i]['envelope eccentricity']
                if ((log_of_x < records_of_the_points_on_envelope[i][x_attribute])
                        and (log_of_x > records_of_the_points_on_envelope[i - 1][x_attribute])):
                    return (records_of_the_points_on_envelope[i]['envelope eccentricity']
                            + (records_of_the_points_on_envelope[i]['envelope eccentricity']
                               - records_of_the_points_on_envelope[i - 1]['envelope eccentricity'])
                            / (records_of_the_points_on_envelope[i][x_attribute]
                               - records_of_the_points_on_envelope[i - 1][x_attribute])
                            * (log_of_x - records_of_the_points_on_envelope[i][x_attribute]))
            return

        def envelope_eccentricity_function_manual(x):
            logx = math.log(x, 10)
            if x_attribute == 'log of orbital period':
                crit1 = -0.11
                crit2 = 0.7
            if x_attribute == 'log of semi major axis over planetary radius':
                crit1 = -2.7
                crit2 = -2.32
            if logx <= crit1:
                return threshold_value_of_envelope_eccentricity
            if logx > crit2:
                return largest_acceptable_value_of_envelope_eccentricity
            if logx > crit1 and logx <= crit2:
                return threshold_value_of_envelope_eccentricity + (logx - crit1) * (
                            largest_acceptable_value_of_envelope_eccentricity - threshold_value_of_envelope_eccentricity) / (
                                   crit2 - crit1)
            return

        if x_attribute == 'log of orbital period':
            self.plot_present_eccentricity_vs_x_attribute(records_of_log_orbital_period_and_eccentricity_now,
                                                          records_of_the_points_on_envelope,
                                                          envelope_eccentricity_function_manual if manual_envelope else envelope_eccentricity_function,
                                                          x_attribute)
        if x_attribute == 'log of semi major axis over planetary radius':
            self.plot_present_eccentricity_vs_x_attribute(
                records_of_log_semi_major_axis_over_planetary_radius_and_eccentricity_now,
                records_of_the_points_on_envelope,
                envelope_eccentricity_function_manual if manual_envelope else envelope_eccentricity_function,
                x_attribute)
        if manual_envelope:
            return envelope_eccentricity_function_manual
        else:
            return envelope_eccentricity_function

    def plot_present_eccentricity_vs_x_attribute(self,
                                                 records_of_x_attribute_and_eccentricity_now,
                                                 records_of_the_points_on_envelope,
                                                 envelope_eccentricity_function,
                                                 x_attribute):
        x = [element[x_attribute] for element in records_of_x_attribute_and_eccentricity_now]
        eccentricity_now = [element['present eccentricity'] for element in records_of_x_attribute_and_eccentricity_now]
        primary_mass = [element['primary mass'] for element in records_of_x_attribute_and_eccentricity_now]
        x_on_envelope = [element[x_attribute] for element in records_of_the_points_on_envelope]
        eccentricity_now_on_envelope = [element['envelope eccentricity'] for element in
                                        records_of_the_points_on_envelope]
        xdata = np.linspace(records_of_x_attribute_and_eccentricity_now[0][x_attribute],
                            records_of_x_attribute_and_eccentricity_now[-1][x_attribute],
                            50)
        ydata = []
        xdata_ = []
        for i in range(0, len(xdata)):
            ydata = ydata + [envelope_eccentricity_function(10 ** xdata[i])]
        for i in range(0, len(xdata)):
            xdata_ = xdata_ + [10**(xdata[i] + 4.37023)]

        x_1 = []
        eccentricity_now_1 = []
        x_2 = []
        eccentricity_now_2 = []
        for i in range(0, len(records_of_x_attribute_and_eccentricity_now)):
            if primary_mass[i] < 1.2:
                x_1 = x_1 + [10**(x[i] + 4.37023)]
                eccentricity_now_1 = eccentricity_now_1 + [eccentricity_now[i]]

            else:
                x_2 = x_2 + [10**(x[i]+4.37023)]
                eccentricity_now_2 = eccentricity_now_2 + [eccentricity_now[i]]
        plt.plot(x_1, eccentricity_now_1, 'x')
        plt.plot(x_2, eccentricity_now_2, 'o')
        plt.plot(xdata_, ydata)
        #plt.plot(x_on_envelope, eccentricity_now_on_envelope, '.')
        # naming the x axis
        if x_attribute == 'log of semi major axis over planetary radius':
            x_attribute_ = 'Semi major axis over planetary radius'
        else:
            x_attribute_ = x_attribute
        plt.xlabel(x_attribute_)
        # naming the y axis
        plt.ylabel('Present Eccentricity')
        # giving a title to my graph
        #string = 'Present Eccentricity vs. ' + x_attribute_
        #plt.title(string)
        plt.xscale("log")
        plt.show()
        return

    def properties_of_ith_binary_system_if_satisfies_constraints(self,
                                                                 i,
                                                                 constraints=constraints()):
        if i > len(self.planet_name) - 1 or i < 0: return None, None, None
        if not (math.isnan(self.orbital_period[i])
                or math.isnan(self.orbital_period_upper_uncertainty[i])
                or math.isnan(self.orbital_period_lower_uncertainty[i])
                or math.isnan(self.primary_mass[i])
                or math.isnan(self.primary_mass_upper_uncertainty[i])
                or math.isnan(self.primary_mass_lower_uncertainty[i])
                or math.isnan(self.secondary_mass[i])
                or math.isnan(self.secondary_mass_upper_uncertainty[i])
                or math.isnan(self.secondary_mass_lower_uncertainty[i])
                or math.isnan(self.stellar_metallicity[i])
                or math.isnan(self.stellar_metallicity_upper_uncertainty[i])
                or math.isnan(self.stellar_metallicity_lower_uncertainty[i])
                or math.isnan(self.secondary_radius[i])
                or math.isnan(self.secondary_radius_upper_uncertainty[i])
                or math.isnan(self.secondary_radius_lower_uncertainty[i])
                or math.isnan(self.stellar_age[i])
                or math.isnan(self.stellar_age_upper_uncertainty[i])
                or math.isnan(self.stellar_age_lower_uncertainty[i])
                or math.isnan(self.eccentricity_now[i])
                or math.isnan(self.eccentricity_now_lower_uncertainty[i])
                or math.isnan(self.eccentricity_now_limit_flag[i])
                or math.isnan(self.eccentricity_now_upper_uncertainty[i])
                or math.isnan(self.semi_major_axis[i])
                or math.isnan(self.semi_major_axis_lower_uncertainty[i])
                or math.isnan(self.semi_major_axis_upper_uncertainty[i])
                or math.isnan(self.stellar_density[i])
                or math.isnan(self.stellar_density_lower_uncertainty[i])
                or math.isnan(self.stellar_density_upper_uncertainty[i])
                or math.isnan(self.stellar_log_g[i])
                or math.isnan(self.stellar_log_g_lower_uncertainty[i])
                or math.isnan(self.stellar_log_g_upper_uncertainty[i])
                or math.isnan(self.stellar_effective_temperature[i])
                or math.isnan(self.stellar_effective_temperature_lower_uncertainty[i])
                or math.isnan(self.stellar_effective_temperature_upper_uncertainty[i])
        ):
            if (constraints_are_satisfied(orbital_period=self.orbital_period[i],
                                          primary_mass=self.primary_mass[i],
                                          secondary_mass=self.secondary_mass[i],
                                          stellar_metallicity=self.stellar_metallicity[i],
                                          eccentricity_now=self.eccentricity_now[i],
                                          stellar_age=self.stellar_age[i],
                                          constraints=constraints)
                    and self.eccentricity_now_limit_flag[i] == 0
            ):
                means = {'primary mass': self.primary_mass[i],
                         'secondary mass': self.secondary_mass[i],
                         'primary radius': self.primary_radius[i],
                         'secondary radius': self.secondary_radius[i],
                         'stellar metallicity': self.stellar_metallicity[i],
                         'orbital period': self.orbital_period[i],
                         'obliquity': self.obliquity[i],
                         'stellar age': self.stellar_age[i],
                         'present eccentricity': self.eccentricity_now[i],
                         'semi major axis': self.semi_major_axis[i],
                         'stellar log g': self.stellar_log_g[i],
                         'stellar density': self.stellar_density[i],
                         'stellar effective temperature': self.stellar_effective_temperature[i],
                         'ratio of planet to stellar radius': self.ratio_of_planet_to_stellar_radius[i]
                         }
                standard_deviations = {'primary_mass_upper_uncertainty': self.primary_mass_upper_uncertainty[i],
                                       'primary_mass_lower_uncertainty': self.primary_mass_lower_uncertainty[i],
                                       'secondary_mass_upper_uncertainty': self.secondary_mass_upper_uncertainty[i],
                                       'secondary_mass_lower_uncertainty': self.secondary_mass_lower_uncertainty[i],
                                       'primary_radius_upper_uncertainty': self.primary_radius_upper_uncertainty[i],
                                       'primary_radius_lower_uncertainty': self.primary_radius_lower_uncertainty[i],
                                       'secondary_radius_upper_uncertainty': self.secondary_radius_upper_uncertainty[i],
                                       'secondary_radius_lower_uncertainty': self.secondary_radius_lower_uncertainty[i],
                                       'stellar_metallicity_upper_uncertainty':
                                           self.stellar_metallicity_upper_uncertainty[i],
                                       'stellar_metallicity_lower_uncertainty':
                                           self.stellar_metallicity_lower_uncertainty[i],
                                       'orbital_period_upper_uncertainty': self.orbital_period_upper_uncertainty[i],
                                       'orbital_period_lower_uncertainty': self.orbital_period_lower_uncertainty[i],
                                       'obliquity_upper_uncertainty': self.obliquity_upper_uncertainty[i],
                                       'obliquity_lower_uncertainty': self.obliquity_lower_uncertainty[i],
                                       'stellar_age_upper_uncertainty': self.stellar_age_upper_uncertainty[i],
                                       'stellar_age_lower_uncertainty': self.stellar_age_lower_uncertainty[i],
                                       'eccentricity_now_upper_uncertainty': self.eccentricity_now_upper_uncertainty[i],
                                       'eccentricity_now_lower_uncertainty': self.eccentricity_now_lower_uncertainty[i],
                                       'semi_major_axis_upper_uncertainty': self.semi_major_axis_upper_uncertainty[i],
                                       'semi_major_axis_lower_uncertainty': self.semi_major_axis_lower_uncertainty[i],
                                       'stellar_log_g_upper_uncertainty': self.stellar_log_g_upper_uncertainty[i],
                                       'stellar_log_g_lower_uncertainty': self.stellar_log_g_lower_uncertainty[i],
                                       'stellar_density_upper_uncertainty': self.stellar_density_upper_uncertainty[i],
                                       'stellar_density_lower_uncertainty': self.stellar_density_lower_uncertainty[i],
                                       'stellar_effective_temperature_upper_uncertainty':
                                           self.stellar_effective_temperature_upper_uncertainty[i],
                                       'stellar_effective_temperature_lower_uncertainty':
                                           self.stellar_effective_temperature_lower_uncertainty[i],
                                       'ratio_of_planet_to_stellar_radius_upper_uncertainty':
                                           self.ratio_of_planet_to_stellar_radius_upper_uncertainty[i],
                                       'ratio_of_planet_to_stellar_radius_lower_uncertainty':
                                           self.ratio_of_planet_to_stellar_radius_lower_uncertainty[i]
                                       }

                return means, standard_deviations, self.planet_name[i]
        return None, None, None

    def print_properties_of_binary_systems_satisfying_constraints(self,
                                                                  constraints=constraints()):
        index_of_binary_system_with_constrained_properties = []
        k = -1
        for i in range(0, len(self.planet_name)):
            means, standard_deviations, planet_name = self.properties_of_ith_binary_system_if_satisfies_constraints(i,
                                                                                                                    constraints=constraints)
            if not (means == None or standard_deviations == None or planet_name == None):
                print('______________________________')
                k = k + 1
                print('k = ', k)
                print('means = ', means, 'standard deviations = ', standard_deviations, ' planet name = ', planet_name)
                print('e_mean ', means['present eccentricity'])
                print('e_now_upper_uncertainty ', standard_deviations['eccentricity_now_upper_uncertainty'])
                print('e_now_lower_uncertainty ', standard_deviations['eccentricity_now_lower_uncertainty'])
                a_over_Rpl = means['semi major axis'] / means['secondary radius']
                print('log of semi major axis over secondary radius = ', math.log(a_over_Rpl, 10))
                print('Envelope eccentricity for semi major axis over secondary radius = ', a_over_Rpl, ' is = ',
                      self.envelope_eccentricity_function(a_over_Rpl))
                index_of_binary_system_with_constrained_properties = index_of_binary_system_with_constrained_properties + [
                    i]
        return index_of_binary_system_with_constrained_properties


class Element:
    def __init__(self, teff, feh, logg, mean_density, debug_plot, lum=None):
        self.Teff = teff
        self.age_cdf_interp_tolerance = 0.0001
        self.debug_plot = debug_plot
        self.debug_plot_dpi = 300
        self.feh = feh
        self.feh_max_cdf_step = 0.1
        self.feh_max_step = 0.1
        self.grid_refine_algorithm = 'worst'
        self.logg = logg
        self.mass_cdf_interp_tolerance = 0.0001
        self.mass_max_step = 0.1
        self.max_discarded_feh_probability = 1e-08
        self.mean_density = mean_density
        self.lum = lum
        self.num_parallel_processes = 4
        self.star_sampler_pickle_fname = 'star_sampler.pkl'
        self.stellar_evolution_interpolator_dir = getStellarEvolutionInterpolatorsDirectory()  #'/home/mmmahmud/poet/stellar_evolution_interpolators'
        self.time_ode_atol = 1e-08
        self.time_ode_max_step = 0.1
        self.time_ode_rtol = 1e-06


class PriorTransform:
    def __init__(self,
                 means,
                 standard_deviations,
                 max_argument_of_phase_lag_function_for_planet=6,
                 min_argument_of_phase_lag_function_for_planet=5,
                 min_log_tidal_break_period=math.log(0.5,10),
                 max_log_tidal_break_period=1,
                 min_power_law_argument=-5,
                 max_power_law_argument=5,
                 max_initial_stellar_spin=15,
                 min_initial_stellar_spin=5
                 ):
        self.means = means
        self.standard_deviations = standard_deviations
        self.max_argument_of_phase_lag_function_for_planet = max_argument_of_phase_lag_function_for_planet
        self.min_argument_of_phase_lag_function_for_planet = min_argument_of_phase_lag_function_for_planet
        self.min_log_tidal_break_period = min_log_tidal_break_period
        self.max_log_tidal_break_period = max_log_tidal_break_period
        self.min_power_law_argument = min_power_law_argument
        self.max_power_law_argument = max_power_law_argument
        self.max_initial_stellar_spin = max_initial_stellar_spin
        self.min_initial_stellar_spin = min_initial_stellar_spin

        #logging.basicConfig(level=logging.DEBUG)

        debug_plot = [('interpolation_performance', 'interp_performance.pdf')]
        teff = split_normal.freeze_error_bar(
            mode=self.means['stellar effective temperature'],
            abs_plus_error=self.standard_deviations['stellar_effective_temperature_upper_uncertainty'],
            abs_minus_error=-self.standard_deviations['stellar_effective_temperature_lower_uncertainty'])
        feh = split_normal.freeze_error_bar(
            mode=self.means['stellar metallicity'],
            abs_plus_error=self.standard_deviations['stellar_metallicity_upper_uncertainty'],
            abs_minus_error=-self.standard_deviations['stellar_metallicity_lower_uncertainty'])
        logg = split_normal.freeze_error_bar(
            mode=self.means['stellar log g'],
            abs_plus_error=self.standard_deviations['stellar_log_g_upper_uncertainty'],
            abs_minus_error=-self.standard_deviations['stellar_log_g_lower_uncertainty'])
        mean_density = split_normal.freeze_error_bar(
            mode=self.means['stellar density'],
            abs_plus_error=self.standard_deviations['stellar_density_upper_uncertainty'],
            abs_minus_error=-self.standard_deviations['stellar_density_lower_uncertainty'])

        config = Element(teff, feh, logg, mean_density, debug_plot)
        constraints = dict()
        constraints['teff'] = config.Teff
        constraints['logg'] = config.logg
        constraints['rho'] = config.mean_density

        likelihood = POETInterpLikelihood(
            **constraints,
            rtol=config.time_ode_rtol,
            atol=config.time_ode_atol,
            max_step=config.time_ode_max_step
        )
        self.star_sampler = StarSampler(likelihood, config)

    def __call__(self, u):
        unit_cube = numpy.array([u[0], u[1], u[2]])
        stellar_metallicity, primary_mass, stellar_age = self.star_sampler.__call__(unit_cube)
        primary_rad = FeHConditionalLikelihoodBase.interpolator('RADIUS', primary_mass, stellar_metallicity)
        primary_radius = primary_rad(stellar_age)
        ratio_of_planet_to_stellar_radius = norm.ppf(u[3], loc=self.means['ratio of planet to stellar radius'], scale=(
                                                                                                                                  self.standard_deviations[
                                                                                                                                      'ratio_of_planet_to_stellar_radius_upper_uncertainty'] -
                                                                                                                                  self.standard_deviations[
                                                                                                                                      'ratio_of_planet_to_stellar_radius_lower_uncertainty']) / 2)
        secondary_radius = (
                                       ratio_of_planet_to_stellar_radius ** 0.5) * primary_radius * const.R_sun.value / const.R_earth.value
        secondary_mass = norm.ppf(u[4], loc=self.means['secondary mass'], scale=(self.standard_deviations[
                                                                                     'secondary_mass_upper_uncertainty'] -
                                                                                 self.standard_deviations[
                                                                                     'secondary_mass_lower_uncertainty']) / 2)
        initial_stellar_spin = self.min_initial_stellar_spin + u[5] * (
                    self.max_initial_stellar_spin - self.min_initial_stellar_spin)
        argument_of_phase_lag_function_for_planet = self.min_argument_of_phase_lag_function_for_planet + u[6] * (
                    self.max_argument_of_phase_lag_function_for_planet - self.min_argument_of_phase_lag_function_for_planet)
        tidal_break_period = 10**(self.min_log_tidal_break_period + u[7] * (
                    self.max_log_tidal_break_period - self.min_log_tidal_break_period))
        power_law_argument = self.min_power_law_argument + u[8] * (
                    self.max_power_law_argument - self.min_power_law_argument)

        parameters_for_evolution = {'primary mass': primary_mass,
                                    'stellar age': stellar_age,
                                    'secondary radius': secondary_radius,
                                    'stellar metallicity': stellar_metallicity,
                                    'secondary mass': secondary_mass,
                                    'initial stellar spin': initial_stellar_spin,
                                    'argument of phase lag function for planet': argument_of_phase_lag_function_for_planet,
                                    'tidal break period': tidal_break_period,
                                    'power law argument': power_law_argument}

        return parameters_for_evolution


class LogLikelihood:
    def __init__(self,
                 prior_transform_instance,
                 orbital_period,
                 obliquity,
                 probability_density_of_eccentricity,
                 e_env,
                 system_name = 'Star-Exoplanet',
                 initial_eccentricity=0.5,
                 constraints=constraints(),
                 spin_frequency_breaks_for_planet=None,
                 spin_frequency_powers_for_planet=np.array([0.0]),
                 Q0 = 5
                 ):
        self.prior_transform_instance = prior_transform_instance
        self.orbital_period = orbital_period
        self.obliquity = obliquity
        self.probability_density_of_eccentricity = probability_density_of_eccentricity
        self.constraints = constraints
        self.initial_eccentricity = initial_eccentricity
        self.spin_frequency_breaks_for_planet = spin_frequency_breaks_for_planet
        self.spin_frequency_powers_for_planet = spin_frequency_powers_for_planet
        self.e_env = e_env
        self.system_name = system_name
        self.calculated_eccentricity_now = None
        self.Q0 = Q0

    def priors(self,
               parameters_for_evolution):
        smallest = self.constraints[0]
        largest = self.constraints[1]

        def prior_parameter(parameter, parameter_name):
            if (parameter > smallest[parameter_name] and parameter < largest[parameter_name]):
                return True
            return False

        priors = True
        for parameter_name in ['primary mass', 'secondary mass', 'stellar metallicity', 'stellar age']:
            priors = priors and prior_parameter(parameters_for_evolution[parameter_name], parameter_name)

        return priors

    def log_prob(self, parameters_for_evolution):
        primary_mass = parameters_for_evolution['primary mass']
        stellar_age = parameters_for_evolution['stellar age']
        secondary_radius = parameters_for_evolution['secondary radius']
        stellar_metallicity = parameters_for_evolution['stellar metallicity']
        secondary_mass = parameters_for_evolution['secondary mass']
        initial_stellar_spin = parameters_for_evolution['initial stellar spin']
        argument_of_phase_lag_function_for_planet = parameters_for_evolution[
            'argument of phase lag function for planet']
        tidal_break_period = parameters_for_evolution['tidal break period']
        power_law_argument = parameters_for_evolution['power law argument']

        print('The parameters for evolution are: ')
        print('primary mass = ', primary_mass)
        print('stellar age = ', stellar_age)
        print('secondary radius = ', secondary_radius)
        print('stellar metallicity = ', stellar_metallicity)
        print('secondary mass = ', secondary_mass)
        print('initial stellar spin = ', initial_stellar_spin)
        print('argument of phase lag function for planet = ', argument_of_phase_lag_function_for_planet)
        print('tidal break period = ', tidal_break_period)
        print('power law argument = ', power_law_argument)

        priors = self.priors({'primary mass': primary_mass,
                              'secondary mass': secondary_mass,
                              'stellar metallicity': stellar_metallicity,
                              'stellar age': stellar_age})

        if not priors:
            return -np.inf

        star_exoplanet_binary_system = System(primary_mass=primary_mass * un.solMass,
                                              secondary_mass=secondary_mass * un.earthMass,
                                              secondary_radius=secondary_radius * un.earthRad,
                                              feh=stellar_metallicity * un.dimensionless_unscaled,
                                              orbital_period=self.orbital_period * un.d,
                                              obliquity=self.obliquity * un.deg,
                                              age=stellar_age * un.Gyr)
        break_frequency = 2 * math.pi / tidal_break_period
        tidal_frequency_breaks_for_planet = None
        tidal_frequency_powers_for_planet = None
        reference_argument_of_phase_lag_function_for_planet = argument_of_phase_lag_function_for_planet
        if power_law_argument < 0:
            tidal_frequency_breaks_for_planet = np.array([break_frequency])
            tidal_frequency_powers_for_planet = np.array([0.0, power_law_argument])

            # tidal_frequency_breaks_for_planet = np.array([2 * math.pi / 20, break_frequency])
            # tidal_frequency_powers_for_planet = np.array([1.0, 0.0, power_law_argument])

        if power_law_argument > 0 or power_law_argument == 0:
            tidal_frequency_breaks_for_planet = np.array([2 * math.pi / 20, break_frequency])
            tidal_frequency_powers_for_planet = np.array([0.0, power_law_argument, 0.0])
            reference_argument_of_phase_lag_function_for_planet += power_law_argument * (math.log(tidal_frequency_breaks_for_planet[0], 10) - math.log(tidal_frequency_breaks_for_planet[1], 10))

            # tidal_frequency_powers_for_planet = np.array([1.0, power_law_argument, 0.0])
            # reference_argument_of_phase_lag_function_for_planet = argument_of_phase_lag_function_for_planet + power_law_argument * (
            # math.log(20.0, 10) - math.log(tidal_break_period, 10))



        #if power_law_argument == 0:
            #reference_argument_of_phase_lag_function_for_planet = argument_of_phase_lag_function_for_planet
            #tidal_frequency_powers_for_planet = np.array([1.0, 0.0])

        dissipation = dict(
            primary=None,
            secondary=dict(
                tidal_frequency_breaks=tidal_frequency_breaks_for_planet,
                spin_frequency_breaks=self.spin_frequency_breaks_for_planet,
                tidal_frequency_powers=tidal_frequency_powers_for_planet,
                spin_frequency_powers=self.spin_frequency_powers_for_planet,
                reference_phase_lag=phase_lag(reference_argument_of_phase_lag_function_for_planet)
            )
        )
        yes = True
        if yes:
            evolutionary_history = find_evolution(system=star_exoplanet_binary_system,
                                                  interpolator=FeHConditionalLikelihoodBase.interpolator,
                                                  dissipation=dissipation,
                                                  max_age=stellar_age * un.Gyr,
                                                  initial_eccentricity=self.initial_eccentricity * un.dimensionless_unscaled,
                                                  initial_obliquity=0.0,
                                                  disk_period=initial_stellar_spin * un.d,
                                                  disk_dissipation_age=2e-2 * un.Gyr,
                                                  primary_wind_strength=0.17,
                                                  primary_wind_saturation=2.78,
                                                  primary_core_envelope_coupling_timescale=0.05 * un.Gyr,
                                                  secondary_wind_strength=0.0,
                                                  secondary_wind_saturation=100.0,
                                                  secondary_core_envelope_coupling_timescale=0.05 * un.Gyr,
                                                  orbital_period_tolerance=1e-6,
                                                  solve=True,
                                                  secondary_is_star=False)

            calculated_eccentricity_now = evolutionary_history.eccentricity[- 1]

        if not yes:
            calculated_eccentricity_now = self.prior_transform_instance.means[
                                              'present eccentricity'] + 0.5 * (
                                                  self.e_env - self.prior_transform_instance.means[
                                              'present eccentricity'])
        self.calculated_eccentricity_now = calculated_eccentricity_now

        if calculated_eccentricity_now >= 0 and calculated_eccentricity_now <= 1:
            probability_density_of_the_calculated_eccentricity = self.probability_density_of_eccentricity(
                calculated_eccentricity_now)
            probability_density = probability_density_of_the_calculated_eccentricity * priors
            if probability_density == 0:
                return -np.inf
            if probability_density < 0:
                logging.warning('Probability density cannot be less than zero.')
                return None
            return np.log(probability_density)
        logging.warning('Calculated present eccentricity can neither be less than zero nor greater than one')
        return None


    def generate_successful_walkers_aux(self,
                                    u,
                                    number_of_discovered_walkers,
                                    p0_file_is_being_updated,
                                    walkers,
                                    nwalkers,
                                    ndim,
                                    p0_file_name,
                                    minprob = 0.0001
                                    ):

        system = self.system_name + '_p0_'
        pid = os.getpid()
        date_time = datetime.now().strftime('%Y%m%d%H%M%S')

        filename = 'logging/' + system + '_processor_' + str(pid)  + 'date_time_' + date_time + '.logging'
        msg_file_name = 'logging/msg_'+str(pid)+ 'date_time_' + date_time + '_.txt'
        msg_file = os.open(msg_file_name,
                           os.O_WRONLY | os.O_TRUNC | os.O_CREAT | os.O_DSYNC,
                           mode=0o666
                           )

        os.dup2(msg_file, 1)
        os.dup2(msg_file, 2)

        logging.basicConfig(
            filename=filename,
            level=logging.DEBUG,
            format='%(levelname)s %(asctime)s %(name)s: %(message)s | %(pathname)s.%(funcName)s:%(lineno)d'
        )

        log_likelihood, parameters_for_evolution = self(u)
        print('u  = ', u)
        print('log p = ', log_likelihood)

        p0_file_exists = os.path.exists(p0_file_name)

        if (not math.isinf(log_likelihood)): # and (log_likelihood > np.log(minprob)):
            print('number of discovered walkers = ', number_of_discovered_walkers.value)
            if p0_file_exists:
                print(p0_file_name, ' file was previously created and now being updated.')
                while True:
                    print('I am here at 1')
                    if  ((p0_file_is_being_updated.value == 0) and (number_of_discovered_walkers.value < nwalkers)):
                        p0_file_is_being_updated.value = 1
                        print('Green signal for updating = ', p0_file_is_being_updated.value)
                        p0_file = open(p0_file_name, 'rb')
                        p0 = np.load(p0_file)
                        p0_file.close()
                        p0_file = open(p0_file_name, 'wb')
                        p0 = np.vstack((p0, u))
                        np.save(p0_file, p0)
                        p0_file.close()
                        walkers.put(u)
                        number_of_discovered_walkers.value = number_of_discovered_walkers.value + 1
                        p0_file_is_being_updated.value = 0
                        break
                    if not (number_of_discovered_walkers.value < nwalkers):
                        break
            else:
                print(p0_file_name, ' file did not previously exist. It will be created now and first walker will be loaded now.')
                while True:
                    print('I am here at 2')
                    if ((p0_file_is_being_updated.value == 0) and (number_of_discovered_walkers.value < nwalkers)):
                        p0_file_is_being_updated.value = 1
                        print('Green signal for updating = ', p0_file_is_being_updated.value)
                        p0_file = open(p0_file_name, 'wb')
                        np.save(p0_file, u)
                        p0_file.close()
                        walkers.put(u)
                        number_of_discovered_walkers.value = number_of_discovered_walkers.value + 1
                        p0_file_is_being_updated.value = 0
                        break
                    if not (number_of_discovered_walkers.value < nwalkers):
                        break



        if number_of_discovered_walkers.value < nwalkers:
            y = randint(1,10)
            for i in range(0, y):
                u = numpy.random.rand(ndim)
            self.generate_successful_walkers_aux(u,
                                             number_of_discovered_walkers,
                                             p0_file_is_being_updated,
                                             walkers,
                                             nwalkers,
                                             ndim,
                                             p0_file_name
                                             )



    def generate_successful_walkers(self,
                                p0_file_name,
                                nwalkers=32,
                                ndim=9,
                                nprocessors = 6):
        array_of_processes = []
        number_of_discovered_walkers = Value('i', 0)
        walkers = Queue()
        p0_file_is_being_updated = Value('i', 0)

        i = 0
        while i<nprocessors:
            u = numpy.random.rand(ndim)
            process = Process(target=self.generate_successful_walkers_aux,
                              args=(u,
                                    number_of_discovered_walkers,
                                    p0_file_is_being_updated,
                                    walkers,
                                    nwalkers,
                                    ndim,
                                    p0_file_name
                                    ))
            array_of_processes.append(process)
            i = i+1

        j = 0
        while j<i:
            array_of_processes[j].start()
            j = j + 1

        p0 = [walkers.get(block=True) for _ in range(nwalkers)]
        for process in array_of_processes:
            process.terminate()
            process.join()

        return p0


    def MCMC(self,
             nwalkers=28,
             ndim=9,
             reset_backend = False):

        config = ConfigObjectForLogging(system=system)
        mcmc_progress_file_name = '%(system)s_mcmc_progress.h5' % dict(system=self.system_name)
        p0_file_name = '%(system)s_p0_file.npy' % dict(system=self.system_name)

        p0_file_exists = os.path.exists(p0_file_name)
        backend_file_exists = os.path.exists(mcmc_progress_file_name)


        if (not p0_file_exists):
            print('Initially the file ', p0_file_name, ' did not exist.')
            print('The walkers are going to be generated for the first time.')
            print('The file ', p0_file_name, ' will be created and the walkers will be stored there.')

            p0 = self.generate_successful_walkers(p0_file_name,
                                              nwalkers,
                                              ndim)
        else:
            print('The file ', p0_file_name, ' existed previously.')
            print('Previously worked out walkers will be loaded in the code for running MCMC.')
            p0_file = open(p0_file_name, 'rb')
            p0 = np.load(p0_file)
            p0_file.close()
            print('The already discovered walkers are: ', p0)
            number_of_already_stored_walkers = p0.size/ndim
            number_of_walkers_yet_to_be_found =(int) (nwalkers - number_of_already_stored_walkers)
            if number_of_walkers_yet_to_be_found > 0:
                print('New walkers are going to be discovered')
                p0 = np.vstack((p0, self.generate_successful_walkers(p0_file_name,
                                                                     number_of_walkers_yet_to_be_found,
                                                                     ndim)))
                print('All walkers are: ', p0)
            if number_of_walkers_yet_to_be_found < 0:
                p0 = p0[0:nwalkers]

        with Pool(config.num_parallel_processes,
                  initializer=setup_process,
                  initargs=[config],
                  maxtasksperchild=1) as pool:
            backend = emcee.backends.HDFBackend(mcmc_progress_file_name)
            if reset_backend:
                backend.reset(nwalkers, ndim)
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.__call__, pool=pool, backend=backend)

            if backend_file_exists:
                print('backend file exists.')
                sampler.run_mcmc(None, 5, progress = True)
            else:
                sampler.run_mcmc(p0, 5, progress = True)


            blobs = sampler.get_blobs(flat=True)

            figure = corner.corner(blobs, labels=['M*', #Mass of the parent star
                                                  'age', #Age
                                                  'Rp', #Planetary radius
                                                  'Fe/H_*', #Stellar Metallicity
                                                  'Mp', #Planetary mass
                                                  'initSpin*', #Initial stellar spin
                                                  'Qpl', #Log of the tidal quality factor for planet, i.e. argument of phase lag function
                                                  'tidal break point',
                                                  'alpha',
                                                  'e_now', #Present eccentricity
                                                  'log(f(e_now))'], #log likelihood of present eccentricity
                                   quantiles=[0.16, 0.5, 0.84],
                                   show_titles=True, title_kwargs={"fontsize": 12})
            plt.show()
            figfilename = "%(system)s_MCMC.pdf" % dict(system=self.system_name)
            figure.savefig(figfilename, bbox_inches='tight')
        return

    def __call__(self, u):
        for i in range(0, 9):
            if u[i] > 1 or u[i] < 0:
                return -np.inf, np.array([None, None, None, None, None, None, None, None, None, None, None])
        parameters_for_evolution = self.prior_transform_instance(u)

        params = np.array([parameters_for_evolution['primary mass'],
                           parameters_for_evolution['stellar age'],
                           parameters_for_evolution['secondary radius'],
                           parameters_for_evolution['stellar metallicity'],
                           parameters_for_evolution['secondary mass'],
                           parameters_for_evolution['initial stellar spin'],
                           parameters_for_evolution['argument of phase lag function for planet'],
                           parameters_for_evolution['tidal break period'],
                           parameters_for_evolution['power law argument']])
        log_prob_parameters_for_evolution = self.log_prob(parameters_for_evolution)
        if np.isinf(-log_prob_parameters_for_evolution):
            return -np.inf, np.array([None, None, None, None, None, None, None, None, None, None, None])
        params = np.append(params, [self.calculated_eccentricity_now, log_prob_parameters_for_evolution])
        return log_prob_parameters_for_evolution, params


class ConfigObjectForLogging:
    def __init__(self,
                 system):
        self.fname_datetime_format = "%m/%d/%Y"
        self.system = system
        self.std_out_err_fname = 'sampling_output/%(system)s_%(now)s_%(pid)d.outerr'
        self.logging_fname = 'logging/%(system)s_%(now)s_%(pid)d.logging'
        self.logging_datetime_format = "%m/%d/%Y"
        self.num_parallel_processes = 6
        self.logging_message_format = '%(levelname)s %(asctime)s %(name)s: %(message)s | %(pathname)s.%(funcName)s:%(lineno)d'
        self.logging_level = logging.WARNING


class InitializationOfSamplingPropertiesOfSystem:
    def __init__(self,
                 serialized_directory = getStellarEvolutionInterpolatorsDirectory(), # '/home/mmmahmud/poet/stellar_evolution_interpolators',
                 eccentricity_expansion_fname= getEccentricityExpansionCoefficientsFile(), # b"/media/mmmahmud/USB/eccentricity_expansion_coef_O400.sqlite"
                 ):

        # mp.set_start_method('forkserver')
        manager = StellarEvolutionManager(serialized_directory)
        interpolator = manager.get_interpolator_by_name('default')
        FeHConditionalLikelihoodBase.set_interpolator(interpolator)
        print('coefficients ')
        orbital_evolution_library.prepare_eccentricity_expansion(
            eccentricity_expansion_fname,
            1e-4,
            True,
            True
        )


class SamplingPropertiesOfSystem:
    def __init__(self,
                 means,
                 standard_deviations,
                 system_name = 'Star-Exoplanet',
                 envelope_eccentricity_function=None,
                 initial_eccentricity=0.5,
                 initial_stellar_spin=5,
                 max_argument_of_phase_lag_function_for_planet=6,
                 min_argument_of_phase_lag_function_for_planet=5,
                 min_tidal_break_period=0.8,
                 max_tidal_break_period=10,
                 min_power_law_argument=-5,
                 max_power_law_argument=5,
                 max_initial_stellar_spin=15,
                 min_initial_stellar_spin=5,
                 constraints=constraints(),
                 spin_frequency_breaks_for_planet=None,
                 spin_frequency_powers_for_planet=np.array([0.0]),
                 find_argument_of_phase_lag_function_for_planet_range_auto=False):

        logging_filename = 'logging/' + system_name + '_start.logging'

        logging.basicConfig(
            filename=logging_filename,
            level=logging.DEBUG
        )

        self.initial_eccentricity = initial_eccentricity
        self.initial_stellar_spin = initial_stellar_spin
        self.max_argument_of_phase_lag_function_for_planet = max_argument_of_phase_lag_function_for_planet
        self.min_argument_of_phase_lag_function_for_planet = min_argument_of_phase_lag_function_for_planet
        self.max_tidal_break_period = max_tidal_break_period
        self.min_tidal_break_period = min_tidal_break_period
        self.max_power_law_argument = max_power_law_argument
        self.min_power_law_argument = min_power_law_argument
        self.max_initial_stellar_spin = max_initial_stellar_spin
        self.min_initial_stellar_spin = min_initial_stellar_spin
        self.spin_frequency_breaks_for_planet = spin_frequency_breaks_for_planet
        self.spin_frequency_powers_for_planet = spin_frequency_powers_for_planet

        if envelope_eccentricity_function == None:
            envelope_eccentricity_distribution_object = EnvelopeEccentricityDistribution()
            self.envelope_eccentricity_function = envelope_eccentricity_distribution_object.envelope_eccentricity_function
        else:
            self.envelope_eccentricity_function = envelope_eccentricity_function

        if (constraints_are_satisfied(orbital_period=means['orbital period'],
                                      primary_mass=means['primary mass'],
                                      secondary_mass=means['secondary mass'],
                                      stellar_metallicity=means['stellar metallicity'],
                                      eccentricity_now=means['present eccentricity'],
                                      stellar_age=means['stellar age'],
                                      constraints=constraints)):
            self.means = means
            self.standard_deviations = standard_deviations
            self.prior_transform_instance = PriorTransform(means,
                                                           standard_deviations,
                                                           max_argument_of_phase_lag_function_for_planet,
                                                           min_argument_of_phase_lag_function_for_planet,
                                                           min_tidal_break_period,
                                                           max_tidal_break_period,
                                                           min_power_law_argument, max_power_law_argument,
                                                           max_initial_stellar_spin, min_initial_stellar_spin)

            self.e_env = self.envelope_eccentricity_function(
                x=self.means['semi major axis'] / self.means['secondary radius'])

            eccentricity_distribution_object = EccentricityDistribution(self.means['present eccentricity'],
                                                                        self.standard_deviations[
                                                                            'eccentricity_now_upper_uncertainty'],
                                                                        self.standard_deviations[
                                                                            'eccentricity_now_lower_uncertainty'],
                                                                        self.e_env)

            eccentricity_distribution_object.plot_probability_density_of_eccentricity_vs_eccentricity_graph()
            self.probability_density_of_eccentricity = eccentricity_distribution_object.probability_density_of_eccentricity

            self.log_likelihood_instance = LogLikelihood(self.prior_transform_instance,
                                                         self.means['orbital period'],
                                                         0,  # obliquity
                                                         self.probability_density_of_eccentricity,
                                                         self.e_env,
                                                         system_name,
                                                         initial_eccentricity,
                                                         constraints,
                                                         spin_frequency_powers_for_planet,
                                                         spin_frequency_powers_for_planet
                                                         )


            self.log_likelihood_instance.MCMC()

            if find_argument_of_phase_lag_function_for_planet_range_auto:
                min_Qpl, max_Qpl = self.determine_a_suitable_range_of_argument_of_phase_lag_function_for_planet()
                self.max_argument_of_phase_lag_function_for_planet = max_Qpl
                self.min_argument_of_phase_lag_function_for_planet = min_Qpl
                print('minimum Qpl = ', min_Qpl, ' maximum Qpl = ', max_Qpl)

    def determine_a_suitable_range_of_argument_of_phase_lag_function_for_planet(self, ndim=9):

        u = numpy.random.rand(ndim)
        parameters_for_evolution = self.prior_transform_instance(u)
        print(parameters_for_evolution)

        min_value_found = False
        max_value_found = False
        i = 0
        init = 5.0
        while not min_value_found:
            parameters_for_evolution['argument of phase lag function for planet'] = init + i * 0.25
            eccentricity = self.log_likelihood_instance.calculated_eccentricity_now
            if eccentricity > self.means['eccentricity']:
                min_value_found = True
            i = i + 1
        min_argument_of_phase_lag_function_for_planet = init + (i - 1) * 0.25
        while not max_value_found:
            parameters_for_evolution['argument of phase lag function for planet'] = init + i * 0.25
            eccentricity = self.log_likelihood_instance.calculated_eccentricity_now
            if eccentricity > self.means['eccentricity']:
                max_value_found = True
            i = i + 1
        max_argument_of_phase_lag_function_for_planet = init + (i - 1) * 0.25

        return min_argument_of_phase_lag_function_for_planet, max_argument_of_phase_lag_function_for_planet

    def testing_log_prob(self, ndim=9):

        u = numpy.random.rand(ndim)
        parameters_for_evolution = self.prior_transform_instance(u)
        print(parameters_for_evolution)

        prob = []
        Qpl = []
        eccentricity = []
        k = -1
        init = 5

        for i in range(0, 16):
            parameters_for_evolution['argument of phase lag function for planet'] = init + i * 0.25
            k = k + 1
            Qpl = Qpl + [init + i * 0.25]
            prob = prob + [self.log_likelihood_instance.log_prob(parameters_for_evolution)]
            eccentricity = eccentricity + [self.log_likelihood_instance.calculated_eccentricity_now]
            print('Qpl = ', Qpl[k], ' log prob = ', prob[k])
            plt.plot(Qpl, prob, label=("Prob vs. Qpl when initial eccentricity = ", self.initial_eccentricity,
                                       ' for orbital period = ', self.means['orbital period']))
            # naming the x axis
            plt.xlabel('Qpl')
            # naming the y axis
            plt.ylabel('log-prob')
            # giving a title to my graph
            plt.title('Log Probability vs Qpl')
            # show a legend on the plot
            plt.legend()
            # function to show the plot
            plt.show()

            plt.plot(Qpl, eccentricity, label=("Calculated eccentricity vs. Qpl when initial eccentricity = ",
                                               self.initial_eccentricity,
                                               ' for orbital period = ', self.means['orbital period'],
                                               ' present eccentricity = ', self.means['present eccentricity'],
                                               ' envelope eccentricity = ', self.e_env))
            # naming the x axis
            plt.xlabel('Qpl')
            # naming the y axis
            plt.ylabel('eccentricity')
            # giving a title to my graph
            plt.title('Eccentricity vs Qpl')
            # show a legend on the plot
            plt.legend()
            # function to show the plot
            plt.show()

        return


if __name__ == '__main__':

    # analysis_on_Nasa_exoplanet_data()
    print('**********************************************************')
    test1 = EccentricityDistribution(mean_e_now=0.39,
                                     e_now_upper_uncertainty=0.2,
                                     e_now_lower_uncertainty=-0.2,
                                     e_env=0.40)
    test1.plot_probability_density_of_eccentricity_vs_eccentricity_graph()
    print('*********************************************************')
    test2 = EnvelopeEccentricityDistribution()
    print('Binary systems whose probability density of eccentricity can be figured out:')
    index = test2.print_properties_of_binary_systems_satisfying_constraints()
    means, standard_deviations, system_name = test2.properties_of_ith_binary_system_if_satisfies_constraints(index[15])
    means['ratio of planet to stellar radius'] = 0.0149
    standard_deviations['ratio_of_planet_to_stellar_radius_upper_uncertainty'] = 0.0002
    standard_deviations['ratio_of_planet_to_stellar_radius_lower_uncertainty'] = -0.0002
    print('Print properties of the chosen binary system: means = ', means, ' standard deviations = ',
          standard_deviations, ' Star-Exoplanet system name = ', system_name)
    print('*********************************************************')
    InitializationOfSamplingPropertiesOfSystem()

    if FeHConditionalLikelihoodBase.interpolator == None:
        print('None')
    else:
        print('good')

    test3 = SamplingPropertiesOfSystem(means,
                                       standard_deviations,
                                       system_name=system_name,
                                       envelope_eccentricity_function=test2.envelope_eccentricity_function
                                       )
