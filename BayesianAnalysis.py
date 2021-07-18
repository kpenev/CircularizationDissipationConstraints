import math
import sys
print(sys.path)
import planetary_system_io
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from abc import ABCMeta, abstractmethod
import emcee
import sys
from scipy.stats import rice
from scipy.stats import norm
from scipy.special import erf
from scipy.optimize import fsolve
from sympy import *
from scipy.special import i0
from scipy.integrate import nquad
##########################
from manual_exoplanet_data import data as manual_data
from astropy.units import Unit, Quantity
#######################
if not sys.warnoptions:
    import warnings

    from sqlalchemy.exc import SAWarning

    warnings.filterwarnings('ignore',
                            r"^Dialect sqlite\+pysqlite does \*not\* support Decimal objects natively\, "
                            "and SQLAlchemy must convert from floating point - rounding errors and other "
                            "issues may occur\. Please consider storing Decimal numbers as strings or "
                            "integers on this platform for lossless storage\.$",
                            SAWarning, r'^sqlalchemy\.sql\.type_api$')

    #warnings.filterwarnings('ignore', r".*support Decimal objects natively", SAWarning, r'^sqlalchemy\.sql\.sqltypes$')

    #warnings.simplefilter("error")

sys.path.append('/home/mmmahmud/poet/PythonPackage')
sys.path.append('../scripts')

from orbital_evolution.evolve_interface import library as \
    orbital_evolution_library

from reproduce_system import *

def analysis_on_Nasa_exoplanet_data():
    eccentricity_expansion_fname = b"/home/mmmahmud/poet/scripts/eccentricity_expansion_coef.txt"
    orbital_evolution_library.read_eccentricity_expansion_coefficients(
        eccentricity_expansion_fname
    )
    serialized_dir = '/home/mmmahmud/poet/stellar_evolution_interpolators'
    manager = StellarEvolutionManager(serialized_dir)
    interpolator = manager.get_interpolator_by_name('default')
    print('ddddd = ', interpolator)
    evolutionary_data = test_Nasa_Exoplanet_data(interpolator=(interpolator, interpolator))
    return evolutionary_data


def test_Nasa_Exoplanet_data(interpolator,
                             path='/home/mmmahmud/CircularizationDissipationConstraints/data/planets_2020.04.10_14.52.24.csv'):
    """
    Testing TransitingExoplanet by Nasa Exoplanet data
    """

    readPlanet = planetary_system_io.read_nasa_planets(path,
                                                       eliminate=('SWEEPS-11',
                                                                  'HD 41004 B',
                                                                  'PSR J1719-1438',
                                                                  'K2-22'),
                                                       need_ages=False,
                                                       )

    planet_name = readPlanet.pl_name
    orbital_period = readPlanet.pl_orbper  # days
    primary_mass = readPlanet.st_mass  # solar mass
    secondary_mass = readPlanet.pl_masse  # Earth mass
    metallicity = readPlanet.st_metfe
    primary_radius = readPlanet.st_rad  # solar radius
    secondary_radius = readPlanet.pl_rade  # earth radius
    stellar_age = readPlanet.st_age  # GYear
    eccentricity = readPlanet.pl_orbeccen
    obliquity = readPlanet.pl_orbincl  # degrees
    vsini = readPlanet.st_vsini  # km/s

    evolutionary_data = []

    j = 0
    for i in range(0, len(planet_name)):
        if not (math.isnan(orbital_period[i])
                or math.isnan(primary_mass[i])
                or math.isnan(secondary_mass[i])
                or math.isnan(metallicity[i])
                or math.isnan(secondary_radius[i])
                or math.isnan(stellar_age[i])
                or math.isnan(eccentricity[i])
                or math.isnan(obliquity[i])
                or math.isnan(vsini[i])):
            if orbital_period[i] <= 10000 and (primary_mass[i] > 0.4 and primary_mass[i] < 1.2) and (
                    metallicity[i] > -1.014 and metallicity[i] < 0.537):
                j = j + 1

                a = System(primary_mass[i] * u.solMass,
                           secondary_mass[i] * u.earthMass,
                           secondary_radius[i] * u.earthRad,
                           metallicity[i] * u.dimensionless_unscaled,
                           orbital_period[i] * u.d,
                           obliquity[i] * u.deg,
                           stellar_age[i] * u.Gyr)

                a.printing()
                print('eccentricity found in the NASA archive is ', eccentricity[i])

                dissipation = dict(
                    primary=None,
                    secondary=dict(
                        #tidal_frequency_breaks=None,
                        tidal_frequency_breaks=np.array([2 * math.pi / 20]),
                        spin_frequency_breaks=None,
                        #tidal_frequency_powers=np.array([0.0]),
                        tidal_frequency_powers=np.array([1.0, 0.0]),
                        spin_frequency_powers=np.array([0.0]),
                        reference_phase_lag=phase_lag(5)
                    )
                )





                final_age = stellar_age[i]
                print(repr(interpolator))







                b = find_evolution(system=a,
                                   interpolator=interpolator,
                                   dissipation=dissipation,
                                   max_age=stellar_age[i] * u.Gyr,
                                   initial_eccentricity=0.5 * u.dimensionless_unscaled,
                                   initial_obliquity=0.0,
                                   disk_period=10 * u.d,
                                   disk_dissipation_age=2e-3 * u.Gyr,
                                   primary_wind_strength=0.17,
                                   primary_wind_saturation=2.78,
                                   primary_core_envelope_coupling_timescale=0.05 * u.Gyr,
                                   secondary_wind_strength=0.0,
                                   secondary_wind_saturation=100.0,
                                   secondary_core_envelope_coupling_timescale=0.05 * u.Gyr,
                                   orbital_period_tolerance=1e-6,
                                   solve=True,
                                   secondary_is_star=False)
                # required_ages=numpy.arange(0.1, final_age, 0.1))

                init = 0
                final = len(b.orbital_period)
                # final = 5000

                y = b.orbital_period[init:final]
                z = b.eccentricity[init:final]
                # x = []
                # for i in range(init, len(y)):
                #    x = x + [math.log(b.age[i])]

                x = b.age[init:final]

                print('length of array =', len(b.orbital_period))

                plt.plot(x, y, label="Orbital period vs. ages")

                # naming the x axis
                plt.xlabel('Age')
                # naming the y axis
                plt.ylabel('Orbital period')
                # giving a title to my graph
                plt.title('Orbital period vs ages')

                # show a legend on the plot
                plt.legend()

                # function to show the plot
                plt.show()

                plt.plot(x, z, label="Eccentricity vs. ages")

                # naming the x axis
                plt.xlabel('Age')
                # naming the y axis
                plt.ylabel('Eccentricity')
                # giving a title to my graph
                plt.title('Eccentricity vs ages')

                # show a legend on the plot
                plt.legend()

                # function to show the plot
                plt.show()
                print('final age in find_evolution = ', x[-1])
                print('present eccentricity according to find_evolution =  ', z[-1])
                print('present orbital period according to find_evolution = ', y[-1])
                print('actual final age = ', final_age)
                print('actual present eccentricity = ', eccentricity[i])
                print('actual orbital period = ', orbital_period[i])

                evolutionary_data = evolutionary_data + [b]
                print(repr(dir(b)))
        if j > 0:
            break
    return evolutionary_data

def phi(z):
    return 0.5 * (1 + erf(z / math.sqrt(2)))

def alpha(ci):
    return 1-ci/100.0

def constraints(smallest_acceptable_value_of_orbital_period = 0,
                largest_acceptable_value_of_orbital_period = 10,
                smallest_acceptable_value_of_primary_mass = 0.4,
                largest_acceptable_value_of_primary_mass = 1.2,
                smallest_acceptable_value_of_secondary_mass = 0,
                largest_acceptable_value_of_secondary_mass = 25000, #mass of a brown dwarf
                smallest_acceptable_value_of_metallicity = -1.014,
                largest_acceptable_value_of_metallicity = 0.537,
                smallest_acceptable_value_of_stellar_age = 0,
                largest_acceptable_value_of_stellar_age = 10,
                smallest_acceptable_value_of_eccentricity_now = 0,
                largest_acceptable_value_of_eccentricity_now = 0.45):
    smallest = {'orbital period': smallest_acceptable_value_of_orbital_period,
                'primary mass': smallest_acceptable_value_of_primary_mass,
                'secondary mass': smallest_acceptable_value_of_secondary_mass,
                'metallicity': smallest_acceptable_value_of_metallicity,
                'stellar age': smallest_acceptable_value_of_stellar_age,
                'present eccentricity': smallest_acceptable_value_of_eccentricity_now}
    largest  = {'orbital period': largest_acceptable_value_of_orbital_period,
                'primary mass': largest_acceptable_value_of_primary_mass,
                'secondary mass': largest_acceptable_value_of_secondary_mass,
                'metallicity': largest_acceptable_value_of_metallicity,
                'stellar age': largest_acceptable_value_of_stellar_age,
                'present eccentricity': largest_acceptable_value_of_eccentricity_now}
    return smallest, largest

def constraints_for_eccentricity_envelope(smallest_acceptable_value_of_secondary_radius = 8,
                                          largest_acceptable_value_of_secondary_radius = math.inf,
                                          smallest_acceptable_value_of_planet_mass_sin_i = 50,
                                          largest_acceptable_value_of_planet_mass_sin_i = math.inf):
    smallest = {'secondary radius': smallest_acceptable_value_of_secondary_radius,
                'planet mass times sin i': smallest_acceptable_value_of_planet_mass_sin_i}
    largest = {'secondary radius': largest_acceptable_value_of_secondary_radius,
               'planet mass times sin i': largest_acceptable_value_of_planet_mass_sin_i}
    return smallest, largest

def constraints_are_satisfied(orbital_period,
                              primary_mass,
                              secondary_mass,
                              metallicity,
                              eccentricity_now,
                              stellar_age,
                              constraints = constraints()):
    smallest = constraints[0]
    largest = constraints[1]
    if((orbital_period <= largest['orbital period'] and orbital_period > smallest['orbital period'])
            and (primary_mass > smallest['primary mass'] and primary_mass < largest['primary mass'])
            and (secondary_mass > smallest['secondary mass'] and secondary_mass < largest['secondary mass'])
            and (metallicity > smallest['metallicity'] and metallicity < largest['metallicity'])
            and (eccentricity_now >= smallest['present eccentricity'] and eccentricity_now <= largest['present eccentricity'])
            and (stellar_age >= smallest['stellar age'] and stellar_age <= largest['stellar age'])):
        return True
    return False

def constraints_for_eccentricity_envelope_are_satisfied(secondary_radius,
                                                        planet_mass_sin_i,
                                                        constraints = constraints_for_eccentricity_envelope()):
    smallest = constraints[0]
    largest = constraints[1]
    if secondary_radius <= largest['secondary radius'] and secondary_radius > smallest['secondary radius']:
        return True
    if planet_mass_sin_i <= largest['planet mass times sin i'] and planet_mass_sin_i > smallest['planet mass times sin i']:
        return True

    return False

class SuperEccentricityDistribution(metaclass=ABCMeta):

    @abstractmethod
    def create_cumulative_density_function_of_present_eccentricity(self):
        pass

    @abstractmethod
    def probability_density_of_eccentricity(self, e):
        pass

class EccentricityDistribution(SuperEccentricityDistribution):

    def __init__(self,
                 mean_e_now,
                 e_now_upper_uncertainty,
                 e_now_lower_uncertainty,
                 e_env,
                 percentile_for_e_now_upper_uncertainty=phi(1),
                 percentile_for_e_now_lower_uncertainty=1 - phi(1)
                 #percentile_for_e_now_upper_uncertainty = 1 - alpha(68.0)/2,
                 #percentile_for_e_now_lower_uncertainty = alpha(68.0)/2
                 ):

        self.mean_e_now = mean_e_now
        self.e_now_upper_uncertainty = e_now_upper_uncertainty
        self.e_now_lower_uncertainty = e_now_lower_uncertainty
        self.percentile_for_e_now_upper_uncertainty = percentile_for_e_now_upper_uncertainty
        self.percentile_for_e_now_lower_uncertainty = percentile_for_e_now_lower_uncertainty
        self.cumulative_density_function_of_present_eccentricity, self.cumulative_density_function_of_present_eccentricity_old = self.create_cumulative_density_function_of_present_eccentricity()
        self.rice_parameters_are_found = True
        self.e_env = e_env

    def equations_to_be_solved_for_Rice_distribution_parameters(self, x):
        b = x[0]
        s = x[1]
        first = rice.cdf((self.mean_e_now+self.e_now_upper_uncertainty), b, scale = s) - self.percentile_for_e_now_upper_uncertainty
        second = rice.cdf((self.mean_e_now+self.e_now_lower_uncertainty), b, scale = s) - self.percentile_for_e_now_lower_uncertainty
        print('b = ', b, 's = ', s)
        if math.isnan(first) or math.isnan(second):
            print('Iteration does not converge.')
            self.rice_parameters_are_found = False
        return [first, second]
    def equation_to_be_solved_for_Rice_distribution_parameter_s_when_b_zero(self, x):
        s = x[0]
        print('s = ', s)
        eqn = rice.cdf((self.mean_e_now + self.e_now_upper_uncertainty), 0,
                         scale=s) - self.percentile_for_e_now_upper_uncertainty

        if math.isnan(eqn):
            print('Iteration does not converge.')
            self.rice_parameters_are_found = False
        return [eqn]

    def roots_for_Rice_parameters(self):
        estimated_s = self.e_now_upper_uncertainty
        if self.mean_e_now == 0:
            try:
                s = fsolve(self.equation_to_be_solved_for_Rice_distribution_parameter_s_when_b_zero,
                           np.asarray([estimated_s]))
                print('s = ', s[0])
            except:
                print('Rice parameters cannot be worked out')
                self.rice_parameters_are_found = False
                return [math.nan, math.nan]
            else:
                self.rice_parameters_are_found = True
                print('s = ', s[0])
                return [0, s[0]]



        estimated_b = self.mean_e_now/(self.e_now_upper_uncertainty)

        roots = [math.nan, math.nan]


        try:
            roots = fsolve(self.equations_to_be_solved_for_Rice_distribution_parameters, np.asarray([estimated_b, estimated_s]))
            print('b = ', roots[0], ' s = ', roots[1])
        except:
            print('Rice parameters cannot be worked out')
            self.rice_parameters_are_found = False
        else:
            self.rice_parameters_are_found = True
            print('b = ', roots[0], ' s = ', roots[1])
        return roots



    def create_cumulative_density_function_of_present_eccentricity(self):
        roots = self.roots_for_Rice_parameters()

        if not(math.isnan(roots[0]) or math.isnan(roots[1])):
            s = roots[1]
            b = roots[0]
            print('s ', s)
            print('b ', b)
            print('e median =  ', self.mean_e_now)
            print(' rice.cdf up to e_median = ', rice.cdf(self.mean_e_now, b, scale=s))
            def pdf(e):
                val = i0((e / s) * b)
                if val == math.inf:
                    return 0  # Then math.exp(-((e/s)**2+b**2)/2) = 0
                return math.exp(-((e / s) ** 2 + b ** 2) / 2) * val

            self.M = pdf

            def pdf_old(e):
                return rice.pdf(e, b, loc=0, scale=s)

            self.M_old = pdf_old

            def cdf(e_now):
                value = nquad(pdf, [[0, e_now]])
                return value[0]

            inv_norm = cdf(1.0)

            def cumulative_density_function_of_present_eccentricity(e_now):
                if inv_norm == 0:
                    return math.inf
                value = cdf(e_now)/inv_norm
                return value

            def cumulative_density_function_of_present_eccentricity_old(e_now):
                value = rice.cdf(e_now, b, loc=0, scale=s)
                return value
            return cumulative_density_function_of_present_eccentricity, cumulative_density_function_of_present_eccentricity_old
        return None, None


    def probability_density_of_eccentricity(self, e):
        if e > 1 or e < 0:
            return 0
        if e <= self.e_env and not(self.cumulative_density_function_of_present_eccentricity == None):
            return self.cumulative_density_function_of_present_eccentricity(e)
        return 0

    def probability_density_of_eccentricity_old(self, e):
        if e > 1 or e < 0:
            return 0
        if e <= self.e_env and not(self.cumulative_density_function_of_present_eccentricity_old == None):
            return self.cumulative_density_function_of_present_eccentricity_old(e)
        return 0

    def plot_probability_density_of_eccentricity_vs_eccentricity_graph(self):
        if not(self.cumulative_density_function_of_present_eccentricity == None or self.cumulative_density_function_of_present_eccentricity_old == None):
            eccentricity = np.linspace(0, 1, 100)
            probability_density_of_eccentricity = []
            probability_density_of_eccentricity_old = []

            for i in range(0, len(eccentricity)):
                probability_density_of_eccentricity = probability_density_of_eccentricity + [
                    self.probability_density_of_eccentricity(eccentricity[i])]
                probability_density_of_eccentricity_old = probability_density_of_eccentricity_old + [
                    self.probability_density_of_eccentricity_old(eccentricity[i])]

            M_cdf = []
            M_cdf_old = []
            for i in range(0, len(eccentricity)):
                M_cdf = M_cdf + [self.cumulative_density_function_of_present_eccentricity(eccentricity[i])]
                M_cdf_old = M_cdf_old + [self.cumulative_density_function_of_present_eccentricity_old(eccentricity[i])]

            M_pdf = []
            M_pdf_old = []
            for i in range(0, len(eccentricity)):
                M_pdf = M_pdf + [self.M(eccentricity[i])]
                M_pdf_old = M_pdf_old + [self.M_old(eccentricity[i])]

            plt.plot(eccentricity, probability_density_of_eccentricity,
                     label="Probality density of eccentricity (f(e)) vs. eccentricity (e)")
            plt.plot(eccentricity, probability_density_of_eccentricity_old, 'x')
            # naming the x axis
            plt.xlabel('Eccentricity (e)')
            # naming the y axis
            plt.ylabel('probability density of eccentricity (f(e))')
            # giving a title to my graph
            plt.title('Probability density of eccentricity vs eccentricity')
            # function to show the plot
            plt.show()

            plt.plot(eccentricity, M_cdf, label="cdf of M(e) vs. eccentricity (e)")
            plt.plot(eccentricity, M_cdf_old, 'x')
            # naming the x axis
            plt.xlabel('Eccentricity (e)')
            # naming the y axis
            plt.ylabel('M_cdf ')
            # giving a title to my graph
            plt.title('cdf of M(e) vs eccentricity')
            # function to show the plot
            plt.show()

            plt.plot(eccentricity, M_pdf, label="pdf of M(e) vs. eccentricity (e)")
            plt.plot(eccentricity, M_pdf_old, 'x')
            # naming the x axis
            plt.xlabel('Eccentricity (e)')
            # naming the y axis
            plt.ylabel('M_pdf ')
            # giving a title to my graph
            plt.title('M_pdf of eccentricity vs eccentricity')
            # function to show the plot
            plt.show()
            return
        print('Probability density distributions of eccentricity according to given mean_e_now and its uncertainties do not exist')


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
        print('Primary mass = ', self.primary_mass, '=', self.primary_mass.to(u.kg))
        print('Secondary mass = ', self.secondary_mass, '=', self.secondary_mass.to(u.kg))
        print('Secondary radius = ', self.secondary_radius, '=', self.secondary_radius.to(u.m))
        print('Metallicity = ', self.feh.to(u.dimensionless_unscaled))
        print('Orbital period = ', self.orbital_period, '=', self.orbital_period.to(u.s))
        print('Obliquity = ', self.obliquity.to(u.deg))
        print('Age = ', self.age, '=', self.age.to(u.s))
############################################################
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
########################################################
class EnvelopeEccentricityDistribution:

    def __init__(self,
                 #path='/home/mmmahmud/CircularizationDissipationConstraints/data/planets_2020.04.10_14.52.24.csv',
                 path='/home/mmmahmud/CircularizationDissipationConstraints/data/PS_2021.07.13_00.12.38.csv',
                 file_name=b"/home/mmmahmud/poet/scripts/eccentricity_expansion_coef.txt",
                 serialized_directory='/home/mmmahmud/poet/stellar_evolution_interpolators',
                 maximum_number_of_data_points=math.inf,
                 threshold_value_of_envelope_eccentricity=0.001,
                 constraints = constraints_for_eccentricity_envelope(),
                 largest_acceptable_value_of_envelope_eccentricity=0.5
                 ):


        self.path = path
        self.file_name = file_name
        self.serialized_directory = serialized_directory

        orbital_evolution_library.read_eccentricity_expansion_coefficients(
            file_name
        )

        manager = StellarEvolutionManager(serialized_directory)

        self.interpolator = manager.get_interpolator_by_name('default')
        ##############################################################################
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
                #system_names = [name.decode() for name in data[:, 1]]
                #The above for loop is written instead of the code system_names = [name.decode() for name in data[:, 1]]
                #couple of name in data[:, 1] were not decoded by decode() method, since they were not numpy.bytes_
                #object. So, I have included a checking command: if type(name) is numpy.bytes_

                delete_indices = []
                for system in eliminate:
                    if system in system_names:
                        delete_indices.append(system_names.index(system))
                a = numpy.delete(data,delete_indices,0)
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

            #data_columns = [col.decode() for col in data[0]]
            #the above loop was written instead of data_columns = [col.decode() for col in data[0]]


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
                        print('data [...]', data[:, column_index][1:])
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

        readPlanet = read_nasa_planets(self.path, eliminate=('SWEEPS-11', 'HD 41004 B', 'PSR J1719-1438', 'K2-22'), need_ages=False,)
        ###########################################################################################

        #readPlanet = planetary_system_io.read_nasa_planets(self.path, eliminate=('SWEEPS-11','HD 41004 B','PSR J1719-1438','K2-22'), need_ages=False,)

        self.planet_name = readPlanet.pl_name
        self.orbital_period = readPlanet.pl_orbper  # days
        self.orbital_period_upper_uncertainty = readPlanet.pl_orbpererr1
        self.orbital_period_lower_uncertainty = readPlanet.pl_orbpererr2
        self.primary_mass = readPlanet.st_mass  # solar mass
        self.primary_mass_upper_uncertainty = readPlanet.st_masserr1
        self.primary_mass_lower_uncertainty = readPlanet.st_masserr2
        self.secondary_mass = readPlanet.pl_masse  # Earth mass
        self.secondary_mass_upper_uncertainty = readPlanet.pl_masseerr1
        self.secondary_mass_lower_uncertainty = readPlanet.pl_masseerr2
        #self.metallicity = readPlanet.st_metfe
        #self.metallicity_upper_uncertainty = readPlanet.st_metfeerr1
        #self.metallicity_lower_uncertainty = readPlanet.st_metfeerr2
        self.metallicity = readPlanet.st_met
        self.metallicity_upper_uncertainty = readPlanet.st_meterr1
        self.metallicity_lower_uncertainty = readPlanet.st_meterr2
        self.semi_major_axis = readPlanet.pl_orbsmax
        self.semi_major_axis_upper_uncertainty = readPlanet.pl_orbsmaxerr1
        self.semi_major_axis_lower_uncertainty = readPlanet.pl_orbsmaxerr2
        self.semi_major_axis_flag_limit = readPlanet.pl_orbsmaxlim
        self.primary_radius = readPlanet.st_rad  # solar radius
        self.primary_radius_upper_uncertainty = readPlanet.st_raderr1
        self.primary_radius_lower_uncertainty = readPlanet.st_raderr2
        self.secondary_radius = readPlanet.pl_rade  # earth radius
        self.secondary_radius_upper_uncertainty = readPlanet.pl_radeerr1
        self.secondary_radius_lower_uncertainty = readPlanet.pl_radeerr2
        self.stellar_age = readPlanet.st_age  # GYear
        self.stellar_age_upper_uncertainty = readPlanet.st_ageerr1
        self.stellar_age_lower_uncertainty = readPlanet.st_ageerr2
        self.eccentricity_now = readPlanet.pl_orbeccen
        self.eccentricity_now_upper_uncertainty = readPlanet.pl_orbeccenerr1
        self.eccentricity_now_lower_uncertainty = readPlanet.pl_orbeccenerr2
        self.eccentricity_now_limit_flag = readPlanet.pl_orbeccenlim
        self.obliquity = readPlanet.pl_orbincl  # degrees
        self.obliquity_upper_uncertainty = readPlanet.pl_orbinclerr1
        self.obliquity_lower_uncertainty = readPlanet.pl_orbinclerr2
        #self.vsini = readPlanet.st_vsini  # km/s
        #self.vsini_upper_uncertainty = readPlanet.st_vsinierr1
        #self.vsini_lower_uncertainty = readPlanet.st_vsinierr2
        self.vsini = readPlanet.st_vsin  # km/s
        self.vsini_upper_uncertainty = readPlanet.st_vsinerr1
        self.vsini_lower_uncertainty = readPlanet.st_vsinerr2
        self.planet_mass_sin_i = readPlanet.pl_msinie # Earth mass
        self.planet_mass_sin_i_upper_uncertainty = readPlanet.pl_msinieerr1
        self.planet_mass_sin_i_lower_uncertainty = readPlanet.pl_msinieerr2

        self.envelope_eccentricity_function = self.create_envelope_eccentricity_function(maximum_number_of_data_points,
                                                                                         threshold_value_of_envelope_eccentricity,
                                                                                         largest_acceptable_value_of_envelope_eccentricity,
                                                                                         constraints)

    def create_envelope_eccentricity_function(self,
                                     maximum_number_of_data_points,
                                     threshold_value_of_envelope_eccentricity,
                                     largest_acceptable_value_of_envelope_eccentricity,
                                     constraints = constraints_for_eccentricity_envelope()):


        #For present eccentricity vs. log of orbital period plot:
        records_of_log_orbital_period_and_eccentricity_now = []
        j = -1
        for i in range(0, len(self.orbital_period)):
            if not (math.isnan(self.orbital_period[i])
                    or math.isnan(self.eccentricity_now[i])
            ) and self.eccentricity_now_limit_flag[i] == 0:
                if (constraints_for_eccentricity_envelope_are_satisfied(
                                              secondary_radius=self.secondary_radius[i],
                                              planet_mass_sin_i=self.planet_mass_sin_i[i],
                                              constraints=constraints)):

                    records_of_log_orbital_period_and_eccentricity_now = (records_of_log_orbital_period_and_eccentricity_now
                                                                          + [{'log of orbital period': math.log(self.orbital_period[i], 10),
                                                                              'present eccentricity': self.eccentricity_now[i],
                                                                              'planet name': self.planet_name[i],
                                                                              'primary mass': self.primary_mass[i],
                                                                              'orbital eccentricity limit flag': self.eccentricity_now_limit_flag[i]}])
                    j = j+1

            if j >= maximum_number_of_data_points-1:
                break


        ##############################For a/Rpl
        records_of_semi_major_axis_over_planetary_radius_vs_eccentricity_now = []
        j = -1

        for i in range(0, len(self.semi_major_axis)):
            if not (math.isnan(self.semi_major_axis[i])
                    or math.isnan(self.secondary_radius[i])
                    or math.isnan(self.eccentricity_now[i])
            ) and self.eccentricity_now_limit_flag[i] != 1 and self.semi_major_axis_flag_limit[i] == 0:
                if (constraints_for_eccentricity_envelope_are_satisfied(
                        secondary_radius=self.secondary_radius[i],
                        planet_mass_sin_i=self.planet_mass_sin_i[i],
                        constraints=constraints)):
                    records_of_semi_major_axis_over_planetary_radius_vs_eccentricity_now = (records_of_semi_major_axis_over_planetary_radius_vs_eccentricity_now
                    + [{'semi major axis over planetary radius': (self.semi_major_axis[i]/self.secondary_radius[i]),
                        'present eccentricity': self.eccentricity_now[i],
                        'planet name': self.planet_name[i],
                        'primary mass': self.primary_mass[i],
                        'orbital eccentricity limit flag': self.eccentricity_now_limit_flag[i]}])

                    j = j + 1

            if j >= maximum_number_of_data_points - 1:
                break

        ##############################

        # Sorting
        records_of_log_orbital_period_and_eccentricity_now = sorted(records_of_log_orbital_period_and_eccentricity_now,
                                                                    key=lambda key_attribute_for_sorting: key_attribute_for_sorting['log of orbital period'])
        records_of_semi_major_axis_over_planetary_radius_vs_eccentricity_now = sorted(records_of_semi_major_axis_over_planetary_radius_vs_eccentricity_now,
                                                                    key=lambda key_attribute_for_sorting:
                                                                    key_attribute_for_sorting['semi major axis over planetary radius'])

        def find_records_of_the_points_on_envelope(records_of_certain_attribute_and_eccentricity_now):
            records_of_the_points_on_envelope = []
            maximum_eccentricity_now = threshold_value_of_envelope_eccentricity
            starting_point_of_the_significant_tidal_dissipation_region_found = False
            end_point_of_the_significant_tidal_dissipation_region_found = False
            for i in range(0, len(records_of_certain_attribute_and_eccentricity_now)):
                present_eccentricity_of_ith_element = records_of_certain_attribute_and_eccentricity_now[i][
                    'present eccentricity']
                if present_eccentricity_of_ith_element > maximum_eccentricity_now:
                    if (not starting_point_of_the_significant_tidal_dissipation_region_found) and i > 0:
                        records_of_the_points_on_envelope = records_of_the_points_on_envelope + [{'log of orbital period': records_of_log_orbital_period_and_eccentricity_now[i - 1]['log of orbital period'],
                                                                                                  'envelope eccentricity': threshold_value_of_envelope_eccentricity}]
                        print('Name of the planet belongs to the binary system on the envelope ',
                              records_of_certain_attribute_and_eccentricity_now[i - 1]['planet name'],
                              'log of orbital period = ',
                              records_of_certain_attribute_and_eccentricity_now[i - 1]['log of orbital period'],
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
                    records_of_the_points_on_envelope = records_of_the_points_on_envelope + [{'log of orbital period':
                                                                                                  records_of_log_orbital_period_and_eccentricity_now[
                                                                                                      i][
                                                                                                      'log of orbital period'],
                                                                                              'envelope eccentricity': maximum_eccentricity_now}]
                    print('Name of the planet belongs to the binary system on the envelope ',
                          records_of_log_orbital_period_and_eccentricity_now[i]['planet name'],
                          'log of orbital period = ',
                          records_of_log_orbital_period_and_eccentricity_now[i]['log of orbital period'],
                          'envelope eccentricity = ', maximum_eccentricity_now,
                          'orbital eccentricity limit flag = ',
                          records_of_log_orbital_period_and_eccentricity_now[i]['orbital eccentricity limit flag'])
                    if end_point_of_the_significant_tidal_dissipation_region_found:
                        break
            return
        records_of_the_points_on_envelope = []
        maximum_eccentricity_now = threshold_value_of_envelope_eccentricity
        starting_point_of_the_significant_tidal_dissipation_region_found = False
        end_point_of_the_significant_tidal_dissipation_region_found = False
        for i in range(0, len(records_of_log_orbital_period_and_eccentricity_now)):
            present_eccentricity_of_ith_element = records_of_log_orbital_period_and_eccentricity_now[i]['present eccentricity']
            if present_eccentricity_of_ith_element>maximum_eccentricity_now:
                if (not starting_point_of_the_significant_tidal_dissipation_region_found) and i>0:
                    records_of_the_points_on_envelope = records_of_the_points_on_envelope + [{'log of orbital period': records_of_log_orbital_period_and_eccentricity_now[i-1]['log of orbital period'],
                                                                                              'envelope eccentricity': threshold_value_of_envelope_eccentricity}]
                    print('Name of the planet belongs to the binary system on the envelope ',records_of_log_orbital_period_and_eccentricity_now[i-1]['planet name'],
                          'log of orbital period = ', records_of_log_orbital_period_and_eccentricity_now[i-1]['log of orbital period'],
                          'envelope eccentricity = ', threshold_value_of_envelope_eccentricity,
                          'orbital eccentricity limit flag = ', records_of_log_orbital_period_and_eccentricity_now[i-1]['orbital eccentricity limit flag'])
                if present_eccentricity_of_ith_element<=largest_acceptable_value_of_envelope_eccentricity:
                    maximum_eccentricity_now = present_eccentricity_of_ith_element
                if present_eccentricity_of_ith_element>largest_acceptable_value_of_envelope_eccentricity:
                    maximum_eccentricity_now = largest_acceptable_value_of_envelope_eccentricity
                    end_point_of_the_significant_tidal_dissipation_region_found = True
                starting_point_of_the_significant_tidal_dissipation_region_found = True
                records_of_the_points_on_envelope = records_of_the_points_on_envelope + [{'log of orbital period': records_of_log_orbital_period_and_eccentricity_now[i]['log of orbital period'],
                                                                                          'envelope eccentricity': maximum_eccentricity_now}]
                print('Name of the planet belongs to the binary system on the envelope ', records_of_log_orbital_period_and_eccentricity_now[i]['planet name'],
                      'log of orbital period = ', records_of_log_orbital_period_and_eccentricity_now[i]['log of orbital period'],
                      'envelope eccentricity = ', maximum_eccentricity_now,
                      'orbital eccentricity limit flag = ', records_of_log_orbital_period_and_eccentricity_now[i]['orbital eccentricity limit flag'])
                if end_point_of_the_significant_tidal_dissipation_region_found:
                    break

        def envelope_eccentricity_function(orbital_period):
            log_of_orbital_period = math.log(orbital_period,10)
            min_log_of_orbital_period = records_of_the_points_on_envelope[0]['log of orbital period']
            max_log_of_orbital_period = records_of_the_points_on_envelope[-1]['log of orbital period']
            if log_of_orbital_period <= min_log_of_orbital_period:
                return threshold_value_of_envelope_eccentricity
            if log_of_orbital_period >= max_log_of_orbital_period:
                return largest_acceptable_value_of_envelope_eccentricity
            for i in range(1, len(records_of_the_points_on_envelope)):
                if log_of_orbital_period == records_of_the_points_on_envelope[i]['log of orbital period']:
                    return records_of_the_points_on_envelope[i]['envelope eccentricity']
                if ((log_of_orbital_period<records_of_the_points_on_envelope[i]['log of orbital period'])
                        and (log_of_orbital_period>records_of_the_points_on_envelope[i-1]['log of orbital period'])):
                    return (records_of_the_points_on_envelope[i]['envelope eccentricity']
                            + (records_of_the_points_on_envelope[i]['envelope eccentricity']
                               - records_of_the_points_on_envelope[i-1]['envelope eccentricity'])
                            /(records_of_the_points_on_envelope[i]['log of orbital period']
                              - records_of_the_points_on_envelope[i-1]['log of orbital period'])
                            *(log_of_orbital_period - records_of_the_points_on_envelope[i]['log of orbital period']))
            print('This situation is not possible')
            return

        def envelope_eccentricity_function_manual(orbital_period):
            log_of_orbital_period = math.log(orbital_period,10)
            crit1 = -0.11
            crit2 = 0.7
            if log_of_orbital_period <=crit1:
                return threshold_value_of_envelope_eccentricity
            if log_of_orbital_period > crit2:
                return largest_acceptable_value_of_envelope_eccentricity
            if log_of_orbital_period>crit1 and log_of_orbital_period <= crit2:
                return threshold_value_of_envelope_eccentricity + (log_of_orbital_period-crit1) * (largest_acceptable_value_of_envelope_eccentricity - threshold_value_of_envelope_eccentricity)/(crit2-crit1)
            print('This situation is not possible')
            return

        def envelope_eccentricity_function_for_eccentricity_vs_semi_major_axis_over_planetary_radius_manual(x):
            crit1 = -0.11
            crit2 = 0.7
            if x <=crit1:
                return threshold_value_of_envelope_eccentricity
            if x > crit2:
                return largest_acceptable_value_of_envelope_eccentricity
            if x>crit1 and x <= crit2:
                return threshold_value_of_envelope_eccentricity + (x-crit1) * (largest_acceptable_value_of_envelope_eccentricity - threshold_value_of_envelope_eccentricity)/(crit2-crit1)
            print('This situation is not possible')

            return

        #self.plot_present_eccentricity_vs_log_period(records_of_log_orbital_period_and_eccentricity_now,
                                                     #records_of_the_points_on_envelope,
                                                     #envelope_eccentricity_function_manual)
        self.plot_present_eccentricity_vs_semi_major_axis_over_planetary_radius(records_of_semi_major_axis_over_planetary_radius_vs_eccentricity_now,
                                                     envelope_eccentricity_function_for_eccentricity_vs_semi_major_axis_over_planetary_radius_manual)



        return envelope_eccentricity_function

    def plot_present_eccentricity_vs_semi_major_axis_over_planetary_radius(self,
                                                                           records_of_semi_major_axis_over_planetary_radius_vs_eccentricity_now,
                                                                           envelope_eccentricity_function):
        log_semi_major_axis_over_planetary_radius = [math.log(element['semi major axis over planetary radius'], 10) for element in records_of_semi_major_axis_over_planetary_radius_vs_eccentricity_now]
        eccentricity_now = [element['present eccentricity'] for element in records_of_semi_major_axis_over_planetary_radius_vs_eccentricity_now]
        primary_mass = [element['primary mass'] for element in records_of_semi_major_axis_over_planetary_radius_vs_eccentricity_now]

        xdata = np.linspace(math.log(records_of_semi_major_axis_over_planetary_radius_vs_eccentricity_now[0]['semi major axis over planetary radius'], 10),
                            math.log(records_of_semi_major_axis_over_planetary_radius_vs_eccentricity_now[-1]['semi major axis over planetary radius'], 10),
                            50)
        ydata = []

        for i in range(0, len(xdata)):
            ydata = ydata + [envelope_eccentricity_function(xdata[i])]
        log_semi_major_axis_over_planetary_radius_1 = []
        eccentricity_now_1 = []
        log_semi_major_axis_over_planetary_radius_2 = []
        eccentricity_now_2 = []
        for i in range(0, len(records_of_semi_major_axis_over_planetary_radius_vs_eccentricity_now)):
            if primary_mass[i] < 1.2:
                log_semi_major_axis_over_planetary_radius_1 = log_semi_major_axis_over_planetary_radius_1 + [log_semi_major_axis_over_planetary_radius[i]]
                eccentricity_now_1 = eccentricity_now_1 + [eccentricity_now[i]]

            else:
                log_semi_major_axis_over_planetary_radius_2 = log_semi_major_axis_over_planetary_radius_2 + [log_semi_major_axis_over_planetary_radius[i]]
                eccentricity_now_2 = eccentricity_now_2 + [eccentricity_now[i]]


        plt.plot(log_semi_major_axis_over_planetary_radius_1, eccentricity_now_1, 'x')
        plt.plot(log_semi_major_axis_over_planetary_radius_2, eccentricity_now_2, 'o')
        #plt.plot(xdata, ydata)
        # naming the x axis
        plt.xlabel('log of semi major axis over planetary radius')
        # naming the y axis
        plt.ylabel('Present Eccentricity')
        # giving a title to my graph
        plt.title('Present Eccentricity vs log of semi major axis over planetary radius')
        # function to show the plot
        #plt.plot(log_orbital_period_on_envelope, eccentricity_now_on_envelope, 'b',
                 #label="Envelope Eccentricities vs. log(Periods)")
        plt.show()
        return

    def plot_present_eccentricity_vs_log_period(self,
                                                records_of_log_orbital_period_and_eccentricity_now,
                                                records_of_the_points_on_envelope,
                                                envelope_eccentricity_function):
        log_orbital_period = [element['log of orbital period'] for element in records_of_log_orbital_period_and_eccentricity_now]
        eccentricity_now = [element['present eccentricity'] for element in records_of_log_orbital_period_and_eccentricity_now]
        primary_mass = [element['primary mass'] for element in records_of_log_orbital_period_and_eccentricity_now]
        log_orbital_period_on_envelope = [element['log of orbital period'] for element in records_of_the_points_on_envelope]
        eccentricity_now_on_envelope = [element['envelope eccentricity'] for element in records_of_the_points_on_envelope]
        xdata = np.linspace(records_of_log_orbital_period_and_eccentricity_now[0]['log of orbital period'],
                            records_of_log_orbital_period_and_eccentricity_now[-1]['log of orbital period'],
                            50)
        ydata = []
        for i in range(0, len(xdata)):
            ydata = ydata + [envelope_eccentricity_function(10 ** xdata[i])]
        log_orbital_period_1 = []
        eccentricity_now_1 = []
        log_orbital_period_2 = []
        eccentricity_now_2 = []
        for i in range(0, len(records_of_log_orbital_period_and_eccentricity_now)):
            if primary_mass[i] < 1.2:
                log_orbital_period_1 = log_orbital_period_1 + [log_orbital_period[i]]
                eccentricity_now_1 = eccentricity_now_1 + [eccentricity_now[i]]

            else:
                log_orbital_period_2 = log_orbital_period_2 + [log_orbital_period[i]]
                eccentricity_now_2 = eccentricity_now_2 + [eccentricity_now[i]]


        plt.plot(log_orbital_period_1, eccentricity_now_1, 'x')
        plt.plot(log_orbital_period_2, eccentricity_now_2, 'o')
        plt.plot(xdata, ydata)
        # naming the x axis
        plt.xlabel('log(Period)')
        # naming the y axis
        plt.ylabel('Present Eccentricity')
        # giving a title to my graph
        plt.title('Present Eccentricity vs log(Period)')
        # function to show the plot
        #plt.plot(log_orbital_period_on_envelope, eccentricity_now_on_envelope, 'b',
                 #label="Envelope Eccentricities vs. log(Periods)")
        plt.show()
        return

    def properties_of_ith_binary_system_if_satisfies_constraints(self, i, constraints=constraints()):
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
                or math.isnan(self.metallicity[i])
                or math.isnan(self.metallicity_upper_uncertainty[i])
                or math.isnan(self.metallicity_lower_uncertainty[i])
                or math.isnan(self.secondary_radius[i])
                or math.isnan(self.secondary_radius_upper_uncertainty[i])
                or math.isnan(self.secondary_radius_lower_uncertainty[i])
                or math.isnan(self.stellar_age[i])
                or math.isnan(self.stellar_age_upper_uncertainty[i])
                or math.isnan(self.stellar_age_lower_uncertainty[i])
                or math.isnan(self.eccentricity_now[i])
                or math.isnan(self.eccentricity_now_lower_uncertainty[i])
                or math.isnan(self.eccentricity_now_upper_uncertainty[i])):
            if (constraints_are_satisfied(orbital_period=self.orbital_period[i],
                                          primary_mass=self.primary_mass[i],
                                          secondary_mass=self.secondary_mass[i],
                                          metallicity=self.metallicity[i],
                                          eccentricity_now=self.eccentricity_now[i],
                                          stellar_age=self.stellar_age[i],
                                          constraints=constraints)):
                means = {'primary mass': self.primary_mass[i],
                         'secondary mass': self.secondary_mass[i],
                         'secondary radius': self.secondary_radius[i],
                         'metallicity': self.metallicity[i],
                         'orbital period': self.orbital_period[i],
                         'obliquity': self.obliquity[i],
                         'stellar age': self.stellar_age[i],
                         'present eccentricity': self.eccentricity_now[i],
                         'v sin i': self.vsini[i]}
                standard_deviations = {'primary_mass_upper_uncertainty': self.primary_mass_upper_uncertainty[i],
                                       'primary_mass_lower_uncertainty': self.primary_mass_lower_uncertainty[i],
                                       'secondary_mass_upper_uncertainty': self.secondary_mass_upper_uncertainty[i],
                                       'secondary_mass_lower_uncertainty': self.secondary_mass_lower_uncertainty[i],
                                       'secondary_radius_upper_uncertainty': self.secondary_radius_upper_uncertainty[i],
                                       'secondary_radius_lower_uncertainty': self.secondary_radius_lower_uncertainty[i],
                                       'metallicity_upper_uncertainty': self.metallicity_upper_uncertainty[i],
                                       'metallicity_lower_uncertainty': self.metallicity_lower_uncertainty[i],
                                       'orbital_period_upper_uncertainty': self.orbital_period_upper_uncertainty[i],
                                       'orbital_period_lower_uncertainty': self.orbital_period_lower_uncertainty[i],
                                       'obliquity_upper_uncertainty': self.obliquity_upper_uncertainty[i],
                                       'obliquity_lower_uncertainty': self.obliquity_lower_uncertainty[i],
                                       'stellar_age_upper_uncertainty': self.stellar_age_upper_uncertainty[i],
                                       'stellar_age_lower_uncertainty': self.stellar_age_lower_uncertainty[i],
                                       'eccentricity_now_upper_uncertainty': self.eccentricity_now_upper_uncertainty[i],
                                       'eccentricity_now_lower_uncertainty': self.eccentricity_now_lower_uncertainty[i],
                                       'vsini_upper_uncertainty': self.vsini_upper_uncertainty[i],
                                       'vsini_lower_uncertainty': self.vsini_lower_uncertainty[i]}
                return means, standard_deviations, self.planet_name[i]
        return None, None, None

    def print_properties_of_binary_systems_satisfying_constraints(self, constraints=constraints()):
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
                print('log of orbital period = ', math.log(means['orbital period'], 10))
                print('Envelope eccentricity for orbital period = ', means['orbital period'], ' is = ',
                      self.envelope_eccentricity_function(means['orbital period']))
                index_of_binary_system_with_constrained_properties = index_of_binary_system_with_constrained_properties + [i]
        return index_of_binary_system_with_constrained_properties



class SamplingPropertiesOfSystem:
    def __init__(self,
                 means,
                 standard_deviations,
                 planet_name='Exo Planet',
                 serialized_directory='/home/mmmahmud/poet/stellar_evolution_interpolators',
                 envelope_eccentricity_function=None,
                 initial_eccentricity = 0.5,
                 max_argument_of_phase_lag_function_for_planet=5,
                 min_argument_of_phase_lag_function_for_planet=10,
                 max_initial_stellar_spin=5,
                 min_initial_stellar_spin=15,
                 constraints = constraints(),
                 tidal_frequency_breaks_for_planet = np.array([2*math.pi/20]),
                 tidal_frequency_powers_for_planet = np.array([1.0, 0.0]),
                 spin_frequency_breaks_for_planet = None,
                 spin_frequency_powers_for_planet = np.array([0.0]),
                 find_argument_of_phase_lag_function_for_planet_range_auto = False):

        self.initial_eccentricity = initial_eccentricity
        self.max_argument_of_phase_lag_function_for_planet = max_argument_of_phase_lag_function_for_planet
        self.min_argument_of_phase_lag_function_for_planet = min_argument_of_phase_lag_function_for_planet
        self.max_initial_stellar_spin = max_initial_stellar_spin
        self.min_initial_stellar_spin = min_initial_stellar_spin

        self.tidal_frequency_breaks_for_planet = tidal_frequency_breaks_for_planet
        self.tidal_frequency_powers_for_planet = tidal_frequency_powers_for_planet
        self.spin_frequency_breaks_for_planet= spin_frequency_breaks_for_planet
        self.spin_frequency_powers_for_planet= spin_frequency_powers_for_planet

        eccentricity_expansion_fname = b"/home/mmmahmud/poet/scripts/eccentricity_expansion_coef.txt"
        orbital_evolution_library.read_eccentricity_expansion_coefficients(
            eccentricity_expansion_fname
        )

        self.serialized_directory = serialized_directory
        manager = StellarEvolutionManager(serialized_directory)
        self.interpolator = manager.get_interpolator_by_name('default')

        if envelope_eccentricity_function == None:
            envelope_eccentricity_distribution_object = EnvelopeEccentricityDistribution()
            self.envelope_eccentricity_function = envelope_eccentricity_distribution_object.envelope_eccentricity_function
        else:
            self.envelope_eccentricity_function = envelope_eccentricity_function

        self.standard_deviations = standard_deviations


        if (constraints_are_satisfied(orbital_period=means['orbital period'],
                                      primary_mass=means['primary mass'],
                                      secondary_mass=means['secondary mass'],
                                      metallicity=means['metallicity'],
                                      eccentricity_now=means['present eccentricity'],
                                      stellar_age=means['stellar age'],
                                      constraints=constraints)):
            self.means = means

            self.means['obliquity'] = 0

            print('orbital period = ',  means['orbital period'])

            self.e_env = self.envelope_eccentricity_function(orbital_period=self.means['orbital period'])
            print(self.means['present eccentricity'],
                  self.standard_deviations['eccentricity_now_upper_uncertainty'],
                  self.standard_deviations['eccentricity_now_lower_uncertainty'],
                  self.e_env)
            eccentricity_distribution_object = EccentricityDistribution(self.means['present eccentricity'],
                                                                        self.standard_deviations['eccentricity_now_upper_uncertainty'],
                                                                        self.standard_deviations['eccentricity_now_lower_uncertainty'],
                                                                        self.e_env)
            eccentricity_distribution_object.plot_probability_density_of_eccentricity_vs_eccentricity_graph()
            self.probability_density_of_eccentricity = eccentricity_distribution_object.probability_density_of_eccentricity

            if find_argument_of_phase_lag_function_for_planet_range_auto:
                min_Qpl, max_Qpl = self.determine_a_suitable_range_of_argument_of_phase_lag_function_for_planet()
                self.max_argument_of_phase_lag_function_for_planet = min_Qpl
                self.min_argument_of_phase_lag_function_for_planet = max_Qpl
                print('minimum Qpl = ', min_Qpl, ' maximum Qpl = ', max_Qpl)

    def pick_a_tuple_from_the_multi_variable_Gaussian_distribution(self, mean, standard_deviation):
        n_cross_n_dimensional_array_of_standard_deviations = np.diag(
            np.array([element ** 2 for element in standard_deviation]))
        temp = np.random.default_rng().multivariate_normal(np.array(mean),
                                                           n_cross_n_dimensional_array_of_standard_deviations).tolist()
        return temp

    def priors(self,
               theta,
               constraints=constraints()):

        smallest = constraints[0]
        largest = constraints[1]

        def prior_primary_mass(primary_mass):
            if (primary_mass > smallest['primary mass'] and primary_mass < largest['primary mass']):
                return 1
            return 0
        def prior_secondary_mass(secondary_mass):
            if (secondary_mass > smallest['secondary mass'] and secondary_mass < largest['secondary mass']):
                return 1
            return 0
        def prior_metallicity(metallicity):
            if metallicity > smallest['metallicity'] and metallicity < largest['metallicity']:
                return 1
            return 0
        def prior_stellar_age(stellar_age):
            if stellar_age > smallest['stellar age'] and stellar_age < largest['stellar age']:
                return 1
            return 0

        priors = (prior_primary_mass(theta[0])
                  * prior_secondary_mass(theta[1])
                  * prior_metallicity(theta[3])
                  * prior_stellar_age(theta[4]))
        return priors


    def walker_satisfies_constraints(self, walker, smallest, largest):
        for i in range(0, len(walker)):
            if not(walker[i]>smallest[i] and walker[i]<largest[i]):
                return False
        return True

    def draw_a_successful_walker_from_Gaussian_distribution(self,
                                                            mean,
                                                            standard_deviation,
                                                            smallest,
                                                            largest):
        success = False
        walker = None
        while not success:
            walker = self.pick_a_tuple_from_the_multi_variable_Gaussian_distribution(mean, standard_deviation)
            if self.walker_satisfies_constraints(walker, smallest, largest):
                success = True
        return walker

    def log_prob(self, theta):
        # theta = (primary_mass, secondary_mass, secondary_radius, metallicity, stellar_age, initial_stellar_spin, argument_of_phase_lag_function_for_planet)
        initial_eccentricity = self.initial_eccentricity
        priors = self.priors(theta)
        if priors == 0:
            return -math.inf

        primary_mass = theta[0]
        secondary_mass = theta[1]
        secondary_radius = theta[2]
        metallicity = theta[3]
        stellar_age = theta[4]
        initial_stellar_spin = theta[5]
        argument_of_phase_lag_function_for_planet = theta[6]
        orbital_period = self.means['orbital period']
        obliquity = self.means['obliquity']

        star_exoplanet_binary_system = System(primary_mass=primary_mass * u.solMass,
                                              secondary_mass=secondary_mass * u.earthMass,
                                              secondary_radius=secondary_radius * u.earthRad,
                                              feh=metallicity * u.dimensionless_unscaled,
                                              orbital_period=orbital_period * u.d,
                                              obliquity=obliquity * u.deg,
                                              age=stellar_age * u.Gyr)

        dissipation = dict(
            primary=None,
            secondary=dict(
                #tidal_frequency_breaks=self.tidal_frequency_breaks_for_planet,
                tidal_frequency_breaks=None,
                spin_frequency_breaks=self.spin_frequency_breaks_for_planet,
                #tidal_frequency_powers=self.tidal_frequency_powers_for_planet,
                tidal_frequency_powers=np.array([0.0]),
                spin_frequency_powers=self.spin_frequency_powers_for_planet,
                reference_phase_lag=phase_lag(argument_of_phase_lag_function_for_planet)
            )
        )

        print(dissipation)
        evolutionary_history = find_evolution(system=star_exoplanet_binary_system,
                                              interpolator=self.interpolator,
                                              dissipation=dissipation,
                                              max_age=stellar_age * u.Gyr,
                                              initial_eccentricity=initial_eccentricity * u.dimensionless_unscaled,
                                              initial_obliquity=0.0,
                                              disk_period=initial_stellar_spin * u.d,
                                              disk_dissipation_age=2e-2 * u.Gyr,
                                              primary_wind_strength=0.17,
                                              primary_wind_saturation=2.78,
                                              primary_core_envelope_coupling_timescale=0.05 * u.Gyr,
                                              secondary_wind_strength=0.0,
                                              secondary_wind_saturation=100.0,
                                              secondary_core_envelope_coupling_timescale=0.05 * u.Gyr,
                                              orbital_period_tolerance=1e-6,
                                              solve=True,
                                              secondary_is_star=False)


        calculated_eccentricity_now = evolutionary_history.eccentricity[- 1]
        self.calculated_eccentricity_now = calculated_eccentricity_now
        print('Calculated eccentricity due to tidal dissipation = ',
              calculated_eccentricity_now,
              ' actual present eccentricity = ',
              self.means['present eccentricity'],
              ' where the envelope eccentricity = ',
              self.e_env,
              ' argument of the phase lag function for planet = ',
              argument_of_phase_lag_function_for_planet,
              ' probability density of calculated eccentricity',
              self.probability_density_of_eccentricity(calculated_eccentricity_now))

        #       print(repr(dir(b)))
        if calculated_eccentricity_now >= 0 and calculated_eccentricity_now <= 1:
            probability_density_of_the_calculated_eccentricity = self.probability_density_of_eccentricity(
                calculated_eccentricity_now)
            probability_density = probability_density_of_the_calculated_eccentricity * priors
            if probability_density == 0:
                return -math.inf
            if probability_density < 0:
                print('Probability density cannot be less than zero.')
                return None
            return np.log(probability_density)
        print('Calculated present eccentricity cannot be less that zero or greater than one')
        return None

    def MCMC(self,
             nwalker=32,
             ndim=7):

        initial_stellar_spin = (self.min_initial_stellar_spin
                                + self.max_initial_stellar_spin)/2
        argument_of_phase_lag_function_for_planet = (self.min_argument_of_phase_lag_function_for_planet
                                                     + self.max_argument_of_phase_lag_function_for_planet)/2

        theta0 = [self.means['primary mass'],
                  self.means['secondary mass'],
                  self.means['secondary radius'],
                  self.means['metallicity'],
                  self.means['stellar age'],
                  initial_stellar_spin,
                  argument_of_phase_lag_function_for_planet]
        p0 = [theta0]

        mean = [self.means['primary mass'],
                self.means['secondary mass'],
                self.means['secondary radius'],
                self.means['metallicity'],
                self.means['stellar age']]
        standard_deviation = [(self.standard_deviations['primary_mass_upper_uncertainty']
                               - self.standard_deviations['primary_mass_lower_uncertainty'])/2,
                              (self.standard_deviations['secondary_mass_upper_uncertainty']
                               - self.standard_deviations['secondary_mass_lower_uncertainty'])/2,
                              (self.standard_deviations['secondary_radius_upper_uncertainty']
                               - self.standard_deviations['secondary_radius_lower_uncertainty'])/2,
                              (self.standard_deviations['metallicity_upper_uncertainty']
                               - self.standard_deviations['metallicity_lower_uncertainty'])/2,
                              (self.standard_deviations['stellar_age_upper_uncertainty']
                               - self.standard_deviations['stellar_age_lower_uncertainty'])/2]

        print('p0 initial = ', p0)

        smallest, largest = constraints()

        smallest_acceptable_values = [smallest['primary mass'],
                                      smallest['secondary mass'],
                                      0,
                                      smallest['metallicity'],
                                      smallest['stellar age']] #secondary radius does not have constraints
        largest_acceptable_values = [largest['primary mass'],
                                     largest['secondary mass'],
                                     math.inf,
                                     largest['metallicity'],
                                     largest['stellar age']]  #secondary radius does not have constraints

        for i in range(1, nwalker):
            temp = self.draw_a_successful_walker_from_Gaussian_distribution(mean,
                                                                            standard_deviation,
                                                                            smallest_acceptable_values,
                                                                            largest_acceptable_values)
            argument_of_phase_lag_function_for_planet = np.random.uniform(low=self.min_argument_of_phase_lag_function_for_planet,
                                    high=self.max_argument_of_phase_lag_function_for_planet,
                                    size=1)
            initial_stellar_spin = np.random.uniform(low=self.min_initial_stellar_spin,
                                    high=self.max_initial_stellar_spin,
                                    size=1)
            p_next = temp +  argument_of_phase_lag_function_for_planet.tolist() + initial_stellar_spin.tolist()
            p0.append(p_next)

        print('Full p0 = ', p0)

        sampler = emcee.EnsembleSampler(nwalker, ndim, self.log_prob)

        state = sampler.run_mcmc(p0, 100)
        sampler.reset()
        sampler.run_mcmc(state, 100)

        samples = sampler.get_chain(flat=True)
        plt.hist(samples[:, 0], 100, color="k", histtype="step")
        plt.xlabel(r"$\theta_1$")
        plt.ylabel(r"$p(\theta_1)$")
        plt.gca().set_yticks([])
        plt.show()

        return

    def determine_a_suitable_range_of_argument_of_phase_lag_function_for_planet(self):

        initial_stellar_spin = 10
        argument_of_phase_lag_function_for_planet = 5.0

        theta0 = [self.means['primary mass'],
                  self.means['secondary mass'],
                  self.means['secondary radius'],
                  self.means['metallicity'],
                  self.means['stellar age'],
                  initial_stellar_spin,
                  argument_of_phase_lag_function_for_planet]

        min_value_found = False
        max_value_found = False
        i=0
        init = 2
        while not min_value_found:
            theta0[6] = init + i * 0.25
            prob = self.log_prob(theta0)
            if prob <0.01:
                min_value_found = True
            i = i+1
        min_argument_of_phase_lag_function_for_planet = init + (i-1)*0.25
        while not max_value_found:
            theta0[6] = init + i * 0.25
            prob = self.log_prob(theta0)
            if prob == - math.inf:
                max_value_found = True
            i = i+1
        max_argument_of_phase_lag_function_for_planet = init + (i - 1) * 0.25

        return min_argument_of_phase_lag_function_for_planet, max_argument_of_phase_lag_function_for_planet

    def testing_log_prob(self):

        initial_stellar_spin = 10
        argument_of_phase_lag_function_for_planet = 5.0

        theta0 = [self.means['primary mass'],
                  self.means['secondary mass'],
                  self.means['secondary radius'],
                  self.means['metallicity'],
                  self.means['stellar age'],
                  initial_stellar_spin,
                  argument_of_phase_lag_function_for_planet]

        prob = []
        Qpl = []
        eccentricity = []
        k = -1
        init = 6.95
        for i in range(0, 20):
            theta0[6] = init + i * 0.001
            k = k + 1
            Qpl = Qpl + [init + i * 0.001]
            prob = prob + [self.log_prob(theta0)]
            eccentricity = eccentricity + [self.calculated_eccentricity_now]
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
    #analysis_on_Nasa_exoplanet_data()
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
    means, standard_deviations, planet_name = test2.properties_of_ith_binary_system_if_satisfies_constraints(index[9])
    print('Print properties of the chosen binary system: means = ', means, ' standard deviations = ',
          standard_deviations, ' planet name = ', planet_name)
    print('*********************************************************')
    test3 = SamplingPropertiesOfSystem(means,
                                       standard_deviations,
                                       planet_name=planet_name,
                                       envelope_eccentricity_function=test2.envelope_eccentricity_function
                                       )

    test3.testing_log_prob()
    #test3.MCMC()






