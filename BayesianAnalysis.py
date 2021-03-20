import math
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
            if orbital_period[i] <= 10 and (primary_mass[i] > 0.4 and primary_mass[i] < 1.2) and (
                    metallicity[i] > -1.014 and metallicity[i] < 0.537):
                j = j + 1
                print(primary_mass[i], (primary_mass[i] > 0.4 and primary_mass[i] < 1.2))
                a = System(primary_mass[i] * u.solMass,
                           secondary_mass[i] * u.earthMass,
                           secondary_radius[i] * u.earthRad,
                           metallicity[i] * u.dimensionless_unscaled,
                           orbital_period[i] * u.d,
                           obliquity[i] * u.deg,
                           stellar_age[i] * u.Gyr,
                           eccentricity[i] * u.dimensionless_unscaled,
                           vsini[i] * u.kilometer / u.second)

                a.printing()
                print('eccentricity found in the NASA archive is ', eccentricity[i])

                dissipation = dict(
                    primary=None,
                    secondary=dict(
                        tidal_frequency_breaks=None,
                        spin_frequency_breaks=None,
                        tidal_frequency_powers=np.array([0.0]),
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
                                   initial_eccentricity=0.5,
                                   initial_obliquity=0.0,
                                   disk_period=None,
                                   disk_dissipation_age=2e-3 * units.Gyr,
                                   primary_wind_strength=0.17,
                                   primary_wind_saturation=2.78,
                                   primary_core_envelope_coupling_timescale=0.05,
                                   secondary_wind_strength=0.0,
                                   secondary_wind_saturation=100.0,
                                   secondary_core_envelope_coupling_timescale=0.05,
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

class SuperEccentricityDistribution(metaclass=ABCMeta):

    @abstractmethod
    def create_probability_density_of_present_eccentricity(self):
        pass

    @abstractmethod
    def probability_density_of_eccentricity(self, e):
        pass

class EccentricityDistribution(SuperEccentricityDistribution):

    def __init__(self,
                 mean_e_now,
                 e_now_upper_uncertainty,
                 e_now_lower_uncertainty,
                 mean_e_env,
                 percentile_for_e_now_upper_uncertainty=phi(1),
                 percentile_for_e_now_lower_uncertainty=1 - phi(1),
                 e_env_upper_uncertainty=0.0,
                 e_env_lower_uncertainty=0.0
                 ):

        self.mean_e_now = mean_e_now
        self.e_now_upper_uncertainty = e_now_upper_uncertainty
        self.e_now_lower_uncertainty = e_now_lower_uncertainty
        self.percentile_for_e_now_upper_uncertainty = percentile_for_e_now_upper_uncertainty
        self.percentile_for_e_now_lower_uncertainty = percentile_for_e_now_lower_uncertainty
        self.probability_density_of_present_eccentricity, self.roots = self.create_probability_density_of_present_eccentricity()
        self.mean_e_env = mean_e_env
        self.e_env_upper_uncertainty = e_env_upper_uncertainty
        self.e_env_lower_uncertainty = e_env_lower_uncertainty

    def equations_to_be_solved_for_Rice_distribution_parameters(self, x):
        return [rice.cdf((self.mean_e_now+self.e_now_upper_uncertainty), x[0], scale = x[1]) - self.percentile_for_e_now_upper_uncertainty,
                rice.cdf((self.mean_e_now+self.e_now_lower_uncertainty), x[0], scale = x[1]) - self.percentile_for_e_now_lower_uncertainty]

    def roots_for_Rice_parameters(self):
        estimated_s = self.e_now_upper_uncertainty
        estimated_b = self.mean_e_now/self.e_now_upper_uncertainty
        roots = fsolve(self.equations_to_be_solved_for_Rice_distribution_parameters, np.asarray([estimated_b, estimated_s]))
        return roots

    def create_probability_density_of_present_eccentricity(self):
        roots = self.roots_for_Rice_parameters()
        def probability_density_of_present_eccentricity(e_now):
            return rice.pdf(e_now, roots[0], scale = roots[1])
        return probability_density_of_present_eccentricity, roots

    def probability_density_of_envelope_eccentricity(self, e_env):
        e_env_stdev = (self.e_env_upper_uncertainty-self.e_env_lower_uncertainty)/2
        value = norm.pdf(e_env, loc = self.mean_e_env, scale = e_env_stdev)
        return value


    def probability_density_of_eccentricity(self, e):
        if e > 1 or e < 0:
            return 0
        def cdf_e_now(e):
            value = rice.cdf(e, b=self.roots[0], loc=0, scale=self.roots[1])
            return value
        def cdf_e_env(e):
            if (self.e_env_upper_uncertainty == 0 or self.e_env_lower_uncertainty == 0):
                if e > self.mean_e_env:
                    return 1
                if e == self.mean_e_env:
                    return 0.5
                return 0
            value = norm.cdf(e, loc=self.mean_e_env, scale=(self.e_env_upper_uncertainty-self.e_env_lower_uncertainty)/2)
            return value
        return cdf_e_now(e)*(1-cdf_e_env(e))

    def plot_probability_density_of_eccentricity_vs_eccentricity_graph(self):
        eccentricity = np.linspace(0, 1, 100)
        probability_density_of_eccentricity = []
        for i in range(0, len(eccentricity)):
            probability_density_of_eccentricity = probability_density_of_eccentricity + [self.probability_density_of_eccentricity(eccentricity[i])]

        plt.plot(eccentricity, probability_density_of_eccentricity, label="Probality density of eccentricity (f(e)) vs. eccentricity (e)")
        # naming the x axis
        plt.xlabel('Eccentricity (e)')
        # naming the y axis
        plt.ylabel('probability density of eccentricity (f(e))')
        # giving a title to my graph
        plt.title('Probability density of eccentricity vs eccentricity')
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
                 age,
                 eccentricity,
                 vsini):
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
        self.eccentricity = eccentricity
        self.age = age
        self.Vsini = vsini

    def printing(self):
        print('Primary mass = ', self.primary_mass, '=', self.primary_mass.to(u.kg))
        print('Secondary mass = ', self.secondary_mass, '=', self.secondary_mass.to(u.kg))
        print('Secondary radius = ', self.secondary_radius, '=', self.secondary_radius.to(u.m))
        print('Metallicity = ', self.feh.to(u.dimensionless_unscaled))
        print('Orbital period = ', self.orbital_period, '=', self.orbital_period.to(u.s))
        print('Obliquity = ', self.obliquity.to(u.deg))
        print('Eccentricity = ', self.eccentricity.to(u.dimensionless_unscaled))
        print('Age = ', self.age, '=', self.age.to(u.s))
        print('Vsini = ', self.Vsini, '=', self.Vsini.to(u.m /u.s))


class EnvelopeEccentricityDistribution:

    def __init__(self,
                 path='/home/mmmahmud/CircularizationDissipationConstraints/data/planets_2020.04.10_14.52.24.csv',
                 file_name=b"/home/mmmahmud/poet/scripts/eccentricity_expansion_coef.txt",
                 serialized_directory='/home/mmmahmud/poet/stellar_evolution_interpolators'):
        self.path = path
        self.file_name = file_name
        self.serialized_directory = serialized_directory

        orbital_evolution_library.read_eccentricity_expansion_coefficients(
            file_name
        )

        manager = StellarEvolutionManager(serialized_directory)

        self.interpolator = manager.get_interpolator_by_name('default')

        readPlanet = planetary_system_io.read_nasa_planets(self.path,
                                                           eliminate=('SWEEPS-11',
                                                                      'HD 41004 B',
                                                                      'PSR J1719-1438',
                                                                      'K2-22'),
                                                           need_ages=False,
                                                           )

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
        self.metallicity = readPlanet.st_metfe
        self.metallicity_upper_uncertainty = readPlanet.st_metfeerr1
        self.metallicity_lower_uncertainty = readPlanet.st_metfeerr2
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
        self.vsini = readPlanet.st_vsini  # km/s
        self.vsini_upper_uncertainty = readPlanet.st_vsinierr1
        self.vsini_lower_uncertainty = readPlanet.st_vsinierr2


        self.envelope_eccentricity_function = self.create_envelope_eccentricity_function()


    def create_envelope_eccentricity_function(self,
                                     maximum_number_of_data_points = 5000,
                                     threshold_value_of_envelope_eccentricity = 0.05,
                                     largest_acceptable_value_of_envelope_eccentricity = 0.5):

        log_orbital_period = [] #Will store 10 based log of orbital period
        eccentricity_now = [] #Will store present eccentricity
        planet_name = [] #Will store planet's name
        j = -1
        for i in range(0, len(self.orbital_period)):
            if not (math.isnan(self.orbital_period[i])
                    or math.isnan(self.eccentricity_now[i])
            ):
                if self.orbital_period[i] <= 10 and (self.primary_mass[i] > 0.4 and self.primary_mass[i] < 1.2) and (
                        self.metallicity[i] > -1.014 and self.metallicity[i] < 0.537):
                    j = j + 1
                    log_orbital_period = log_orbital_period + [math.log(self.orbital_period[i], 10)]
                    eccentricity_now = eccentricity_now + [self.eccentricity_now[i]]
                    planet_name = planet_name +  [self.planet_name[i]]
                    #print('planet name = ', self.planet_name[i],
                          #', period = ', self.orbital_period[i],
                          #', log(period) = ', math.log(self.orbital_period[i], 10),
                          #', eccentricity_now = ', self.eccentricity_now[i])
            if j >= maximum_number_of_data_points-1:
                break

        # Sorting
        for i in range(0, j):
            for k in range(i + 1, j+1):
                if log_orbital_period[i] > log_orbital_period[k]:
                    temp = log_orbital_period[k]
                    temp2 = eccentricity_now[k]
                    temp3 = planet_name[k]
                    log_orbital_period[k] = log_orbital_period[i]
                    eccentricity_now[k] = eccentricity_now[i]
                    planet_name[k] = planet_name[i]
                    log_orbital_period[i] = temp
                    eccentricity_now[i] = temp2
                    planet_name[i] = temp3


        eccentricity_now_on_envelope = []
        log_orbital_period_on_envelope = []

        for i in range(0, j+1):
            if eccentricity_now[i] <= threshold_value_of_envelope_eccentricity:
                log_orbital_period_on_envelope = log_orbital_period_on_envelope + [log_orbital_period[i]]
                eccentricity_now_on_envelope = eccentricity_now_on_envelope + [threshold_value_of_envelope_eccentricity]
            if eccentricity_now[i] > threshold_value_of_envelope_eccentricity:
                break

        maximum_eccentricity_now = threshold_value_of_envelope_eccentricity

        log_orbital_period_on_the_curved_segment_of_envelope = []
        eccentricity_now_on_the_curved_segment_of_envelope = []

        for k in range(i, j+1):
            if eccentricity_now[k] >= maximum_eccentricity_now and eccentricity_now[k] <= largest_acceptable_value_of_envelope_eccentricity:
                maximum_eccentricity_now = eccentricity_now[k]
                log_orbital_period_on_envelope = log_orbital_period_on_envelope + [log_orbital_period[k]]
                eccentricity_now_on_envelope = eccentricity_now_on_envelope + [eccentricity_now[k]]
                log_orbital_period_on_the_curved_segment_of_envelope = log_orbital_period_on_the_curved_segment_of_envelope + [log_orbital_period[k]]
                eccentricity_now_on_the_curved_segment_of_envelope = eccentricity_now_on_the_curved_segment_of_envelope + [eccentricity_now[k]]
            if eccentricity_now[k] > largest_acceptable_value_of_envelope_eccentricity:
                break


        maximum_eccentricity_now = largest_acceptable_value_of_envelope_eccentricity
        topmost_point_of_the_curved_segment_of_the_envelope_not_yet_found = true
        for m in range(k, j+1):
            if eccentricity_now[m]>=maximum_eccentricity_now:
                log_orbital_period_on_envelope = log_orbital_period_on_envelope + [log_orbital_period[m]]
                eccentricity_now_on_envelope = eccentricity_now_on_envelope + [largest_acceptable_value_of_envelope_eccentricity]
                if topmost_point_of_the_curved_segment_of_the_envelope_not_yet_found:
                    log_orbital_period_on_the_curved_segment_of_envelope = log_orbital_period_on_the_curved_segment_of_envelope + [log_orbital_period[m]]
                    eccentricity_now_on_the_curved_segment_of_envelope = eccentricity_now_on_the_curved_segment_of_envelope + [largest_acceptable_value_of_envelope_eccentricity]
                    topmost_point_of_the_curved_segment_of_the_envelope_not_yet_found = false




        # figuring out the equation of envelop curve
        coefficients_and_constant_of_envelope_eccentricity_function_for_curved_segment =np.polyfit(log_orbital_period_on_the_curved_segment_of_envelope,
                                                                                eccentricity_now_on_the_curved_segment_of_envelope,
                                                                                (len(log_orbital_period_on_the_curved_segment_of_envelope)-1))
        envelope_eccentricity_function_for_curved_segment = np.poly1d(coefficients_and_constant_of_envelope_eccentricity_function_for_curved_segment)

        def envelope_eccentricity_function(orbital_period):
            x = math.log(orbital_period, 10)
            if x<log_orbital_period_on_the_curved_segment_of_envelope[0]:
                return threshold_value_of_envelope_eccentricity
            if x>log_orbital_period_on_the_curved_segment_of_envelope[-1]:
                return largest_acceptable_value_of_envelope_eccentricity
            return envelope_eccentricity_function_for_curved_segment(x)

        self.plot_present_eccentricity_vs_log_period(log_orbital_period,
                                                       eccentricity_now,
                                                       log_orbital_period_on_envelope,
                                                       eccentricity_now_on_envelope,
                                                       envelope_eccentricity_function)

        return envelope_eccentricity_function


    def plot_present_eccentricity_vs_log_period(self,
                                                  log_orbital_period,
                                                  eccentricity_now,
                                                  log_orbital_period_on_envelope,
                                                  eccentricity_now_on_envelope,
                                                  envelope_eccentricity_function):
        xdata = np.linspace(log_orbital_period_on_envelope[0], log_orbital_period_on_envelope[-1], 50)

        ydata = []
        for i in range(0, len(xdata)):
            ydata = ydata + [envelope_eccentricity_function(10**xdata[i])]
        plt.plot(log_orbital_period, eccentricity_now, 'x')
        plt.plot(xdata, ydata)
        # naming the x axis
        plt.xlabel('log(Period)')
        # naming the y axis
        plt.ylabel('Present Envelope Eccentricity')
        # giving a title to my graph
        plt.title('Present Envelope Eccentricity vs log(Period)')
        # function to show the plot
        plt.plot(log_orbital_period_on_envelope, eccentricity_now_on_envelope, 'o', label="Envelope Eccentricities vs. log(Periods)")
        plt.show()
        return

    def create_initial_theta_and_standard_deviations(self,
                             index_of_the_binary_system = 15,
                             argument_of_phase_lag_function=6.5):

        if index_of_the_binary_system > len(self.planet_name) - 1 or index_of_the_binary_system < 0: return None, None, None

        k = -1

        for i in range(0, len(self.planet_name)):
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
                    or self.eccentricity_now_lower_uncertainty[i] == 0
                    or math.isnan(self.eccentricity_now_upper_uncertainty[i])
                    or self.eccentricity_now_upper_uncertainty[i] == 0
                    or math.isnan(self.obliquity[i])
                    or math.isnan(self.obliquity_upper_uncertainty[i])
                    or math.isnan(self.obliquity_lower_uncertainty[i])
                    or math.isnan(self.vsini[i])
                    or math.isnan(self.vsini_upper_uncertainty[i])
                    or math.isnan(self.vsini_lower_uncertainty[i])):
                if self.orbital_period[i] <= 10 and (self.primary_mass[i] > 0.4 and self.primary_mass[i] < 1.2) and (
                        self.metallicity[i] > -1.014 and self.metallicity[i] < 0.537):
                    k = k + 1
                    if k == index_of_the_binary_system:
                        theta0 = [self.primary_mass[i],
                                  self.secondary_mass[i],
                                  self.secondary_radius[i],
                                  self.metallicity[i],
                                  self.orbital_period[i],
                                  self.obliquity[i],
                                  self.stellar_age[i],
                                  self.eccentricity_now[i],
                                  self.vsini[i],
                                  argument_of_phase_lag_function]

                        self.position_of_binary_system = i
                        standard_deviations = [self.primary_mass_upper_uncertainty[i],
                                               self.primary_mass_lower_uncertainty[i],
                                               self.secondary_mass_upper_uncertainty[i],
                                               self.secondary_mass_lower_uncertainty[i],
                                               self.secondary_radius_upper_uncertainty[i],
                                               self.secondary_radius_lower_uncertainty[i],
                                               self.metallicity_upper_uncertainty[i],
                                               self.metallicity_lower_uncertainty[i],
                                               self.orbital_period_upper_uncertainty[i],
                                               self.orbital_period_lower_uncertainty[i],
                                               self.obliquity_upper_uncertainty[i],
                                               self.obliquity_lower_uncertainty[i],
                                               self.stellar_age_upper_uncertainty[i],
                                               self.stellar_age_lower_uncertainty[i],
                                               self.eccentricity_now_upper_uncertainty[i],
                                               self.eccentricity_now_lower_uncertainty[i],
                                               self.vsini_upper_uncertainty[i],
                                               self.vsini_lower_uncertainty[i]]
                        return theta0, standard_deviations, self.planet_name[i]


        return None, None, None



    def print_binary_systems_whose_probability_density_of_eccentricity_can_be_figured_out(self,
                                                                                          argument_of_phase_lag_function=6.5,):
        k = 0
        for j in range(0, len(self.orbital_period) - 1):
            theta0, standard_deviations, planet_name = self.create_initial_theta_and_standard_deviations(j, argument_of_phase_lag_function)

            if not (theta0 == None or standard_deviations == None or planet_name == None):
                print('______________________________')
                k = k + 1
                print('k = ', k)
                print('theta0 = ', theta0, 'standard deviations = ', standard_deviations, ' planet name = ', planet_name)





class SamplingPropertiesOfSystem:

    def __init__(self,
                 primary_mass,
                 secondary_mass,
                 secondary_radius,
                 metallicity,
                 orbital_period,
                 obliquity,
                 stellar_age,
                 eccentricity_now,
                 vsini,
                 primary_mass_upper_uncertainty,
                 primary_mass_lower_uncertainty,
                 secondary_mass_upper_uncertainty,
                 secondary_mass_lower_uncertainty,
                 secondary_radius_upper_uncertainty,
                 secondary_radius_lower_uncertainty,
                 metallicity_upper_uncertainty,
                 metallicity_lower_uncertainty,
                 orbital_period_upper_uncertainty,
                 orbital_period_lower_uncertainty,
                 obliquity_upper_uncertainty,
                 obliquity_lower_uncertainty,
                 stellar_age_upper_uncertainty,
                 stellar_age_lower_uncertainty,
                 eccentricity_now_upper_uncertainty,
                 eccentricity_now_lower_uncertainty,
                 vsini_upper_uncertainty,
                 vsini_lower_uncertainty,
                 argument_of_phase_lag_function=6.5,
                 upper_uncertainty_associated_with_argument_of_phase_lag_function=1,
                 lower_uncertainty_associated_with_argument_of_phase_lag_function=-1,
                 planet_name = 'Exo Planet',
                 serialized_directory='/home/mmmahmud/poet/stellar_evolution_interpolators',
                 envelope_eccentricity_function = None,
                 highest_acceptable_value_of_orbital_period = 10,
                 lowest_acceptable_value_of_orbital_period = 0,
                 highest_acceptable_value_of_metallicity = 0.537,
                 lowest_acceptable_value_of_metallicity = -1.014,
                 highest_acceptable_value_of_primary_mass = 1.2,
                 lowest_acceptable_value_of_primary_mass = 0.4):

        self.primary_mass = primary_mass
        self.secondary_mass = secondary_mass
        self.secondary_radius = secondary_radius
        self.metallicity = metallicity
        self.orbital_period = orbital_period
        self.obliquity = obliquity
        self.stellar_age = stellar_age
        self.eccentricity_now = eccentricity_now
        self.vsini = vsini
        self.primary_mass_upper_uncertainty = primary_mass_upper_uncertainty
        self.primary_mass_lower_uncertainty = primary_mass_lower_uncertainty
        self.secondary_mass_upper_uncertainty = secondary_mass_upper_uncertainty
        self.secondary_mass_lower_uncertainty = secondary_mass_lower_uncertainty
        self.secondary_radius_upper_uncertainty = secondary_radius_upper_uncertainty
        self.secondary_radius_lower_uncertainty = secondary_radius_lower_uncertainty
        self.metallicity_upper_uncertainty = metallicity_upper_uncertainty
        self.metallicity_lower_uncertainty = metallicity_lower_uncertainty
        self.orbital_period_upper_uncertainty = orbital_period_upper_uncertainty
        self.orbital_period_lower_uncertainty = orbital_period_lower_uncertainty
        self.obliquity_upper_uncertainty = obliquity_upper_uncertainty
        self.obliquity_lower_uncertainty = obliquity_lower_uncertainty
        self.stellar_age_upper_uncertainty = stellar_age_upper_uncertainty
        self.stellar_age_lower_uncertainty = stellar_age_lower_uncertainty
        self.eccentricity_now_upper_uncertainty = eccentricity_now_upper_uncertainty
        self.eccentricity_now_lower_uncertainty = eccentricity_now_lower_uncertainty
        self.vsini_upper_uncertainty = vsini_upper_uncertainty
        self.vsini_lower_uncertainty = vsini_lower_uncertainty
        self.planet_name = planet_name
        self.highest_acceptable_value_of_orbital_period = highest_acceptable_value_of_orbital_period,
        self.lowest_acceptable_value_of_orbital_period = lowest_acceptable_value_of_orbital_period,
        self.highest_acceptable_value_of_metallicity = highest_acceptable_value_of_metallicity,
        self.lowest_acceptable_value_of_metallicity = lowest_acceptable_value_of_metallicity,
        self.highest_acceptable_value_of_primary_mass = highest_acceptable_value_of_primary_mass,
        self.lowest_acceptable_value_of_primary_mass = lowest_acceptable_value_of_primary_mass

        self.argument_of_phase_lag_function = argument_of_phase_lag_function
        self.upper_uncertainty_associated_with_argument_of_phase_lag_function=upper_uncertainty_associated_with_argument_of_phase_lag_function
        self.lower_uncertainty_associated_with_argument_of_phase_lag_function=lower_uncertainty_associated_with_argument_of_phase_lag_function

        self.serialized_directory = serialized_directory
        manager = StellarEvolutionManager(serialized_directory)
        self.interpolator = manager.get_interpolator_by_name('default')

        if  envelope_eccentricity_function == None:
            envelope_eccentricity_distribution_object = EnvelopeEccentricityDistribution()
            self.envelope_eccentricity_function = envelope_eccentricity_distribution_object.envelope_eccentricity_function
        else:
            self.envelope_eccentricity_function = envelope_eccentricity_function


        if (not (math.isnan(self.orbital_period)
                or math.isnan(self.orbital_period_upper_uncertainty)
                or math.isnan(self.orbital_period_lower_uncertainty)
                or math.isnan(self.primary_mass)
                or math.isnan(self.primary_mass_upper_uncertainty)
                or math.isnan(self.primary_mass_lower_uncertainty)
                or math.isnan(self.secondary_mass)
                or math.isnan(self.secondary_mass_upper_uncertainty)
                or math.isnan(self.secondary_mass_lower_uncertainty)
                or math.isnan(self.metallicity)
                or math.isnan(self.metallicity_upper_uncertainty)
                or math.isnan(self.metallicity_lower_uncertainty)
                or math.isnan(self.secondary_radius)
                or math.isnan(self.secondary_radius_upper_uncertainty)
                or math.isnan(self.secondary_radius_lower_uncertainty)
                or math.isnan(self.stellar_age)
                or math.isnan(self.stellar_age_upper_uncertainty)
                or math.isnan(self.stellar_age_lower_uncertainty)
                or math.isnan(self.eccentricity_now)
                or math.isnan(self.eccentricity_now_lower_uncertainty)
                or self.eccentricity_now_lower_uncertainty == 0
                or math.isnan(self.eccentricity_now_upper_uncertainty)
                or self.eccentricity_now_upper_uncertainty == 0
                or math.isnan(self.obliquity)
                or math.isnan(self.obliquity_upper_uncertainty)
                or math.isnan(self.obliquity_lower_uncertainty)
                or math.isnan(self.vsini)
                or math.isnan(self.vsini_upper_uncertainty)
                or math.isnan(self.vsini_lower_uncertainty))
            and ((self.orbital_period <= highest_acceptable_value_of_orbital_period and self.orbital_period > lowest_acceptable_value_of_orbital_period)
                and (self.primary_mass > lowest_acceptable_value_of_primary_mass and self.primary_mass < highest_acceptable_value_of_primary_mass)
                and (self.metallicity > lowest_acceptable_value_of_metallicity and self.metallicity < highest_acceptable_value_of_metallicity))):

            self.theta0 = [self.primary_mass,
                           self.secondary_mass,
                           self.secondary_radius,
                           self.metallicity,
                           self.orbital_period,
                           self.obliquity,
                           self.stellar_age,
                           self.eccentricity_now,
                           self.vsini,
                           self.argument_of_phase_lag_function]

            self.mean_e_env = self.envelope_eccentricity_function(orbital_period=self.orbital_period)
            eccentricity_distribution_object = EccentricityDistribution(self.eccentricity_now,
                                                                        self.eccentricity_now_upper_uncertainty,
                                                                        self.eccentricity_now_lower_uncertainty,
                                                                        self.mean_e_env)
            self.probability_density_of_eccentricity = eccentricity_distribution_object.probability_density_of_eccentricity



    def pick_a_tuple_from_the_multi_variable_Gaussian_distribution(self, mean, standard_deviation):
        n_cross_n_dimensional_array_of_standard_deviations = np.diag(
            np.array([element ** 2 for element in standard_deviation]))
        temp = np.random.default_rng().multivariate_normal(np.array(mean),
                                                           n_cross_n_dimensional_array_of_standard_deviations).tolist()
        return temp

    def priors(self, theta):
        def prior_orbital_period(orbital_period):
            if (orbital_period <= self.highest_acceptable_value_of_orbital_period and orbital_period > self.lowest_acceptable_value_of_orbital_period):
                return 1
            return 0
        def prior_primary_mass(primary_mass):
            if (primary_mass > self.lowest_acceptable_value_of_primary_mass and primary_mass < self.highest_acceptable_value_of_primary_mass):
                return 1
            return 0
        def prior_metallicity(metallicity):
            if metallicity > self.lowest_acceptable_value_of_metallicity and metallicity < self.highest_acceptable_value_of_metallicity:
                return 1
            return 0
        priors = prior_orbital_period(theta[4]) * prior_primary_mass(theta[0]) * prior_metallicity(theta[3])
        return priors

    def draw_a_successful_walker_from_Gaussian_distribution(self, mean, standard_deviation):
        success = False
        walker = None
        while not success:
            walker = self.pick_a_tuple_from_the_multi_variable_Gaussian_distribution(mean, standard_deviation)
            if ((walker[4] <= self.highest_acceptable_value_of_orbital_period and walker[4]>self.lowest_acceptable_value_of_orbital_period)
                    and (walker[0] > self.lowest_acceptable_value_of_primary_mass and walker[0] < self.highest_acceptable_value_of_primary_mass)
                    and (walker[3] > self.lowest_acceptable_value_of_metallicity and walker[3] < self.highest_acceptable_value_of_metallicity)):
                success = True
        return walker

    def log_prob(self, theta, initial_eccentricity = 0.5):
        # theta = (primary_mass, secondary_mass, secondary_radius, feh, orbital_period, obliquity, age, vsini)

        primary_mass = theta[0]
        secondary_mass = theta[1]
        secondary_radius = theta[2]
        metallicity = theta[3]
        orbital_period = theta[4]
        obliquity = theta[5]
        age = theta[6]
        eccentricity_now = theta[7]
        vsini = theta[8]
        argument_of_phase_lag_function = theta[9]

        priors = self.priors(theta)
        if priors == 0:
            return -math.inf

        a = System(primary_mass * u.solMass,
                   secondary_mass * u.earthMass,
                   secondary_radius * u.earthRad,
                   metallicity * u.dimensionless_unscaled,
                   orbital_period * u.d,
                   obliquity * u.deg,
                   age * u.Gyr,
                   eccentricity_now * u.dimensionless_unscaled,
                   vsini * u.kilometer / u.second
                   )

        dissipation = dict(
            primary=None,
            secondary=dict(
                tidal_frequency_breaks=None,
                spin_frequency_breaks=None,
                tidal_frequency_powers=np.array([0.0]),
                spin_frequency_powers=np.array([0.0]),
                reference_phase_lag=phase_lag(argument_of_phase_lag_function)
            )
        )



        b = find_evolution(system=a,
                           interpolator=self.interpolator,
                           dissipation=dissipation,
                           max_age=age * u.Gyr,
                           initial_eccentricity=initial_eccentricity,
                           initial_obliquity=0.0,
                           disk_period=None,
                           disk_dissipation_age=2e-3 * u.Gyr,
                           primary_wind_strength=0.17,
                           primary_wind_saturation=2.78,
                           primary_core_envelope_coupling_timescale=0.05,
                           secondary_wind_strength=0.0,
                           secondary_wind_saturation=100.0,
                           secondary_core_envelope_coupling_timescale=0.05,
                           orbital_period_tolerance=1e-6,
                           solve=True,
                           secondary_is_star=False)


        calculated_eccentricity_now = b.eccentricity[- 1]
        print('Calculated eccentricity_now = ',
              calculated_eccentricity_now,
              ' actual present eccentricity = ',
              eccentricity_now,
              ' at calculated age = ', b.age[- 1],
              ', where actual age of the star = ', age)


        #       print(repr(dir(b)))
        if  calculated_eccentricity_now >= 0 and calculated_eccentricity_now <=1:
            probability_density_of_eccentricity_now = self.probability_density_of_eccentricity(calculated_eccentricity_now)
            probability_density = probability_density_of_eccentricity_now * priors
            if probability_density == 0:
                return -math.inf

            if probability_density < 0:
                print('Probability density cannot be less than zero.')
                return None
            return np.log(probability_density)
        print('Calculated present eccentricity cannot be less that zero or greater than one')
        return None


    def MCMC(self, nwalker = 32, ndim = 9):
        mean = self.theta0
        standard_deviation = [(self.primary_mass_upper_uncertainty - self.primary_mass_lower_uncertainty) / 2,
                              (self.secondary_mass_upper_uncertainty - self.secondary_mass_lower_uncertainty) / 2,
                              (self.secondary_radius_upper_uncertainty - self.secondary_radius_lower_uncertainty) / 2,
                              (self.metallicity_upper_uncertainty - self.metallicity_lower_uncertainty) / 2,
                              (self.orbital_period_upper_uncertainty - self.orbital_period_lower_uncertainty) / 2,
                              (self.obliquity_upper_uncertainty - self.obliquity_lower_uncertainty) / 2,
                              (self.stellar_age_upper_uncertainty - self.stellar_age_lower_uncertainty) / 2,
                              (self.eccentricity_now_upper_uncertainty - self.eccentricity_now_lower_uncertainty)/2,
                              (self.vsini_upper_uncertainty - self.vsini_lower_uncertainty) / 2,
                              (self.upper_uncertainty_associated_with_argument_of_phase_lag_function - self.lower_uncertainty_associated_with_argument_of_phase_lag_function) / 2]


        p0 = [self.theta0]
        print('p0 initial = ', p0)
        for i in range(1, nwalker):
            p0.append(self.draw_a_successful_walker_from_Gaussian_distribution(mean, standard_deviation))
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



    def testing_log_prob(self):
        prob = []
        Qpl = []
        for i in range(3,9):
            self.theta0[9] = i*1
            Qpl = Qpl + [i*1]
            prob = prob + [self.log_prob(self.theta0)]

        plt.plot(Qpl, prob, label="Prob vs. Qpl")
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
        return




if __name__ == '__main__':
    #analysis_on_Nasa_exoplanet_data()
    print('**********************************************************')
    test1 = EccentricityDistribution(mean_e_now=0.059,
                                    e_now_upper_uncertainty=0.05,
                                    e_now_lower_uncertainty=-0.037,
                                    mean_e_env=0.46887,
                                    e_env_lower_uncertainty=0.0,
                                    e_env_upper_uncertainty=0.0)
    test1.plot_probability_density_of_eccentricity_vs_eccentricity_graph()
    print('*********************************************************')
    test2 = EnvelopeEccentricityDistribution()
    print('Binary systems whose probability density of eccentricity can be figured out:')
    test2.print_binary_systems_whose_probability_density_of_eccentricity_can_be_figured_out()
    theta0, standard_deviations, planet_name = test2.create_initial_theta_and_standard_deviations()
    print('Print chosen theta0 and corresponding standard deviations: theta0 = ', theta0, ' standard deviations = ', standard_deviations, ' planet name = ', planet_name)
    print('*********************************************************')
    test3 = SamplingPropertiesOfSystem(primary_mass=theta0[0],
                                       secondary_mass=theta0[1],
                                       secondary_radius=theta0[2],
                                       metallicity=theta0[3],
                                       orbital_period=theta0[4],
                                       obliquity=theta0[5],
                                       stellar_age=theta0[6],
                                       eccentricity_now=theta0[7],
                                       vsini=theta0[8],
                                       primary_mass_upper_uncertainty=standard_deviations[0],
                                       primary_mass_lower_uncertainty=standard_deviations[1],
                                       secondary_mass_upper_uncertainty=standard_deviations[2],
                                       secondary_mass_lower_uncertainty=standard_deviations[3],
                                       secondary_radius_upper_uncertainty=standard_deviations[4],
                                       secondary_radius_lower_uncertainty=standard_deviations[5],
                                       metallicity_upper_uncertainty=standard_deviations[6],
                                       metallicity_lower_uncertainty=standard_deviations[7],
                                       orbital_period_upper_uncertainty=standard_deviations[8],
                                       orbital_period_lower_uncertainty=standard_deviations[9],
                                       obliquity_upper_uncertainty=standard_deviations[10],
                                       obliquity_lower_uncertainty=standard_deviations[11],
                                       stellar_age_upper_uncertainty=standard_deviations[12],
                                       stellar_age_lower_uncertainty=standard_deviations[13],
                                       eccentricity_now_upper_uncertainty=standard_deviations[14],
                                       eccentricity_now_lower_uncertainty=standard_deviations[15],
                                       vsini_upper_uncertainty=standard_deviations[16],
                                       vsini_lower_uncertainty=standard_deviations[17],
                                       planet_name=planet_name
                                       )
    test3.testing_log_prob()

