import math
import planetary_system_io
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy import constants as cnst
from abc import ABCMeta, abstractmethod
from scipy.integrate import quad, dblquad, nquad
import random
import emcee
import sys
from scipy.optimize import curve_fit
from scipy.stats import rice
from scipy.special import i0
from scipy.special import i1
from scipy.special import erf
from scipy.optimize import fsolve
from scipy.optimize import root
from scipy.optimize import broyden1
from sympy import *

from scipy.optimize import root_scalar
from scipy.optimize import newton



sys.path.append('/home/mmmahmud/poet/PythonPackage')
sys.path.append('../scripts')

# from matplotlib import pyplot
# from stellar_evolution.manager import StellarEvolutionManager

from orbital_evolution.evolve_interface import library as \
    orbital_evolution_library
from orbital_evolution.binary import Binary
from orbital_evolution.transformations import phase_lag
from orbital_evolution.star_interface import EvolvingStar
from orbital_evolution.planet_interface import LockedPlanet
from orbital_evolution.initial_condition_solver import InitialConditionSolver
from basic_utils import Structure
import numpy
from astropy import units, constants
from reproduce_system import *


class SuperEccentricityDistribution(metaclass=ABCMeta):
    def __init__(self,
                 mean_e_now,
                 e_now_upper_uncertainty,
                 e_now_lower_uncertainty,
                 mean_e_env,
                 percentile_for_e_now_upper_uncertainty = 0.5*(1+erf(1/math.sqrt(2))),
                 percentile_for_e_now_lower_uncertainty = 1-0.5*(1+erf(1/math.sqrt(2))),
                 e_env_upper_uncertainty=0,
                 e_env_lower_uncertainty=0
                 ):
        self.mean_e_now = mean_e_now
        self.e_now_upper_uncertainty = e_now_upper_uncertainty
        self.e_now_lower_uncertainty = e_now_lower_uncertainty
        self.percentile_for_e_now_upper_uncertainty = percentile_for_e_now_upper_uncertainty
        self.percentile_for_e_now_lower_uncertainty = percentile_for_e_now_lower_uncertainty

        self.distribution_of_present_eccentricity = None



        self.mean_e_env = mean_e_env
        self.e_env_upper_uncertainty = e_env_upper_uncertainty
        self.e_env_lower_uncertainty = e_env_lower_uncertainty

        #self.e_env_upper_uncertainty = e_now_upper_uncertainty #This should be modified
        #self.e_env_lower_uncertainty = e_now_lower_uncertainty #This should be modified


    @abstractmethod
    def distribution_of_present_eccentricity(self, e_now):
        pass

    @abstractmethod
    def distribution_of_envelope_eccentricity(self, e_env):
        pass


class EccentricityDistribution(SuperEccentricityDistribution):


    def distribution_of_present_eccentricity_old(self, e_now):
        e_now_stdev = (self.e_now_upper_uncertainty-self.e_now_lower_uncertainty)/2
        if e_now_stdev == 0:
            if e_now == self.mean_e_now:
                return math.inf
            else:
                return 0
        return math.exp(-0.5 * ((e_now - self.mean_e_now) / e_now_stdev) ** 2) / e_now_stdev / math.sqrt(
            2 * math.pi)

    def phi(self, z):
        return 0.5 * (1 + erf(z / math.sqrt(2)))

    def equations_to_be_solved_for_Rice_distribution_parameters(self, x):
        return [rice.cdf((self.mean_e_now+self.e_now_upper_uncertainty), x[0], scale = x[1]) - self.percentile_for_e_now_upper_uncertainty,
                rice.cdf((self.mean_e_now+self.e_now_lower_uncertainty), x[0], scale = x[1]) - self.percentile_for_e_now_lower_uncertainty]


    def root_for_Rice_parameters(self):
        root = fsolve(self.equations_to_be_solved_for_Rice_distribution_parameters, [5, 0.1])
        return root

    def create_distribution_of_present_eccentricity(self):
        root = self.root_for_Rice_parameters()
        def distribution_of_present_eccentricity(e_now):
            return rice.pdf(e_now, root[0], scale = root[1])
        self.distribution_of_present_eccentricity = distribution_of_present_eccentricity
        return distribution_of_present_eccentricity

    def distribution_of_present_eccentricity(self, e_now):
        return self.distribution_of_present_eccentricity(e_now)


    def distribution_of_envelope_eccentricity(self, e_env):
        e_env_stdev = (self.e_env_upper_uncertainty-self.e_env_lower_uncertainty)/2
        if e_env_stdev == 0:
            if e_env == self.mean_e_env:
                return math.inf
            else:
                return 0
        return math.exp(-0.5 * ((e_env - self.mean_e_env) / e_env_stdev) ** 2) / e_env_stdev / math.sqrt(
            2 * math.pi)


    def distribution_of_eccentricity_by_nquad(self, e, e_env_exists=True):
        if e > 1 or e < 0:
            return 0
        if (not e_env_exists):
            # For the left part of the eccentricity vs log(orbital period) graph where datapoints are on the x axis:
            if self.mean_e_env == 0:
                return 0
            # For the right part of the eccentricity vs. log(orbital period) graph where eccentricity excedes 0.5:
            w = lambda e_now: self.distribution_of_present_eccentricity(e_now) / (1 - e_now)
            value = nquad(w, [[0, e]])

            return value
        # For the middle part of the eccentricity vs. log(orbital period) graph where we have to find the envelop.
        return nquad(lambda e_now, e_envelope: self.distribution_of_present_eccentricity(
            e_now) * self.distribution_of_envelope_eccentricity(e_envelope) / (e_envelope - e_now),
                     [self.bounds_e_now(e), self.bounds_e_env(e)])

    def bounds_e_now(self, e):
        return [0, e]

    def bounds_e_env(self, e):
        return [e, 1]


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


class BayesianAnalysis:

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


        self.envelope_eccentricity_function = self.workout_envelope_eccentricity(nsegments=10)  # nsegments is chosen in
        # such a way that we get a clear eccentricity vs log(orbital period) graph, apparently free of noise.

        self.probability_density_distribution_of_eccentricity = None

        self.position_of_binary_system = -1  # null value




    def pick_a_tuple_from_the_multi_variable_Gaussian_distribution(self, mean, standard_deviation):
        n_cross_n_dimensional_array_of_standard_deviations = np.diag(
            np.array([element ** 2 for element in standard_deviation]))
        temp = np.random.default_rng().multivariate_normal(numpy.array(mean),
                                                           n_cross_n_dimensional_array_of_standard_deviations).tolist()

        return temp

    def workout_envelope_eccentricity(self,
                                     nsegments=10,
                                     maximum_number_of_data_points = 5000,
                                     threshold_value_of_envelope_eccentricity = 0.05,
                                     largest_acceptable_value_of_envelope_eccentricity = 0.5):
        x = [] #Will store 10 based log of orbital period
        y = [] #Will store eccentricity
        z = [] #Will store planet's name
        j = 0
        for i in range(0, len(self.orbital_period)):
            if not (math.isnan(self.orbital_period[i])
                    or math.isnan(self.eccentricity_now[i])
            ):
                if self.orbital_period[i] <= 10 and (self.primary_mass[i] > 0.4 and self.primary_mass[i] < 1.2) and (
                        self.metallicity[i] > -1.014 and self.metallicity[i] < 0.537):
                    j = j + 1
                    x = x + [math.log(self.orbital_period[i], 10)]
                    y = y + [self.eccentricity_now[i]]
                    z = z +  [self.planet_name[i]]
                    print('planet name = ', self.planet_name[i], ', period = ', self.orbital_period[i], ', log(period) = ', math.log(self.orbital_period[i], 10), ', e_now = ',self.eccentricity_now[i])
            if j > maximum_number_of_data_points:
                break

        # Sorting
        for i in range(0, j - 1):
            for k in range(i + 1, j):
                if x[i] > x[k]:
                    temp = x[k]
                    temp2 = y[k]
                    temp3 = z[k]
                    x[k] = x[i]
                    y[k] = y[i]
                    z[k] = z[i]
                    x[i] = temp
                    y[i] = temp2
                    z[i] = temp3

        step = (x[-1] - x[0]) / nsegments

        # Figuring out the maximum value of eccentricity in different intervals of log(orbital period)
        i = 0
        j = 0
        v = []
        u = []

        while j < len(x):
            while x[j] < x[i] + step:
                j = j + 1
                if j > len(x) - 1: break
            if j > len(x) - 1: break
            if x[j] > x[i] + step:
                v = v + [max(y[i:j])]
                m = max(y[i:j])
                for k in range(i, j):
                    if y[k]==m:
                        print('Planet on the envelope = ', z[k], ', actual log(period) = ', x[k], ' log(period) for envelope =',((x[i] + x[j]) / 2), ', eccentricity = ', m )

                u = u + [(x[i] + x[j]) / 2]  # if we assume the maximum v occurs at the midpoint of u-interval
                # u = u + [x[i]] #We are assuming the maximum v occurs at the left end of u-interval. We are overestimating the envelop-eccentricity by doing so
                i = j

        # Figuring out the value of log(orbital period) upto which envelop eccentricity is close to zero, i.e. less than 0.05
        i = 0
        #while u[i] < 0:
        while v[i] < threshold_value_of_envelope_eccentricity:
            i = i + 1
        i = i - 1

        # Figuring out the critical value of log(orbital period) after which envelop eccentricity > 0.5
        # We are assuming that there is no envelop when envelop eccentricity exceeds 0.5
        j = i

        while v[j] < largest_acceptable_value_of_envelope_eccentricity:
            j = j + 1
        j = j+1

        # figuring out the equation of envelop curve
        popt, pcov = curve_fit(self.trial_func_for_envelope_eccentricity, u[i:j], v[i:j])
        xdata = np.linspace(u[i], u[j - 1], 50)
        plt.plot(x, y, 'x')
        plt.plot(xdata, self.trial_func_for_envelope_eccentricity(xdata, *popt))
        # naming the x axis
        plt.xlabel('log(Period)')
        # naming the y axis
        plt.ylabel('Present Envelope Eccentricity')
        # giving a title to my graph
        plt.title('Present Envelope Eccentricity vs log(Period)')
        # function to show the plot
        plt.plot(u, v, 'o', label="Envelope Eccentricities vs. log(Periods)")
        plt.show()

        def envelope_eccentricity_function(orbital_period):
            if math.log(orbital_period, 10) < u[i]:
                return threshold_value_of_envelope_eccentricity
            envelope_eccentricity = self.trial_func_for_envelope_eccentricity(math.log(orbital_period, 10), *popt)
            if envelope_eccentricity > largest_acceptable_value_of_envelope_eccentricity:
                return -1  # when the returned value is -1 then it is realized that there is no envelope eccentricity
            return envelope_eccentricity

        return envelope_eccentricity_function

    def trial_func_for_envelope_eccentricity(self, x, a, b, c):

        return a * x ** 2 + b * x + c

    def construct_probability_density_distribution_of_eccentricity(self,
                                                                   mean_e_now,
                                                                   e_now_upper_uncertainty,
                                                                   e_now_lower_uncertainty,
                                                                   orbital_period):

        mean_e_env = self.envelope_eccentricity_function(orbital_period=orbital_period)

        b = EccentricityDistribution(mean_e_now, e_now_upper_uncertainty, e_now_lower_uncertainty, mean_e_env)

        e_env_exists = True

        if mean_e_env == -1:
            print('There is no envelope.')
            e_env_exists = False
        if mean_e_env == 0:
            print('There is no envelope')
            e_env_exists = False

        def f(e):
            return b.distribution_of_eccentricity_by_nquad(e, e_env_exists)

        return f

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
                tidal_frequency_powers=numpy.array([0.0]),
                spin_frequency_powers=numpy.array([0.0]),
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
            print('QQQQQQQQ calculated eccentricity now = ', calculated_eccentricity_now)
            probability_density_of_eccentricity_now = self.probability_density_distribution_of_eccentricity(calculated_eccentricity_now)[0]

            print('********************* probability density ', probability_density_of_eccentricity_now)
            print('sddsfdsfd priors ', priors)

            probability_density = probability_density_of_eccentricity_now * priors
            if probability_density == 0:
                return -math.inf

            if probability_density < 0:
                print('probability density cannot be less than zero.')
                return
            return np.log(probability_density)
        print('calculated_eccentricity_now cannot be less that zero or greater than one')
        return

    def priors(self, theta):
        def prior_orbital_period(orbital_period):
            if orbital_period <= 10:
                return 1
            return 0

        def prior_primary_mass(primary_mass):
            if primary_mass > 0.4 and primary_mass < 1.2:
                return 1
            return 0

        def prior_metallicity(metallicity):
            if metallicity > -1.014 and metallicity < 0.537:
                return 1
            return 0

        priors = prior_orbital_period(theta[4]) * prior_primary_mass(theta[0]) * prior_metallicity(theta[3])

        print('priors = ', priors)

        return priors

    def create_initial_theta_and_probability_density_distribution_of_eccentricity(self,
                                                                                  index_of_the_binary_system = 15,
                                                                                  argument_of_phase_lag_function=6.5):

        if index_of_the_binary_system > len(self.planet_name) - 1 or index_of_the_binary_system < 0: return [-5]

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

                        self.probability_density_distribution_of_eccentricity = self.construct_probability_density_distribution_of_eccentricity(
                            mean_e_now=self.eccentricity_now[i],
                            e_now_upper_uncertainty = self.eccentricity_now_upper_uncertainty[i],
                            e_now_lower_uncertainty = self.eccentricity_now_lower_uncertainty[i],
                            orbital_period=self.orbital_period[i])
                        self.position_of_binary_system = i
                        return theta0
        theta0 = [-5]

        return theta0

    def draw_a_successful_walker_from_Gaussian_distribution(self, mean, standard_deviation):
        success = False
        walker = None
        while not success:
            walker = self.pick_a_tuple_from_the_multi_variable_Gaussian_distribution(mean, standard_deviation)
            if walker[4] <= 10 and (walker[0] > 0.4 and walker[0] < 1.2) and (
                    walker[3] > -1.014 and walker[3] < 0.537):
                success = True

        return walker

    def print_binary_systems_whose_probability_density_of_eccentricity_can_be_figured_out(self,
                                                                                          argument_of_phase_lag_function=6.5,):
        k = 0
        for j in range(0, len(self.orbital_period) - 1):
            theta0 = self.create_initial_theta_and_probability_density_distribution_of_eccentricity(j, argument_of_phase_lag_function)

            if not theta0[0] == -5:
                print('______________________________')
                k = k + 1
                print('k = ', k)
                print('theta0 = ', theta0)
                print('probability_density_distribution_of_eccentricity(0.2) = ',
                      self.probability_density_distribution_of_eccentricity(0.2))

        return

    def MCMC(self,
             index_of_the_binary_system=15,
             argument_of_phase_lag_function=6.5,
             upper_uncertainty_associated_with_argument_of_phase_lag_function=1,
             lower_uncertainty_associated_with_argument_of_phase_lag_function=-1):


        theta0 = self.create_initial_theta_and_probability_density_distribution_of_eccentricity(
            index_of_the_binary_system=index_of_the_binary_system,
            argument_of_phase_lag_function=argument_of_phase_lag_function)

        mean = theta0
        print('theta0 = ', theta0, ' for the label of binary system = ', index_of_the_binary_system)
        standard_deviation = [(self.primary_mass_upper_uncertainty[self.position_of_binary_system] -
                               self.primary_mass_lower_uncertainty[self.position_of_binary_system]) / 2,
                              (self.secondary_mass_upper_uncertainty[self.position_of_binary_system] -
                               self.secondary_mass_lower_uncertainty[self.position_of_binary_system]) / 2,
                              (self.secondary_radius_upper_uncertainty[self.position_of_binary_system] -
                               self.secondary_radius_lower_uncertainty[self.position_of_binary_system]) / 2,
                              (self.metallicity_upper_uncertainty[self.position_of_binary_system] -
                               self.metallicity_lower_uncertainty[self.position_of_binary_system]) / 2,
                              (self.orbital_period_upper_uncertainty[self.position_of_binary_system] -
                               self.orbital_period_lower_uncertainty[self.position_of_binary_system]) / 2,
                              (self.obliquity_upper_uncertainty[self.position_of_binary_system] -
                               self.obliquity_lower_uncertainty[self.position_of_binary_system]) / 2,
                              (self.stellar_age_upper_uncertainty[self.position_of_binary_system] -
                               self.stellar_age_lower_uncertainty[self.position_of_binary_system]) / 2,
                              (self.eccentricity_now_upper_uncertainty[self.position_of_binary_system]-
                               self.eccentricity_now_lower_uncertainty[self.position_of_binary_system])/2,
                              (self.vsini_upper_uncertainty[self.position_of_binary_system] -
                               self.vsini_lower_uncertainty[self.position_of_binary_system]) / 2,
                              (upper_uncertainty_associated_with_argument_of_phase_lag_function - lower_uncertainty_associated_with_argument_of_phase_lag_function) / 2]

        nwalker = 32
        ndim = 9
        p0 = [theta0]
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
        k = 0
        for j in range(0, len(self.orbital_period) - 1):
            theta0 = self.create_initial_theta_and_probability_density_distribution_of_eccentricity(j)

            if not theta0[0] == -5:
                print('______________________________')
                k = k + 1
                print('k = ', k)
                print('theta0 = ', theta0)
                print('probability_density_distribution_of_eccentricity(0.2) = ',
                      self.probability_density_distribution_of_eccentricity(0.2))

        theta0 = self.create_initial_theta_and_probability_density_distribution_of_eccentricity(index_of_the_binary_system=15)  # I am choosing index = 15
        theta0[8] = 6.5
        prob = []
        Qpl = []
        print('log_prob for Qpl = 6.5,', self.log_prob(theta0))
        for i in range(1,9):
            theta0[8] = i*1
            Qpl = Qpl + [i*1]
            prob = prob + [self.log_prob(theta0)]
            print('>>>>>>>>>>>>>>>>>>', i, 'Qpl ', Qpl, 'prob ', prob)


        print('Qpl = ', Qpl)
        print('prob = ', prob)

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

    def test_Nasa_Exoplanet_data(self, interpolator):

        """
        Testing TransitingExoplanet by Nasa Exoplanet data
        """

        readPlanet = planetary_system_io.read_nasa_planets(self.path,
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
                            tidal_frequency_powers=numpy.array([0.0]),
                            spin_frequency_powers=numpy.array([0.0]),
                            reference_phase_lag=phase_lag(8)
                        )
                    )
                    final_age = stellar_age[i]
                    print(repr(interpolator))

                    b = find_evolution(system = a,
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
                                       #required_ages=numpy.arange(0.1, final_age, 0.1))

                    init = 0
                    final = len(b.orbital_period)
                    #final = 5000

                    y = b.orbital_period[init:final]
                    z = b.eccentricity[init:final]
                    #x = []
                    #for i in range(init, len(y)):
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
                    print('final age in find_evolution = ',x[-1])
                    print('present eccentricity according to find_evolution =  ',z[-1])
                    print('present orbital period according to find_evolution = ', y[-1])
                    print('actual final age = ', final_age)
                    print('actual present eccentricity = ', eccentricity[i])
                    print('actual orbital period = ', orbital_period[i])

                    evolutionary_data = evolutionary_data + [b]
                    print(repr(dir(b)))
            if j > 0:
                break
        return evolutionary_data


if __name__ == '__main__':
    #print(4)
    test = EccentricityDistribution(mean_e_now=0.5,
                                    e_now_upper_uncertainty=0.1,
                                    e_now_lower_uncertainty=-0.1,
                                    mean_e_env=0.3,
                                    percentile_for_e_now_upper_uncertainty=erf(1),
                                    percentile_for_e_now_lower_uncertainty=(1-erf(1)))

    #print('Testing Laguerre polynomial of degree half = ', test.laguerre_polynomial_of_degree_half(0.7) )
    #print('Testing Rice distribution = ', test.mean_of_Rice_distribution(0.2))
    #print('ppf of rice distribution = ', test.percent_point_function_for_Rice_distribution(percentile = 0.16, b=0.5, local=0.1, scale=2))
    #print('mean of rice distribution minus mean e_now = ', test.mean_of_Rice_distribution_minus_mean_e_now(0.2))
    #print('solve for b ', test.solve_for_b())

    print(test.root_for_Rice_parameters())
    a = test.create_distribution_of_present_eccentricity()
    print('a ', a(0.2))
    print('distribution of present eccentricity = ',test.distribution_of_present_eccentricity(0.2))


    #print('Testing rice distribution: Starts')
    #fig, ax = plt.subplots(1, 1)
    #print(ax)

    #b = 0.5
    #mean, var, skew, kurt = rice.stats(b, moments='mvsk')
    #print('mean =', mean)
    #print('var =', var)
    #print('skew =', skew)
    #print('kurt =', kurt)
    #x = np.linspace(rice.ppf(0.01, b),
                    #rice.ppf(0.99, b), 100)
    #print('ppf .99 is ', rice.ppf(.99, b))
    #print('x = ', x)
    #ax.plot(x, rice.pdf(x, b),'r-', lw=5, alpha=0.6, label='rice pdf')
    #rv = rice(b)
    #print('rv ', rv)
    #ax.plot(x, rv.pdf(x), 'k-', lw=2, label='frozen pdf')
    #vals = rice.ppf([0.001, 0.5, 0.999], b)
    #np.allclose([0.001, 0.5, 0.999], rice.cdf(vals, b))
    #r = rice.rvs(b, size=1000)
    #ax.hist(r, density=True, histtype='stepfilled', alpha=0.2)
    #ax.legend(loc='best', frameon=False)
    #plt.show()
    #print('End')

    #Testing evolution of binary systems using test_Nasa_exoplanet_data
    #print('_____________________________________________________________________________')
    #print('Testing evolution of binary systems using test_Nasa_exoplanet_data.txt: Start')
    #test = BayesianAnalysis()
    #eccentricity_expansion_fname = b"/home/mmmahmud/poet/scripts/eccentricity_expansion_coef.txt"
    #orbital_evolution_library.read_eccentricity_expansion_coefficients(
        #eccentricity_expansion_fname
    #)
    #serialized_dir = '/home/mmmahmud/poet/stellar_evolution_interpolators'
    #manager = StellarEvolutionManager(serialized_dir)

    #interpolator = manager.get_interpolator_by_name('default')

    #evolutionary_data = test.test_Nasa_Exoplanet_data(interpolator=(interpolator, interpolator))

    #print('End')
    #print('_____________________________________________________________________________')

    #Testing testing_log_prob, MCMC and other methods

    #print('Testing testing_log_prob, MCMC and other methods: Start')
    #test = BayesianAnalysis()

    #test.testing_log_prob()
    #test.MCMC()

    #    b = EccentricityDistribution(mean_e_now=0.2, e_now_stdev=0.5, mean_e_env=0.8, e_env_stdev=0.5)
    #    print('The value of b is nquad = ', b.distribution_of_eccentricity_by_nquad(e=0.6))

    #   test = BayesianAnalysis()
    #   test.MCMC()
    #   print('envelope eccentricity = ',test.workout_envelope_eccentricity(planet_orbital_period=0.5))
    #   f = test.construct_probability_density_distribution_of_eccentricity(mean_e_now=0.2, e_now_stdev=0.5, orbital_period=5.6)
    #    print('f(.3) = ', f( e=0.3))

    #  c = test.pick_a_tuple_from_the_multi_variable_Gaussian_distribution(mean = [2,3,4], standard_deviation=[.2,.4,.5])
    #   print('a random tuple from gaussian distribution: ',c)









