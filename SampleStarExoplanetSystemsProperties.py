import sys
import os
import logging
from datetime import datetime
import math
import json
import h5py


sys.path.append('/home1/08529/mmmahmud/CircularizationDissipationConstraints/source')
sys.path.append('/home1/08529/mmmahmud/general_purpose_python_modules')
sys.path.append('/home1/08529/mmmahmud/CircularizationDissipationConstraints/data')
sys.path.append('/home1/08529/mmmahmud/poet')
sys.path.append('/home1/08529/mmmahmud/lib')
sys.path.append('/home1/08529/mmmahmud/.local/lib/python3.9/site-packages')

from split_normal_distribution import split_normal
#from bayesian.stellar_param_sampling.poet_interp_likelihood import POETInterpLikelihood
#from bayesian.stellar_param_sampling.star_sampler import StarSampler
import numpy
#from bayesian.stellar_param_sampling.feh_conditional_likelihood_base import \
    #FeHConditionalLikelihoodBase
from scipy.stats import norm
import astropy.constants as const
import Constraints_for_selecting_systems
from astropy import units as un
import StarExoplanetSystem
#from reproduce_system import *
from random import random, randint
from multiprocessing import Pool, Queue, Process, Value
import emcee
import corner
import matplotlib.pyplot as plt
#from orbital_evolution.evolve_interface import library as \
    #orbital_evolution_library
import EnvelopeEccentricityDistribution
import EccentricityDistribution
import argparse
from bayesian.hacked_emcee_hdf5_backend import HDFBackend, TempHDFBackend
from bayesian.stellar_param_sampling.poet_interp_likelihood import POETInterpLikelihood
from bayesian.stellar_param_sampling.star_sampler import StarSampler
from bayesian.stellar_param_sampling.feh_conditional_likelihood_base import \
    FeHConditionalLikelihoodBase
from reproduce_system import *
from orbital_evolution.evolve_interface import library as \
    orbital_evolution_library

if not sys.warnoptions:
    import warnings

    from sqlalchemy.exc import SAWarning

    warnings.filterwarnings('ignore',
                            r"^Dialect sqlite\+pysqlite does \*not\* support Decimal objects natively\, "
                            "and SQLAlchemy must convert from floating point - rounding errors and other "
                            "issues may occur\. Please consider storing Decimal numbers as strings or "
                            "integers on this platform for lossless storage\.$",
                            SAWarning, r'^sqlalchemy\.sql\.type_api$')

def getStellarEvolutionInterpolatorsDirectory():
    return '/home1/08529/mmmahmud/poet/stellar_evolution_interpolators' 
def getEccentricityExpansionCoefficientsFile():
    return b"/work/08529/mmmahmud/ls6/eccentricity_expansion_coef_O400.sqlite" 

def setup_basic_logging(logging_file_name, msg_file_name):
    def ensure_directory(fname):
        """Make sure the directory containing the given name exists."""
        dirname = os.path.dirname(fname)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    ensure_directory(logging_file_name)
    ensure_directory(msg_file_name)
    msg_file = os.open(msg_file_name,
                       os.O_WRONLY | os.O_TRUNC | os.O_CREAT | os.O_DSYNC,
                       mode=0o666
                       )

    os.dup2(msg_file, 1)
    os.dup2(msg_file, 2)

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        handler.close()
        logging_config = dict(
            filename=logging_file_name,
            level=logging.DEBUG,
            format='%(levelname)s %(asctime)s %(name)s: %(message)s | %(pathname)s.%(funcName)s:%(lineno)d',
            datefmt='%Y%m%d%H%M%S'
        )
    logging.basicConfig(**logging_config)
    return

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
        self.num_parallel_processes = 16
        self.star_sampler_pickle_fname = 'star_sampler.pkl'
        self.stellar_evolution_interpolator_dir = getStellarEvolutionInterpolatorsDirectory()  #'/home/mmmahmud/poet/stellar_evolution_interpolators'
        self.time_ode_atol = 1e-08
        self.time_ode_max_step = 0.1
        self.time_ode_rtol = 1e-06

class PriorTransform:
    def __init__(self,
                 means,
                 standard_deviations,
                 max_argument_of_phase_lag_function_for_planet=12,
                 min_argument_of_phase_lag_function_for_planet=5,
                 min_log_tidal_break_period=math.log(0.5,10),
                 max_log_tidal_break_period=1,
                 min_power_law_argument=-5,
                 max_power_law_argument=5,
                 max_initial_stellar_spin=15,
                 min_initial_stellar_spin=5,
                 power_of_the_ratio_of_planetary_and_stellar_radius = 2.0,
                 logger=None
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
        self.power_of_the_ratio_of_planetary_and_stellar_radius = power_of_the_ratio_of_planetary_and_stellar_radius


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

        if logger is not None:
            logger.debug("POETInterpLikelihood instance is going to be created.")
        likelihood = POETInterpLikelihood(
            **constraints,
            rtol=config.time_ode_rtol,
            atol=config.time_ode_atol,
            max_step=config.time_ode_max_step
        )
        if logger is not None:
            logger.debug("POETInterpLikelihood instance is created. StarSampler instance is going to be created.")
        self.star_sampler = StarSampler(likelihood, config)
        if logger is not None:
            logger.debug("StarSampler instance is created.")

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

        n = 1/self.power_of_the_ratio_of_planetary_and_stellar_radius
        secondary_radius = (
                                       ratio_of_planet_to_stellar_radius ** n) * primary_radius * const.R_sun.value / const.R_earth.value
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
                 initial_eccentricity = 0.8,
                 constraints = Constraints_for_selecting_systems.constraints(),
                 spin_frequency_breaks_for_planet=None,
                 spin_frequency_powers_for_planet=numpy.array([0.0]),
                 Q0 = 5, logger = None, number_of_parallel_processes = 16
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
        self.logger = logger
        self.number_of_parallel_processes = number_of_parallel_processes

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

        logging.info("log prob of eccetricity will be calculated for the following parameters.")
        logging.info('The parameters for evolution are: ')
        logging.info('primary mass = %(m)f '% dict(m= primary_mass))
        logging.info('stellar age = %(age)f'% dict(age= stellar_age))
        logging.info('secondary radius = %(r)f' % dict(r= secondary_radius))
        logging.info('stellar metallicity = %(feh)f '% dict(feh= stellar_metallicity))
        logging.info('secondary mass = %(m)f'% dict(m= secondary_mass))
        logging.info('initial stellar spin = %(spin)f '% dict(spin=initial_stellar_spin))
        logging.info('argument of phase lag function for planet = %(ar)f '% dict(ar= argument_of_phase_lag_function_for_planet))
        logging.info('tidal break period = %(bp)f '% dict(bp= tidal_break_period))
        logging.info('power law argument = %(alpha)f' % dict(alpha= power_law_argument))
        priors = self.priors({'primary mass': primary_mass,
                              'secondary mass': secondary_mass,
                              'stellar metallicity': stellar_metallicity,
                              'stellar age': stellar_age})

        if not priors:
            return -numpy.inf

        star_exoplanet_binary_system = StarExoplanetSystem.System(primary_mass=primary_mass * un.solMass,
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
            tidal_frequency_breaks_for_planet = numpy.array([break_frequency])
            tidal_frequency_powers_for_planet = numpy.array([0.0, power_law_argument])

            # tidal_frequency_breaks_for_planet = np.array([2 * math.pi / 20, break_frequency])
            # tidal_frequency_powers_for_planet = np.array([1.0, 0.0, power_law_argument])

        if power_law_argument > 0 or power_law_argument == 0:
            tidal_frequency_breaks_for_planet = numpy.array([2 * math.pi / 20, break_frequency])
            tidal_frequency_powers_for_planet = numpy.array([0.0, power_law_argument, 0.0])
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
        logging.debug("Evolution is being worked out")
        if self.logger is not None: self.logger.debug("Evolution is being worked out")
        evolution_complete = False
        e_in = self.initial_eccentricity
        while (not evolution_complete):
            try:
                 if self.logger is not None: self.logger.debug("find_evolution method is being called.")
                 evolutionary_history = find_evolution(system=star_exoplanet_binary_system,
                                                       interpolator=FeHConditionalLikelihoodBase.interpolator,
                                                       dissipation=dissipation,
                                                       max_age=stellar_age * un.Gyr,
                                                       initial_eccentricity=e_in * un.dimensionless_unscaled,
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
            except AssertionError:
                 e_in = e_in - 0.01
                 logging.warning('Calculating evolution failed, trying initial eccentricity = %(e)f' % dict(e=e_in))
                 if self.logger is not None: self.logger.warning('Calculating evolution failed, trying initial eccentricity = %(e)f' % dict(e=e_in))
                 print('Calculating evolution failed, trying initial eccentricity = %(e)f' % dict(e=e_in))
                 #if self.logger is not None: self.logger.warning('Calculating evolution failed, trying initial eccentricity = %(e)f' % dict(e=e_in))
                 evolution_complete = False
            except ValueError as error:
                 logging.error('Invalid parameter values encountered: %(error)s' % dict(error= str(error)))
                 if self.logger is not None: self.logger.error('Invalid parameter values encountered: %(error)s' % dict(error= str(error)))
                 print('Invalid parameter values encountered: %(error)s' % dict(error= str(error)))
                 return None
            else:
                 calculated_eccentricity_now = evolutionary_history.eccentricity[- 1]
                 if self.logger is not None: self.logger.debug("Calculated eccentricity now is %(e)f" % dict(e=calculated_eccentricity_now))
                 logging.debug("Calculated eccentricity now is %(e)f" % dict(e=calculated_eccentricity_now))
                 print("Calculated eccentricity now is %(e)f" % dict(e=calculated_eccentricity_now))
                 if calculated_eccentricity_now < 1:
                      evolution_complete = True
                 else:
                      return None
 
        self.calculated_eccentricity_now = calculated_eccentricity_now

        if calculated_eccentricity_now >= 0 and calculated_eccentricity_now <= 1:
            probability_density_of_the_calculated_eccentricity = self.probability_density_of_eccentricity(
                calculated_eccentricity_now)
            probability_density = probability_density_of_the_calculated_eccentricity * priors
            if self.logger is not None: self.logger.debug("Probability density of the calculated eccentricity is %(p)f " % dict(p = probability_density))
            if probability_density == 0:
                logging.debug("log of probability density of e is %(x)f " % dict(x=-numpy.inf))
                if self.logger is not None: self.logger.debug("log of probability density of e is %(x)f " % dict(x=-numpy.inf))
                return -numpy.inf
            if probability_density < 0:
                logging.warning('Probability density cannot be less than zero.')
                return None
            logging.debug("log of probability density of e is %(x)f " % dict(x=numpy.log(probability_density)))
            if self.logger is not None: self.logger.debug("log of probability density of e is %(x)f " % dict(x=numpy.log(probability_density)))
            return numpy.log(probability_density)
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
                                    min_log_likelihood = - 0.0001,
                                    output_dirname = "/home1/08529/mmmahmud/scratch/circularization_exoplanet_system/sampling_output"
                                    ):
        numpy.random.seed()
        pid = os.getpid()
        date_time = datetime.now().strftime('%Y%m%d%H%M%S')
        logging_file_name = '%(dirname)s/%(system)s/P0_processor_logging/p0_%(now)s_%(pid)d.logging' % dict(dirname=output_dirname,
                                                                                                            system=self.system_name,
                                                                                                            pid=pid,
                                                                                                            now=date_time)
        msg_file_name = '%(dirname)s/%(system)s/P0_processor_message/p0_%(now)s_%(pid)d.txt' % dict(dirname=output_dirname,
                                                                                                    system=self.system_name,
                                                                                                    pid=pid,
                                                                                                    now=date_time)

        setup_basic_logging(logging_file_name, msg_file_name)
        if self.logger is not None: self.logger.debug("generate_successsful_walkers_aux method is called. ")
        print("u = ", u)

        lgQpl_max = self.prior_transform_instance.max_argument_of_phase_lag_function_for_planet
        lgQpl_min = self.prior_transform_instance.min_argument_of_phase_lag_function_for_planet
        lgPbr_max = self.prior_transform_instance.max_log_tidal_break_period
        lgPbr_min = self.prior_transform_instance.min_log_tidal_break_period
        alpha_max = self.prior_transform_instance.max_power_law_argument
        alpha_min = self.prior_transform_instance.min_power_law_argument
        init_spin_max = self.prior_transform_instance.max_initial_stellar_spin
        init_spin_min = self.prior_transform_instance.min_initial_stellar_spin

        del_lgQpl_u = 0.25/(lgQpl_max - lgQpl_min)
        del_lgPbr_u = 0.10/(lgPbr_max - lgPbr_min)
        del_alpha_u = 0.5/(alpha_max - alpha_min)
        del_spin_u = 1/(init_spin_max - init_spin_min)

        n_lgQpl = int(1.0/del_lgQpl_u)
        n_lgPbr = int(1.0/del_lgPbr_u)
        n_alpha = int(1.0/del_alpha_u)
        n_spin  = int(1.0/del_spin_u)

        w_start = numpy.random.randint(0, (n_alpha + 1))
        x_start = numpy.random.randint(0, (n_lgPbr + 1))
        y_start = numpy.random.randint(0, (n_spin + 1))

        w_end = w_start - 1
        if w_end < 0: w_end = n_alpha
        x_end = x_start - 1
        if x_end < 0: x_end = n_lgPbr
        y_end = y_start - 1
        if y_end < 0: y_end = n_spin

        loop_w_complete = False
        loop_x_complete = False
        loop_y_complete = False
        walker_found = False

        w = w_start
        alpha_u = w * del_alpha_u + numpy.random.rand() * del_alpha_u
        while not (loop_w_complete or walker_found):
            x = x_start
            lgPbr_u = (x + numpy.random.rand()) * del_lgPbr_u
            while not (loop_x_complete or walker_found):
                y = y_start
                spin_u = (y + numpy.random.rand()) * del_spin_u
                while not (loop_y_complete or walker_found):
                    u[5] = spin_u
                    u[6] = numpy.random.rand() * del_lgQpl_u
                    u[7] = lgPbr_u
                    u[8] = alpha_u
                    print("Modified u is ", u)
                    log_likelihood, parameters_for_evolution = self(u)
                    print("log likelihood for u = ", u, " is ", log_likelihood, " with parameters ", parameters_for_evolution) 
                    if (not math.isinf(log_likelihood)) and (log_likelihood < min_log_likelihood):
                        print("khela suru: log likelihood is not negative infinity for this u")
                        a = u[6]
                        b = 1.0
                        c = (a+b)/2.0
                        u[6] = c
                        log_likelihood, parameters_for_evolution = self(u)
                        print("u = ", u, " log likelihood ", log_likelihood, " parameters ", parameters_for_evolution)
                        while log_likelihood < min_log_likelihood:
                            while math.isinf(log_likelihood):
                                b = c
                                c = (a+b)/2.0
                                u[6] = c
                                logging.debug("c = %(s)f  " % dict(s = c))
                                if self.logger is not None: self.logger.debug("c =  %(s)f " % dict(s = c))
                                log_likelihood, parameters_for_evolution = self(u)
                                logging.debug("log likelihood = %(s)f  " % dict(s = log_likelihood))
                                if self.logger is not None: self.logger.debug("log likelihood =  %(s)f " % dict(s = log_likelihood))
                            if log_likelihood >  min_log_likelihood: break
                            a = c
                            c = (a+b)/2.0
                            u[6] = c
                            logging.debug("c = %(s)f  " % dict(s = c))
                            if self.logger is not None: self.logger.debug("c =  %(s)f " % dict(s = c))
                            log_likelihood, parameters_for_evolution = self(u)
                            logging.debug("log likelihood = %(s)f  " % dict(s = log_likelihood))
                            if self.logger is not None: self.logger.debug("log likelihood =  %(s)f " % dict(s = log_likelihood))
                    if (not math.isinf(log_likelihood)) and (log_likelihood > min_log_likelihood):
                        print("we got a range of u[6] for the acceptable walkers. The extreme u = ", u, " with log likelihood ", log_likelihood)
                        print("correspondin extreme parameters are: ", parameters_for_evolution)

                        x = numpy.random.uniform(0.0, u[6], 1)
                        u[6] = x[0]
                        print("a value of u[6] picked up randomly from 0 and the extremum u[6]. That is ", u[6])
                        log_likelihood, parameters_for_evolution = self(u)
                        if not math.isinf(log_likelihood):
                            logging.debug('The discovered walker is u  = %(u)s' % dict(u=numpy.array2string(u)))
                            logging.debug('log p = %(logp)f' % dict(logp = log_likelihood))
                            logging.debug('parameters for evolution = %(params)s' % dict(params=numpy.array2string(parameters_for_evolution)))
                            if self.logger is not None:
                                self.logger.debug('u  = %(u)s' % dict(u=numpy.array2string(u)))
                                self.logger.debug('log p = %(logp)f' % dict(logp = log_likelihood))
                                self.logger.debug('parameters for evolution = %(params)s' % dict(params=numpy.array2string(parameters_for_evolution)))

                            p0_file_exists = os.path.exists(p0_file_name)
                            logging.debug('number of discovered walkers = %(x)f' % dict(x=number_of_discovered_walkers.value))
                            if p0_file_exists:
                                while True:
                                     if ((p0_file_is_being_updated.value == 0) and (number_of_discovered_walkers.value < nwalkers)):
                                         p0_file_is_being_updated.value = 1
                                         p0_file = open(p0_file_name, 'rb')
                                         p0 = numpy.load(p0_file)
                                         p0_file.close()
                                         p0_file = open(p0_file_name, 'wb')
                                         p0 = numpy.vstack((p0, u))
                                         numpy.save(p0_file, p0)
                                         p0_file.close()
                                         walkers.put(u)
                                         number_of_discovered_walkers.value = number_of_discovered_walkers.value + 1
                                         p0_file_is_being_updated.value = 0 
                                         break
                                     if not (number_of_discovered_walkers.value < nwalkers):
                                         break
                            else:
                                while True:
                                     if ((p0_file_is_being_updated.value == 0) and (number_of_discovered_walkers.value < nwalkers)):
                                         p0_file_is_being_updated.value = 1
                                         p0_file = open(p0_file_name, 'wb')
                                         numpy.save(p0_file, u)
                                         p0_file.close()
                                         walkers.put(u)
                                         number_of_discovered_walkers.value = number_of_discovered_walkers.value + 1
                                         p0_file_is_being_updated.value = 0
                                         break
                                     if not (number_of_discovered_walkers.value < nwalkers):
                                         break
                            walker_found = True
                    if y == y_end: loop_y_complete = True
                    y = y + 1
                    if y > n_spin: y = 0
                    spin_u = (y + numpy.random.rand()) * del_spin_u
                if x == x_end: loop_x_complete = True
                x = x + 1
                if x > n_lgPbr: x = 0
                lgPbr_u = (x + numpy.random.rand()) *  del_lgPbr_u
            if w == w_end: loop_w_complete = True
            w = w + 1
            if w > n_alpha: w = 0
            alpha_u = (w + numpy.random.rand()) * del_alpha_u

        if number_of_discovered_walkers.value < nwalkers:
            numpy.random.seed(pid)
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
                                nwalkers=64,
                                ndim=9,
                                nprocessors = 16):

        if self.logger is not None: self.logger.debug("Initial walkers are going to be generated now.")
        array_of_processes = []
        number_of_discovered_walkers = Value('i', 0)
        walkers = Queue()
        p0_file_is_being_updated = Value('i', 0)

        i = 0
        while i<nprocessors:
            numpy.random.seed()
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
             nwalkers=64,
             ndim=9,
             reset_backend = False,
             iterations = 16):

        mcmc_progress_file_name = '%(system)s_mcmc_progress.h5' % dict(system=self.system_name)
        p0_file_name = '%(system)s_p0_file.npy' % dict(system=self.system_name)

        p0_file_exists = os.path.exists(p0_file_name)
        backend_file_exists = os.path.exists(mcmc_progress_file_name)


        if (not p0_file_exists):
            if backend_file_exists:
                if reset_backend:
                    p0 = self.generate_successful_walkers(p0_file_name,
                                                          nwalkers,
                                                          ndim)
                else:
                    backend_file_reader = emcee.backends.HDFBackend(mcmc_progress_file_name, read_only= True)
                    ndim_prev = backend_file_reader.shape[1]
                    if not (ndim == ndim_prev):
                        reset_backend = True
                        p0 = self.generate_successful_walkers(p0_file_name,
                                                              nwalkers,
                                                              ndim)
                    else:
                        if (not backend_file_reader.initialized) or backend_file_reader.iteration <= 0:
                            reset_backend = True
                            p0 = self.generate_successful_walkers(p0_file_name,
                                                                  nwalkers,
                                                                  ndim)
            else:
                if self.logger is not None: 
                    self.logger.debug('Initially the file %(fname)s did not exist' % dict(fname=p0_file_name))
                    self.logger.debug('The walkers are going to be generated for the first time.')
                    self.logger.debug('The file %(fname)s will be created and the walkers will be stored there.' % dict(fname=p0_file_name))
                reset_backend = True
                p0 = self.generate_successful_walkers(p0_file_name,
                                                      nwalkers,
                                                      ndim)
        else:
            if backend_file_exists and (not reset_backend):
                backend_file_reader = emcee.backends.HDFBackend(mcmc_progress_file_name, read_only=True)
                ndim_prev = backend_file_reader.shape[1]
                if ndim != ndim_prev:
                    reset_backend = True
                if (not backend_file_reader.initialized) or backend_file_reader.iteration <= 0:
                    reset_backend = True
            if (not backend_file_exists) or reset_backend:
                if self.logger is not None:
                    self.logger.debug('The file %(fname)s existed previously' % dict(fname=p0_file_name))
                    self.logger.debug('Previously worked out walkers will be loaded in the code for running MCMC.')
                p0_file = open(p0_file_name, 'rb')
                p0 = numpy.load(p0_file)
                p0_file.close()
                if self.logger is not None:
                    self.logger.debug('The already discovered walkers are: %(p0)s' % dict(p0=numpy.array2string(p0)))
                number_of_already_stored_walkers = p0.size / ndim
                number_of_walkers_yet_to_be_found = (int)(nwalkers - number_of_already_stored_walkers)
                if number_of_walkers_yet_to_be_found > 0:
                    if self.logger is not None: self.logger.debug('New walkers are going to be discovered')
                    p0 = numpy.vstack((p0, self.generate_successful_walkers(p0_file_name,
                                                                         number_of_walkers_yet_to_be_found,
                                                                         ndim)))
                    if self.logger is not None: self.logger.debug('All walkers are: %(p0)s' % dict(p0=numpy.array2string(p0)))
                if number_of_walkers_yet_to_be_found <= 0:
                    p0 = p0[0:nwalkers]


        def setup_basic_logging_for_mcmc():
            pid = os.getpid()
            date_time = datetime.now().strftime('%Y%m%d%H%M%S')
            logging_file_name='/home1/08529/mmmahmud/scratch/circularization_exoplanet_system/sampling_output/%(system)s/mcmc_processor_logging/mcmc_%(now)s_%(pid)d.logging'%dict(system=self.system_name,
                                                                                                                                                                                   pid=pid,
                                                                                                                                                                                   now=date_time)
            msg_file_name='/home1/08529/mmmahmud/scratch/circularization_exoplanet_system/sampling_output/%(system)s/mcmc_processor_message/mcmc_%(now)s_%(pid)d.txt' % dict(system=self.system_name,
                                                                                                                                                                             pid=pid,
                                                                                                                                                                             now=date_time)
            setup_basic_logging(logging_file_name, msg_file_name)

        with Pool(self.number_of_parallel_processes,
                  initializer=setup_basic_logging_for_mcmc,
                  #initargs=[config],
                  maxtasksperchild=1) as pool:

            #backend = emcee.backends.HDFBackend(mcmc_progress_file_name)
            backend = HDFBackend(mcmc_progress_file_name)
            if reset_backend:
                backend.reset(nwalkers, ndim)
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.__call__, pool=pool, backend=backend)

            if backend_file_exists:
                if self.logger is not None: self.logger.debug('Backend file exists.')
                print("Backend file exists.")
                chain_exists = backend.initialized and (backend.iteration > 0)
                if self.logger is not None:
                    self.logger.debug('Backend iterations = %(x)f ' % dict(x= backend.iteration))
                    self.logger.debug('backend initialized = %(x)s' % dict(x= backend.initialized))
                print('Backend iterations = %(x)f ' % dict(x= backend.iteration))
                print('backend initialized = %(x)s' % dict(x= backend.initialized))
                if reset_backend == False and chain_exists:
                    if self.logger is not None:
                        self.logger.debug('Backend file is not subject to reset. The chain size is not zero.')
                        self.logger.debug('Next samples will be drawn from the end of the previously worked out chain.')
                    print('Backend file is not subject to reset. The chain size is not zero. Next samples will be drawn from the end of the previously worked out chain.')
                    sampler.run_mcmc(None, iterations, progress = True)
                else:
                    if self.logger is not None:
                        self.logger.debug('Either the backend file is subject to reset or previously calculated chain size is zero.')
                        self.logger.debug('Samples will be drawn for the first time.')
                    print('Either the backend file is subject to reset or previously calculated chain size is zero.')
                    print('Samples will be drawn for the first time.')
                    backend.reset(nwalkers, ndim)
                    sampler.run_mcmc(p0, iterations, progress = True)
            else:
                print('Backend file did not exist previously')
                if self.logger is not None: self.logger.debug('Backend file did not exist previously')
                sampler.run_mcmc(p0, iterations, progress = True)


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
                return -numpy.inf, numpy.array([None, None, None, None, None, None, None, None, None, None, None])
        parameters_for_evolution = self.prior_transform_instance(u)

        params = numpy.array([parameters_for_evolution['primary mass'],
                           parameters_for_evolution['stellar age'],
                           parameters_for_evolution['secondary radius'],
                           parameters_for_evolution['stellar metallicity'],
                           parameters_for_evolution['secondary mass'],
                           parameters_for_evolution['initial stellar spin'],
                           parameters_for_evolution['argument of phase lag function for planet'],
                           parameters_for_evolution['tidal break period'],
                           parameters_for_evolution['power law argument']])
        log_prob_parameters_for_evolution = self.log_prob(parameters_for_evolution)
        if numpy.isinf(-log_prob_parameters_for_evolution):
            print("Here it happens HHHHH")
            print("Actual params are ", parameters_for_evolution)
            return -numpy.inf, numpy.array([None, None, None, None, None, None, None, None, None, None, None])
        params = numpy.append(params, [self.calculated_eccentricity_now, log_prob_parameters_for_evolution])
        return log_prob_parameters_for_evolution, params

class InitializationOfSamplingPropertiesOfSystem:
    serialized_directory = getStellarEvolutionInterpolatorsDirectory()
    eccentricity_expansion_fname = getEccentricityExpansionCoefficientsFile()

    @classmethod
    def set_serialized_directory(cls, name):
        cls.serialized_directory = name

    @classmethod
    def get_serialized_directory(cls):
        return cls.serialized_directory

    @classmethod
    def set_eccentricity_expansion_fname(cls, name):
        cls.eccentricity_expansion_fname = name

    @classmethod
    def get_eccentricity_expansion_fname(cls):
        return cls.eccentricity_expansion_fname

    def __init__(self):
        # mp.set_start_method('forkserver')
        print('serialized directory ', self.serialized_directory)
        manager = StellarEvolutionManager(self.serialized_directory)
        interpolator = manager.get_interpolator_by_name('default')
        FeHConditionalLikelihoodBase.set_interpolator(interpolator)
        orbital_evolution_library.prepare_eccentricity_expansion(
            self.eccentricity_expansion_fname,
            1e-4,
            True,
            True
        )

def setup_logger(name,
                 log_file,
                 level=logging.DEBUG,
                 formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')):
    """To setup as many loggers as you want"""

    def ensure_directory(fname):
        """Make sure the directory containing the given name exists."""
        dirname = os.path.dirname(fname)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

    ensure_directory(log_file)

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger


class SamplingPropertiesOfSystem:
    def __init__(self,
                 means,
                 standard_deviations,
                 system_name = 'Star-Exoplanet',
                 envelope_eccentricity_function=EnvelopeEccentricityDistribution.envelope_eccentricity_function,
                 initial_eccentricity=0.8,
                 initial_stellar_spin=5,
                 max_argument_of_phase_lag_function_for_planet=12,
                 min_argument_of_phase_lag_function_for_planet=5,
                 min_log_of_tidal_break_period=math.log(0.5, 10),
                 max_log_of_tidal_break_period=1,
                 min_power_law_argument=-5,
                 max_power_law_argument=5,
                 max_initial_stellar_spin=15,
                 min_initial_stellar_spin=5,
                 constraints=Constraints_for_selecting_systems.constraints(),
                 spin_frequency_breaks_for_planet=None,
                 spin_frequency_powers_for_planet=numpy.array([0.0]),
                 power_of_the_ratio_of_planetary_and_stellar_radius = 2.0,
                 output_dirname="/home1/08529/mmmahmud/scratch/circularization_exoplanet_system/sampling_output",
                 logging_level=logging.DEBUG):


        logging_fname = "%(dirname)s/%(system)s/logger_for_%(system)s.log" % dict(dirname=output_dirname, system=system_name)
        logger_name = "%(dirname)s/%(system)s/logger_for_%(system)s" % dict(dirname=output_dirname, system=system_name)
        logger = setup_logger(logger_name, logging_fname, level=logging_level)

        logger.info("The logger file for the system %(system_name)s is created." % dict(system_name = system_name))
        logger.info("The initial eccentricity = %(e)f " % dict(e= initial_eccentricity))
        self.initial_eccentricity = initial_eccentricity
        logger.info("The initial stellar spin = %(spin)f " % dict(spin= initial_stellar_spin))
        self.initial_stellar_spin = initial_stellar_spin
        logger.info("Maximum argument of phase lag function for planet = %(Q)f" % dict(Q=max_argument_of_phase_lag_function_for_planet))
        self.max_argument_of_phase_lag_function_for_planet = max_argument_of_phase_lag_function_for_planet
        logger.info("Minimum argument of phase lag function for planet = %(Q)f" % dict(Q=min_argument_of_phase_lag_function_for_planet))
        self.min_argument_of_phase_lag_function_for_planet = min_argument_of_phase_lag_function_for_planet
        logger.info("Maximum log of tidal break period = %(p)f" % dict(p = max_log_of_tidal_break_period))
        self.max_log_of_tidal_break_period = max_log_of_tidal_break_period
        logger.info("Minimum log of tidal break period = %(p)f" % dict(p = min_log_of_tidal_break_period))
        self.min_log_of_tidal_break_period = min_log_of_tidal_break_period
        logger.info("Maximum value of power law argument = %(a)f" % dict(a = max_power_law_argument))
        self.max_power_law_argument = max_power_law_argument
        logger.info("Minimum value of power law argument = %(a)f" % dict(a = min_power_law_argument))
        self.min_power_law_argument = min_power_law_argument
        logger.info("Maximum initial stellar spin = %(a)f" % dict(a = max_initial_stellar_spin))
        self.max_initial_stellar_spin = max_initial_stellar_spin
        logger.info("Minimum initial stellar spin = %(a)f" % dict(a = min_initial_stellar_spin))
        self.min_initial_stellar_spin = min_initial_stellar_spin
        self.spin_frequency_breaks_for_planet = spin_frequency_breaks_for_planet
        self.spin_frequency_powers_for_planet = spin_frequency_powers_for_planet
        self.power_of_the_ratio_of_planetary_and_stellar_radius = power_of_the_ratio_of_planetary_and_stellar_radius


        self.envelope_eccentricity_function = envelope_eccentricity_function

        if (Constraints_for_selecting_systems.constraints_are_satisfied(orbital_period=means['orbital period'],
                                      primary_mass=means['primary mass'],
                                      secondary_mass=means['secondary mass'],
                                      stellar_metallicity=means['stellar metallicity'],
                                      eccentricity_now=means['present eccentricity'],
                                      stellar_age=means['stellar age'],
                                      constraints=constraints) ):
            logger.info("Measured values are %(means)s " % dict(means = means))
            logger.info("Standard deviations in the measured values are %(stdev)s " % dict(stdev = standard_deviations))
            self.means = means
            self.standard_deviations = standard_deviations
            logger.info("PriorTransform instance is going to be created for the %(system)s" % dict(system=system_name))
            self.prior_transform_instance = PriorTransform(means,
                                                           standard_deviations,
                                                           max_argument_of_phase_lag_function_for_planet,
                                                           min_argument_of_phase_lag_function_for_planet,
                                                           min_log_of_tidal_break_period,
                                                           max_log_of_tidal_break_period,
                                                           min_power_law_argument, max_power_law_argument,
                                                           max_initial_stellar_spin, min_initial_stellar_spin,
                                                           power_of_the_ratio_of_planetary_and_stellar_radius, logger)


            self.e_env = self.envelope_eccentricity_function(
                x=self.means['semi major axis'] / self.means['secondary radius'], logger=logger)
            logger.info("The envelope eccentricity for the system %(system)s is %(eenv)f" % dict(system=system_name, eenv=self.e_env))
            logger.debug("EnccentricityDistribution instance is going to be created.")
            eccentricity_distribution_object = EccentricityDistribution.EccentricityDistribution(self.means['present eccentricity'],
                                                                        self.standard_deviations[
                                                                            'eccentricity_now_upper_uncertainty'],
                                                                        self.standard_deviations[
                                                                            'eccentricity_now_lower_uncertainty'],
                                                                        self.e_env,
                                                                        system_name, logger = logger)
            logger.debug("EccentricityDistribution instance is created. Now probability density of eccentricity vs. eccentricity graph will be plotted.")

            eccentricity_distribution_object.plot_probability_density_of_eccentricity_vs_eccentricity_graph()
            logger.debug("The graph is plotted")
            self.probability_density_of_eccentricity = eccentricity_distribution_object.probability_density_of_eccentricity
            logger.debug("LogLikelihood instance is going to be created")
            self.log_likelihood_instance = LogLikelihood(self.prior_transform_instance,
                                                         self.means['orbital period'],
                                                         0,  # obliquity
                                                         self.probability_density_of_eccentricity,
                                                         self.e_env,
                                                         system_name,
                                                         initial_eccentricity,
                                                         constraints,
                                                         spin_frequency_breaks_for_planet,
                                                         spin_frequency_powers_for_planet,
                                                         logger=logger
                                                         )
            logger.debug("LogeLikelihod instance is created. Now MCMC is going to run")
            self.log_likelihood_instance.MCMC()
            logger.debug("MCMC is done")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--path_of_the_stellar_evolution_interpolators_directory',
                        help='Store the path of the directory of the stellar evolution interpolators'
                        )
    parser.add_argument('--path_of_the_eccentricity_expansion_coefficients_file',
                        help='Store the path of the eccentricity expansion coefficients file')
    parser.add_argument('--measured_values',
                        help='stores a dictionary containing the measured values of primary mass, '
                             'secondary mass, primary radius, secondary radius, stellar metallicity, '
                             'orbital period, obliquity, stellar age, present eccentricity, semi-major axis,'
                             'stellar log g, stellar density, stellar effective temperature, ratio of planet to stellar radius',
                        type=json.loads)
    parser.add_argument('--standard_deviations',
                        help='stores a dictionary containing the standard deviations associated with the measured'
                             'values of the quantities',
                        type=json.loads)
    parser.add_argument('--power_of_the_ratio_of_planetary_and_stellar_radius',
                        help = 'stores the power of the ratio of planetary and stellar radius',
                        type = float)
    parser.add_argument('--system',
                        help = 'stores the name of the star-exoplanet system')
    args = parser.parse_args()

    if args.path_of_the_stellar_evolution_interpolators_directory:
        InitializationOfSamplingPropertiesOfSystem.set_serialized_directory(args.path_of_the_stellar_evolution_interpolators_directory)
    if args.path_of_the_eccentricity_expansion_coefficients_file:
        InitializationOfSamplingPropertiesOfSystem.set_eccentricity_expansion_fname(args.path_of_the_eccentricity_expansion_coefficients_file)

    InitializationOfSamplingPropertiesOfSystem()

    test3 = SamplingPropertiesOfSystem(args.measured_values,
                                       args.standard_deviations,
                                       system_name=args.system,
                                       envelope_eccentricity_function=EnvelopeEccentricityDistribution.envelope_eccentricity_function,
                                       power_of_the_ratio_of_planetary_and_stellar_radius = args.power_of_the_ratio_of_planetary_and_stellar_radius if args.power_of_the_ratio_of_planetary_and_stellar_radius else 2                                      
                                       )

