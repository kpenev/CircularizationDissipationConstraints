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
import numpy
from scipy.stats import norm
import astropy.constants as const
import Constraints_for_selecting_systems
from astropy import units as un
import StarExoplanetSystem
from random import random, randint
from multiprocessing import Pool, Queue, Process, Value
import emcee
import corner
import matplotlib.pyplot as plt
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
import traceback

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

def ensure_directory(fname):
        """Make sure the directory containing the given name exists."""
        dirname = os.path.dirname(fname)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def setup_basic_logging(logging_file_name, msg_file_name):
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
    def __init__(self, teff=None, feh=None, logg=None, mean_density=None, debug_plot=None):
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
        self.num_parallel_processes = 16
        self.star_sampler_pickle_fname = 'star_sampler.pkl'
        self.stellar_evolution_interpolator_dir = getStellarEvolutionInterpolatorsDirectory()
        self.time_ode_atol = 1e-08
        self.time_ode_max_step = 0.1
        self.time_ode_rtol = 1e-06

class PriorTransform:
    def __init__(self,
                 means,
                 standard_deviations,
                 system = "Star-Exoplanet",
                 dirname = "/work/08529/mmmahmud",
                 max_argument_of_phase_lag_function_for_planet=12,
                 min_argument_of_phase_lag_function_for_planet=3,
                 max_argument_of_phase_lag_function_for_star=12, #
                 min_argument_of_phase_lag_function_for_star=5, #
                 min_log_tidal_break_period=math.log(0.5,10),
                 max_log_tidal_break_period=1,
                 min_power_law_argument=-5,
                 max_power_law_argument=5,
                 max_initial_stellar_spin=15,
                 min_initial_stellar_spin=14.5,
                 logger=None
                 ):
        self.means = means
        self.standard_deviations = standard_deviations
        self.system = system
        self.dirname = "%(dirname)s/star_sampler" % dict(dirname=dirname)
        self.max_argument_of_phase_lag_function_for_planet = max_argument_of_phase_lag_function_for_planet
        self.min_argument_of_phase_lag_function_for_planet = min_argument_of_phase_lag_function_for_planet
        self.max_argument_of_phase_lag_function_for_star = max_argument_of_phase_lag_function_for_star #
        self.min_argument_of_phase_lag_function_for_star = min_argument_of_phase_lag_function_for_star #

        self.min_log_tidal_break_period = min_log_tidal_break_period
        self.max_log_tidal_break_period = max_log_tidal_break_period
        self.min_power_law_argument = min_power_law_argument
        self.max_power_law_argument = max_power_law_argument
        self.max_initial_stellar_spin = max_initial_stellar_spin
        self.min_initial_stellar_spin = min_initial_stellar_spin
        self.logger = logger
        self.logging_fname = None
        if self.logger is not None:
            handler = logger.handlers[0]
            self.logging_fname = handler.baseFilename
            self.logger.debug("The name of the log file for the system %(s)s is %(f)s " % dict(s=system, f=self.logging_fname))
            self.logger.debug("We are forming prior transform instance for %(s)s" % dict(s=system))
        if self.logging_fname is not None: logging.basicConfig(filename = self.logging_fname, level=logging.DEBUG, force = True, format='%(asctime)s %(message)s')
        logging.debug("Basic Config for the log file is done for the system %(s)s" % dict(s=system))

        def construct_star_sampler():
            debug_plot = [('interpolation_performance', 'interp_performance.pdf')]
            teff = None
            feh = None
            logg = None
            mean_density = None
            if 'stellar effective temperature' in self.means:
                if not math.isnan(self.means['stellar effective temperature']):
                    teff = split_normal.freeze_error_bar(
                        mode=self.means['stellar effective temperature'],
                        abs_plus_error=self.standard_deviations['stellar_effective_temperature_upper_uncertainty'],
                        abs_minus_error=-self.standard_deviations['stellar_effective_temperature_lower_uncertainty'])
            if 'stellar metallicity' in self.means:
                if not math.isnan(self.means['stellar metallicity']):
                    feh = split_normal.freeze_error_bar(
                        mode=self.means['stellar metallicity'],
                        abs_plus_error=self.standard_deviations['stellar_metallicity_upper_uncertainty'],
                        abs_minus_error=-self.standard_deviations['stellar_metallicity_lower_uncertainty'])
            if 'stellar log g' in self.means:
                if not math.isnan(self.means['stellar log g']):
                    logg = split_normal.freeze_error_bar(
                        mode=self.means['stellar log g'],
                        abs_plus_error=self.standard_deviations['stellar_log_g_upper_uncertainty'],
                        abs_minus_error=-self.standard_deviations['stellar_log_g_lower_uncertainty'])
            if 'stellar density' in self.means:
                if not math.isnan(self.means['stellar density']):
                    mean_density = split_normal.freeze_error_bar(
                                mode=self.means['stellar density'],
                                abs_plus_error=self.standard_deviations['stellar_density_upper_uncertainty'],
                                abs_minus_error=-self.standard_deviations['stellar_density_lower_uncertainty'])

            config = Element(teff=teff, feh=feh, logg=logg, mean_density=mean_density, debug_plot=debug_plot)
            constraints = dict()
            if not (config.Teff is None): constraints['teff'] = config.Teff
            if not (config.logg is None): constraints['logg'] = config.logg
            if not (config.mean_density is None): constraints['rho'] = config.mean_density

            if self.logger is not None:
                self.logger.debug("POETInterpLikelihood instance for %(system)s is going to be created." % dict(system=self.system))
            likelihood = None
            try:
                likelihood = POETInterpLikelihood(
                    **constraints,
                    rtol=config.time_ode_rtol,
                    atol=config.time_ode_atol,
                    max_step=config.time_ode_max_step
                )
            except:
                if self.logger is not None: self.logger.debug("An exception occurs while creating POETInterpLikelihood instance for %(s)s" % dict(s=system))
            else:
                if self.logger is not None: self.logger.debug("POETInterpLikelihood instance is successfully created for %(s)s" % dict(s=system))
            star_sampler = None
            try:
                if likelihood is not None:
                    star_sampler = StarSampler(likelihood, config)
            except:
                if self.logger is not None: self.logger.debug("An exception occurs while creating star sampler for %(s)s" % dict(s=self.system))
                star_sampler = None
            else:
                if self.logger is not None and star_sampler is not None: self.logger.debug("star sampler is successfully created for %(s)s" % dict(s=self.system))
            return star_sampler

        self.star_sampler_fname = "%(dirname)s/%(system)s_star_sampler.pkl" % dict(dirname=self.dirname, system=self.system)
        ensure_directory(self.star_sampler_fname)
        file_exists = os.path.exists(self.star_sampler_fname)
        self.star_sampler = None
        #if file_exists:
            #self.star_sampler = pickle.load(open(self.star_sampler_fname, "rb"))
        #if self.star_sampler is None:
            #self.star_sampler = construct_star_sampler()
            #save_object(self.star_sampler, self.star_sampler_fname)

    def __call__(self, u):
        unit_cube = numpy.array([u[0], u[1], u[2]])
        #stellar_metallicity, primary_mass, stellar_age = self.star_sampler.__call__(unit_cube)
        stellar_metallicity = norm.ppf(u[0], loc=self.means['stellar metallicity'], scale=(self.standard_deviations[
                                                                                          'stellar_metallicity_upper_uncertainty'] -
                                                                                          self.standard_deviations[
                                                                                          'stellar_metallicity_lower_uncertainty'])/2)
        primary_mass = norm.ppf(u[1], loc=self.means['primary mass'], scale=(self.standard_deviations[
                                                                            'primary_mass_upper_uncertainty'] -
                                                                             self.standard_deviations[
                                                                            'primary_mass_lower_uncertainty'])/2)
        stellar_age = norm.ppf(u[2], loc=self.means['stellar age'], scale=(self.standard_deviations[
                                                                           'stellar_age_upper_uncertainty'] -
                                                                           self.standard_deviations[
                                                                           'stellar_age_lower_uncertainty'])/2)
        if self.logger is not None: self.logger.debug("stellar metallicity = %(a)f, primary mass = %(b)f and stellar age = %(c)f" % dict(a=stellar_metallicity, b=primary_mass, c=stellar_age))
        if not FeHConditionalLikelihoodBase.interpolator.in_range(primary_mass, stellar_metallicity):
            if self.logger is not None: self.logger.debug("primary mass and/or stellar metallicity is not within the range of poet")
            return None
        primary_rad = FeHConditionalLikelihoodBase.interpolator('RADIUS', primary_mass, stellar_metallicity)
        if stellar_age <= primary_rad.min_age or stellar_age >= primary_rad.max_age:
            if self.logger is not None: self.logger.debug("stellar age is not within the range of poet")
            return None
        primary_radius = primary_rad(stellar_age)
        if self.logger is not None: self.logger.debug("primary_radius is %(x)f" % dict(x=primary_radius))
        iconv = FeHConditionalLikelihoodBase.interpolator('ICONV', primary_mass, stellar_metallicity)
        if min(iconv(numpy.linspace(iconv.min_age, stellar_age, 1000))) <= 0:
            if self.logger is not None: self.logger.debug("The planet will collapse into the star")
            return None

        secondary_radius_is_found = False
        a = 'transit depth' in self.means
        if a:
            aa = 'transit_depth_upper_uncertainty' in self.standard_deviations and 'transit_depth_lower_uncertainty' in self.standard_deviations
            if aa:
               aaa = not(math.isnan(self.standard_deviations['transit_depth_upper_uncertainty']) or math.isnan(self.standard_deviations['transit_depth_lower_uncertainty']))
               if aaa:
                   transit_depth = norm.ppf(u[3], loc=self.means['transit depth'],
                                            scale=(self.standard_deviations['transit_depth_upper_uncertainty']-self.standard_deviations['transit_depth_lower_uncertainty'])/2)/100
                   secondary_radius = (transit_depth**0.5)*primary_radius * const.R_sun.value/const.R_earth.value
                   secondary_radius_is_found = True
        b = 'ratio of planet to stellar radius' in self.means
        if b and (not secondary_radius_is_found):
            bb = 'ratio_of_planet_to_stellar_radius_upper_uncertainty' in self.standard_deviations and 'ratio_of_planet_to_stellar_radius_lower_uncertainty' in self.standard_deviations
            if bb:
                bbb = not(math.isnan(self.standard_deviations['ratio_of_planet_to_stellar_radius_upper_uncertainty'])
                          or math.isnan(self.standard_deviations['ratio_of_planet_to_stellar_radius_lower_uncertainty']))
                if bbb:
                    ratio_of_planet_to_stellar_radius = norm.ppf(u[3], loc = self.means['ratio of planet to stellar radius'],
                                                                 scale = (self.standard_deviations['ratio_of_planet_to_stellar_radius_upper_uncertainty']
                                                                          - self.standard_deviations['ratio_of_planet_to_stellar_radius_lower_uncertainty'])/2)
                    secondary_radius = ratio_of_planet_to_stellar_radius * primary_radius * const.R_sun.value/const.R_earth.value
                    secondary_radius_is_found = True
        c = 'secondary radius' in self.means
        if c and (not secondary_radius_is_found):
            cc = 'secondary_radius_upper_uncertainty' in self.standard_deviations and 'secondary_radius_lower_uncertainty' in self.standard_deviations
            if cc:
                ccc = not(math.isnan(self.standard_deviations['secondary_radius_upper_uncertainty'])
                          or math.isnan(self.standard_deviations['secondary_radius_lower_uncertainty']))
                if ccc:
                    secondary_radius = norm.ppf(u[3], loc = self.means['secondary radius'],
                                                scale = (self.standard_deviations['secondary_radius_upper_uncertainty']
                                                         - self.standard_deviations['secondary_radius_lower_uncertainty'])/2)
                    secondary_radius_is_found = True
        if not secondary_radius_is_found: secondary_radius = None
        if self.logger is not None: self.logger.debug("secondary radius is %(x)s" % dict(x=repr(secondary_radius)))
        secondary_mass = norm.ppf(u[4], loc=self.means['secondary mass'], scale=(self.standard_deviations[
                                                                                     'secondary_mass_upper_uncertainty'] -
                                                                                 self.standard_deviations[
                                                                                     'secondary_mass_lower_uncertainty']) / 2)
        if self.logger is not None: self.logger.debug("secondary mass is %(x)s" % dict(x=repr(secondary_mass)))
        initial_stellar_spin = self.min_initial_stellar_spin + u[5] * (
                    self.max_initial_stellar_spin - self.min_initial_stellar_spin)
        argument_of_phase_lag_function_for_planet = self.min_argument_of_phase_lag_function_for_planet + u[6] * (
                    self.max_argument_of_phase_lag_function_for_planet - self.min_argument_of_phase_lag_function_for_planet)
        tidal_break_period = 10**(self.min_log_tidal_break_period + u[7] * (
                    self.max_log_tidal_break_period - self.min_log_tidal_break_period))
        power_law_argument = self.min_power_law_argument + u[8] * (
                    self.max_power_law_argument - self.min_power_law_argument)
        argument_of_phase_lag_function_for_star = self.min_argument_of_phase_lag_function_for_star + u[9] * ( #
                    self.max_argument_of_phase_lag_function_for_star - self.min_argument_of_phase_lag_function_for_star) #


        parameters_for_evolution = {'primary mass': primary_mass,
                                    'stellar age': stellar_age,
                                    'secondary radius': secondary_radius,
                                    'stellar metallicity': stellar_metallicity,
                                    'secondary mass': secondary_mass,
                                    'initial stellar spin': initial_stellar_spin,
                                    'argument of phase lag function for planet': argument_of_phase_lag_function_for_planet,
                                    'argument of phase lag function for star': argument_of_phase_lag_function_for_star, #
                                    'tidal break period': tidal_break_period,
                                    'power law argument': power_law_argument}
        if self.logger is not None: self.logger.debug("params for evolution: %(x)s" % dict(x=repr(parameters_for_evolution)))
        return parameters_for_evolution

class LogLikelihood:
    def __init__(self,
                 prior_transform_instance,
                 orbital_period,
                 obliquity,
                 eccentricity_distribution,
                 e_env,
                 system_name = 'Star-Exoplanet',
                 directory_name = '/work/08529/mmmahmud/p0andmcmc',
                 initial_eccentricity = 0.8,
                 constraints = Constraints_for_selecting_systems.constraints(),
                 spin_frequency_breaks_for_planet=None,
                 spin_frequency_powers_for_planet=numpy.array([0.0]),
                 spin_frequency_breaks_for_star=None, #
                 spin_frequency_powers_for_star=numpy.array([0.0]), #
                 logger = None, number_of_parallel_processes = 16
                 ):

        self.prior_transform_instance = prior_transform_instance
        self.orbital_period = orbital_period
        self.obliquity = obliquity
        self.eccentricity_distribution = eccentricity_distribution
        self.constraints = constraints
        self.initial_eccentricity = initial_eccentricity
        self.spin_frequency_breaks_for_planet = spin_frequency_breaks_for_planet
        self.spin_frequency_powers_for_planet = spin_frequency_powers_for_planet
        self.spin_frequency_breaks_for_star = spin_frequency_breaks_for_star #
        self.spin_frequency_powers_for_star = spin_frequency_powers_for_star #
        self.e_env = e_env
        self.system_name = system_name
        self.directory_name = directory_name
        self.logger = logger
        self.number_of_parallel_processes = number_of_parallel_processes

        self.means = self.prior_transform_instance.means

        if (not (self.eccentricity_distribution is None)) and (not (self.e_env is None)):
            if 'secondary radius' in self.means:
                rad = self.means['secondary radius']
                if not self.logger is None: self.logger.debug("secondary radius is directly found in the list of parameters that is %(x)f" % dict(x=rad))
            elif ('ratio of planet to stellar radius' in self.means) and ('primary radius' in self.means):
                rad = self.means['ratio of planet to stellar radius'] * self.means['primary radius'] * const.R_sun.value / const.R_earth.value
                if not self.logger is None: self.logger.debug("secondary radius is calculated from Rp/Rs and Rs that is %(x)f" % dict(x=rad))
            elif ('transit depth' in self.means) and ('primary radius' in self.means):
                rad = ((self.means['transit depth']/100)**0.5) *  self.means['primary radius'] * const.R_sun.value / const.R_earth.value
                if not self.logger is None: self.logger.debug("secondary radius is calculated from transit depth and primary radius that is %(x)f" % dict(x=rad))
            else:
                rad = None
            if not(rad is None):
                rad_j = rad * const.R_earth.value / const.R_jup.value
                if not self.logger is None: self.logger.debug("secondary radius in Jupiter mass is %(x)f" % dict(x=rad_j))
                if rad_j > 0.6:
                    self.prior_transform_instance.min_argument_of_phase_lag_function_for_planet = 3
                    if self.logger is not None: self.logger.debug("min argument of phase lag function for this BIG planet %(x)f"
                                                                  % dict(x=self.prior_transform_instance.min_argument_of_phase_lag_function_for_planet))
                else:
                    self.prior_transform_instance.min_argument_of_phase_lag_function_for_planet = 1
                    if self.logger is not None: self.logger.debug("min argument of phase lag function for this SMALL planet %(x)f"
                                                                  % dict(x=self.prior_transform_instance.min_argument_of_phase_lag_function_for_planet))

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
        if parameters_for_evolution is None: return -numpy.inf, self.initial_eccentricity
        primary_mass = parameters_for_evolution['primary mass']
        stellar_age = parameters_for_evolution['stellar age']
        secondary_radius = parameters_for_evolution['secondary radius']
        stellar_metallicity = parameters_for_evolution['stellar metallicity']
        secondary_mass = parameters_for_evolution['secondary mass']
        initial_stellar_spin = parameters_for_evolution['initial stellar spin']
        argument_of_phase_lag_function_for_planet = parameters_for_evolution[
            'argument of phase lag function for planet']
        argument_of_phase_lag_function_for_star = parameters_for_evolution[ #
            'argument of phase lag function for star'] #

        tidal_break_period = parameters_for_evolution['tidal break period']
        power_law_argument = parameters_for_evolution['power law argument']

        if not self.logger is None:
            self.logger.info("log prob of eccetricity will be calculated for the following parameters.")
            self.logger.info('The parameters for evolution are: ')
            self.logger.info('primary mass = %(m)f '% dict(m= primary_mass))
            self.logger.info('stellar age = %(age)f'% dict(age= stellar_age))
            self.logger.info('secondary radius = %(r)f' % dict(r= secondary_radius))
            self.logger.info('stellar metallicity = %(feh)f '% dict(feh= stellar_metallicity))
            self.logger.info('secondary mass = %(m)f'% dict(m= secondary_mass))
            self.logger.info('initial stellar spin = %(spin)f '% dict(spin=initial_stellar_spin))
            self.logger.info('argument of phase lag function for planet = %(ar)f '% dict(ar= argument_of_phase_lag_function_for_planet))
            self.logger.info('argument of phase lag function for star = %(ar)f '% dict(ar= argument_of_phase_lag_function_for_star))
            self.logger.info('tidal break period = %(bp)f '% dict(bp= tidal_break_period))
            self.logger.info('power law argument = %(alpha)f' % dict(alpha= power_law_argument))

        priors = self.priors({'primary mass': primary_mass,
                              'secondary mass': secondary_mass,
                              'stellar metallicity': stellar_metallicity,
                              'stellar age': stellar_age})

        if not priors:
            return -numpy.inf, self.initial_eccentricity

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
        if power_law_argument > 0 or power_law_argument == 0:
            tidal_frequency_breaks_for_planet = numpy.array([2 * math.pi / 20, break_frequency])
            tidal_frequency_powers_for_planet = numpy.array([0.0, power_law_argument, 0.0])
            reference_argument_of_phase_lag_function_for_planet += power_law_argument * math.log10(tidal_frequency_breaks_for_planet[1]/tidal_frequency_breaks_for_planet[0])

        tidal_frequency_breaks_for_star = None #
        tidal_frequency_powers_for_star = numpy.array([0.0]) #
        primary=dict( #
            tidal_frequency_breaks=tidal_frequency_breaks_for_star, #
            spin_frequency_breaks=self.spin_frequency_breaks_for_star, #
            tidal_frequency_powers=tidal_frequency_powers_for_star, #
            spin_frequency_powers=self.spin_frequency_powers_for_star, #
            reference_phase_lag=phase_lag(argument_of_phase_lag_function_for_star) #
        )
        dissipation = dict(
            primary=primary, #
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
        e_in_of_previous_iteration = e_in + 0.01
        if self.logger is not None: self.logger.debug("****************************** the initial eccentricity is %(e)f" % dict(e=e_in))

        dummy = False
        calculated_eccentricity_now = 0.8 * (argument_of_phase_lag_function_for_planet**2)/144.0
        while (not evolution_complete) and (not dummy):
            try:
                if self.logger is not None:
                    self.logger.debug("find_evolution method for %(system)s is being called." % dict(system=self.system_name))
                    self.logger.debug("dissipation: %(dissipation)s" % dict(dissipation = repr(dissipation)))
                evolutionary_history = find_evolution(system=star_exoplanet_binary_system,
                                                      interpolator=FeHConditionalLikelihoodBase.interpolator,
                                                      dissipation=dissipation,
                                                      max_age=parameters_for_evolution['stellar age'] * un.Gyr,
                                                      initial_eccentricity=e_in * un.dimensionless_unscaled,
                                                      initial_obliquity=0.0,
                                                      disk_period=parameters_for_evolution['initial stellar spin'] * un.d,
                                                      disk_dissipation_age=2e-2 * un.Gyr,
                                                      primary_wind_strength=0.17,
                                                      primary_wind_saturation=2.78,
                                                      primary_core_envelope_coupling_timescale=0.005 * un.Gyr,
                                                      secondary_wind_strength=0.0,
                                                      secondary_wind_saturation=100.0,
                                                      secondary_core_envelope_coupling_timescale=0.005 * un.Gyr,
                                                      orbital_period_tolerance=1e-6,
                                                      solve=True,
                                                      secondary_is_star=False)
            except RuntimeError as runerror:
                logging.error('Runtime error occurs for %(s)s: %(error)s' % dict(s=self.system_name, error= str(runerror)))
                if self.logger is not None: self.logger.warning(traceback.format_exc())
                e_in_of_previous_iteration = e_in
                e_in = e_in - 0.01
                if (e_in < max(self.e_env + 0.2, 0.7)):
                    if self.logger is not None: self.logger.debug("e_in becomes less than e_env + 0.2, so loglikelihood is -inf")
                    return -numpy.inf, self.initial_eccentricity

                logging.warning('Calculating evolution for %(s)s with initial eccentricity %(ep)f failed, now trying initial eccentricity = %(e)f'
                                 % dict(e=e_in, s=self.system_name, ep=e_in_of_previous_iteration ))
                if self.logger is not None: self.logger.warning('Calculating evolution failed, trying initial eccentricity = %(e)f' % dict(e=e_in))
                evolution_complete = False

            except ValueError as verror:
                if self.logger is not None: self.logger.warning(traceback.format_exc())
                logging.error('Invalid parameter values encountered for %(s)s: %(error)s' % dict(s=self.system_name, error= str(verror)))
                if self.logger is not None: self.logger.error('Invalid parameter values encountered for %(s)s: %(error)s' % dict(error= str(verror), s=self.system_name))
                return -numpy.inf, self.initial_eccentricity
            except Exception as exc:
                if self.logger is not None: self.logger.warning(traceback.format_exc())
                logging.error("General Error occurs while calculating evolution for %(s)s" % dict(s=self.system_name))
                if self.logger is not None: self.logger.error("Error occurs while calculating evolution for %(s)s" % dict(s=self.system_name))
                return -numpy.inf, self.initial_eccentricity
            else:
                evolution_complete = True
                calculated_eccentricity_now = evolutionary_history.eccentricity[-1]
                age_upto_which_evolution_is_calculated = evolutionary_history.age[-1]
                if self.logger is not None:
                    self.logger.debug("Evolution is calculated for %(s)s." % dict(s=self.system_name))
                    self.logger.debug("%(s)s : calculated e = %(a)f, e_env = %(b)f, stellar age = %(c)f, calculated stellar age = %(d)f" % dict(a=calculated_eccentricity_now,
                                                                                                                                                b=self.e_env,
                                                                                                                                                c=parameters_for_evolution['stellar age'],
                                                                                                                                                d=age_upto_which_evolution_is_calculated,
                                                                                                                                                s=self.system_name))
                if math.fabs(stellar_age - age_upto_which_evolution_is_calculated) <0.00001:
                    self.logger.debug("Evolution of %(s)s is calculated up to the stellar age." % dict(s=self.system_name))

                    if calculated_eccentricity_now < self.e_env:
                        if self.logger is not None:
                            self.logger.debug("calculated e = %(a)f, e_env = %(b)f, stellar age = %(c)f, calculated stellar age = %(d)f" % dict(a=calculated_eccentricity_now,
                                                                                                                                                b=self.e_env,
                                                                                                                                                c=stellar_age,
                                                                                                                                                d=age_upto_which_evolution_is_calculated))
                            self.logger.debug("calculated e is less than the envelope e and stellar age is almost same to the age upto which the evolution is calculated.")
                    else:
                        if self.logger is not None:
                            self.logger.debug("calculated e = %(a)f, e_env = %(b)f, stellar age = %(c)f, calculated stellar age = %(d)f" % dict(a=calculated_eccentricity_now,
                                                                                                                                                b=self.e_env,
                                                                                                                                                c=stellar_age,
                                                                                                                                                d=age_upto_which_evolution_is_calculated))
                        self.logger.debug("The calculated e is more than the envelope e.")
                        return -numpy.inf, self.initial_eccentricity
                else:
                    self.logger.debug("Evolution of %(s)s is NOT calculated up to the stellar age." % dict(s=self.system_name))
                    return -numpy.inf, self.initial_eccentricity

        if calculated_eccentricity_now >= 0 and calculated_eccentricity_now <= 1:
            if calculated_eccentricity_now>self.e_env: return -numpy.inf, self.initial_eccentricity
            if math.fabs(calculated_eccentricity_now)<0.0001:
                integral = self.eccentricity_distribution.pdf(0)
            else:
                integral = self.eccentricity_distribution.cdf(
                    calculated_eccentricity_now)/calculated_eccentricity_now
            probability = integral * priors
            if self.logger is not None: self.logger.debug("Probability of the calculated eccentricity is %(p)f " % dict(p = probability))
            if probability == 0:
                logging.debug("log of probability of e is %(x)f " % dict(x=-numpy.inf))
                if self.logger is not None: self.logger.debug("log of probability of e is %(x)f " % dict(x=-numpy.inf))
                return -numpy.inf, self.initial_eccentricity
            if probability < 0:
                logging.warning('Probability cannot be less than zero.')
                if self.logger is not None: self.logger.warning('Probability cannot be less than zero.')
                return -numpy.inf, self.initial_eccentricity
            log_likelihood = numpy.log(probability)
            logging.debug("log of probability of e is %(x)f " % dict(x=log_likelihood))
            if not self.logger is None: self.logger.debug("log of probability of e is %(x)f " % dict(x=log_likelihood))
            return log_likelihood, calculated_eccentricity_now
        logging.warning('Calculated present eccentricity can neither be less than zero nor greater than one')
        if not self.logger is None: self.logger.warning('Calculated present eccentricity can neither be less than zero nor greater than one')
        return -numpy.inf, self.initial_eccentricity


    def generate_successful_walkers_aux(self,
                                    u,
                                    number_of_discovered_walkers,
                                    p0_file_is_being_updated,
                                    walkers,
                                    nwalkers,
                                    ndim,
                                    p0_file_name,
                                    min_log_likelihood = - 0.0001,
                                    output_dirname = "/work/08529/mmmahmud/scratch/circularization_exoplanet_system/sampling_output"
                                    ):
        numpy.random.seed()
        #pid = os.getpid()
        #date_time = datetime.now().strftime('%Y%m%d%H%M%S')
        #logging_file_name = '%(outdir)s/%(system)s/P0_processor_logging/p0_%(now)s_%(pid)d.logging' % dict(outdir=output_dirname, system=self.system_name, now=date_time, pid=pid)
        #msg_file_name = '%(outdir)s/%(system)s/P0_processor_message/p0_%(now)s_%(pid)d.txt' % dict(outdir=output_dirname, system=self.system_name, now=date_time, pid=pid)
        pid = os.getpid()
        logging_folder_name = '%(outdir)s/%(system)s/P0_processor_logging/p0_%(pid)d/' % dict(outdir=output_dirname, system=self.system_name, pid=pid)
        msg_folder_name = '%(outdir)s/%(system)s/P0_processor_message/p0_%(pid)d/' % dict(outdir=output_dirname, system=self.system_name, pid=pid)
        ensure_directory(logging_folder_name)
        ensure_directory(msg_folder_name)
        def setup_logging_messaging_file(i):
            date_time = datetime.now().strftime('%Y%m%d%H%M%S')
            logging_file_name = '%(folder)sp0_%(now)s_%(pid)d_i_%(i)d.logging' % dict(folder=logging_folder_name, now=date_time, pid=pid, i = i)
            msg_file_name = '%(folder)sp0_%(now)s_%(pid)d_i_%(i)d.txt' % dict(folder=msg_folder_name, now=date_time, pid=pid, i = i)
            setup_basic_logging(logging_file_name, msg_file_name)


        #setup_basic_logging(logging_file_name, msg_file_name)
        if self.logger is not None:
            self.logger.debug("generate_successsful_walkers_aux method is called. ")
            self.logger.debug("u = %(u)s " % dict(u=numpy.array2string(u)))

        lgQpl_max = self.prior_transform_instance.max_argument_of_phase_lag_function_for_planet
        lgQpl_min = self.prior_transform_instance.min_argument_of_phase_lag_function_for_planet
        lgQst_max = self.prior_transform_instance.max_argument_of_phase_lag_function_for_star #
        lgQst_min = self.prior_transform_instance.min_argument_of_phase_lag_function_for_star #
        lgPbr_max = self.prior_transform_instance.max_log_tidal_break_period
        lgPbr_min = self.prior_transform_instance.min_log_tidal_break_period
        alpha_max = self.prior_transform_instance.max_power_law_argument
        alpha_min = self.prior_transform_instance.min_power_law_argument
        init_spin_max = self.prior_transform_instance.max_initial_stellar_spin
        init_spin_min = self.prior_transform_instance.min_initial_stellar_spin

        del_lgQpl_u = 0.25/(lgQpl_max - lgQpl_min)
        del_lgQst_u = 0.25/(lgQst_max - lgQst_min) #
        del_lgPbr_u = 0.10/(lgPbr_max - lgPbr_min)
        del_alpha_u = 0.5/(alpha_max - alpha_min)
        del_spin_u = 0.125/(init_spin_max - init_spin_min)

        n_lgQpl = int(1.0/del_lgQpl_u)
        n_lgQst = int(1.0/del_lgQst_u) #
        n_lgPbr = int(1.0/del_lgPbr_u)
        n_alpha = int(1.0/del_alpha_u)
        n_spin  = int(1.0/del_spin_u)

        v_start = numpy.random.randint(0, n_lgQst) #
        w_start = numpy.random.randint(0, n_alpha)
        x_start = numpy.random.randint(0, n_lgPbr)
        y_start = numpy.random.randint(0, n_spin)

        v_end = v_start - 1 #
        if v_end < 0: v_end = n_lgQst - 1 #
        w_end = w_start - 1
        if w_end < 0: w_end = n_alpha - 1
        x_end = x_start - 1
        if x_end < 0: x_end = n_lgPbr - 1
        y_end = y_start - 1
        if y_end < 0: y_end = n_spin - 1

        loop_v_complete = False #
        loop_w_complete = False
        loop_x_complete = False
        loop_y_complete = False
        walker_found = False
        i = 0
        v = v_start #
        lgQst_u = v * del_lgQst_u + numpy.random.rand() * del_lgQst_u #
        while not (loop_v_complete or walker_found):  #
            w = w_start
            alpha_u = w * del_alpha_u + numpy.random.rand() * del_alpha_u
            while not (loop_w_complete or walker_found):
                x = x_start
                lgPbr_u = (x + numpy.random.rand()) * del_lgPbr_u
                while not (loop_x_complete or walker_found):
                    y = y_start
                    spin_u = (y + numpy.random.rand()) * del_spin_u
                    while not (loop_y_complete or walker_found):
                        u[5] = spin_u % 1 if spin_u > 1 else spin_u
                        temp = numpy.random.rand() * del_lgQpl_u
                        u[6] = temp % 1 if temp > 1 else temp
                        u[7] = lgPbr_u % 1 if lgPbr_u > 1 else lgPbr_u
                        u[8] = alpha_u % 1 if alpha_u > 1 else alpha_u
                        i = i+1
                        setup_logging_messaging_file(i)
                        logging.debug("Modified u is %(u)s" % dict(u=numpy.array2string(u)))
                        if self.logger is not None: self.logger.debug("Modified u is %(u)s" % dict(u=numpy.array2string(u)))
                        log_likelihood, parameters_for_evolution = self(u)
                        logging.debug("log likelihood for u = %(u)s is %(n)f with parameters %(p)s" % dict(u=numpy.array2string(u), n=log_likelihood, p=numpy.array2string(parameters_for_evolution)))
                        if self.logger is not None: self.logger.debug("log likelihood for u = %(u)s is %(n)f with parameters %(p)s" % dict(u=numpy.array2string(u),
                                                                                                                                           n=log_likelihood,
                                                                                                                                           p=numpy.array2string(parameters_for_evolution)))
                        t=u[6]
                        if math.isinf(log_likelihood):
                            u[6] = numpy.random.rand()
                            i = i+1
                            setup_logging_messaging_file(i)
                            logging.debug("Loglikelihood is found to be -inf even for lowest lgQpl. Now we are trying with any value of lgQpl between min and max")
                            if self.logger is not None:
                                self.logger.debug("Loglikelihood is found to be -inf even for lowest lgQpl. Now we are trying with any value of lgQpl between min and max")
                            log_likelihood, parameters_for_evolution = self(u)
                            logging.debug("log likelihood for u = %(u)s is %(n)f with parameters %(p)s" % dict(u=numpy.array2string(u), n=log_likelihood, p=numpy.array2string(parameters_for_evolution)))
                            if self.logger is not None:
                                self.logger.debug("log likelihood for u = %(u)s is %(n)f with parameters %(p)s" % dict(u=numpy.array2string(u),
                                                                                                                       n=log_likelihood,
                                                                                                                       p=numpy.array2string(parameters_for_evolution)))
                            if not math.isinf(log_likelihood): t=u[6]
                        if (not math.isinf(log_likelihood)):
                            logging.debug("Start: log likelihood is not negative infinity for this u")
                            if self.logger is not None: self.logger.debug("Start: log likelihood is not negative infinity for this u")
                            a = u[6]
                            b = 1.0
                            i = i+1
                            setup_logging_messaging_file(i)
                            if self.logger is not None:
                                self.logger.debug("loglikelihood is going to be calculated for maximum lgQpl")
                            u[6] = b
                            log_likelihood, parameters_for_evolution = self(u)
                            logging.debug("loglikelihood for maximum lgQpl is %(x)f" % dict(x=log_likelihood))
                            if self.logger is not None:
                                self.logger.debug("loglikelihood for maximum lgQpl is %(x)f" % dict(x=log_likelihood))
                            if (not math.isinf(log_likelihood)):
                                i = i + 1
                                setup_logging_messaging_file(i)
                                logging.debug("loglikelihoods are finite for maximum and minimum lgQpl. We are now randomly picking up any value of lgQpl between them" )
                                if self.logger is not None:
                                    self.logger.debug("loglikelihoods are finite for maximum and minimum lgQpl. We are now randomly picking up any value of lgQpl between them" )
                                u[6] = numpy.random.uniform(a, b, 1)
                                log_likelihood, parameters_for_evolution = self(u)
                                if not math.isinf(log_likelihood):
                                    if self.logger is not None:
                                        self.logger.debug("log likelihood is finite: %(x)f" % dict(x=log_likelihood))
                            else:
                                c = (a+b)/2.0
                                c = c % 1 if c > 1 else c
                                u[6] = c
                                i = i+1
                                setup_logging_messaging_file(i)
                                log_likelihood, parameters_for_evolution = self(u)
                                logging.debug("u = %(u)s log likelihood = %(n)f for parameters: %(p)s" % dict(u=numpy.array2string(u), n=log_likelihood, p=numpy.array2string(parameters_for_evolution)))
                                if self.logger is not None: self.logger.debug("u = %(u)s log likelihood = %(n)f for parameters: %(p)s" % dict(u=numpy.array2string(u), n=log_likelihood,
                                                                                                                                              p=numpy.array2string(parameters_for_evolution)))
                                while math.fabs(b-c)>0.0001:
                                    while math.isinf(log_likelihood):
                                        b = c
                                        c = (a+b)/2.0
                                        c = c % 1 if c>1 else c
                                        u[6] = c
                                        logging.debug("c = %(s)f  " % dict(s = c))
                                        if self.logger is not None: self.logger.debug("c =  %(s)f " % dict(s = c))
                                        i = i+1
                                        setup_logging_messaging_file(i)
                                        log_likelihood, parameters_for_evolution = self(u)
                                        logging.debug("log likelihood = %(s)f  " % dict(s = log_likelihood))
                                        if self.logger is not None: self.logger.debug("log likelihood =  %(s)f " % dict(s = log_likelihood))
                                    a = c
                                    c = (a+b)/2.0
                                    c = c % 1 if c > 1 else c
                                    u[6] = c
                                    logging.debug("c = %(s)f  " % dict(s = c))
                                    if self.logger is not None: self.logger.debug("c =  %(s)f " % dict(s = c))
                                    i = i+1
                                    setup_logging_messaging_file(i)
                                    log_likelihood, parameters_for_evolution = self(u)
                                    logging.debug("log likelihood = %(s)f  " % dict(s = log_likelihood))
                                    if self.logger is not None: self.logger.debug("log likelihood =  %(s)f " % dict(s = log_likelihood))
                                if not math.isinf(log_likelihood):
                                    a = c
                                logging.debug("we got a range of u[6] for the acceptable walkers. The extreme u = %(u)s with log likelihood %(n)f" % dict(u=numpy.array2string(u), n=log_likelihood))
                                if self.logger is not None: self.logger.debug(logging.debug("we got a range of u[6] for the acceptable walkers. The extreme u = %(u)s with log likelihood %(n)f"
                                                                                           % dict(u=numpy.array2string(u), n=log_likelihood)))
                                logging.debug("corresponding extreme parameters are: %(p)s" % dict(p=numpy.array2string(parameters_for_evolution)))
                                if self.logger is not None: self.logger.debug("corresponding extreme parameters are: %(p)s" % dict(p=numpy.array2string(parameters_for_evolution)))
                                x = numpy.random.uniform(t, a, 1)
                                u[6] = x[0]
                                logging.debug("a value of u[6] picked up randomly from init_u[6] and the extremum u[6]. That is %(x)f" % dict(x= u[6]))
                                if self.logger is not None: self.logger.debug("a value of u[6] picked up randomly from init_u[6] and the extremum u[6]. That is %(x)f" % dict(x= u[6]))
                                i = i+1
                                setup_logging_messaging_file(i)
                                log_likelihood, parameters_for_evolution = self(u)
                                if not math.isinf(log_likelihood):
                                    logging.debug('The discovered walker is u  = %(u)s' % dict(u=numpy.array2string(u)))
                                    logging.debug('log p = %(logp)f' % dict(logp = log_likelihood))
                                    logging.debug('parameters for evolution = %(params)s' % dict(params=numpy.array2string(parameters_for_evolution)))
                                    if self.logger is not None:
                                        self.logger.debug('u  = %(u)s' % dict(u=numpy.array2string(u)))
                                        self.logger.debug('log p = %(logp)f' % dict(logp = log_likelihood))
                                        self.logger.debug('parameters for evolution = %(params)s' % dict(params=numpy.array2string(parameters_for_evolution)))
                                else:
                                    u[6]=t
                            p0_file_exists = os.path.exists(p0_file_name)
                            logging.debug('number of discovered walkers = %(x)f' % dict(x=number_of_discovered_walkers.value))
                            if self.logger is not None: self.logger.debug('number of discovered walkers = %(x)f' % dict(x=number_of_discovered_walkers.value))
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
                        if y > n_spin-1: y = 0
                        spin_u = (y + numpy.random.rand()) * del_spin_u
                        spin_u = spin_u % 1 if spin_u > 1 else spin_u
                    if x == x_end: loop_x_complete = True
                    x = x + 1
                    if x > n_lgPbr-1: x = 0
                    lgPbr_u = (x + numpy.random.rand()) *  del_lgPbr_u
                    lgPbr_u = lgPbr_u % 1 if lgPbr_u > 1 else lgPbr_u
                if w == w_end: loop_w_complete = True
                w = w + 1
                if w > n_alpha-1: w = 0
                alpha_u = (w + numpy.random.rand()) * del_alpha_u
                alpha_u = alpha_u % 1 if alpha_u > 1 else alpha_u
            if v == v_end: loop_v_complete = True #
            v = v + 1 #
            if v > n_lgQst-1: v = 0 #
            lgQst_u = (v + numpy.random.rand()) * del_lgQst_u #
            lgQst_u = lgQst_u % 1 if lgQst_u > 1 else lgQst_u #


        #*****************
        if number_of_discovered_walkers.value < nwalkers:
            numpy.random.seed()
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
                                ndim=10, #
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
             ndim=10, #
             reset_backend = False,
             iterations = 16):

        mcmc_progress_file_name = '%(dirname)s/%(system)s/%(system)s_mcmc_progress.h5' % dict(dirname=self.directory_name, system=self.system_name)
        p0_file_name = '%(dirname)s/%(system)s/%(system)s_p0_file.npy' % dict(dirname=self.directory_name, system=self.system_name)
        ensure_directory(mcmc_progress_file_name)
        ensure_directory(p0_file_name)

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
            logging_file_name='/work/08529/mmmahmud/scratch/circularization_exoplanet_system/sampling_output/%(system)s/mcmc_processor_logging/mcmc_%(now)s_%(pid)d.logging'%dict(system=self.system_name,
                                                                                                                                                                                  pid=pid,
                                                                                                                                                                                  now=date_time)
            msg_file_name='/work/08529/mmmahmud/scratch/circularization_exoplanet_system/sampling_output/%(system)s/mcmc_processor_message/mcmc_%(now)s_%(pid)d.txt' % dict(system=self.system_name,
                                                                                                                                                                            pid=pid,
                                                                                                                                                                            now=date_time)
            ensure_directory(logging_file_name)
            ensure_directory(msg_file_name)
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
                logging.debug("Backend file exists.")
                chain_exists = backend.initialized and (backend.iteration > 0)
                if self.logger is not None:
                    self.logger.debug('Backend iterations = %(x)f ' % dict(x= backend.iteration))
                    self.logger.debug('backend initialized = %(x)s' % dict(x= backend.initialized))
                logging.debug('Backend iterations = %(x)f ' % dict(x= backend.iteration))
                logging.debug('backend initialized = %(x)s' % dict(x= backend.initialized))
                if reset_backend == False and chain_exists:
                    if self.logger is not None:
                        self.logger.debug('Backend file is not subject to reset. The chain size is not zero.')
                        self.logger.debug('Next samples will be drawn from the end of the previously worked out chain.')
                    logging.debug('Backend file is not subject to reset. The chain size is not zero. Next samples will be drawn from the end of the previously worked out chain.')
                    sampler.run_mcmc(None, iterations, progress = True)
                else:
                    if self.logger is not None:
                        self.logger.debug('Either the backend file is subject to reset or previously calculated chain size is zero.')
                        self.logger.debug('Samples will be drawn for the first time.')
                    logging.debug('Either the backend file is subject to reset or previously calculated chain size is zero.')
                    logging.debug('Samples will be drawn for the first time.')
                    backend.reset(nwalkers, ndim)
                    sampler.run_mcmc(p0, iterations, progress = True)
            else:
                logging.debug('Backend file did not exist previously')
                if self.logger is not None: self.logger.debug('Backend file did not exist previously')
                sampler.run_mcmc(p0, iterations, progress = True)


            blobs = sampler.get_blobs(flat=True)

            figure = backend.corner(blobs, labels=['M*', #Mass of the parent star
                                                  'age', #Age
                                                  'Rp', #Planetary radius
                                                  'Fe/H_*', #Stellar Metallicity
                                                  'Mp', #Planetary mass
                                                  'initSpin*', #Initial stellar spin
                                                  'lgQpl', #Log of the tidal quality factor for planet, i.e. argument of phase lag function
                                                  'tidal break point',
                                                  'alpha',
                                                  'lgQst', #
                                                  'e_now', #Present eccentricity
                                                  'log(f(e_now))'], #log likelihood of present eccentricity
                                   quantiles=[0.16, 0.5, 0.84],
                                   show_titles=True, title_kwargs={"fontsize": 12})
            plt.show()
            figfilename = "%(dirname)s/%(system)s/%(system)s_MCMC.pdf" % dict(dirname=self.directory_name, system=self.system_name)
            ensure_directory(figfilename)
            figure.savefig(figfilename, bbox_inches='tight')
        return

    def __call__(self, u):
        for i in range(0, 10): #
            if u[i] > 1 or u[i] < 0:
                logging.debug("ui cannot be greater than 1 and less than zero.")
                return -numpy.inf, numpy.array([None, None, None, None, None, None, None, None, None, None, None, None]) #
        parameters_for_evolution = self.prior_transform_instance(u)
        if parameters_for_evolution is None:
            logging.debug("parameters for evolution is none.")
            return -numpy.inf, numpy.array([None, None, None, None, None, None, None, None, None, None, None, None])
        params = numpy.array([parameters_for_evolution['primary mass'],
                           parameters_for_evolution['stellar age'],
                           parameters_for_evolution['secondary radius'],
                           parameters_for_evolution['stellar metallicity'],
                           parameters_for_evolution['secondary mass'],
                           parameters_for_evolution['initial stellar spin'],
                           parameters_for_evolution['argument of phase lag function for planet'],
                           parameters_for_evolution['tidal break period'],
                           parameters_for_evolution['power law argument'], #
                           parameters_for_evolution['argument of phase lag function for star']]) #
        log_prob_parameters_for_evolution, calculated_eccentricity_now = self.log_prob(parameters_for_evolution)
        if numpy.isinf(-log_prob_parameters_for_evolution):
            logging.debug("Actual params are: %(x)s " % dict(x=json.dumps(parameters_for_evolution)))
            return -numpy.inf, numpy.array([None, None, None, None, None, None, None, None, None, None, None, None]) #
        params = numpy.append(params, [calculated_eccentricity_now, log_prob_parameters_for_evolution])
        logging.debug("The params are %(p)s" % dict(p=repr(params)))
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
        #print('serialized directory ', self.serialized_directory)
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
                 dirname = "/work/08529/mmmahmud",
                 envelope_eccentricity_function=EnvelopeEccentricityDistribution.envelope_eccentricity_function,
                 initial_eccentricity=0.8,
                 max_argument_of_phase_lag_function_for_planet=12,
                 min_argument_of_phase_lag_function_for_planet=3,
                 max_argument_of_phase_lag_function_for_star=12, #
                 min_argument_of_phase_lag_function_for_star=5, #
                 min_log_of_tidal_break_period=math.log(0.5, 10),
                 max_log_of_tidal_break_period=1,
                 min_power_law_argument=-5,
                 max_power_law_argument=5,
                 max_initial_stellar_spin=15,
                 min_initial_stellar_spin=14.5,
                 constraints=Constraints_for_selecting_systems.constraints(),
                 spin_frequency_breaks_for_planet=None,
                 spin_frequency_powers_for_planet=numpy.array([0.0]),
                 spin_frequency_breaks_for_star=None, #
                 spin_frequency_powers_for_star=numpy.array([0.0]), #
                 output_dirname="/work/08529/mmmahmud/scratch/circularization_exoplanet_system/sampling_output",
                 logging_level=logging.DEBUG):


        logging_fname = "%(dirname)s/%(system)s/logger_for_%(system)s.log" % dict(dirname=output_dirname, system=system_name)
        logger_name = "%(dirname)s/%(system)s/logger_for_%(system)s" % dict(dirname=output_dirname, system=system_name)
        logger = setup_logger(logger_name, logging_fname, level=logging_level)

        logger.info("The logger file for the system %(system_name)s is created." % dict(system_name = system_name))
        logger.info("The initial eccentricity = %(e)f " % dict(e= initial_eccentricity))
        self.initial_eccentricity = initial_eccentricity
        logger.info("Maximum argument of phase lag function for planet = %(Q)f" % dict(Q=max_argument_of_phase_lag_function_for_planet))
        self.max_argument_of_phase_lag_function_for_planet = max_argument_of_phase_lag_function_for_planet
        logger.info("Minimum argument of phase lag function for planet = %(Q)f" % dict(Q=min_argument_of_phase_lag_function_for_planet))
        self.min_argument_of_phase_lag_function_for_planet = min_argument_of_phase_lag_function_for_planet
        logger.info("Maximum argument of phase lag function for star = %(Q)f" % dict(Q=max_argument_of_phase_lag_function_for_star)) #
        self.max_argument_of_phase_lag_function_for_star = max_argument_of_phase_lag_function_for_star #
        logger.info("Minimum argument of phase lag function for star = %(Q)f" % dict(Q=min_argument_of_phase_lag_function_for_star)) #
        self.min_argument_of_phase_lag_function_for_star = min_argument_of_phase_lag_function_for_star #
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
        self.spin_frequency_breaks_for_star = spin_frequency_breaks_for_star #
        self.spin_frequency_powers_for_star = spin_frequency_powers_for_star #

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

            if 'secondary radius' in self.means:
                rad = self.means['secondary radius']
            elif ('ratio of planet to stellar radius' in self.means) and ('primary radius' in self.means):
                rad = self.means['ratio of planet to stellar radius'] * self.means['primary radius'] * const.R_sun.value / const.R_earth.value
            elif ('transit depth' in self.means) and ('primary radius' in self.means):
                rad = ((self.means['transit depth']/100)**0.5) *  self.means['primary radius'] * const.R_sun.value / const.R_earth.value
            else:
                rad = None

            rad_j = 0
            if rad is not None:
                rad_j = rad * const.R_earth.value / const.R_jup.value
                self.e_env = self.envelope_eccentricity_function(x=self.means['semi major axis'] / rad, logger=logger)
            else:
                self.e_env = 0.5
            logger.info("The envelope eccentricity for the system %(system)s is %(eenv)f" % dict(system=system_name, eenv=self.e_env))
            if rad is not None and rad_j>0.6 and self.e_env > self.means['present eccentricity']:
                logger.debug("EccentricityDistribution instance is going to be created.")
                eccentricity_distribution_object = EccentricityDistribution.EccentricityDistribution(self.means['present eccentricity'],
                                                                            self.standard_deviations[
                                                                                'eccentricity_now_upper_uncertainty'],
                                                                            self.standard_deviations[
                                                                                'eccentricity_now_lower_uncertainty'],
                                                                            eccentricity_flag_limit = self.means['eccentricity now limit flag'],
                                                                            system_name = system_name, logger = logger)

                logger.debug("EccentricityDistribution instance is created. Now probability density of eccentricity vs. eccentricity graph will be plotted.")

                eccentricity_distribution_object.plot_probability_density_of_eccentricity_vs_eccentricity_graph()
                logger.debug("The graph is plotted")
                self.eccentricity_distribution = eccentricity_distribution_object.eccentricity_distribution
                logger.info("PriorTransform instance is going to be created for the %(system)s" % dict(system=system_name))
                self.prior_transform_instance = PriorTransform(means,
                                                               standard_deviations,
                                                               system_name,
                                                               dirname,
                                                               max_argument_of_phase_lag_function_for_planet,
                                                               min_argument_of_phase_lag_function_for_planet,
                                                               max_argument_of_phase_lag_function_for_star, #
                                                               min_argument_of_phase_lag_function_for_star, #
                                                               min_log_of_tidal_break_period,
                                                               max_log_of_tidal_break_period,
                                                               min_power_law_argument, max_power_law_argument,
                                                               max_initial_stellar_spin, min_initial_stellar_spin,
                                                               logger)

                logger.debug("LogLikelihood instance is going to be created")
                self.log_likelihood_instance = LogLikelihood(prior_transform_instance = self.prior_transform_instance,
                                                             orbital_period = self.means['orbital period'],
                                                             obliquity = 0,  # obliquity
                                                             eccentricity_distribution = self.eccentricity_distribution,
                                                             e_env = self.e_env,
                                                             system_name = system_name,
                                                             directory_name = '/work/08529/mmmahmud/p0andmcmc',
                                                             initial_eccentricity = initial_eccentricity,
                                                             constraints = constraints,
                                                             spin_frequency_breaks_for_planet = spin_frequency_breaks_for_planet,
                                                             spin_frequency_powers_for_planet = spin_frequency_powers_for_planet,
                                                             spin_frequency_breaks_for_star = spin_frequency_breaks_for_star, #
                                                             spin_frequency_powers_for_star = spin_frequency_powers_for_star, #
                                                             logger=logger, number_of_parallel_processes = 16
                                                             )

                logger.debug("LogeLikelihod instance is created. Now MCMC is going to run")
                self.log_likelihood_instance.MCMC()
                logger.debug("MCMC is done")
            else:
                if rad_j < 0.6:
                    logger.debug("The planet's radius is less than 0.6 R_j")
                else:
                    logger.debug("Envelope eccentricity is lower than the measured present eccentricity for this system.")

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
    parser.add_argument('--system',
                        help = 'stores the name of the star-exoplanet system')
    args = parser.parse_args()

    if args.path_of_the_stellar_evolution_interpolators_directory:
        InitializationOfSamplingPropertiesOfSystem.set_serialized_directory(args.path_of_the_stellar_evolution_interpolators_directory)
    if args.path_of_the_eccentricity_expansion_coefficients_file:
        InitializationOfSamplingPropertiesOfSystem.set_eccentricity_expansion_fname(args.path_of_the_eccentricity_expansion_coefficients_file)

    InitializationOfSamplingPropertiesOfSystem()
    if args.measured_values and args.standard_deviations and args.system:
        test3 = SamplingPropertiesOfSystem(args.measured_values,
                                           args.standard_deviations,
                                           system_name=args.system,
                                           envelope_eccentricity_function=EnvelopeEccentricityDistribution.envelope_eccentricity_function
                                           )


