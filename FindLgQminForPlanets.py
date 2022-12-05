import sys
import traceback
import os
import logging
from datetime import datetime
import math
import json
import h5py
import pickle
from types import SimpleNamespace
from functools import partial

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

if not sys.warnoptions:
    import warnings

    from sqlalchemy.exc import SAWarning

    warnings.filterwarnings('ignore',
                            r"^Dialect sqlite\+pysqlite does \*not\* support Decimal objects natively\, "
                            "and SQLAlchemy must convert from floating point - rounding errors and other "
                            "issues may occur\. Please consider storing Decimal numbers as strings or "
                            "integers on this platform for lossless storage\.$",
                            SAWarning, r'^sqlalchemy\.sql\.type_api$')


from bayesian.stellar_param_sampling.feh_conditional_likelihood_base import \
    FeHConditionalLikelihoodBase



def getStellarEvolutionInterpolatorsDirectory():
    return '/home1/08529/mmmahmud/poet/stellar_evolution_interpolators'

def getEccentricityExpansionCoefficientsFile():
    return b"/work/08529/mmmahmud/ls6/eccentricity_expansion_coef_O400.sqlite"

def ensure_directory(fname):
        """Make sure the directory containing the given name exists."""
        dirname = os.path.dirname(fname)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

class Initialization:
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
        manager = StellarEvolutionManager(self.serialized_directory)
        interpolator = manager.get_interpolator_by_name('default')
        FeHConditionalLikelihoodBase.set_interpolator(interpolator)
        orbital_evolution_library.prepare_eccentricity_expansion(
            self.eccentricity_expansion_fname,
            1e-4,
            True,
            True
        )

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
                 e_env,
                 system='Star-exoplanet',
                 obliquity=0,
                 initial_eccentricity=0.8,
                 min_log_tidal_break_period=math.log(0.5,10),
                 max_log_tidal_break_period=1,
                 max_initial_stellar_spin=15,
                 min_initial_stellar_spin=5,
                 spin_frequency_breaks_for_planet=None,
                 spin_frequency_powers_for_planet=numpy.array([0.0]),
                 logger=None
                 ):
        self.means = means
        self.standard_deviations = standard_deviations
        self.system = system
        self.obliquity = obliquity
        self.initial_eccentricity = initial_eccentricity
        self.e_env=e_env
        self.min_log_tidal_break_period = min_log_tidal_break_period
        self.max_log_tidal_break_period = max_log_tidal_break_period
        self.max_initial_stellar_spin = max_initial_stellar_spin
        self.min_initial_stellar_spin = min_initial_stellar_spin
        self.spin_frequency_breaks_for_planet=spin_frequency_breaks_for_planet
        self.spin_frequency_powers_for_planet=spin_frequency_powers_for_planet
        self.logger = logger
        self.logging_fname = None
        if self.logger is not None:
            handler = logger.handlers[0]
            self.logging_fname = handler.baseFilename
        if self.logger is not None:
            self.logger.debug("The name of the log file for the system %(s)s is %(f)s " % dict(s=system, f=self.logging_fname))
        if self.logger is not None: self.logger.debug("We are forming prior transform instance for %(s)s" % dict(s=system))
        if self.logging_fname is not None: logging.basicConfig(filename = self.logging_fname, level=logging.DEBUG, force = True, format='%(asctime)s %(message)s')
        logging.debug("Basic Config for the log file is done for the system %(s)s" % dict(s=system))
        debug_plot = [('interpolation_performance', 'interp_performance.pdf')]
        teff = None
        feh = None
        logg = None
        mean_density = None
        if 'stellar effective temperature' in self.means:
            if not (math.isnan(self.means['stellar effective temperature']) or math.isnan(self.standard_deviations['stellar_effective_temperature_upper_uncertainty']) or math.isnan(self.standard_deviations['stellar_effective_temperature_lower_uncertainty'])):
                teff = split_normal.freeze_error_bar(
                    mode=self.means['stellar effective temperature'],
                    abs_plus_error=self.standard_deviations['stellar_effective_temperature_upper_uncertainty'],
                    abs_minus_error=-self.standard_deviations['stellar_effective_temperature_lower_uncertainty'])
        if 'stellar metallicity' in self.means:
            if not (math.isnan(self.means['stellar metallicity']) or math.isnan(self.standard_deviations['stellar_metallicity_upper_uncertainty']) or math.isnan(self.standard_deviations['stellar_metallicity_lower_uncertainty'])):
                feh = split_normal.freeze_error_bar(
                    mode=self.means['stellar metallicity'],
                    abs_plus_error=self.standard_deviations['stellar_metallicity_upper_uncertainty'],
                    abs_minus_error=-self.standard_deviations['stellar_metallicity_lower_uncertainty'])
        if 'stellar log g' in self.means:
            if not (math.isnan(self.means['stellar log g']) or math.isnan(self.standard_deviations['stellar_log_g_upper_uncertainty']) or math.isnan(self.standard_deviations['stellar_log_g_lower_uncertainty'])):
                logg = split_normal.freeze_error_bar(
                    mode=self.means['stellar log g'],
                    abs_plus_error=self.standard_deviations['stellar_log_g_upper_uncertainty'],
                    abs_minus_error=-self.standard_deviations['stellar_log_g_lower_uncertainty'])
        if 'stellar density' in self.means:
            if not (math.isnan(self.means['stellar density']) or math.isnan(self.standard_deviations['stellar_density_upper_uncertainty']) or math.isnan(self.standard_deviations['stellar_density_lower_uncertainty'])):
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
            likelihood = POETInterpLikelihood(**constraints, rtol=config.time_ode_rtol, atol=config.time_ode_atol, max_step=config.time_ode_max_step)
        except:
            if self.logger is not None: self.logger.debug("An exception occurs while creating POETInterpLikelihood instance for %(s)s" % dict(s=system))
        else:
            if self.logger is not None: self.logger.debug("POETInterpLikelihood instance is successfully created for %(s)s" % dict(s=system))

        if (self.logger is not None) and (likelihood is not None):
            self.logger.debug("POETInterpLikelihood instance %(system)s is created. StarSampler instance is going to be created." % dict(system=self.system))
        try:
            if likelihood is not None:
                self.star_sampler = StarSampler(likelihood, config)
            else:
                self.star_sampler = None
        except:
            if self.logger is not None: self.logger.debug("An exception occurs while creating star sampler for %(s)s" % dict(s=self.system))
        else:
            if self.logger is not None: self.logger.debug("star sampler is successfully created for %(s)s" % dict(s=self.system))
        if (self.logger is not None) and (self.star_sampler is not None):
            self.logger.debug("StarSampler instance for %(system)s is created." % dict(system=self.system))
            u = numpy.random.rand(7)
            unit_cube = numpy.array([u[0], u[1], u[2]])
            try:
                stellar_metallicity, primary_mass, stellar_age = self.star_sampler.__call__(unit_cube)
            except ValueError as verror:
                if self.logger is not None:
                    self.logger.warning(traceback.format_exc())
                    self.logger.error("Value Error occurs while sampling metallicity, primary mass and age for  %(s)s" % dict(s=self.system))
                    self.logger.error('error occurs for %(s)s: %(error)s' % dict(s=self.system, error= str(verror)))
            except RuntimeError as rerror:
                if self.logger is not None:
                    self.logger.warning(traceback.format_exc())
                    self.logger.error("Runtime Error occurs while sampling metallicity, primary mass and age for  %(s)s" % dict(s=self.system))
                    self.logger.error('Runtime error occurs for %(s)s: %(error)s' % dict(s=self.system, error= str(rerror)))
            except Exception as expt:
                if self.logger is not None:
                    self.logger.warning(traceback.format_exc())
                    self.logger.error("General error occurs")
            else:
                if self.logger is not None: self.logger.debug("stellar metallicity = %(x)f, primary mass = %(y)f, stellar age = %(z)f" % dict(x=stellar_metallicity, y=primary_mass, z=stellar_age))


    def evolution_is_calculated_up_to_stellar_age(self, lgQmin):
        if self.logging_fname is not None: logging.basicConfig(filename = self.logging_fname, level=logging.DEBUG, force = True, format='%(asctime)s %(message)s')
        if self.logger is not None: self.logger.debug("evolution_is_calculated_up_to_stellar_age method is called for %(s)s" % dict(s=self.system))

        u = numpy.random.rand(7)
        parameters_for_evolution = self.__call__(u)

        parameters_for_evolution['argument of phase lag function for planet'] = lgQmin
        parameters_for_evolution['power law argument'] = 0
        stellar_age = parameters_for_evolution['stellar age']
        if self.logger is not None: self.logger.debug("parameters for evolution: %(x)s" % dict(x="Khalum"))
        star_exoplanet_binary_system = SimpleNamespace(primary_mass=parameters_for_evolution['primary mass'] * un.solMass,
                                              secondary_mass=parameters_for_evolution['secondary mass'] * un.earthMass,
                                              secondary_radius=parameters_for_evolution['secondary radius'] * un.earthRad,
                                              feh=parameters_for_evolution['stellar metallicity'] * un.dimensionless_unscaled,
                                              orbital_period=self.means['orbital period'] * un.d,
                                              obliquity=self.obliquity * un.deg,
                                              age=parameters_for_evolution['stellar age'] * un.Gyr)

        break_frequency = 2 * math.pi / parameters_for_evolution['tidal break period']
        reference_argument_of_phase_lag_function_for_planet = parameters_for_evolution['argument of phase lag function for planet']

        tidal_frequency_breaks_for_planet = numpy.array([2 * math.pi / 20, break_frequency])
        tidal_frequency_powers_for_planet = numpy.array([0.0, 0.0, 0.0])
        reference_phase_lag=phase_lag(reference_argument_of_phase_lag_function_for_planet)

        dissipation = dict(
            primary=None,
            secondary=dict(
                tidal_frequency_breaks=tidal_frequency_breaks_for_planet,
                spin_frequency_breaks=self.spin_frequency_breaks_for_planet,
                tidal_frequency_powers=tidal_frequency_powers_for_planet,
                spin_frequency_powers=self.spin_frequency_powers_for_planet,
                reference_phase_lag=reference_phase_lag
                )
	        )
        dissipation_string = dict(
            primary = None,
            secondary = dict(
                tidal_frequency_breaks=numpy.array2string(tidal_frequency_breaks_for_planet) if tidal_frequency_breaks_for_planet is not None else None,
                spin_frequency_breaks=numpy.array2string(self.spin_frequency_breaks_for_planet) if self.spin_frequency_breaks_for_planet is not None else None,
                tidal_frequency_powers=numpy.array2string(tidal_frequency_powers_for_planet) if tidal_frequency_powers_for_planet is not None else None,
                spin_frequency_powers=numpy.array2string(self.spin_frequency_powers_for_planet) if self.spin_frequency_powers_for_planet is not None else None,
                reference_phase_lag=str(reference_phase_lag)
                )
                )
        if self.logger is not None: self.logger.debug("dissipation again: %(x)s" % dict(x=str(dissipation_string)))
        if self.logger is not None: self.logger.debug("Evolution of %(system)s is being worked out" % dict(system = self.system))
        evolution_complete = False
        e_in = self.initial_eccentricity
        e_in_of_previous_iteration = e_in + 0.01

        if self.logger is not None: self.logger.debug("VVVVVVVVVVVVVVVVV  the initial eccentricity is %(e)f for %(system)s" % dict(e=e_in, system=self.system))

        while (not evolution_complete):
            try:
                if self.logger is not None:
                    self.logger.debug("find_evolution method for %(system)s is being called." % dict(system=self.system))
                    self.logger.debug("dissipation: %(dissipation)s" % dict(dissipation = json.dumps(dissipation_string)))
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
                                                      primary_core_envelope_coupling_timescale=0.05 * un.Gyr,
                                                      secondary_wind_strength=0.0,
                                                      secondary_wind_saturation=100.0,
                                                      secondary_core_envelope_coupling_timescale=0.05 * un.Gyr,
                                                      orbital_period_tolerance=1e-6,
                                                      solve=True,
                                                      secondary_is_star=False)
            except RuntimeError as runerror:
                logging.error('Runtime error occurs for %(s)s: %(error)s' % dict(s=self.system, error= str(runerror)))
                if self.logger is not None: self.logger.warning(traceback.format_exc())
                e_in_of_previous_iteration = e_in
                e_in = e_in - 0.01
                if (e_in < self.e_env + 0.2):
                    if self.logger is not None: self.logger.debug("e_in becomes less than e_env + 0.2, so loglikelihood is -inf")
                    return 0

                logging.warning('Calculating evolution for %(s)s with initial eccentricity %(ep)f failed, now trying initial eccentricity = %(e)f'
                                 % dict(e=e_in, s=self.system, ep=e_in_of_previous_iteration ))
                if self.logger is not None: self.logger.warning('Calculating evolution failed, trying initial eccentricity = %(e)f' % dict(e=e_in))
                evolution_complete = False

            except ValueError as verror:
                if self.logger is not None: self.logger.warning(traceback.format_exc())
                logging.error('Invalid parameter values encountered for %(s)s: %(error)s' % dict(s=self.system, error= str(verror)))
                if self.logger is not None: self.logger.error('Invalid parameter values encountered for %(s)s: %(error)s' % dict(error= str(verror), s=self.system))
                return 0
            except Exception as exc:
                if self.logger is not None: self.logger.warning(traceback.format_exc())
                logging.error("General Error occurs while calculating evolution for %(s)s" % dict(s=self.system))
                if self.logger is not None: self.logger.error("Error occurs while calculating evolution for %(s)s" % dict(s=self.system))
                return 0
            else:
                evolution_complete = True
                calculated_eccentricity_now = evolutionary_history.eccentricity[-1]
                age_upto_which_evolution_is_calculated = evolutionary_history.age[-1]
                if self.logger is not None:
                    self.logger.debug("Evolution is calculated for %(s)s." % dict(s=self.system))
                    self.logger.debug("%(s)s : calculated e = %(a)f, e_env = %(b)f, stellar age = %(c)f, calculated stellar age = %(d)f" % dict(a=calculated_eccentricity_now,
                                                                                                                                                b=self.e_env,
                                                                                                                                                c=parameters_for_evolution['stellar age'],
                                                                                                                                                d=age_upto_which_evolution_is_calculated,
                                                                                                                                                s=self.system))
                if math.fabs(stellar_age - age_upto_which_evolution_is_calculated) <0.00001:
                     self.logger.debug("Evolution of %(s)s is calculated up to the stellar age." % dict(s=self.system))
                     return 1
                else:
                     self.logger.debug("Evolution of %(s)s is NOT calculated up to the stellar age." % dict(s=self.system))
                     return 0


    def __call__(self, u):
        unit_cube = numpy.array([u[0], u[1], u[2]])
        if self.logger is not None: self.logger.debug("Checkpoint 1")
        stellar_metallicity, primary_mass, stellar_age = self.star_sampler.__call__(unit_cube)
        if self.logger is not None: self.logger.debug("Checkpoint 2: Fe/H: %(x)f, M*: %(y)f, age: %(z)f" % dict(x=stellar_metallicity, y=primary_mass, z=stellar_age))
        primary_rad = FeHConditionalLikelihoodBase.interpolator('RADIUS', primary_mass, stellar_metallicity)
        primary_radius = primary_rad(stellar_age)
        secondary_radius = None
        if 'transit depth' in self.means:
            if self.standard_deviations['transit_depth_upper_uncertainty'] is not None and self.standard_deviations['transit_depth_lower_uncertainty'] is None:
                square_of_the_ratio_of_planet_to_stellar_radius = norm.ppf(u[3], loc=self.means['transit depth'],
                                                                           scale=(self.standard_deviations['transit_depth_upper_uncertainty']
                                                                                 - self.standard_deviations['transit_depth_lower_uncertainty'])/2)/100.0
                secondary_radius = (square_of_the_ratio_of_planet_to_stellar_radius ** 0.5) * primary_radius * const.R_sun.value / const.R_earth.value
        if 'ratio of planet to stellar radius' in self.means:
            if self.standard_deviations['ratio_of_planet_to_stellar_radius_upper_uncertainty'] is not None and self.standard_deviations['ratio_of_planet_to_stellar_radius_lower_uncertainty'] is not None:
                ratio_of_planet_to_stellar_radius = norm.ppf(u[3], loc=self.means['ratio of planet to stellar radius'],
                                                             scale=(self.standard_deviations['ratio_of_planet_to_stellar_radius_upper_uncertainty']
                                                                   - self.standard_deviations['ratio_of_planet_to_stellar_radius_lower_uncertainty'])/2)
                secondary_radius = ratio_of_planet_to_stellar_radius * primary_radius * const.R_sun.value / const.R_earth.value
        if 'secondary radius' in self.means:
            if self.standard_deviations['secondary_radius_upper_uncertainty'] is not None and self.standard_deviations['secondary_radius_lower_uncertainty'] is not None and secondary_radius is None:
                secondary_radius = norm.ppf(u[3], loc=self.means['secondary radius'],
                                            scale=(self.standard_deviations['secondary_radius_upper_uncertainty']
                                                  - self.standard_deviations['secondary_radius_lower_uncertainty'])/2)
        secondary_mass = norm.ppf(u[4], loc=self.means['secondary mass'], scale=(self.standard_deviations[
                                                                                     'secondary_mass_upper_uncertainty'] -
                                                                                 self.standard_deviations[
                                                                                     'secondary_mass_lower_uncertainty']) / 2)
        initial_stellar_spin = self.min_initial_stellar_spin + u[5] * (
                    self.max_initial_stellar_spin - self.min_initial_stellar_spin)
        tidal_break_period = 10**(self.min_log_tidal_break_period + u[6] * (
                    self.max_log_tidal_break_period - self.min_log_tidal_break_period))

        parameters_for_evolution = {'primary mass': primary_mass,
                                    'stellar age': stellar_age,
                                    'secondary radius': secondary_radius,
                                    'stellar metallicity': stellar_metallicity,
                                    'secondary mass': secondary_mass,
                                    'initial stellar spin': initial_stellar_spin,
                                    'tidal break period': tidal_break_period
                                    }
        return parameters_for_evolution

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

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


class FindLgQmin:
    def __init__(self, array_of_prior_transform_instances, starting=3, ending=7, n=10, tol=0.8, directory="/work/08529/mmmahmud/directory_n_successes_and_iterations", reset=False):
        self.array_of_prior_transform_instances = array_of_prior_transform_instances
        self.starting = starting
        self.ending = ending
        self.n = n
        self.tol = tol
        self.directory = directory
        self.reset = reset
        self.lgQmin = self.determine_lgQmin()
    def number_of_successes_of_calculating_evolution(self, prior_transform_instance, lgQmin):
        success = 0
        iteration = 0
        system_name = prior_transform_instance.system
        success_file_name = "%(d)s/%(lgQmin)s/n_success_of_%(s)s.pkl" % dict(d=self.directory, lgQmin=lgQmin, s=system_name)
        iteration_file_name = "%(d)s/%(lgQmin)s/i_of_%(s)s.pkl" % dict(d=self.directory, lgQmin=lgQmin, s=system_name)
        ensure_directory(success_file_name)
        ensure_directory(iteration_file_name)
        if not self.reset:
            if prior_transform_instance.logger is not None:
                prior_transform_instance.logger.debug("We are not resetting previous calculations. Let's see whether any iteration and success number previously calculated can be retrieved.")
            success_file_exists = os.path.exists(success_file_name)
            if prior_transform_instance.logger is not None and success_file_exists:
                prior_transform_instance.logger.debug("File storing the number of successful calculations of evolution previously calculated exists")
            iteration_file_exists = os.path.exists(iteration_file_name)
            if prior_transform_instance.logger is not None and iteration_file_exists:
                prior_transform_instance.logger.debug("File storing the number of iterations previously done, exists")
            if success_file_exists and iteration_file_exists:
                success = pickle.load(open(success_file_name,"rb"))
                iteration = pickle.load(open(iteration_file_name, "rb"))
                if prior_transform_instance.logger is not None:
                    prior_transform_instance.logger.debug("Number of successful calculations of evolution, previously calculated is %(x)d" % dict(x=success))
                    prior_transform_instance.logger.debug("Number of iterations, previously done is %(x)d" % dict(x=iteration))
            else:
                if prior_transform_instance.logger is not None:
                    prior_transform_instance.logger.debug("No calculations were done previously. New files are created to store default values of success and iterations")
                save_object(success, success_file_name)
                save_object(iteration, iteration_file_name)
                success_file_exists = os.path.exists(success_file_name)
                iteration_file_exists = os.path.exists(iteration_file_name)
                if (prior_transform_instance.logger is not None) and success_file_exists and iteration_file_exists:
                    success_temp = pickle.load(open(success_file_name,"rb"))
                    iteration_temp = pickle.load(open(iteration_file_name, "rb"))
                    prior_transform_instance.logger.debug("default value of success = %(x)d" % dict(x=success_temp))
                    prior_transform_instance.logger.debug("default value of iteration = %(x)d" % dict(x=iteration_temp))

        f = False
        if prior_transform_instance.logger is not None: f = True
        if f: prior_transform_instance.logger.debug("We are now calculating the number of successes of calculating evolution of the system %(s)s for lgQmin %(q)f"
                                                    % dict(s=prior_transform_instance.system, q=lgQmin))
        if (not self.reset) and (iteration > self.n) and (iteration !=0):
            r_success = int(success/iteration * self.n)
            if prior_transform_instance.logger is not None:
                prior_transform_instance.logger.debug("Number of previously done iterations is greater than the expectation. Expected n is %(x)d, where previously done %(y)d"
                                                      % dict(x=self.n,y=iteration))
                prior_transform_instance.logger.debug("success rate on the basis of previous results = %(x)f" % dict(x=r_success))
            return r_success
        if not self.reset:
            i = iteration
        else:
            i = 0
        if prior_transform_instance.logger is not None:
            prior_transform_instance.logger.debug("iteration = %(i)d" % dict(i=i))
            prior_transform_instance.logger.debug("self.n = %(n)d" % dict(n=self.n))
        while i < self.n:
            prev_success = success
            success = success + prior_transform_instance.evolution_is_calculated_up_to_stellar_age(lgQmin)
            if success > prev_success:
                save_object(success, success_file_name)
                if prior_transform_instance.logger is not None:
                    prior_transform_instance.logger.debug("New successful calculation of evolution  is found and the success_file is updated. Previous number of success = %(x)d" % dict(x=prev_success))
                    temp = pickle.load(open(success_file_name,"rb"))
                    prior_transform_instance.logger.debug("Number of successes retrieved from the updated success_file is %(x)d" % dict(x=temp))
            i = i+1
            if prior_transform_instance.logger is not None:
                    prior_transform_instance.logger.debug("Number of iteration is going to be updated in the iteration_file")
            save_object(i, iteration_file_name)
            temp = pickle.load(open(iteration_file_name, "rb"))
            if prior_transform_instance.logger is not None:
                    prior_transform_instance.logger.debug("Recently updated number of iteration retrieved from the iteration_file is %(x)d" % dict(x=temp))

        if f: prior_transform_instance.logger.debug("number of successful evolutions for %(s)s with lgQmin= %(q)f is = %(d)d" % dict(s=prior_transform_instance.system, q=lgQmin, d=success))
        return success

    def rate_of_successful_evolution_calculations_for_all_systems(self, lgQmin):

        success_rate_file_name = "%(d)s/%(lgQmin)s/r_success_%(lgQmin)s.pkl" % dict(d=self.directory, lgQmin=lgQmin)
        ensure_directory(success_rate_file_name)
        if not self.reset:
            success_rate_file_exists = os.path.exists(success_rate_file_name)
            if success_rate_file_exists:
                logging.debug("success_rate_file exists for lgQmin = %(x)f" % dict(x=lgQmin))
                rSuccess = pickle.load(open(success_rate_file_name,"rb"))
                logging.debug("previously calculated rate of success for lgQmin %(a)f is = %(x)f" % dict(a=lgQmin, x=rSuccess))
                return rSuccess
        logging.debug("success_rate_file still does not exists")
        func = partial(self.number_of_successes_of_calculating_evolution, lgQmin=lgQmin)
        with Pool(processes=len(self.array_of_prior_transform_instances)) as pool:
            checks = pool.map(func, self.array_of_prior_transform_instances)
        nSystems = len(self.array_of_prior_transform_instances)
        logging.debug("nSystems = %(s)s" % dict(s=nSystems))
        nTrials = nSystems * self.n
        nSuccess = numpy.sum(list(checks))
        logging.debug("nSuccess = %(s)f for lgQmin = %(x)f" % dict(s=nSuccess, x=lgQmin))
        rSuccess = (nSuccess * 1.00)/nTrials
        logging.debug("rSuccess = %(s)f for lgQmin = %(x)f" % dict(s=rSuccess, x = lgQmin))
        save_object(rSuccess, success_rate_file_name)
        temp = pickle.load(open(success_rate_file_name,"rb"))
        logging.debug("success_rate_file is created and rSuccess is stored in it. The retrieved value of rSuccess for lgQmin %(y)f from this file is %(x)f" % dict(y=lgQmin, x=temp))
        return rSuccess

    def determine_lgQmin(self):
        logging.debug("We are now determining lgQmin")
        lgQmin = self.starting
        while lgQmin < self.ending:
            r = self.rate_of_successful_evolution_calculations_for_all_systems(lgQmin)
            logging.debug("Rate of success for lgQmin = %(x)f is %(r)f" % dict(x=lgQmin, r=r))
            if r >= self.tol:
                return lgQmin
            lgQmin = lgQmin + 1
        return lgQmin

if __name__ == '__main__':
    Initialization()
    envelope_eccentricity_function=EnvelopeEccentricityDistribution.envelope_eccentricity_function
    output_dirname = "/work/08529/mmmahmud/scratch/circularization_exoplanet_system/sampling_output"
    logging_fname = "%(dirname)s/logger_for_lgQmin.log" % dict(dirname=output_dirname)
    logger_name = "%(dirname)s/logger_for_lgQmin" % dict(dirname=output_dirname)
    logging_level = logging.DEBUG
    logger = setup_logger(logger_name, logging_fname, level=logging_level)

    envelope_eccentricity_distribution_instance = EnvelopeEccentricityDistribution.EnvelopeEccentricityDistribution()
    index = envelope_eccentricity_distribution_instance.print_properties_of_binary_systems_satisfying_constraints()
    logger.debug("^^^^index of the selected systems = %(i)s" % dict(i = ' '.join(map(str,index))))
    logger.debug("length of index is = %(z)f" % dict(z=len(index)))
    measured_values = []
    standard_deviations = []
    system_name = []
    e_env = []
    index_of_systems_of_big_planets = []
    prior_transform_instance = []
    j = -1
    for i in range(0, len(index)):
        measured_values_, standard_deviations_, system_name_ = envelope_eccentricity_distribution_instance.properties_of_ith_binary_system_if_satisfies_constraints(index[i])
        if 'secondary radius' in measured_values_:
            rad = measured_values_['secondary radius']
        elif ('ratio of planet and stellar radius' in measured_values_) and ('primary radius' in measured_values_):
            rad = measured_values_['ratio of planet and stellar radius'] * measured_values_['primary radius'] * const.R_sun.value / const.R_earth.value
        elif ('transit depth' in measured_values_) and ('primary radius' in measured_values_):
            rad = ((measured_values_['transit depth']/100)**0.5) *  measured_values_['primary radius'] * const.R_sun.value / const.R_earth.value
        else:
            rad = None
        if not(rad is None):
            rad_j = rad * const.R_earth.value / const.R_jup.value
            if rad_j > 0.6:
                logger.debug("Radius of the planet of %(s)s is %(x)f RJ" % dict(s=system_name_, x = rad_j))
                j = j + 1
                envelope_eccentricity = envelope_eccentricity_function(x=(measured_values_['semi major axis'] / rad), logger=logger)
                e_env.append(envelope_eccentricity)
                measured_values.append(measured_values_)
                standard_deviations.append(standard_deviations_)
                system_name.append(system_name_)
                x = json.dumps(measured_values[j])
                y = json.dumps(standard_deviations[j])
                logger.debug('%(j)f: inserted measured values are %(x)s with standard deviations %(y)s for system %(s)s:' % dict(j=j, x=x, y=y, s = system_name[j]))
                index_of_systems_of_big_planets.append(index[i])
                #prior_transform_instance.append(PriorTransform(measured_values[j], standard_deviations[j], e_env[j], system=system_name[j], logger=logger))
            else:
                logger.debug("Radius of the planet of %(s)s is %(x)f RJ" % dict(s=system_name_, x=rad_j))
    logger.debug("There are %(x)f number of systems having big planet." % dict(x=len(index_of_systems_of_big_planets)))
    logger.debug("Index of the systems of big planets are: %(x)s" % dict(x = ' '.join(map(str,index_of_systems_of_big_planets))))

    def get_prior_transform_instance(i):
        output_dirname = "/work/08529/mmmahmud/scratch/circularization_exoplanet_system/sampling_output"
        logging_fname = "%(dirname)s/logger_for_lgQmin_of_%(s)s_index_%(i)d.log" % dict(dirname=output_dirname, s=system_name[i], i = i)
        logger_name = "%(dirname)s/logger_for_lgQmin_of_%(s)s_index_%(i)d" % dict(dirname=output_dirname, s=system_name[i], i = i)
        logging_level = logging.DEBUG
        logger = setup_logger(logger_name, logging_fname, level=logging_level)
        x = PriorTransform(measured_values[i], standard_deviations[i], e_env[i], system=system_name[i], logger=logger)
        return x
    directory = "/work/08529/mmmahmud/prior_transform_instances"
    for i in range(0, j+1):
        file_name = "%(d)s/prior_transform_instance_for_%(s)s.pkl" % dict(d=directory, s=system_name[i])
        ensure_directory(file_name)
        file_exists = os.path.exists(file_name)
        a = None
        if not file_exists:
            logger.debug("pickled file did not exists. Prior transform instance is created for the first time for this system %(s)s" % dict(s=system_name[i]))
            a = get_prior_transform_instance(i)
            save_object(a, file_name)
        if file_exists:
            a = pickle.load(open( file_name, "rb" ) )
            logger.debug("pickled file containing the prior transform instance for this system %(s)s exists" % dict(s=system_name[i]))
            if a is None:
                a = get_prior_transform_instance(i)
                save_object(a, file_name)
        if a is not None: prior_transform_instance.append(a)

    k = prior_transform_instance[54:55]
    g = k[0].system
    logging.debug("The value of present g is %(s)s" % dict(s=g))

    p = FindLgQmin(k)
    g = p.lgQmin
    logging.debug("The value of g is %(x)f" % dict(x=g))
    logging.debug("DONE")

