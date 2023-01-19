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
import pandas as pd
from scipy.optimize import fsolve
from rice_distribution_utils import rice_from_error_bars
from scipy import integrate
from scipy.misc import derivative
from scipy.special import i0

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
            os.makedirs(dirname, exist_ok=True)

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def get_alpha():
    alpha = 3
    return alpha

def get_Pbr():
    Pbr = 1
    return Pbr

def get_lgQpl():
    lgQpl = 5
    return lgQpl

def get_target_e_f():
    target_e_f = 0.3
    return target_e_f

def reset():
    reset = False
    return reset

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


class FinalVsInitialEccentricity:
    def __init__(self, logging_fname = "/work/08529/mmmahmud/scratch/finalVsInitialEccentricity/finalVsInitialEccentricity.log", reset = False):
        self.logging_fname = logging_fname
        ensure_directory(self.logging_fname)
        logger_name = "/work/08529/mmmahmud/scratch/finalVsInitialEccentricity/logger_for_finalVsInitialEccentricity"
        logging_level = logging.DEBUG
        self.logger = setup_logger(logger_name, logging_fname, level=logging_level)
        self.obliquity = 0
        self.spin_frequency_breaks_for_planet=None
        self.spin_frequency_powers_for_planet=numpy.array([0.0])
        self.directory = "/work/08529/mmmahmud/finalVsInitialEccentricity/recordsOnEfinalVsPorb"
        self.primary_mass = 1
        self.secondary_mass = 1
        self.secondary_radius = 1
        self.initial_stellar_spin = 10
        self.tidal_break_period = get_Pbr()
        self.argument_of_phase_lag_function_for_planet = get_lgQpl()
        self.power_law_argument = get_alpha()
        self.stellar_age = 4.57
        self.stellar_metallicity = 0
        self.reset = reset

    def find_Porb_for_a_target_e_final(self,
                                       efinal,
                                       e_i = 0.8,
                                       min_Porb=1.0,
                                       max_Porb=7.0,
                                       alpha = get_alpha(),
                                       Pbr = get_Pbr(),
                                       lgQpl = get_lgQpl(),
                                       tol=0.001,
                                       n=61):
        self.power_law_argument = alpha
        self.tidal_break_period = Pbr
        self.argument_of_phase_lag_function_for_planet = lgQpl
        a = min_Porb
        b = max_Porb

        dirname =  "%(directory)s/alpha_%(alpha)f/Pbr_%(Pbr)f/lgQpl_%(lgQpl)f" % dict(directory = self.directory,
                                                                                      alpha = alpha,
                                                                                      Pbr = Pbr,
                                                                                      lgQpl = lgQpl)
        filename = "%(dirname)s/alpha_%(alpha)f_Pbr_%(Pbr)f_lgQpl_%(lgQpl)f_efinalVsPorb_min_Porb_%(a)f_max_Porb_%(b)f.pkl" % dict(dirname=dirname,
                                                                                                                                   alpha=alpha,
                                                                                                                                   Pbr=Pbr,
                                                                                                                                   lgQpl=lgQpl, a=a, b=b)
        self.logger.debug("The name of the file is %(filename)s." % dict(filename=filename))
        ensure_directory(filename)
        file_exists = os.path.exists(filename)

        if file_exists and (not self.reset):
            self.logger.debug("The file with previous records exists and we are retrieving those records.")
            efinalVsPorb_instance = pickle.load(open(filename, "rb"))
            checks = efinalVsPorb_instance.checks
            arg = efinalVsPorb_instance.arg
        else:
            self.logger.debug("Either file with previous records does not exist or we are resetting")
            arg = []
            delta = (b-a)/(n-1)
            for i in numpy.arange(a, b+delta, delta):
                arg.append(i)
            self.logger.debug("Porbs are %(porbs)s" % dict(porbs=repr(arg)))

            func = partial(self.find_e_final, e_init = e_i)
            with Pool(processes=len(arg)) as pool:
                checks = pool.map(func, arg)
            self.logger.debug("e_finals are %(ef)s" % dict(ef=repr(checks)))

            checks_excluding_None = []
            arg_excluding_None = []
            for i in range(0, len(checks)):
                if not pd.isnull(checks[i]):
                    checks_excluding_None.append(checks[i])
                    arg_excluding_None.append(arg[i])
            checks = checks_excluding_None
            arg = arg_excluding_None
            self.logger.debug("e_finals excluding Nan are %(ef)s" % dict(ef=repr(checks)))
            self.logger.debug("Porbs excluding those values for which e_finals are Nan, are %(porbs)s" % dict(porbs=repr(arg)))
            efinalVsPorb_instance = EfinalVsPorb(checks, arg)
            save_object(efinalVsPorb_instance, filename)
         
        if math.fabs(checks[0]-efinal)<tol: return arg[0]
        if math.fabs(checks[-1]-efinal)<tol: return arg[-1]
        if checks[0]==checks[-1]: return None
        if (checks[0]<checks[-1] and checks[-1]<efinal) or (checks[0]>checks[-1] and checks[-1]>efinal):
            new_a = arg[-1]
            new_b = 2*arg[-1] - arg[0]
            return self.find_Porb_for_a_target_e_final(efinal,
                                                       e_i = e_i,
                                                       min_Porb=new_a,
                                                       max_Porb=new_b,
                                                       alpha = alpha,
                                                       Pbr = Pbr,
                                                       tol=tol,
                                                       n=n)
        if (efinal<checks[0] and checks[0]<checks[-1]) or (efinal>checks[0] and checks[0]>checks[-1]):
            new_b = arg[0]
            new_a = arg[0]/10
            return self.find_Porb_for_a_target_e_final(efinal,
                                                       e_i = e_i,
                                                       min_Porb=new_a,
                                                       max_Porb=new_b,
                                                       alpha = alpha,
                                                       Pbr = Pbr,
                                                       tol=tol,
                                                       n=n)
        if (checks[0]<efinal and efinal<checks[-1]) or (checks[0]>efinal and efinal>checks[-1]):
            for i in range(0, len(checks)-1):
                if math.fabs(checks[i]-efinal)<tol: return arg[i]
                if math.fabs(checks[i+1]-efinal)<tol: return arg[i+1]
                if (checks[i]<efinal and efinal<checks[i+1]) or (checks[i]>efinal and efinal>checks[i+1]):
                    new_a = arg[i]
                    new_b = arg[i+1]
                    return self.find_Porb_for_a_target_e_final(efinal,
                                                       e_i = e_i,
                                                       min_Porb=new_a,
                                                       max_Porb=new_b,
                                                       alpha = alpha,
                                                       Pbr = Pbr,
                                                       tol=tol,
                                                       n=n)

    def find_e_final(self, Porb, e_init):
        logging.basicConfig(filename = self.logging_fname, level=logging.DEBUG, force = True, format='%(asctime)s %(message)s')
        parameters_for_evolution = {'primary mass': self.primary_mass,
                                    'secondary radius': self.secondary_radius,
                                    'secondary mass': self.secondary_mass,
                                    'initial stellar spin': self.initial_stellar_spin,
                                    'tidal break period': self.tidal_break_period, #10**((math.log10(0.5)+1)/2),
                                    'argument of phase lag function for planet': self.argument_of_phase_lag_function_for_planet,
                                    'power law argument': self.power_law_argument,
                                    'stellar age': self.stellar_age,
                                    'stellar metallicity': self.stellar_metallicity
                                    }

        star_exoplanet_binary_system = SimpleNamespace(primary_mass=parameters_for_evolution['primary mass'] * un.solMass,
                                              secondary_mass=parameters_for_evolution['secondary mass'] * un.jupiterMass,
                                              secondary_radius=parameters_for_evolution['secondary radius'] * un.jupiterRad,
                                              feh=parameters_for_evolution['stellar metallicity'] * un.dimensionless_unscaled,
                                              orbital_period=Porb * un.d,
                                              obliquity=self.obliquity * un.deg,
                                              age=parameters_for_evolution['stellar age'] * un.Gyr)

        break_frequency = 2 * math.pi / parameters_for_evolution['tidal break period']
        reference_argument_of_phase_lag_function_for_planet = parameters_for_evolution['argument of phase lag function for planet']
        if parameters_for_evolution['power law argument'] < 0:
            tidal_frequency_breaks_for_planet = numpy.array([break_frequency])
            tidal_frequency_powers_for_planet = numpy.array([0.0, parameters_for_evolution['power law argument']])
        if parameters_for_evolution['power law argument'] > 0 or parameters_for_evolution['power law argument'] == 0:
            tidal_frequency_breaks_for_planet = numpy.array([2 * math.pi / 20, break_frequency])
            tidal_frequency_powers_for_planet = numpy.array([0.0, parameters_for_evolution['power law argument'], 0.0])
            reference_argument_of_phase_lag_function_for_planet += parameters_for_evolution['power law argument'] * math.log10(tidal_frequency_breaks_for_planet[1]/tidal_frequency_breaks_for_planet[0])
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
        try:
            self.logger.debug("find_evolution method for is being called.")
            self.logger.debug("dissipation: %(dissipation)s" % dict(dissipation = repr(dissipation)))
            evolutionary_history = find_evolution(system=star_exoplanet_binary_system,
                                                  interpolator=FeHConditionalLikelihoodBase.interpolator,
                                                  dissipation=dissipation,
                                                  max_age=parameters_for_evolution['stellar age'] * un.Gyr,
                                                  initial_eccentricity=e_init * un.dimensionless_unscaled,
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
            self.logger.error('Runtime error occurs: %(error)s' % dict(error= str(runerror)))
            self.logger.warning(traceback.format_exc())
            return None
        except ValueError as verror:
            self.logger.warning(traceback.format_exc())
            self.logger.error('Invalid parameter values encountered: %(error)s' % dict(error= str(verror)))
            return None
        except Exception as exc:
            self.logger.warning(traceback.format_exc())
            self.logger.error("General Error occurs while calculating evolution: %(error)s" % dict(error=str(exc)))
            return None
        else:
            calculated_eccentricity_now = evolutionary_history.eccentricity[-1]
            age_upto_which_evolution_is_calculated = evolutionary_history.age[-1]
            self.logger.debug("Evolution is calculated.")
            if math.fabs(parameters_for_evolution['stellar age'] - age_upto_which_evolution_is_calculated) <0.00001:
                self.logger.debug("Evolution is calculated up to the stellar age.")
                return calculated_eccentricity_now
            else:
                self.logger.debug("Evolution is NOT calculated up to the stellar age.")
                return None

class EfinalVsPorb:
    def __init__(self, checks, arg):
        self.checks = checks
        self.arg = arg

class argument:
    def __init__(self, e_i, Porb):
        self.e_i = e_i
        self.Porb = Porb
        self.e_f = None


class e_final_vs_e_initial:
    def __init__(self, e_i_array, e_f_array):
        self.e_i_array = e_i_array
        self.e_f_array = e_f_array
        self.raw_e_f_as_a_smooth_function_of_e_i = self.build_lagrangian_function(self.e_i_array, self.e_f_array)
        e_f_array_for_smooth_inverse_function, e_i_array_for_smooth_inverse_function = self.build_x_array_and_y_array_for_smooth_inverse_function()
        self.raw_e_i_as_a_smooth_function_of_e_f = self.build_lagrangian_function(e_f_array_for_smooth_inverse_function, e_i_array_for_smooth_inverse_function)
        e_i_array_for_derivative_of_e_f_wrt_e_i_smooth, d_e_f_wrt_e_i_array_for_derivative_of_e_f_wrt_e_i_smooth = self.build_x_array_and_y_array_for_derivative_of_e_f_wrt_e_i_smooth()
        self.derivative_of_e_f_wrt_e_i_smooth = self.build_lagrangian_function(e_i_array_for_derivative_of_e_f_wrt_e_i_smooth, d_e_f_wrt_e_i_array_for_derivative_of_e_f_wrt_e_i_smooth)

    def piecewise_linear_function(self, e_i):
        a = 0
        length = len(self.e_i_array)
        b = length -1
        while b-a>1:
            mid_index = int((a+b)/2)
            mid = self.e_i_array[mid_index]
            if e_i == mid:
                return self.e_f_array[mid_index]
            if e_i < mid:
                b = mid_index
            if e_i > mid:
                a = mid_index
        if self.e_i_array[a]==e_i: return self.e_f_array[a]
        if self.e_i_array[b]==e_i: return self.e_f_array[b]
        m = (self.e_f_array[b]-self.e_f_array[a])/(self.e_i_array[b]-self.e_i_array[a])
        e_f = self.e_f_array[a] + (e_i - self.e_i_array[a])*m
        if e_f<0: e_f=0
        if e_f>1: e_f=1
        return e_f

    def linear_function(self, e_i):
        m = (self.e_f_array[-1]-self.e_f_array[0])/(self.e_i_array[-1]-self.e_i_array[0])
        e_f = self.e_f_array[0] + (e_i - self.e_i_array[0])*m
        if e_f<0: e_f=0
        if e_f>1: e_f=1
        return e_f

    def piecewise_linear_inverse_function(self, e_f):
        a = 0
        length = len(self.e_f_array)
        b = length -1
        while b-a>1:
            mid_index = int((a+b)/2)
            mid = self.e_f_array[mid_index]
            if e_f == mid:
                return self.e_i_array[mid_index]
            if e_f < mid:
                b = mid_index
            if e_f > mid:
                a = mid_index
        if self.e_f_array[a]==e_f: return self.e_i_array[a]
        if self.e_f_array[b]==e_f: return self.e_i_array[b]
        m = (self.e_i_array[b]-self.e_i_array[a])/(self.e_f_array[b]-self.e_f_array[a])
        e_i = self.e_i_array[a] + (e_f - self.e_f_array[a])*m
        if e_i<0: e_i=0
        if e_i>1: e_i=1
        return e_i

    def linear_inverse_function(self, e_f):
        m = (self.e_i_array[-1]-self.e_i_array[0])/(self.e_f_array[-1]-self.e_f_array[0])
        e_i = self.e_i_array[0] + (e_f - self.e_f_array[0])*m
        if e_i<0: e_i=0
        if e_i>1: e_i=1
        return e_i

    def build_lagrangian_function(self, x_array, y_array):
        def lagrangian_function(x):
            y = 0
            i = 0
            while i<len(x_array):
                A = y_array[i]
                B = 1
                C = 1
                j = 0
                while j<len(x_array):
                    if j != i:
                        B = B * (x - x_array[j])
                        C = C * (x_array[i] - x_array[j])
                    j = j + 1
                y = y + A * B / C
                i = i + 1
            return y
        return lagrangian_function

    def smooth_function(self, e_i):
        e_f = self.raw_e_f_as_a_smooth_function_of_e_i(e_i)
        if e_f <0: return 0
        if e_f>1: return 1
        return e_f

    def smooth_inverse_function_aux(self, e_f):
        delta = (self.e_f_array[-1] - self.e_f_array[0])/1000.0
        e_i_a = self.e_i_array[0]
        e_i_b = self.e_i_array[-1]
        e_f_a = self.smooth_function(e_i_a)
        e_f_b = self.smooth_function(e_i_b)
        if math.fabs(e_f-e_f_a) < delta: return e_i_a
        if math.fabs(e_f-e_f_b) < delta: return e_i_b
        if e_f>e_f_b and math.fabs(e_f-e_f_b)>delta:
            e_i_a = e_i_b
            e_i_b = 1.0
            e_f_a = e_f_b
            e_f_b = self.smooth_function(e_i_b)
        if e_f<e_f_a and math.fabs(e_f-e_f_a)>delta:
            e_i_b = e_i_a
            e_i_a = 0.0
            e_f_b = e_f_a
            e_f_a = self.smooth_function(e_i_a)
        while (e_f>e_f_a and e_f<e_f_b):
            e_i_c = (e_i_a + e_i_b)/2.0
            e_f_c = self.smooth_function(e_i_c)
            if math.fabs(e_f_c - e_f)<delta:
                e_i = e_i_c
                if e_i<0: e_i=0
                if e_i>1: e_i=1
                return e_i
            if e_f_c < e_f:
                e_i_a = e_i_c
                e_f_a = self.smooth_function(e_i_a)
            if e_f_c > e_f:
                e_i_b = e_i_c
                e_f_b = self.smooth_function(e_i_b)
            if math.fabs(e_f-e_f_a) < delta: return e_i_a
            if math.fabs(e_f-e_f_b) < delta: return e_i_b

    def build_x_array_and_y_array_for_smooth_inverse_function(self):
        e_i_array = self.e_i_array
        e_f_array = self.e_f_array
        k =(e_i_array[-1]-e_i_array[0])/(e_i_array[1]-e_i_array[0])
        del_e_f = (e_f_array[-1] - e_f_array[0])/(k)

        x_array = []
        y_array = []

        e_f = e_f_array[0]
        while e_f < (e_f_array[-1]+del_e_f):
            e_i = self.smooth_inverse_function_aux(e_f)
            if e_i is not None:
                x_array.append(e_f)
                y_array.append(e_i)
            e_f = e_f + del_e_f
        return x_array, y_array

    def smooth_inverse_function(self, e_f):
        e_i = self.raw_e_i_as_a_smooth_function_of_e_f(e_f)
        if e_i < 0: return 0
        if e_i > 1: return 1
        return e_i

    def build_x_array_and_y_array_for_derivative_of_e_f_wrt_e_i_smooth(self):
        e_i_array = self.e_i_array
        e_f_array = self.e_f_array
        x_array = []
        y_array = []

        k = (e_i_array[-1] - e_i_array[0])/(e_i_array[1] - e_i_array[0])
        del_e_i = (e_i_array[-1] - e_i_array[0])/(5*k)

        e_i = e_i_array[0]
        while e_i < e_i_array[-1]:
            d_e_f_wrt_e_i = (self.smooth_function(e_i + del_e_i) - self.smooth_function(e_i))/del_e_i
            if d_e_f_wrt_e_i is not None:
                x_array.append(e_i)
                y_array.append(d_e_f_wrt_e_i)
            e_i = e_i + del_e_i

        return x_array, y_array

    def derivative_of_e_final_wrt_e_initial(self, e_i, function):
        delta = 0.05
        d = derivative(function, e_i, dx=delta)
        return d


class eccentricity_distribution_class:
    def __init__(self, sigma_square, nu_square):
        self.sigma_square = sigma_square
        self.nu_square = nu_square
        self.nu = math.sqrt(self.nu_square)

    def eccentricity_distribution(self, e):
        if self.nu_square == 0:
            return (1/2/math.pi/self.sigma_square)*math.exp(-e**2/2/self.sigma_square)
        return (1/2/math.pi/self.sigma_square)*math.exp(-(e**2 + self.nu_square)/2/self.sigma_square)* i0(e * self.nu/self.sigma_square)

    def prior_distribution_of_e_i(self, e_i):
        if e_i > 1: return 0
        if e_i < 0: return 0
        return 1

if __name__ == '__main__':
    Initialization()
    dirname = "/work/08529/mmmahmud/finalVsInitialEccentricity"
    arg = []
    find_e_final_vs_e_init_for_different_Porb = False
    find_Porb_with_target_e_f = True
    a = FinalVsInitialEccentricity()
    def func(ar):
       filename = "%(dirname)s/arguments/Porb_%(Porb)f/alpha_%(alpha)f/lgQpl_%(lgQpl)f/Porb_%(Porb)f_e_i_%(e_i)f_when_Pbr_%(Pbr)f_alpha_%(alpha)f_lgQpl_%(lgQpl)f.pkl" % dict(dirname = dirname,
                                                                                                                                                                              Porb=ar.Porb,
                                                                                                                                                                              e_i=ar.e_i,
                                                                                                                                                                              alpha=a.power_law_argument,
                                                                                                                                                                              Pbr=a.tidal_break_period,
                                                                                                                                                                              lgQpl=a.argument_of_phase_lag_function_for_planet)
       ensure_directory(filename)
       a.logger.debug("fileName = %(x)s" % dict(x=filename))
       file_exists = os.path.exists(filename)
       if file_exists and (not reset()):
           a.logger.debug("file exists. We are retrieving data.")
           element = pickle.load(open(filename, "rb"))
           if element.e_f is not None:
               a.logger.debug("element e_final is = %(x)f" % dict(x=element.e_f))
               return element.e_f
       a.logger.debug("File was not there. We are calculating e_final.")
       e = a.find_e_final(ar.Porb, ar.e_i)
       a.logger.debug("Calculated e_final is %(x)s" % dict(x=repr(e)))
       if e is not None: ar.e_f = e
       save_object(ar, filename)
       return e

    if find_e_final_vs_e_init_for_different_Porb:
       arg = []
       for Porb in numpy.arange(1.0, 7, 0.5):
           for e_init in numpy.arange(0.1, 0.9, 0.1):
               x = argument(e_init, Porb)
               arg.append(x)
       with Pool(processes=len(arg)) as pool:
           checks = pool.map(func, arg)

       for Porb in numpy.arange(1.0, 7, 0.5):
           e_i = []
           e_f = []
           for e_init in numpy.arange(0.1, 0.9, 0.1):
               filename = "%(dirname)s/arguments/Porb_%(Porb)f/alpha_%(alpha)f/lgQpl_%(lgQpl)f/Porb_%(Porb)f_e_i_%(e_i)f_when_Pbr_%(Pbr)f_alpha_%(alpha)f_lgQpl_%(lgQpl)f.pkl" % dict(dirname=dirname,
                                                                                                                                                                                      Porb=Porb,
                                                                                                                                                                                      e_i=e_init,
                                                                                                                                                                                      alpha=get_alpha(),
                                                                                                                                                                                      Pbr=get_Pbr(),
                                                                                                                                                                                      lgQpl=get_lgQpl())
               file_exists = os.path.exists(filename)
               if file_exists:
                   element = pickle.load(open(filename, "rb"))
                   if element.e_f is not None:
                       e_i.append(e_init)
                       e_f.append(element.e_f)
           plt.plot(e_i, e_f)
           plt.xlabel("Initial Eccentricity")
           plt.ylabel('Final Eccentricity')
           fig_file_name = "%(dirname)s/plots/e_f_vs_e_in_for_Porb_%(Porb)f_when_Pbr_%(Pbr)f_alpha_%(alpha)f_lgQpl_%(lgQpl)f.pdf" % dict(dirname=dirname,
                                                                                                                                         Porb=Porb,
                                                                                                                                         alpha=get_alpha(),
                                                                                                                                         Pbr=get_Pbr(),
                                                                                                                                         lgQpl=get_lgQpl())
           ensure_directory(fig_file_name)
           plt.savefig(fig_file_name)
           plt.cla()
           plt.clf()

    if find_Porb_with_target_e_f:
       #Porb_1 = a.find_Porb_for_a_target_e_final(0.3, alpha=0, Pbr = 1.0, lgQpl = 5)
       #Porb_2 = a.find_Porb_for_a_target_e_final(0.3, alpha=3, Pbr = 1.0, lgQpl = 5)
       #Porb_3 = a.find_Porb_for_a_target_e_final(0.1, alpha=-3, Pbr = 5.1, lgQpl = 5)
       #Porb_4 = a.find_Porb_for_a_target_e_final(0.5, alpha=0, Pbr = 1.0, lgQpl = 5)
       Porb_5 = a.find_Porb_for_a_target_e_final(0.5, alpha=-3, Pbr = 1.0, lgQpl = 5)
       target_e_f = 0.5
       alpha = -3
       Pbr = 1.0
       lgQpl = 5
       Porb = Porb_5
       a.logger.debug("Porb = %(x)f" % dict(x=Porb))
       def plot_e_i_vs_e_f_for_the_finetuned_Porb_for_which_a_target_e_f_can_be_reached_at_present_age_by_starting_evolution_from_e_i_max(Porb_, alpha, Pbr, lgQpl):
           Porb = Porb_
           a.power_law_argument = alpha
           a.tidal_break_period = Pbr
           a.argument_of_phase_lag_function_for_planet = lgQpl
           arg = []

           for e_init in numpy.arange(0.1, 0.9, 0.1):
               x = argument(e_init, Porb)
               arg.append(x)

           with Pool(processes=len(arg)) as pool:
               checks = pool.map(func, arg)
           e_i = []
           e_f = []
           e_i.append(0)
           e_f.append(0)
           e_init = 0.01
           for e_init in numpy.arange(0.1, 0.9, 0.1):
               filename = "%(dirname)s/arguments/Porb_%(Porb)f/alpha_%(alpha)f/lgQpl_%(lgQpl)f/Porb_%(Porb)f_e_i_%(e_i)f_when_Pbr_%(Pbr)f_alpha_%(alpha)f_lgQpl_%(lgQpl)f.pkl" % dict(dirname = dirname,
                                                                                                                                                                                      Porb=Porb,
                                                                                                                                                                                      e_i=e_init,
                                                                                                                                                                                      alpha=alpha,
                                                                                                                                                                                      Pbr=Pbr,
                                                                                                                                                                                      lgQpl=lgQpl)
               file_exists = os.path.exists(filename)
               if file_exists:
                   element = pickle.load(open(filename, "rb"))
                   if element.e_f is not None:
                       e_i.append(e_init)
                       e_f.append(element.e_f)

           plt.plot(e_i, e_f)
           plt.xlabel("Initial Eccentricity")
           plt.ylabel('Final Eccentricity')
           fig_file_name = "%(dirname)s/plots/comp_e_f_vs_e_in_for_Porb_%(Porb)f_when_alpha_%(alpha)f_Pbr_%(Pbr)f_lgQpl_%(lgQpl)f.pdf" % dict(dirname=dirname,
                                                                                                                                              Porb=Porb,
                                                                                                                                              alpha=a.power_law_argument,
                                                                                                                                              Pbr=a.tidal_break_period,
                                                                                                                                              lgQpl=a.argument_of_phase_lag_function_for_planet)
           ensure_directory(fig_file_name)
           plt.savefig(fig_file_name)
           plt.cla()
           plt.clf()
           return e_i, e_f
       e_i_array, e_f_array = plot_e_i_vs_e_f_for_the_finetuned_Porb_for_which_a_target_e_f_can_be_reached_at_present_age_by_starting_evolution_from_e_i_max(Porb, alpha=alpha, Pbr=Pbr, lgQpl = lgQpl)
       e_final_vs_e_initial_instance = e_final_vs_e_initial(e_i_array, e_f_array)
       e_i = []
       e_f = []
       delta = (e_i_array[-1] - e_i_array[0])/1000.0
       for e_init in numpy.arange(e_i_array[0], e_i_array[-1]+delta, delta):
           e_i.append(e_init)
           e_f.append(e_final_vs_e_initial_instance.piecewise_linear_function(e_init))
       plt.plot(e_i, e_f)
       plt.xlabel("Initial Eccentricity")
       plt.ylabel('Final Eccentricity')
       fig_file_name = "%(dirname)s/plots/test59_e_f_vs_e_in_for_Porb_%(Porb)f_when_alpha_%(alpha)f_Pbr_%(Pbr)f_lgQpl_%(lgQpl)f.pdf" % dict(dirname=dirname,
                                               	                                                                                            Porb=Porb,
                                                    	                                                                                    alpha=alpha,
                                                                                                                                            Pbr=Pbr,
                                                                                                                                            lgQpl=lgQpl)
       ensure_directory(fig_file_name)
       plt.savefig(fig_file_name)
       plt.cla()
       plt.clf()

       e_i = []
       e_f = []
       delta = (e_f_array[-1] - e_f_array[0])/1000.0
       for e_final in numpy.arange(e_f_array[0], e_f_array[-1]+delta , delta):
           e_f.append(e_final)
           e_i.append(e_final_vs_e_initial_instance.piecewise_linear_inverse_function(e_final))
       plt.plot(e_i, e_f)
       plt.xlabel("Initial Eccentricity")
       plt.ylabel('Final Eccentricity')
       fig_file_name = "%(dirname)s/plots/test669_e_f_vs_e_in_for_Porb_%(Porb)f_when_alpha_%(alpha)f_Pbr_%(Pbr)f_lgQpl_%(lgQpl)f.pdf" % dict(dirname=dirname,
                                                                                                                                             Porb=Porb,
                                                                                                                                             alpha=alpha,
                                                                                                                                             Pbr=Pbr,
                                                                                                                                             lgQpl=lgQpl)
       ensure_directory(fig_file_name)
       plt.savefig(fig_file_name)
       plt.cla()
       plt.clf()

       d_e_i = []
       e_i = []
       e_f = []
       delta = (e_i_array[1] - e_i_array[0])/4
       i = 0
       #for e_initial in numpy.arange(e_i_array[0], e_i_array[-1] + 4*delta , 4*delta):
       while i<len(e_i_array):
           #e_i.append(e_initial)
           e_initial = e_i_array[i]
           e_final = e_final_vs_e_initial_instance.piecewise_linear_function(e_initial)
           #r = e_final_vs_e_initial_instance.derivative_of_e_f_wrt_e_i_smooth(e_initial)
           r = (e_final_vs_e_initial_instance.piecewise_linear_function(e_initial+delta)-e_final_vs_e_initial_instance.piecewise_linear_function(e_initial))/delta
           if not (math.fabs(r)<0.000001):
               e_f.append(e_final)
               d_e_i.append(1/r)
           i = i+1
           #d_e_i.append(1/e_final_vs_e_initial_instance.derivative_of_e_final_wrt_e_initial(e_initial, e_final_vs_e_initial_instance.smooth_function))
       plt.plot(e_f, d_e_i)
       plt.xlabel("final Eccentricity")
       plt.ylabel('Derivative of initial Eccentricity wrt final Eccentricity')
       fig_file_name = "%(dirname)s/plots/test79_d_e_i_vs_e_f_for_Porb_%(Porb)f_when_alpha_%(alpha)f_Pbr_%(Pbr)f_lgQpl_%(lgQpl)f.pdf" % dict(dirname=dirname,
                                                                                                                                             Porb=Porb,
                                                                                                                                             alpha=alpha,
                                                                                                                                             Pbr=Pbr,
                                                                                                                                             lgQpl=lgQpl)
       ensure_directory(fig_file_name)
       plt.savefig(fig_file_name)
       plt.cla()
       plt.clf()

       measured_e_now = 0.1
       e_now_upper_uncertainty = 0.05
       e_now_lower_uncertainty = -0.05
       rice_distribution = rice_from_error_bars(measured_e_now, e_now_upper_uncertainty, math.fabs(e_now_lower_uncertainty))
       a.logger.debug("The rice of 0.3 is %(x)f" % dict(x=rice_distribution.pdf(0.3)))
       raw_moment1 = rice_distribution.moment(1)
       raw_moment2 = rice_distribution.moment(2)
       raw_moment3 = rice_distribution.moment(3)
       raw_moment4 = rice_distribution.moment(4)
       a.logger.debug("raw moment 1 = %(x)f" % dict(x=raw_moment1))
       a.logger.debug("raw moment 2 = %(x)f" % dict(x=raw_moment2))
       a.logger.debug("raw moment 3 = %(x)f" % dict(x=raw_moment3))
       a.logger.debug("raw moment 4 = %(x)f" % dict(x=raw_moment4))
       if math.fabs(2 * (raw_moment2 ** 2) - raw_moment4)< 0.000001:
           nu_square = 0
       else:
           nu_square = math.sqrt(2 * (raw_moment2 ** 2) - raw_moment4)
       sigma_square = (raw_moment2 - nu_square)/2
       a.logger.debug("nu square = %(x)f" % dict(x=nu_square))
       a.logger.debug("sigma square = %(x)f" % dict(x=sigma_square))

       eccentricity_distribution_instance = eccentricity_distribution_class(sigma_square, nu_square)

       def integrand(e_f):
           p = eccentricity_distribution_instance.eccentricity_distribution(e_f)
           e_i = e_final_vs_e_initial_instance.piecewise_linear_inverse_function(e_f)
           #r = e_final_vs_e_initial_instance.derivative_of_e_f_wrt_e_i_smooth(e_i)
           r = (e_final_vs_e_initial_instance.piecewise_linear_function(e_i+delta)-e_final_vs_e_initial_instance.piecewise_linear_function(e_i))/delta
           #r = (e_final_vs_e_initial_instance.smooth_function(e_i+0.005) - e_final_vs_e_initial_instance.smooth_function(e_i))/0.005
           #r = derivative(e_final_vs_e_initial_instance.smooth_function, e_i, dx=0.05)
           #q = eccentricity_distribution_instance.prior_distribution_of_e_i(e_i)
           #r = e_final_vs_e_initial_instance.derivative_of_e_final_wrt_e_initial(e_i, e_final_vs_e_initial_instance.smooth_function)
           if r == 0: return math.inf
           return p/r
       def integrand_linear(e_f):
           p = eccentricity_distribution_instance.eccentricity_distribution(e_f)
           e_i = e_final_vs_e_initial_instance.linear_inverse_function(e_f)
           #r = e_final_vs_e_initial_instance.derivative_of_e_f_wrt_e_i_smooth(e_i)
           r = (e_final_vs_e_initial_instance.linear_function(e_i+delta) - e_final_vs_e_initial_instance.linear_function(e_i))/delta
           #r = derivative(e_final_vs_e_initial_instance.linear_function, e_i, dx=0.05)
           #q = eccentricity_distribution_instance.prior_distribution_of_e_i(e_i)
           #r = e_final_vs_e_initial_instance.derivative_of_e_final_wrt_e_initial(e_i, e_final_vs_e_initial_instance.linear_function)
           if r == 0: return math.inf
           return p/r
       e_f_max = e_final_vs_e_initial_instance.piecewise_linear_function(0.8)
       result = integrate.quad(integrand, 0, e_f_max, limit=10000)
       result_linear = integrate.quad(integrand_linear, 0, e_f_max, limit = 10000)
       a.logger.debug("The result for piecewise linear curve is %(x)f" % dict(x=result[0]))
       a.logger.debug("TTTTT The result for linear curve is %(x)f" % dict(x=result_linear[0]))
       integ = []
       integ_linear = []
       p = []
       e_f = []
       delta = (e_f_array[-1] - e_f_array[0])/1000.0
       for e_final in numpy.arange(e_f_array[0], e_f_array[-1]+delta , delta):
           e_f.append(e_final)
           integ.append(integrand(e_final))
           integ_linear.append(integrand_linear(e_final))
           p.append(eccentricity_distribution_instance.eccentricity_distribution(e_final))
       plt.plot(e_f, integ)
       plt.xlabel("Final Eccentricity")
       plt.ylabel('integrand_piecewise_linear')
       fig_file_name = "%(dirname)s/plots/test80_integrand_vs_e_f_for_Porb_%(Porb)f_when_alpha_%(alpha)f_Pbr_%(Pbr)f_lgQpl_%(lgQpl)f.pdf" % dict(dirname=dirname,
                                                                                                                                                 Porb=Porb,
                                                                                                                                                 alpha=alpha,
                                                                                                                                                 Pbr=Pbr,
                                                                                                                                                 lgQpl=lgQpl)
       ensure_directory(fig_file_name)
       plt.savefig(fig_file_name)
       plt.cla()
       plt.clf()
       
       plt.plot(e_f, integ_linear)
       plt.xlabel("Final Eccentricity")
       plt.ylabel('integrand_linear')
       fig_file_name = "%(dirname)s/plots/test90_integrand_linear_vs_e_f_for_Porb_%(Porb)f_when_alpha_%(alpha)f_Pbr_%(Pbr)f_lgQpl_%(lgQpl)f.pdf" % dict(dirname=dirname,
                                                                                                                                                        Porb=Porb,
                                                                                                                                                        alpha=alpha,
                                                                                                                                                        Pbr=Pbr,
                                                                                                                                                        lgQpl=lgQpl)
       ensure_directory(fig_file_name)
       plt.savefig(fig_file_name)
       plt.cla()
       plt.clf()

       plt.plot(e_f, p)
       plt.xlabel("Final Eccentricity")
       plt.ylabel('rice over 2 pi e_f')
       fig_file_name = "%(dirname)s/plots/test990_p_vs_e_f_for_Porb_%(Porb)f_when_alpha_%(alpha)f_Pbr_%(Pbr)f_lgQpl_%(lgQpl)f.pdf" % dict(dirname=dirname,
                                                                                                                                          Porb=Porb,
                                                                                                                                          alpha=alpha,
                                                                                                                                          Pbr=Pbr,
                                                                                                                                          lgQpl=lgQpl)
       ensure_directory(fig_file_name)
       plt.savefig(fig_file_name)
       plt.cla()
       plt.clf()
