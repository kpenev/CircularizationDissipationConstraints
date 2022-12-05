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
            os.makedirs(dirname, exist_ok=True)

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

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
    def __init__(self, logging_fname = "/work/08529/mmmahmud/scratch/finalVsInitialEccentricity/finalVsInitialEccentricity.log"):
        self.logging_fname = logging_fname
        ensure_directory(self.logging_fname)
        logger_name = "/work/08529/mmmahmud/scratch/finalVsInitialEccentricity/logger_for_finalVsInitialEccentricity"
        logging_level = logging.DEBUG
        self.logger = setup_logger(logger_name, logging_fname, level=logging_level)

        self.obliquity = 0
        self.spin_frequency_breaks_for_planet=None
        self.spin_frequency_powers_for_planet=numpy.array([0.0])
    def find_e_final(self, Porb, e_init):
        logging.basicConfig(filename = self.logging_fname, level=logging.DEBUG, force = True, format='%(asctime)s %(message)s')
        parameters_for_evolution = {'primary mass': 1,
                                    'secondary radius': 1,
                                    'secondary mass': 1,
                                    'initial stellar spin': 10,
                                    'tidal break period': 10**((math.log10(0.5)+1)/2),
                                    'argument of phase lag function for planet': 5,
                                    'power law argument': 0,
                                    'stellar age': 4.57,
                                    'stellar metallicity': 0
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

class argument:
    def __init__(self, e_i, Porb):
        self.e_i = e_i
        self.Porb = Porb
        self.e_f = None

if __name__ == '__main__':
    Initialization()
    dirname = "/work/08529/mmmahmud/finalVsInitialEccentricity"
    arg = []
    for Porb in numpy.arange(1.0, 7, 0.5):
       for e_init in numpy.arange(0.1, 0.9, 0.1):
           x = argument(e_init, Porb)
           arg.append(x)
    a = FinalVsInitialEccentricity()
    def func(ar):
       e = a.find_e_final(ar.Porb, ar.e_i)
       if e is not None: ar.e_f = e
       filename = "%(dirname)s/arguments/Porb_%(Porb)f/Porb_%(Porb)f_e_i_%(e_i)f.pkl" % dict(dirname=dirname, Porb=ar.Porb, e_i=ar.e_i)
       ensure_directory(filename)
       save_object(ar, filename)
       return e


    with Pool(processes=len(arg)) as pool:
           checks = pool.map(func, arg)

    for Porb in numpy.arange(1.0, 7, 0.5):
       e_i = []
       e_f = []
       for e_init in numpy.arange(0.1, 0.9, 0.1):
           filename = "%(dirname)s/arguments/Porb_%(Porb)f/Porb_%(Porb)f_e_i_%(e_i)f.pkl" % dict(dirname=dirname, Porb=Porb, e_i=e_init)
           file_exists = os.path.exists(filename)
           if file_exists:
               element = pickle.load(open(filename, "rb"))
               if element.e_f is not None:
                   e_i.append(e_init)
                   e_f.append(element.e_f)
       plt.plot(e_i, e_f)
       plt.xlabel("Initial Eccentricity")
       plt.ylabel('Final Eccentricity')
       fig_file_name = "%(dirname)s/plots/e_f_vs_e_in_for_Porb_%(Porb)f.pdf" % dict(dirname=dirname, Porb=Porb)
       ensure_directory(fig_file_name)
       plt.savefig(fig_file_name)
       plt.cla()
       plt.clf()

