import sys
import os
import logging
from datetime import datetime
import math
from split_normal_distribution import split_normal
from bayesian.stellar_param_sampling.poet_interp_likelihood import POETInterpLikelihood
from bayesian.stellar_param_sampling.star_sampler import StarSampler
import numpy
from bayesian.stellar_param_sampling.feh_conditional_likelihood_base import \
    FeHConditionalLikelihoodBase
from scipy.stats import norm
import astropy.constants as const
import Constraints_for_selecting_systems
from astropy import units as un
import StarExoplanetSystem
from reproduce_system import *
from random import random, randint
from multiprocessing import Pool, Queue, Process, Value
import emcee
import corner
import matplotlib.pyplot as plt
from orbital_evolution.evolve_interface import library as \
    orbital_evolution_library
import EnvelopeEccentricityDistribution
import EccentricityDistribution

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
    return '/home/mmmahmud/poet/stellar_evolution_interpolators' # '/home/mmm161430/projects/git/poet/stellar_evolution_interpolators'
def getEccentricityExpansionCoefficientsFile():
    return b"/media/mmmahmud/USB/eccentricity_expansion_coef_O400.sqlite" # b"/home/mmm161430/projects/git/poet/eccentricity_expansion_coef_O400.sqlite"

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
                 initial_eccentricity = 0.5,
                 constraints = Constraints_for_selecting_systems.constraints(),
                 spin_frequency_breaks_for_planet=None,
                 spin_frequency_powers_for_planet=numpy.array([0.0]),
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


        self.calculated_eccentricity_now = calculated_eccentricity_now

        if calculated_eccentricity_now >= 0 and calculated_eccentricity_now <= 1:
            probability_density_of_the_calculated_eccentricity = self.probability_density_of_eccentricity(
                calculated_eccentricity_now)
            probability_density = probability_density_of_the_calculated_eccentricity * priors
            if probability_density == 0:
                return -numpy.inf
            if probability_density < 0:
                logging.warning('Probability density cannot be less than zero.')
                return None
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
                print(p0_file_name, ' file did not previously exist. It will be created now and first walker will be loaded now.')
                while True:
                    print('I am here at 2')
                    if ((p0_file_is_being_updated.value == 0) and (number_of_discovered_walkers.value < nwalkers)):
                        p0_file_is_being_updated.value = 1
                        print('Green signal for updating = ', p0_file_is_being_updated.value)
                        p0_file = open(p0_file_name, 'wb')
                        numpy.save(p0_file, u)
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

        config = ConfigObjectForLogging(system=self.system_name)
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
                print('Initially the file ', p0_file_name, ' did not exist.')
                print('The walkers are going to be generated for the first time.')
                print('The file ', p0_file_name, ' will be created and the walkers will be stored there.')
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
                print('The file ', p0_file_name, ' existed previously.')
                print('Previously worked out walkers will be loaded in the code for running MCMC.')
                p0_file = open(p0_file_name, 'rb')
                p0 = numpy.load(p0_file)
                p0_file.close()
                print('The already discovered walkers are: ', p0)
                number_of_already_stored_walkers = p0.size / ndim
                number_of_walkers_yet_to_be_found = (int)(nwalkers - number_of_already_stored_walkers)
                if number_of_walkers_yet_to_be_found > 0:
                    print('New walkers are going to be discovered')
                    p0 = numpy.vstack((p0, self.generate_successful_walkers(p0_file_name,
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
                print('Backend file exists.')
                chain_exists = backend.initialized and (backend.iteration > 0)
                if reset_backend == False and chain_exists:
                    print('Backend file is not subject to reset. The chain size is not zero.')
                    print('Next samples will be drawn from the end of the previously worked out chain.')
                    sampler.run_mcmc(None, 5, progress = True)
                else:
                    print('Either the backend file is subject to reset or previously calculated chain size is zero.')
                    print('Samples will be drawn for the first time.')
                    sampler.run_mcmc(p0, 5, progress = True)
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
            return -numpy.inf, numpy.array([None, None, None, None, None, None, None, None, None, None, None])
        params = numpy.append(params, [self.calculated_eccentricity_now, log_prob_parameters_for_evolution])
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
                 envelope_eccentricity_function=EnvelopeEccentricityDistribution.envelope_eccentricity_function,
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
                 constraints=Constraints_for_selecting_systems.constraints(),
                 spin_frequency_breaks_for_planet=None,
                 spin_frequency_powers_for_planet=numpy.array([0.0])):

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


        self.envelope_eccentricity_function = envelope_eccentricity_function

        if (Constraints_for_selecting_systems.constraints_are_satisfied(orbital_period=means['orbital period'],
                                      primary_mass=means['primary mass'],
                                      secondary_mass=means['secondary mass'],
                                      stellar_metallicity=means['stellar metallicity'],
                                      eccentricity_now=means['present eccentricity'],
                                      stellar_age=means['stellar age'],
                                      constraints=constraints) ):
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

            eccentricity_distribution_object = EccentricityDistribution.EccentricityDistribution(self.means['present eccentricity'],
                                                                        self.standard_deviations[
                                                                            'eccentricity_now_upper_uncertainty'],
                                                                        self.standard_deviations[
                                                                            'eccentricity_now_lower_uncertainty'],
                                                                        self.e_env,
                                                                        system_name)

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

if __name__ == '__main__':

    test2 = EnvelopeEccentricityDistribution.EnvelopeEccentricityDistribution()
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
