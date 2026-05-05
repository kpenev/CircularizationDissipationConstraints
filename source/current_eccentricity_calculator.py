"""Define a class for multiprocessing calculation of e(lgQ)."""

import numpy
from astropy import units

from orbital_evolution.transformations import phase_lag
from general_purpose_python_modules.reproduce_system import \
    find_evolution,\
    check_if_secondary_is_star

from io_utilities import pickle_new_result

from josh_scripts import get_dissipation

#This is intended to serve as the callable for multiprocessing pool.
#pylint: disable=too-few-public-methods
class CurrentEccentricityCalculator:
    """
    Class to use in multiprocessing pool to calculate current eccentricities.

    Saves successful calculations in a pickle file to allow resuming after
    interruption.

    Attributes:
        interpolator:    See same name argument to __init__().

        primary_lgQ:    See same name argument to __init__().

        progress_pickle_fname(str):    See same name argument to __init__().

        _progress_lock:    See same name argument to __init__().

        _progress(dict):    See return value of load_progress_pickle(). Only
            includes results loaded from the progress file supplied to __init__.
    """

    #Suffix _lgQ is most readable.
    #pylint: disable=invalid-name
    def __init__(self,
                 *,
                 primary_lgQ,
                 interpolator,
                 progress,
                 progress_pickle_fname,
                 progress_lock,
                 disk_period=10.0 * units.day,
                 disk_dissipation_age=20.0 * units.Myr):
        """
        Set-up the object to calculate present day eccentricities for systems.

        Args:
            interpolator:     A stellar evolution interpolator instance used to
                create the star in the system.

            primary_lgQ(float):    The value to assume for log10(Q*').

            progress(dict):    The unpickled results calculated during a
                previous run (not re-calculated).

            progress_pickle_fname(str):     The filename containing previously
                calculated results and which gets updated with any newly
                calculated results.

            progress_lock:     A lock to guard against multiple processes
                writing to the progress pickle file at once.

        Returns:
            None
        """

        self.interpolator = interpolator
        self.primary_lgQ = primary_lgQ
        self.progress_pickle_fname = progress_pickle_fname
        self._progress_lock = progress_lock
        self.progress = progress
        self.disk_period = disk_period
        self.disk_dissipation_age = disk_dissipation_age
    #pylint: enable=invalid-name

    def __call__(self, job):
        """
        Calculate the evolution for a single system/lgQ combination.

        Args:
            job(tuple):    Should have two entries:

                * system(SimpleNamespace): containing the system to evolve.

                * lgQ(float): log10(Q*') to assume for calculating the
                  evolution.

                * initial_eccentricity:    The eccentricity to start system
                  evolution with.

        Returns:
            float:
                The present day eccentricity that the system has when evolved
                with an initial orbital period such that the present day orbital
                period is reproduced.
        """

        #lgQ is more readable than alternatives
        #pylint: disable=invalid-name
        system, lgQ, initial_eccentricity = job
        #pylint: enable=invalid-name

        #False positive
        #pylint: disable=no-member
        if system.primary_mass > 1.2 * units.M_sun:
        #pylint: enable=no-member
            print('Skipping %s, lgQ = %g, e0 = %g' % (system.hostname,
                                                      lgQ,
                                                      initial_eccentricity))
            return None

        print('Trying %s, lgQ = %g, e0 = %g' % (system.hostname,
                                                lgQ,
                                                initial_eccentricity))

        default_dissipation = dict(
            tidal_frequency_breaks=None,
            spin_frequency_breaks=None,
            tidal_frequency_powers=numpy.array([0.0]),
            spin_frequency_powers=numpy.array([0.0])
        )

        if system.hostname in self.progress:
            progress_entry = self.progress[system.hostname]
            attempt_id = (lgQ, initial_eccentricity)
            if attempt_id in progress_entry:
                return progress_entry[attempt_id]

        secondary_dissipation = dict(
            default_dissipation,
            reference_phase_lag=phase_lag(lgQ)
        )
        if numpy.isfinite(self.primary_lgQ):
            primary_dissipation = dict(
                default_dissipation,
                reference_phase_lag=phase_lag(self.primary_lgQ)
            )
        elif check_if_secondary_is_star(system):
            primary_dissipation = dict(
                default_dissipation,
                reference_phase_lag=phase_lag(lgQ)
            )
            if system.secondary_mass > 1.2 * units.M_sun:
                secondary_dissipation = None
        else:
            primary_dissipation = None

        lgQ_inertial_boost = 0.0
        lgQ_inertial_sharpness = 10.0
        lgQ_break_period = 1.0 * units.day
        lgQ_powerlaw = 0.0
        star_dissipation = get_dissipation(lgQ,lgQ_inertial_boost,lgQ_inertial_sharpness,lgQ_break_period,lgQ_powerlaw)
        primary_wind_strength = 0.17
        primary_wind_saturation = 2.45
        primary_core_envelope_coupling_timescale = 0.01# * units.Gyr
        secondary_disk_lock_period = None
        secondary_wind_strength = 0.17
        secondary_wind_saturation = 2.45
        secondary_core_envelope_coupling_timescale = 0.01# * units.Gyr
        try:
            evolution = find_evolution(
                system=system,
                interpolator=self.interpolator,
                dissipation=dict(
                    primary=star_dissipation,
                    secondary=star_dissipation
                ),
                initial_eccentricity=initial_eccentricity,
                #False positive.
                #pylint: disable=no-member
                disk_period=self.disk_period,
                disk_dissipation_age=self.disk_dissipation_age,
                #
                primary_wind_strength=primary_wind_strength,
                primary_wind_saturation=primary_wind_saturation,
                primary_core_envelope_coupling_timescale=primary_core_envelope_coupling_timescale * units.Gyr,
                secondary_wind_strength=secondary_wind_strength,
                secondary_wind_saturation=secondary_wind_saturation,
                secondary_core_envelope_coupling_timescale=secondary_core_envelope_coupling_timescale * units.Gyr,
                secondary_disk_period=secondary_disk_lock_period,
                # max_age=system.age,
                solve=False,
                max_iterations=4900,
                secondary_is_star=(check_if_secondary_is_star(system)
                                   and
                                   system.secondary_mass <= 1.2 * units.M_sun),
                carepackage = None,
                precision = 1e-5,
                #pylint: enable=no-member
                eccentricity_expansion_fname = '/home/vortebo/eccentricity_expansion_coef_O400.sqlite'.encode('ascii')
            )
        except:
            print('Failed %s, lgQ = %g, e0 = %g' % (system.hostname,
                                                    lgQ,
                                                    initial_eccentricity))
            return numpy.nan

        print('Evolution:')
        for varname, var_evol in vars(evolution).items():
            print('\t' + varname + ': ' + repr(var_evol))

        #False positive
        #pylint: disable=no-member
        if numpy.allclose(evolution.age[-1],
                          system.age.to_value('Gyr'),
                          rtol=1e-10,
                          atol=1e-10):

            final_eccentricity = evolution.eccentricity[-1]
            #pylint: enable=no-member

            print(
                'Solved %s, lgQ = %g, e0 = %g, ef = %g'
                %
                (
                    system.hostname,
                    lgQ,
                    initial_eccentricity,
                    final_eccentricity
                )
            )

            self._progress_lock.acquire()
            pickle_new_result(system.hostname,
                              lgQ,
                              initial_eccentricity,
                              final_eccentricity,
                              self.progress_pickle_fname)
            self._progress_lock.release()

            return final_eccentricity

        return None
#pylint: disable=too-few-public-methods
