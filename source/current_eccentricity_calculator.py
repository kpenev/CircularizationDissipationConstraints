"""Define a class for multiprocessing calculation of e(lgQ)."""

import numpy
from astropy import units

from orbital_evolution.transformations import phase_lag
from reproduce_system import find_evolution

from io_utilities import pickle_new_result

#This is intended to serve as the callable for multiprocessing pool.
#pylint: disable=too-few-public-methods
class CurrentEccentricityCalculator:
    """
    Class to use in multiprocessing pool to calculate current eccentricities.

    Saves successful calculations in a pickle file to allow resuming after
    interruption.

    Attributes:
        initial_eccentricity:    See same name argument to __init__().

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
                 initial_eccentricity,
                 primary_lgQ,
                 interpolator,
                 progress,
                 progress_pickle_fname,
                 progress_lock):
        """
        Set-up the object to calculate present day eccentricities for systems.

        Args:
            initial_eccentricity(float):     The eccentricity to start system
                evolution with.

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

        self.initial_eccentricity = initial_eccentricity
        self.interpolator = interpolator
        self.primary_lgQ = primary_lgQ
        self.progress_pickle_fname = progress_pickle_fname
        self._progress_lock = progress_lock
        self.progress = progress
    #pylint: enable=invalid-name

    def __call__(self, job):
        """
        Calculate the evolution for a single system/lgQ combination.

        Args:
            job(tuple):    Should have two entries:

                * system(SimpleNamespace): containing the system to evolve.

                * lgQ(float): log10(Q*') to assume for calculating the
                evolution.

        Returns:
            float:
                The present day eccentricity that the system has when evolved
                with an initial orbital period such that the present day orbital
                period is reproduced.
        """

        #lgQ is more readable than alternatives
        #pylint: disable=invalid-name
        system, lgQ = job
        #pylint: enable=invalid-name

        if system.hostname in self.progress:
            progress_entry = self.progress[system.hostname]
            if lgQ in progress_entry:
                progress_entry = progress_entry[lgQ]
                assert self.initial_eccentricity == progress_entry[0]
                return progress_entry[1]

        if numpy.isfinite(self.primary_lgQ):
            primary_dissipation = dict(
                reference_phase_lag=phase_lag(self.primary_lgQ),
                tidal_frequency_breaks=None,
                spin_frequency_breaks=None,
                tidal_frequency_powers=numpy.array([0.0]),
                spin_frequency_powers=numpy.array([0.0])
            )
        else:
            primary_dissipation = None

        evolution = find_evolution(
            system=system,
            interpolator=self.interpolator,
            dissipation=dict(
                primary=primary_dissipation,
                secondary=dict(
                    reference_phase_lag=phase_lag(lgQ),
                    tidal_frequency_breaks=None,
                    spin_frequency_breaks=None,
                    tidal_frequency_powers=numpy.array([0.0]),
                    spin_frequency_powers=numpy.array([0.0])
                )
            ),
            initial_eccentricity=self.initial_eccentricity,
            #False positive.
            #pylint: disable=no-member
            disk_period=(7.0 * units.day),
            disk_dissipation_age=(5e-3 * units.Gyr),
            #pylint: enable=no-member
            max_age=system.age
        )
        print(evolution.format())
        #False positive
        #pylint: disable=no-member
        if numpy.allclose(evolution.age[-1],
                          system.age.to_value('Gyr'),
                          rtol=1e-10,
                          atol=1e-10):
            final_eccentricity = evolution.eccentricity[-1]
            #pylint: enable=no-member

            self._progress_lock.acquire()
            pickle_new_result(system.hostname,
                              lgQ,
                              self.initial_eccentricity,
                              final_eccentricity,
                              self.progress_pickle_fname)
            self._progress_lock.release()

            return final_eccentricity

        return None
#pylint: disable=too-few-public-methods
