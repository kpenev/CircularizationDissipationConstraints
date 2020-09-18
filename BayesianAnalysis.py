import math
import planetary_system_io
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy import constants as cnst
import matplotlib.pyplot as plt

import sys
sys.path.append('/home/mmmahmud/poet/PythonPackage')
sys.path.append('../scripts')

from matplotlib import pyplot
from stellar_evolution.manager import StellarEvolutionManager

from orbital_evolution.evolve_interface import library as\
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

class BayesianAnalysis:

    def __init__(self):
        print('BayesianAnalysis class object is made')

    def eccentricity_orbitalperiod_evolution(self, interpolator):
        planet_name = readPlanet.pl_name
        orbital_period = readPlanet.pl_orbper #days
        primary_mass = readPlanet.st_mass #solar mass
        secondary_mass = readPlanet.pl_masse #Earth mass
        metallicity = readPlanet.st_metfe
        primary_radius = readPlanet.st_rad #solar radius
        secondary_radius = readPlanet.pl_rade #earth radius
        stellar_age = readPlanet.st_age #GYear
        eccentricity = readPlanet.pl_orbeccen
        obliquity = readPlanet.pl_orbincl #degrees
        vsini = readPlanet.st_vsini #km/s


    def test_Nasa_Exoplanet_data(self, path, interpolator):

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
        orbital_period = readPlanet.pl_orbper #days
        primary_mass = readPlanet.st_mass #solar mass
        secondary_mass = readPlanet.pl_masse #Earth mass
        metallicity = readPlanet.st_metfe
        primary_radius = readPlanet.st_rad #solar radius
        secondary_radius = readPlanet.pl_rade #earth radius
        stellar_age = readPlanet.st_age #GYear
        eccentricity = readPlanet.pl_orbeccen
        obliquity = readPlanet.pl_orbincl #degrees
        vsini = readPlanet.st_vsini #km/s


        evolutionary_data = []

        j = 0
        for i in range(0, len(planet_name)):
           if not(math.isnan(orbital_period[i])
                  or math.isnan(primary_mass[i])
                  or math.isnan(secondary_mass[i])
                  or math.isnan(metallicity[i])
                  or math.isnan(secondary_radius[i])
                  or math.isnan(stellar_age[i])
                  or math.isnan(eccentricity[i])
                  or math.isnan(obliquity[i])
                  or math.isnan(vsini[i])):
               if orbital_period[i]<=10 and (primary_mass[i]>0.4 and primary_mass[i]<1.2) and (metallicity[i]>-1.014 and metallicity[i]<0.537):
                   j = j+1
                   print(primary_mass[i], (primary_mass[i]>0.4 and primary_mass[i]<1.2))
                   a = System(primary_mass[i] * u.solMass,
                          secondary_mass[i]*u.earthMass,
                          secondary_radius[i]*u.earthRad,
                          metallicity[i]*u.dimensionless_unscaled,
                          orbital_period[i]*u.d,
                          obliquity[i]*u.deg,
                          stellar_age[i]*u.Gyr,
                          eccentricity[i],
                          vsini[i]*u.kilometer/u.second)

                   dissipation = dict(
                       primary=None,
                       secondary=dict(
                           tidal_frequency_breaks=None,
                           spin_frequency_breaks=None,
                           tidal_frequency_powers=numpy.array([0.0]),
                           spin_frequency_powers=numpy.array([0.0]),
                           reference_phase_lag=phase_lag(6.0)
                       )
                   )
                   final_age = 500
                   print(repr(interpolator))
                   b = find_evolution(system = a,
                                      interpolator = interpolator,
                                      dissipation = dissipation,
                                      solve=True,
                                      required_ages = numpy.arange(0.1, final_age, 0.1))


                   y = b.orbital_period[1:42534]
                   z = b.eccentricity[1:42534]
                   x = []
                   for i in range(1,42534):
                       x = x + [math.log(b.age[i])]


             

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

                   evolutionary_data = evolutionary_data + [b]
                   print(repr(dir(b)))
           if j>0:
               break
        return evolutionary_data



class System:
    def __init__(self, primary_mass, secondary_mass, secondary_radius, feh, orbital_period, obliquity, age, eccentricity, vsini):

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
        print(self.planet_name)
        print(self.primary_mass)
        print(self.secondary_mass)
        print(self.secondary_radius)
        print(self.feh)
        print(self.orbital_period)
        print(self.obliquity)
        print(self.eccentricity)
        print(self.age)
        print(self.Vsini)


if __name__ == '__main__':
    test = BayesianAnalysis()
    eccentricity_expansion_fname = b"/home/mmmahmud/poet/scripts/eccentricity_expansion_coef.txt"
    orbital_evolution_library.read_eccentricity_expansion_coefficients(
        eccentricity_expansion_fname
    )
    serialized_dir = '/home/mmmahmud/poet/stellar_evolution_interpolators'
    manager = StellarEvolutionManager(serialized_dir)


    interpolator = manager.get_interpolator_by_name('default')


#    evolutionary_data = test.test_Nasa_Exoplanet_data(path='/home/mmmahmud/CircularizationDissipationConstraints/data/planets_2020.04.10_14.52.24.csv',
#                                  interpolator= (interpolator, interpolator))

    print('end')








