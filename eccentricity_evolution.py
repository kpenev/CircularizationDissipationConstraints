#!/usr/bin/env python3

import matplotlib
matplotlib.use('TkAgg')

import sys
sys.path.append('../PythonPackage')
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

wsun = 2.0 * numpy.pi / 25.34

def create_planet(mass, radius,
                  planet_phase_lag) :
    """Return a configured planet to use in the evolution."""

    planet = LockedPlanet(
        mass = mass,
        radius = radius
    )
    if phase_lag:
        print('Setting planet dissipation')
        planet.set_dissipation(tidal_frequency_breaks = None,
                               spin_frequency_breaks = None,
                               tidal_frequency_powers = numpy.array([0.0]),
                               spin_frequency_powers = numpy.array([0.0]),
                               reference_phase_lag = planet_phase_lag)
    return planet

def create_star(interpolator, convective_phase_lag) :
    """Create the star to use in the evolution."""

    star = EvolvingStar(mass = 1.0,
                        metallicity = 0.0,
                        wind_strength = 0.17,
                        wind_saturation_frequency = 2.45,
                        diff_rot_coupling_timescale = 5.0e-3,
                        interpolator = interpolator)
    if convective_phase_lag:
        print('Setting star dissipation')
        star.set_dissipation(zone_index = 0,
                             tidal_frequency_breaks = None,
                             spin_frequency_breaks = None,
                             tidal_frequency_powers = numpy.array([0.0]),
                             spin_frequency_powers = numpy.array([0.0]),
                             reference_phase_lag = convective_phase_lag)
    star.select_interpolation_region(star.core_formation_age())
    return star

def create_system(star,
                  planet,
                  disk_lock_frequency,
                  porb_initial,
                  initial_eccentricity) :
    """Create the system which to evolve from the given star and planet."""

    porb_initial = porb_initial
    disk_dissipation_age = 4e-3
    binary = Binary(primary = star,
                    secondary = planet,
                    initial_orbital_period = porb_initial,
                    initial_eccentricity = initial_eccentricity,
                    initial_inclination = 0.0,
                    disk_lock_frequency = disk_lock_frequency,
                    disk_dissipation_age = disk_dissipation_age,
                    secondary_formation_age = disk_dissipation_age)
    binary.configure(age = star.core_formation_age(),
                     semimajor = float('nan'),
                     eccentricity = float('nan'),
                     spin_angmom = numpy.array([0.0]),
                     inclination = None,
                     periapsis = None,
                     evolution_mode = 'LOCKED_SURFACE_SPIN')
    planet.configure(age = disk_dissipation_age,
                     companion_mass = star.mass,
                     semimajor = binary.semimajor(porb_initial),
                     eccentricity = initial_eccentricity,
                     spin_angmom = numpy.array([6.3 * 0.3 * planet.mass * planet.radius**2]), #uses the ang_vel for earth of 6.3 rad/day
                     inclination = None,
                     periapsis = None,
                     locked_surface = False,
                     zero_outer_inclination = True,
                     zero_outer_periapsis = True)
    star.detect_stellar_wind_saturation()
    return binary

def test_evolution(interpolator, mass, radius, density,
                   planet_phase_lag,
                   convective_phase_lag = phase_lag(5.5)) :
    """Run a single orbital evolution calculation and plot the results."""

    for pdisk, color, wsat_enabled in [(1.4, 'r', '1')] :#,
#                                       (3.0, 'g', '2'),
#                                       (7.0, 'b', '3')] :
        star = create_star(interpolator = interpolator,
                           convective_phase_lag=convective_phase_lag)
        porb_range = numpy.array([0.5,1.0,1.5,2.0,2.5,3.0])
        eccen_range = numpy.array([0.0,0.1,0.2,0.3,0.4])

        planet = create_planet(mass, radius, planet_phase_lag)
        for porb_initial in porb_range:
        
            for initial_eccentricity in eccen_range:
                binary = create_system(star, planet, 2.0 * numpy.pi / pdisk, porb_initial, initial_eccentricity)
        
                binary.evolve(10.0, 0.1, 1e-6, None)
               # print('====== FINAL STATE ======')
               # print(binary.final_state().format())
               # print('=========================')
                evolution_quantities = ['age',
                                        'semimajor',
                                        'eccentricity',
                                        'envelope_angmom',
                                        'core_angmom',
                                        'wind_saturation',
                                        'planet_angmom']
                evolution = binary.get_evolution(evolution_quantities)
                worb = (2.0 * numpy.pi / binary.orbital_period(evolution.semimajor)
                        /
                        wsun)
                wenv = (evolution.envelope_angmom
                        /
                        binary.primary.envelope_inertia(evolution.age)) / wsun
                wcore = (evolution.core_angmom
                         /
                         binary.primary.core_inertia(evolution.age)) / wsun
        
                planet_inertia = 0.3 * planet.mass * planet.radius**2
        
                #print('Lplanet = ' + repr(evolution.planet_angmom))
        
                wplanet = (evolution.planet_angmom / planet_inertia) / wsun
        
                #print('Wplanet = ' + repr(wplanet))
        
                numpy.savetxt(
                    'Pdisk=%f.evol' % pdisk,
                    numpy.dstack([evolution.age,
                                  worb,
                                  wenv,
                                  wcore,
                                  wplanet,
                                  evolution.wind_saturation])[0],
                    fmt = '%25s',
                    header = ' '.join(
                        ['%25s' % q
                         for q in ['t', 'worb', 'wenv', 'wcore', 'wplanet', 'wind_sat']]
                             )
                    )
                pyplot.plot(evolution.age, evolution.eccentricity, '.')
#        pyplot.loglog(
#            evolution.age,
#            (
#                2.0 * numpy.pi / binary.orbital_period(evolution.semimajor)
#                /
#                wsun
#            ),
#            '-r'
#        )
# =============================================================================
#         pyplot.loglog(evolution.age,  worb, '-' + 'k')
#         pyplot.loglog(evolution.age[evolution.wind_saturation],
#                       wenv[evolution.wind_saturation],
#                       'o' + color,
#                       markerfacecolor=color)
#         pyplot.loglog(
#             evolution.age[numpy.logical_not(evolution.wind_saturation)],
#             wenv[numpy.logical_not(evolution.wind_saturation)],
#             'o' + color,
#             markerfacecolor='none'
#         )
#         pyplot.loglog(evolution.age,
#                       wcore,
#                       '.' + 'c')
#         pyplot.loglog(evolution.age,
#                       wplanet,
#                       '.' + 'g')
# 
# #        wind_sat = numpy.zeros(evolution.age.shape)
# #        wind_sat[evolution.wind_saturation] = wsat_enabled
# #        pyplot.loglog(evolution.age, wind_sat, '.')
# 
#         pyplot.show()
# =============================================================================

            pyplot.title("Eccentricity Evolution for " + str(density) + "g/cm^3 Planet with Initial Orbital Period=" + str(porb_initial))
            pyplot.xlabel("Age (Gyrs)")
            pyplot.ylabel("Eccentricity")
            pyplot.show()

        star.delete()
        planet.delete()
        binary.delete()
    pyplot.loglog([4.6], [1.0], 'o')
    pyplot.loglog([4.6], [1.0], 'x')
#    pyplot.axhline(2.45 / wsun)
#    pyplot.ylim((0.1, 100))

def test_ic_solver(interpolator) :
    """Find initial condition to reproduce some current state and plot."""

    find_ic = InitialConditionSolver(disk_dissipation_age = 5e-3,
                                     evolution_max_time_step = 1e-2)
    target = Structure(age = 5.0,
                       Porb = 3.0,
                       Psurf = 10.0,
                       planet_formation_age = 5e-3)
    star = create_star(interpolator)
    planet = create_planet()
    initial_porb, initial_psurf = find_ic(target = target,
                                          star = star,
                                          planet = planet)
    print('IC: Porb0 = %s, P*0 = %s' % (repr(initial_porb),
                                        repr(initial_psurf)))

if __name__ == '__main__' :
    orbital_evolution_library.read_eccentricity_expansion_coefficients(
        b'/home/annamtetz/repositories/poet/eccentricity_expansion_coef.txt'
    )
    serialized_dir = '../stellar_evolution_interpolators'
    manager = StellarEvolutionManager(serialized_dir)
    interpolator = manager.get_interpolator_by_name('default')
    
    #The file used for this is just a simple tab separated file that includes the density (in g/cm^3)
    #and mass and radius (in solar units) for two theorectical planets.
    #For the purpose of recreating my results the values I used are as follows:

    #rocky planet: density=5.0     mass=0.00002179      radius=0.01832
    #gaseous planet: density=2.0     mass=0.000008716     radius=0.01832

    for line in open(b'/home/annamtetz/repositories/poet/scripts/density_mass_radius.txt'):
        density,mass,radius = line.split()
        density = float(density)
        mass = float(mass)
        radius = float(radius)
        
        if density == 2.0:
            lgQ = 6
        else:
            lgQ = 3

        test_evolution(interpolator, mass, radius, density,
                       planet_phase_lag=phase_lag(lgQ),
                       convective_phase_lag=0.0*phase_lag(6.0))

#    test_ic_solver()
