import math
import planetary_system_io
from minimumEccentricity import TransitingExoplanet


#Data from planets_2020.04.10_14.52.24.csv found in the Data folder


readPlanet = planetary_system_io.read_nasa_planets('C:/Users/moham/OneDrive/Documents/GitHub/CircularizationDissipationConstraints/data/planets_2020.04.10_14.52.24.csv',
                     eliminate=('SWEEPS-11',
                                'HD 41004 B',
                                'PSR J1719-1438',
                                'K2-22'),
                     need_ages=False,
                     )

#Now we are taking data on planets from the file



planet_id = readPlanet.pl_name

semi_major_axis = readPlanet.pl_orbsmax
semi_major_axis_upper_uncertainty = readPlanet.pl_orbsmaxerr1
semi_major_axis_lower_uncertainty = readPlanet.pl_orbsmaxerr2

orbital_period = readPlanet.pl_orbper
orbital_period_upper_uncertainty = readPlanet.pl_orbpererr1
orbital_period_lower_uncertainty = readPlanet.pl_orbpererr2

transit_duration = readPlanet.pl_trandur
transit_duration_upper_uncertainty = readPlanet.pl_trandurerr1
transit_duration_lower_uncertainty = readPlanet.pl_trandurerr2


planet_radius = readPlanet.pl_rade
planet_radius_upper_uncertainty = readPlanet.pl_radeerr1
planet_radius_lower_uncertainty = readPlanet.pl_radeerr2

star_radius = readPlanet.st_rad
star_radius_upper_uncertainty = readPlanet.st_raderr1
star_radius_lower_uncertainty = readPlanet.st_raderr2

impact_parameter = readPlanet.pl_imppar

#Following command is to print the attributes of the table
print('Planet Name', '    ', '  semi_major_axis', '    ', '    orbital_period', '    ', '    transit_duration', '    ', '    planet_radius', '   ', ' star_radius', ' ', '    impact_parameter', '    ', '         transit_duration_if_circular_orbit', '    ','             delta','   ','              emin')


planet = []
j = -1
for i in range(0, (len(planet_id)-1)):
    if not(math.isnan(semi_major_axis[i]) or math.isnan(orbital_period[i])or math.isnan(transit_duration[i])or math.isnan(planet_radius[i]) or math.isnan(star_radius[i]) or math.isnan(impact_parameter[i])):
        j = j + 1
        planet = planet + [TransitingExoplanet(planet_id[i], semi_major_axis[i], orbital_period[i], transit_duration[i]*24, planet_radius[i], star_radius[i], impact_parameter[i])]
        planet[j].print_attributes()
