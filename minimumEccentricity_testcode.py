import math
import planetary_system_io
from minimumEccentricity import TransitingExoplanet
#______________________________________________________________________________________________________________________________________________________________________________

#Testing the class TransitingExoplanet

#Data on exoplanets found in Barnes' paper's table 1:
planet_id = [01.01, 02.01, 03.01, 04.01, 05.01, 05.02, 07.01, 10.01, 17.01, 18.01, 20.01]
semi_major_axis = [0.036, 0.039, 0.052, 0.056, 0.058, 0.075, 0.044, 0.047, 0.045, 0.052, 0.056]
orbital_period = [2.471, 2.205, 4.888, 3.849, 4.780, 7.052, 3.214, 3.522, 3.235, 3.548, 4.438]
transit_duration = [1.732, 3.877, 2.368, 2.928, 2.012, 3.688, 4.111, 3.198, 3.602, 4.081, 4.671] 
planet_radius = [14.42, 22.29, 04.67, 11.79, 05.65, 00.66, 03.72, 15.88, 11.06, 17.37, 17.58]
star_radius = [1.06, 2.71, 0.74, 2.60, 1.42, 1.42, 1.27, 1.56, 1.08, 2.02, 1.38]*56378*1000 
impact_parameter = [0.822, 0.128, 0.029, 0.946, 0.951, 0.750, 0.640, 0.029, 0.006, 0.018]
transit_duration_if_circular_orbit = [1.984, 5.810, 2.612, 2.764, 1.716, 3.169, 2.431, 3.682, 3.015, 5.282, 4.338]

#We are going to reproduce the table 1 of Barnes' paper.
#Following command is to print the attributes of the table
print('planet_id', '    ', '  semi_major_axis', '    ', '    orbital_period', '    ', '    transit_duration', '    ', '    planet_radius', '   ', ' star_radius', ' ', '    impact_parameter', '    ', '         transit_duration_if_circular_orbit', '    ','             delta','   ','              minimum_eccentricity')

#Let's define an array called planet which will store
#11 instances of class TransitingExoplanet
#The first element of this array is initialized in the following
#way:
planet = []

#Now we are running a loop to insert records of rest of the planets
#in the table:

for i in range(0, (len(planet_id)-1)):
   planet = planet + [TransitingExoplanet(planet_id[i], semi_major_axis[i], orbital_period[i], transit_duration[i], planet_radius[i], star_radius[i], impact_parameter[i])]
   planet[i].print_attributes()

#_______________________________________________________________________________________________________________________________________________________________________________

#Data from planets_2019.09.18_13.19.49.csv found in the Data folder


readPlanet = planetary_system_io.read_nasa_planets('C:/Users/moham/OneDrive/Documents/GitHub/CircularizationDissipationConstraints/data/planets_2019.09.18_13.19.49.csv',
                     eliminate=('SWEEPS-11',
                                'HD 41004 B',
                                'PSR J1719-1438',
                                'K2-22'),
                     need_ages=False,
                     )

#Now we are taking data on planets from the file
planet_id = readPlanet.pl_name
semi_major_axis = readPlanet.pl_orbsmax
orbital_period = readPlanet.pl_orbper
transit_duration = readPlanet.pl_trandur
planet_radius = readPlanet.pl_rade
star_radius = readPlanet.st_rad
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
        
#____________________________________________________________________________________________________________________________________________________________________________    
#Data from updated_planets_koi_2020.03.16_21.25.09.csv found in the Data folder
print('Now Printing results from updated_planets_koi_2020.03.16_21.25.09.csv')

readPlanet = planetary_system_io.read_nasa_planets('C:/Users/moham/OneDrive/Documents/GitHub/CircularizationDissipationConstraints/data/updated_planets_koi_2020.03.16_21.25.09.csv',
                     eliminate=('SWEEPS-11',
                                'HD 41004 B',
                                'PSR J1719-1438',
                                'K2-22'),
                     need_ages=False,
                     )


#Now we are taking data on planets from the file

planet_id = readPlanet.pl_name
semi_major_axis = readPlanet.pl_orbsmax
orbital_period = readPlanet.pl_orbper
transit_duration = readPlanet.pl_trandur
planet_radius = readPlanet.pl_rade
star_radius = readPlanet.st_rad
impact_parameter = readPlanet.pl_imppar

#Following command is to print the attributes of the table
print('Planet Name', '    ', '  semi_major_axis', '    ', '    orbital_period', '    ', '    transit_duration', '    ', '    planet_radius', '   ', ' star_radius', ' ', '    impact_parameter', '    ', '         transit_duration_if_circular_orbit', '    ','             delta','   ','              emin')


planet = []
j = -1
for i in range(0, (len(planet_id)-1)):
    if not(math.isnan(semi_major_axis[i]) or math.isnan(orbital_period[i])or math.isnan(transit_duration[i])or math.isnan(planet_radius[i]) or math.isnan(star_radius[i]) or math.isnan(impact_parameter[i])):
        j = j + 1
        planet = planet + [TransitingExoplanet(planet_id[i], semi_major_axis[i], orbital_period[i], transit_duration[i], planet_radius[i], star_radius[i], impact_parameter[i])]
        planet[j].print_attributes()

#_______________________________________________________________________________________________________________________________________________________________________________

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
orbital_period = readPlanet.pl_orbper
transit_duration = readPlanet.pl_trandur
planet_radius = readPlanet.pl_rade
star_radius = readPlanet.st_rad
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
