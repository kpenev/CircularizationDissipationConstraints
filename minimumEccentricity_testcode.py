import math
import planetary_system_io
from minimumEccentricity import TransitingExoplanet



class TestingTransitingExoplanet:
    """ 
    Class for testing Transiting Exoplanet class 

    """

    def __init__(self):
       print('This is for testing TransitingExoplanet class')

    def test_Barnes_data(self):
       """
       Testing TransitingExoplanet by
       """
       planet_id = [01.01, 02.01, 03.01, 04.01, 05.01, 05.02, 07.01, 10.01, 17.01, 18.01, 20.01]
       semi_major_axis = [0.036, 0.039, 0.052, 0.056, 0.058, 0.075, 0.044, 0.047, 0.045, 0.052, 0.056]
       star_radius = [1.06, 2.71, 0.74, 2.60, 1.42, 1.42, 1.27, 1.56, 1.08, 2.02, 1.38]
       x = 1.496*(10**11)/696342000
       semi_major_axis_over_star_radius = [0.036/1.06*x, 0.039/2.71*x, 0.052/0.74*x, 0.056/2.60*x, 0.058/1.42*x, 0.075/1.42*x, 0.044/1.27*x, 0.047/1.56*x, 0.045/1.08*x, 0.052/2.02*x, 0.056/1.38*x]
       orbital_period = [2.471, 2.205, 4.888, 3.849, 4.780, 7.052, 3.214, 3.522, 3.235, 3.548, 4.438]
       transit_duration = [1.732, 3.877, 2.368, 2.928, 2.012, 3.688, 4.111, 3.198, 3.602, 4.081, 4.671] 
       planet_radius = [14.42, 22.29, 04.67, 11.79, 05.65, 00.66, 03.72, 15.88, 11.06, 17.37, 17.58]
       y = 6.3781 * (10**6)/696342000
       planet_radius_over_star_radius = [14.42/1.06*y, 22.29/2.71*y, 04.67/0.74*y, 11.79/2.60*y, 05.65/1.42*y, 00.66/1.42*y, 3.72/1.27*y, 15.88/1.56*y, 11.06/1.08*y, 17.37/2.02*y, 17.58/1.38*y] 
       impact_parameter = [0.822, 0.128, 0.029, 0.946, 0.951, 0.750, 0.640, 0.029, 0.006, 0.018]
       transit_duration_if_circular_orbit = [1.984, 5.810, 2.612, 2.764, 1.716, 3.169, 2.431, 3.682, 3.015, 5.282, 4.338]

       #We are going to reproduce the table 1 of Barnes' paper.
       #Following command is to print the attributes of the table

       #Let's define an array called planet which will store
       #11 instances of class TransitingExoplanet
       #The first element of this array is initialized in the following
       #way:
       planet = []

       #Now we are running a loop to insert records of rest of the planets
       #in the table:

       for i in range(0, (len(planet_id)-1)):
          planet = planet + [TransitingExoplanet(planet_id[i],
                                          semi_major_axis_over_star_radius[i], 0, 0,
                                          orbital_period[i], 0,0,
                                          transit_duration[i], 0,0,
                                          planet_radius_over_star_radius[i], 0,0,
                                          impact_parameter[i],0,0)]
          planet[i].print_attributes()


   

    def test_Nasa_Exoplanet_data(self, path, tolerance):
       
       """
       Testing TransitingExoplanet by Nasa Exoplanet data
       """


       #Data from planets_2020.04.10_14.52.24.csv found in the Data folder



       readPlanet = planetary_system_io.read_nasa_planets(path,
                     eliminate=('SWEEPS-11',
                                'HD 41004 B',
                                'PSR J1719-1438',
                                'K2-22'),
                     need_ages=False,
                     )

       #Now we are taking data on planets from the file

       



       planet_id = readPlanet.pl_name

       semi_major_axis_over_star_radius = readPlanet.pl_ratdor
       semi_major_axis_over_star_radius_upper_uncertainty = readPlanet.pl_ratdorerr1
       semi_major_axis_over_star_radius_lower_uncertainty = readPlanet.pl_ratdorerr2

       orbital_period = readPlanet.pl_orbper
       orbital_period_upper_uncertainty = readPlanet.pl_orbpererr1
       orbital_period_lower_uncertainty = readPlanet.pl_orbpererr2

       transit_duration = readPlanet.pl_trandur
       transit_duration_upper_uncertainty = readPlanet.pl_trandurerr1
       transit_duration_lower_uncertainty = readPlanet.pl_trandurerr2


       planet_radius_over_star_radius = readPlanet.pl_ratror
       planet_radius_over_star_radius_upper_uncertainty = readPlanet.pl_ratrorerr1
       planet_radius_over_star_radius_lower_uncertainty = readPlanet.pl_ratrorerr2

       impact_parameter = readPlanet.pl_imppar
       impact_parameter_upper_uncertainty = readPlanet.pl_impparerr1
       impact_parameter_lower_uncertainty = readPlanet.pl_impparerr2


       planet = []
       j = -1
       for i in range(0, (len(planet_id)-1)):
          if not(math.isnan(semi_major_axis_over_star_radius[i])
                 or math.isnan(semi_major_axis_over_star_radius_upper_uncertainty[i])
                 or math.isnan(semi_major_axis_over_star_radius_lower_uncertainty[i])
                 or math.isnan(orbital_period[i])
                 or math.isnan(orbital_period_upper_uncertainty[i])
                 or math.isnan(orbital_period_lower_uncertainty[i])
                 or math.isnan(transit_duration[i])
                 or math.isnan(transit_duration_upper_uncertainty[i])
                 or math.isnan(transit_duration_lower_uncertainty[i])
                 or math.isnan(planet_radius_over_star_radius[i])
                 or math.isnan(planet_radius_over_star_radius_upper_uncertainty[i])
                 or math.isnan(planet_radius_over_star_radius_lower_uncertainty[i])
                 or math.isnan(impact_parameter[i])
                 or math.isnan(impact_parameter_upper_uncertainty[i])
                 or math.isnan(impact_parameter_lower_uncertainty[i]))and ((semi_major_axis_over_star_radius_upper_uncertainty[i] <= tolerance*semi_major_axis_over_star_radius[i])
                                                                           and (-semi_major_axis_over_star_radius_lower_uncertainty[i] <= tolerance*semi_major_axis_over_star_radius[i])
                                                                           and (orbital_period_upper_uncertainty[i] <= tolerance*orbital_period[i])
                                                                           and (-orbital_period_lower_uncertainty[i] <= tolerance*orbital_period[i])
                                                                           and (transit_duration_upper_uncertainty[i] <= tolerance*transit_duration[i])
                                                                           and (-transit_duration_lower_uncertainty[i] <= tolerance*transit_duration[i])
                                                                           and (planet_radius_over_star_radius_upper_uncertainty[i] <= tolerance*planet_radius_over_star_radius[i])
                                                                           and (-planet_radius_over_star_radius_lower_uncertainty[i] <= tolerance*planet_radius_over_star_radius[i])
                                                                           and (impact_parameter_upper_uncertainty[i] <= tolerance*impact_parameter[i])
                                                                           and (-impact_parameter_lower_uncertainty[i] <= tolerance*impact_parameter[i])):
             j = j + 1
             planet = planet + [TransitingExoplanet(planet_id[i],
                                                    semi_major_axis_over_star_radius[i],
                                                    semi_major_axis_over_star_radius_upper_uncertainty[i],
                                                    semi_major_axis_over_star_radius_lower_uncertainty[i],
                                                    orbital_period[i],
                                                    orbital_period_upper_uncertainty[i],
                                                    orbital_period_lower_uncertainty[i],
                                                    transit_duration[i]*24,
                                                    transit_duration_upper_uncertainty[i]*24,
                                                    transit_duration_lower_uncertainty[i]*24,
                                                    planet_radius_over_star_radius[i],
                                                    planet_radius_over_star_radius_upper_uncertainty[i],
                                                    planet_radius_over_star_radius_lower_uncertainty[i],
                                                    impact_parameter[i],
                                                    impact_parameter_upper_uncertainty[i],
                                                    impact_parameter_lower_uncertainty[i])]
             print('Number: ', (j+1))
             
             planet[j].print_attributes()




test = TestingTransitingExoplanet()
test.test_Barnes_data()
test.test_Nasa_Exoplanet_data(path = 'C:/Users/moham/OneDrive/Documents/GitHub/CircularizationDissipationConstraints/data/planets_2020.04.10_14.52.24.csv', tolerance = 0.1)
test.test_Nasa_Exoplanet_data(path = 'C:/Users/moham/OneDrive/Documents/GitHub/CircularizationDissipationConstraints/data/planets_2019.09.18_13.19.49.csv', tolerance = 0.4)


