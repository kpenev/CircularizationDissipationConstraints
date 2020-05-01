import math
import planetary_system_io
class TransitingExoplanet:
    """ 
    This is a class for calculating the minimum eccentricity
    of the orbit of the transiting exoplanet.

    A transiting exoplanet is defined by following
    attributes: planet's id (planet_id), length of semimajor axis (semi_major_axis),
    orbital period (orbital_period), planet's radius (planet_radius),
    parent star's radius (star_radius), impact parameter (impact_parameter)
    actual transit duration (transit_duration), transit duration if its
    orbit were circular (transit_duration_if_circular_orbit)
      
    Attributes: 
        planet_id                            (string): The identification label of the exoplanet 
        semi_major_axis                       (float): The length of the semi-major axis of the exoplanet
        orbital_period                        (float): Orbital period in days
        transit_duration                      (float): Transit time in hours
        planet_radius                         (float): Radius of the planet in the unit of the Earth's radius, EARTH_RADIUS
        star_radius                           (float): Radius of the star in the unit of the Sun's radius, SOLAR_RADIUS
        impact_parameter                      (float): Impact parameter in the unit of the parent star's radius
        transit_duration_if_circular_orbit    (float): Duration for circular orbit in hours
        minimum_eccentricity                  (float): Minimum eccentricity of the exoplanet
        delta                                 (float): Delta factor
        SOLAR_RADIUS                          (float): The radius of the Sun in meters. This is a constant quantity
        EARTH_RADIUS                          (float): The radius of the Earth in meters. This is a constant quantity
        AU                                    (float): Astronomical Unit in meters
        SOLAR_RADIUS_OVER_AU                  (float): The Sun's radius divided by 1 AU
        EARTH_RADIUS_OVER_SOLAR_RADIUS        (float): The Earth's radius divided by the Sun's radius

    Methods:
        find_transit_duration_if_circular_orbit   : Works out the transit duration, in hours, if the orbit were circular 
        find_delta                                : Works out the delta factor
        find_emin                                 : Works out the minimum eccentricity of the orbit of the exoplanet
        print_attributes                          : Prints out the attributes of the exoplanet  
        
    """
    
      
    SOLAR_RADIUS = 6.96 * pow(10,8)
    EARTH_RADIUS = 6378*1000
    AU = 1.496* pow(10,11)
    SOLAR_RADIUS_OVER_AU = 0.00465 #is the Sun's radius divided by 1AU
    EARTH_RADIUS_OVER_SOLAR_RADIUS = 0.00916 #is the Earth's radius divided by the Sun's radius

    
    
    def __init__(self, planet_id, semi_major_axis, orbital_period, transit_time, planet_radius, star_radius, impact_parameter):

        """ 
        The constructor for MinimumEccentricityOfTheOrbitOfTheTransitingExoplanet class.
        It takes the below mentioned attributes of the exoplanet as paramenters and initializes:
        duration for circular orbit, transit_duration_if_circular_orbit in the unit of days
        delta factor, delta which is dimensionless
        minimum eccentricity of the orbit, minimum_eccentricity which is dimensionless
        
  
        Parameters: 
           planet_id           (string): The identification label of the exoplanet 
           semi_major_axis     (float): Semi-major axis of the orbit of the exoplanet in Astronomical Unit (AU)
           orbital_period      (float): Orbital period of the exoplanet in days
           transit_time        (float): Transit time of the exoplanet in hours
           planet_radius       (float): Radius of the exoplanet in the Earth's radius, EARTH_RADIUS
           star_radius         (float): Radius of the parent star in the Sun's radius, SOLAR_RADIUS
           impact_parameter    (float): Impact parameter in the unit of the parent star's radius
           
        """

        self.planet_id = planet_id
        self.semi_major_axis = semi_major_axis 
        self.orbital_period = orbital_period 
        self.transit_duration =  transit_time   
        self.planet_radius = planet_radius 
        self.star_radius = star_radius 
        self.impact_parameter = impact_parameter 
        self.transit_duration_if_circular_orbit = self.find_transit_duration_if_circular_orbit(self.star_radius, self.planet_radius, self.impact_parameter, self.semi_major_axis, self.orbital_period) 
        self.delta = self.find_delta(self.transit_duration, self.transit_duration_if_circular_orbit) 
        self.minimum_eccentricity = self.find_emin(self.delta) 

       
    def find_transit_duration_if_circular_orbit(self, star_radius, planet_radius, impact_parameter, semi_major_axis, orbital_period):

        """ 
        The function to work out the transit duration if the orbit were circular, transit_duration_if_circular_orbit in the unit of days. 
  
        Parameters: 
            star_radius      (float): Radius of the parent star in the unit of the Sun's radius, SOLAR_RADIUS
            planet_radius    (float): Radius of the exoplanet in the unit of the Earth's radius, EARTH_RADIUS
            impact_parameter (float): Impact parameter in the unit of the parent star's radius
            semi_major_axis  (float): Length of the semi-major axis of the exoplanet's orbit in Astronomical Unit, AU
            orbital_period   (float): Orbital period in days
          
        Returns: 
            tc               (float): Transit Duration, in hours, if the orbit were circular
            
        """

        
        tc = (1+planet_radius/star_radius * self.EARTH_RADIUS_OVER_SOLAR_RADIUS)        
        tc = tc*tc - impact_parameter*impact_parameter
        tc = math.sqrt(tc)        
        tc = tc * star_radius/3.1416/semi_major_axis * orbital_period * self.SOLAR_RADIUS_OVER_AU * 24 #last 24 is for converting 1 day to 24 hours       
        return tc #in hours

    

    def find_delta(self, transit_time, transitTimeForCircularOrbit):
        """ 
        The function to calculate delta factor, i.e. the ratio of the transit time
        to the transit time for circular orbit.
  
        Parameters: 
            transit_time                 (float): The transit time of the exoplanet
            transitTimeForCircularOrbit (float): The transit time for circular orbit
          
        Returns: 
            transit_time/transitTimeForCircularOrbit (float): delta 
        """
        return transit_time/transitTimeForCircularOrbit
    
        
    def find_emin(self, delta):
        """ 
        The function to calculate minimum eccentricity of the exoplanet's orbit
  
        Parameters: 
            delta (float): Delta factor, i.e. the ratio of the transit time to the transit time for circular orbit.
                      
        Returns: 
            minimum eccentricity of the exoplanet's orbit (float)
        """
        
        return abs((delta * delta - 1)/(delta * delta+1))

    
    def print_attributes(self):
        """ 
        The function to print attributes of exoplanet
          
        """
        print(self.planet_id, '   ', self.semi_major_axis, '    ', self.orbital_period, '    ', self.transit_duration, '    ', self.planet_radius, '  ', self.star_radius, '  ', self.impact_parameter, '    ', self.transit_duration_if_circular_orbit, '  ', self.delta, '    ', self.minimum_eccentricity)

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
