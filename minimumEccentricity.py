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
        semi_major_axis                       (float): The length of the semi-major axis of the exoplanet in AU
        semi_major_axis_upper_uncertainty     (float): The upper uncertainty associated with the length of the semi-major axis of the exoplanet in AU
        semi_major_axis_lower_uncertainty     (float): The lower uncertainty associated with the length of the semi-major axis of the exoplanet in AU
        orbital_period                        (float): Orbital period in days
        orbital_period_upper_uncertainty      (float): The upper uncertainty associated with the orbital period in days
        orbital_period_lower_uncertainty      (float): The lower uncertainty associated with the orbital period in days
        
        transit_duration                      (float): Transit duration in hours
        transit_duration_upper_uncertainty    (float): The upper uncertainty associated with the transit duration in hours
        transit_duration_lower_uncertainty    (float): The lower uncertainty associated with the transit duration in hours


        planet_radius                         (float): Radius of the planet in the unit of the Earth's radius, EARTH_RADIUS
        planet_radius_upper_uncertainty       (float): The upper uncertainty associated with the radius of the planet in the unit of the Earth's radius, EARTH_RADIUS
        planet_radius_lower_uncertainty       (float): The lower uncertainty associated with the radius of the planet in the unit of the Earth's radius, EARTH_RADIUS
        
        star_radius                           (float): Radius of the star in the unit of the Sun's radius, SOLAR_RADIUS
        star_radius_upper_uncertainty         (float): The upper uncertainty associated with the radius of the star in the unit of the Sun's radius, SOLAR_RADIUS
        star_radius_lower_uncertainty         (float): The lower uncertainty associated with the radius of the star in the unit of the Sun's radius, SOLAR_RADIUS


        
        impact_parameter                      (float): Impact parameter in the unit of the parent star's radius
        impact_parameter_upper_uncertainty    (float): The upper uncertainty associated with the impact parameter in the unit of the parent star's radius
        impact_parameter_lower_uncertainty    (float): The lower uncertainty associated with the impact parameter in the unit of the parent star's radius

        
        transit_duration_if_circular_orbit                      (float): Transit duration in hours if the orbit is circular 
        transit_duration_if_circular_orbit_upper_uncertainty    (float): The upper uncertainty associated with the transit duration in hours if the orbit is circular
        transit_duration_if_circular_orbit_lower_uncertainty    (float): The lower uncertainty associated with the transit duration in hours if the orbit is circular

        
        minimum_eccentricity                        (float): Minimum eccentricity of the exoplanet
        minimum_eccentricity_upper_uncertainty      (float): The upper uncertainty associated with the minimum eccentricity of the exoplanet
        minimum_eccentricity_lower_uncertainty      (float): The lower uncertainty associated with the minimum eccentricity of the exoplanet

        
        delta                                 (float): Delta factor
        delta_upper_uncertainty               (float): The upper uncertainty associated with the delta factor
        delta_lower_uncertainty               (float): The lower uncertainty associated with the delta factor
        
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
    
      
    SOLAR_RADIUS = 696342000
    SOLAR_RADIUS_UPPER_UNCERTAINTY = 65000
    SOLAR_RADIUS_LOWER_UNCERTAINTY = 65000
    
    EARTH_RADIUS = 6378000
    EARTH_RADIUS_UPPER_UNCERTAINTY = 0
    EARTH_RADIUS_LOWER_UNCERTAINTY = 0


    
    AU = 149600000000
    SOLAR_RADIUS_OVER_AU = SOLAR_RADIUS / AU #is the Sun's radius divided by 1AU
    SOLAR_RADIUS_OVER_AU_UPPER_UNCERTAINTY = SOLAR_RADIUS_UPPER_UNCERTAINTY / AU
    SOLAR_RADIUS_LOWER_AU_UPPER_UNCERTAINTY = SOLAR_RADIUS_LOWER_UNCERTAINTY / AU
    
    EARTH_RADIUS_OVER_SOLAR_RADIUS = EARTH_RADIUS / SOLAR_RADIUS #is the Earth's radius divided by the Sun's radius
    EARTH_RADIUS_OVER_SOLAR_RADIUS_UPPER_UNCERTAINTY = (EARTH_RADIUS + EARTH_RADIUS_UPPER_UNCERTAINTY)/(SOLAR_RADIUS - SOLAR_RADIUS_LOWER_AU_UPPER_UNCERTAINTY)-EARTH_RADIUS_OVER_SOLAR_RADIUS
    EARTH_RADIUS_OVER_SOLAR_RADIUS_LOWER_UNCERTAINTY = EARTH_RADIUS_OVER_SOLAR_RADIUS - (EARTH_RADIUS - EARTH_RADIUS_LOWER_UNCERTAINTY)/(SOLAR_RADIUS + SOLAR_RADIUS_UPPER_AU_UPPER_UNCERTAINTY)

    
    
    def __init__(self,
                 planet_id,
                 semi_major_axis,
                 semi_major_axis_upper_uncertainty,
                 semi_major_axis_lower_uncertainty,
                 orbital_period,
                 orbital_period_upper_uncertainty,
                 orbital_period_lower_uncertainty,
                 transit_duration,
                 transit_duration_upper_uncertainty,
                 transit_duration_lower_uncertainty,
                 planet_radius,
                 planet_radius_upper_uncertainty,
                 planet_radius_lower_uncertainty                 
                 star_radius,
                 star_radius_upper_uncertainty,
                 star_radius_lower_uncertainty,
                 impact_parameter,
                 impact_parameter_upper_uncertainty,
                 impact_parameter_lower_uncertainty):

        """ 
        The constructor for TransitingExoplanet class.
        It takes the below mentioned attributes of the exoplanet as paramenters and initializes:
        transit duration, in hours, if the orbit were circular (transit_duration_if_circular_orbit) 
        delta factor (delta) which is dimensionless
        minimum eccentricity of the orbit (minimum_eccentricity) which is dimensionless
        
  
        Parameters: 
           planet_id           (string): The identification label of the exoplanet 
           semi_major_axis     (float): Semi-major axis of the orbit of the exoplanet in Astronomical Unit (AU)
           orbital_period      (float): Orbital period of the exoplanet in days
           transit_duration    (float): Transit duration of the exoplanet in hours
           planet_radius       (float): Radius of the exoplanet in the Earth's radius, EARTH_RADIUS
           star_radius         (float): Radius of the parent star in the Sun's radius, SOLAR_RADIUS
           impact_parameter    (float): Impact parameter in the unit of the parent star's radius
           
        """

        self.planet_id = planet_id
        self.semi_major_axis = semi_major_axis 
        self.orbital_period = orbital_period 
        self.transit_duration =  transit_duration   
        self.planet_radius = planet_radius 
        self.star_radius = star_radius 
        self.impact_parameter = impact_parameter 
        self.transit_duration_if_circular_orbit = self.find_transit_duration_if_circular_orbit(self.star_radius,
                                                                                               self.planet_radius,
                                                                                               self.impact_parameter,
                                                                                               self.semi_major_axis,
                                                                                               self.orbital_period) 
        self.delta = self.find_delta(self.transit_duration, self.transit_duration_if_circular_orbit) 
        self.minimum_eccentricity = self.find_emin(self.delta) 

       
    def find_transit_duration_if_circular_orbit(self,
                                                star_radius,
                                                planet_radius,
                                                impact_parameter,
                                                semi_major_axis,
                                                orbital_period):

        """ 
        The function to work out the transit duration if the orbit were circular
        (transit_duration_if_circular_orbit) in the unit of hours. 
  
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
        tc = tc * star_radius/3.1416/semi_major_axis * orbital_period * self.SOLAR_RADIUS_OVER_AU * 24
        #last 24 is for converting 1 day to 24 hours
        
        return tc #in hours

    

    def find_delta(self, transit_time, transitTimeForCircularOrbit):
        """ 
        The function to calculate delta factor, i.e. the ratio of the transit time
        to the transit time for circular orbit.
  
        Parameters: 
            transit_time                 (float): The transit time of the exoplanet
            transitTimeForCircularOrbit  (float): The transit time for circular orbit
          
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
        print(self.planet_id, '   ', self.semi_major_axis, '    ', self.orbital_period, '    ',
              self.transit_duration, '    ', self.planet_radius, '  ', self.star_radius, '  ',
              self.impact_parameter, '    ', self.transit_duration_if_circular_orbit, '  ',
              self.delta, '    ', self.minimum_eccentricity)

