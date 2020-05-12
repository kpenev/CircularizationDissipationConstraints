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
    SOLAR_RADIUS_LOWER_UNCERTAINTY = -65000
    
    EARTH_RADIUS = 6378000
    EARTH_RADIUS_UPPER_UNCERTAINTY = 0
    EARTH_RADIUS_LOWER_UNCERTAINTY = 0


    
    AU = 149600000000
    SOLAR_RADIUS_OVER_AU = SOLAR_RADIUS / AU #is the Sun's radius divided by 1AU
    SOLAR_RADIUS_OVER_AU_UPPER_UNCERTAINTY = SOLAR_RADIUS_UPPER_UNCERTAINTY / AU
    SOLAR_RADIUS_OVER_AU_LOWER_UNCERTAINTY = - SOLAR_RADIUS_LOWER_UNCERTAINTY / AU
    
    EARTH_RADIUS_OVER_SOLAR_RADIUS = EARTH_RADIUS / SOLAR_RADIUS #is the Earth's radius divided by the Sun's radius
    EARTH_RADIUS_OVER_SOLAR_RADIUS_UPPER_UNCERTAINTY = (EARTH_RADIUS + EARTH_RADIUS_UPPER_UNCERTAINTY)/(SOLAR_RADIUS + SOLAR_RADIUS_LOWER_UNCERTAINTY)-EARTH_RADIUS_OVER_SOLAR_RADIUS
    EARTH_RADIUS_OVER_SOLAR_RADIUS_LOWER_UNCERTAINTY = -(EARTH_RADIUS_OVER_SOLAR_RADIUS - (EARTH_RADIUS + EARTH_RADIUS_LOWER_UNCERTAINTY)/(SOLAR_RADIUS + SOLAR_RADIUS_UPPER_UNCERTAINTY))

    
    
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
                 planet_radius_lower_uncertainty,                 
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
           
        """

        self.planet_id = planet_id
        self.semi_major_axis = semi_major_axis
        self.semi_major_axis_upper_uncertainty = semi_major_axis_upper_uncertainty
        self.semi_major_axis_lower_uncertainty = semi_major_axis_lower_uncertainty
        self.orbital_period = orbital_period
        self.orbital_period_upper_uncertainty = orbital_period_upper_uncertainty
        self.orbital_period_lower_uncertainty = orbital_period_lower_uncertainty
        self.transit_duration =  transit_duration
        self.transit_duration_upper_uncertainty = transit_duration_upper_uncertainty
        self.transit_duration_lower_uncertainty = transit_duration_lower_uncertainty
        self.planet_radius = planet_radius
        self.planet_radius_upper_uncertainty = planet_radius_upper_uncertainty
        self.planet_radius_lower_uncertainty = planet_radius_lower_uncertainty
        self.star_radius = star_radius
        self.star_radius_upper_uncertainty = star_radius_upper_uncertainty
        self.star_radius_lower_uncertainty = star_radius_lower_uncertainty
        self.impact_parameter = impact_parameter
        self.impact_parameter_upper_uncertainty = impact_parameter_upper_uncertainty
        self.impact_parameter_lower_uncertainty = impact_parameter_lower_uncertainty
        self.transit_duration_if_circular_orbit = self.find_transit_duration_if_circular_orbit(self.star_radius,
                                                                                               self.planet_radius,
                                                                                               self.impact_parameter,
                                                                                               self.semi_major_axis,
                                                                                               self.orbital_period)
        self.transit_duration_upper_uncertainty_if_circular_orbit = -self.transit_duration_if_circular_orbit + self.find_transit_duration_upper_limit_if_circular_orbit(
                                                            self.star_radius,
                                                            self.star_radius_upper_uncertainty,                                                            
                                                            self.planet_radius,
                                                            self.planet_radius_upper_uncertainty,                                                            
                                                            self.impact_parameter,
                                                            self.impact_parameter_lower_uncertainty,
                                                            self.semi_major_axis,
                                                            self.semi_major_axis_lower_uncertainty,
                                                            self.orbital_period,
                                                            self.orbital_period_upper_uncertainty)
        self.transit_duration_lower_uncertainty_if_circular_orbit = -self.transit_duration_if_circular_orbit + self.find_transit_duration_lower_limit_if_circular_orbit(
                                                            self.star_radius,                                                            
                                                            self.star_radius_lower_uncertainty,
                                                            self.planet_radius,                                                            
                                                            self.planet_radius_lower_uncertainty,
                                                            self.impact_parameter,
                                                            self.impact_parameter_upper_uncertainty,                                                            
                                                            self.semi_major_axis,
                                                            self.semi_major_axis_upper_uncertainty,
                                                            self.orbital_period,
                                                            self.orbital_period_lower_uncertainty)
        self.delta = self.find_delta(self.transit_duration, self.transit_duration_if_circular_orbit)
        self.delta_upper_uncertainty = - self.delta + self.find_delta_upper_limit(self.transit_duration,
                                                                                  self.transit_duration_if_circular_orbit,
                                                                                  self.transit_duration_upper_uncertainty,
                                                                                  self.transit_duration_lower_uncertainty_if_circular_orbit)
        self.delta_lower_uncertainty = -self.delta + self.find_delta_lower_limit(self.transit_duration,
                                                                                 self.transit_duration_if_circular_orbit,
                                                                                 self.transit_duration_lower_uncertainty,
                                                                                 self.transit_duration_upper_uncertainty_if_circular_orbit) 
        self.minimum_eccentricity = self.find_emin(self.delta)
        self.minimum_eccentricity_upper_uncertainty = self.find_emin_upper_limit(self.delta, self.delta_upper_uncertainty, self.delta_lower_uncertainty) - self.minimum_eccentricity
        self.minimum_eccentricity_lower_uncertainty = self.find_emin_lower_limit(self.delta, self.delta_upper_uncertainty, self.delta_lower_uncertainty) - self.minimum_eccentricity

        #if self.minimum_eccentricity + self.minimum_eccentricity_lower_uncertainty <0:
        #    self.minimum_eccentricity_lower_uncertainty = self.minimum_eccentricity
            
    


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
        tc = max(tc, 0)
        tc = math.sqrt(tc)        
        tc = tc * star_radius/3.1416/semi_major_axis * orbital_period * self.SOLAR_RADIUS_OVER_AU * 24
        #last 24 is for converting 1 day to 24 hours
        
        return tc #in hours

    def find_transit_duration_upper_limit_if_circular_orbit(self,
                                                            star_radius,
                                                            star_radius_upper_uncertainty,                                                           
                                                            planet_radius,
                                                            planet_radius_upper_uncertainty,                                                            
                                                            impact_parameter,
                                                            impact_parameter_lower_uncertainty,
                                                            semi_major_axis,
                                                            semi_major_axis_lower_uncertainty,
                                                            orbital_period,
                                                            orbital_period_upper_uncertainty):
        """ 
        The function to work out the upper uncertainty limit of the transit duration if the orbit were circular
        (transit_duration_if_circular_orbit) in the unit of hours. 
  
        Parameters: 
            star_radius                           (float): Radius of the star in the unit of the Sun's radius, SOLAR_RADIUS
            star_radius_upper_uncertainty         (float): The upper uncertainty associated with the radius of the star in the unit of the Sun's radius, SOLAR_RADIUS
              
            planet_radius                         (float): Radius of the planet in the unit of the Earth's radius, EARTH_RADIUS
            planet_radius_upper_uncertainty       (float): The upper uncertainty associated with the radius of the planet in the unit of the Earth's radius, EARTH_RADIUS
            
            impact_parameter                      (float): Impact parameter in the unit of the parent star's radius
            impact_parameter_lower_uncertainty    (float): The lower uncertainty associated with the impact parameter in the unit of the parent star's radius
            
            semi_major_axis                       (float): The length of the semi-major axis of the exoplanet in AU
            
            semi_major_axis_lower_uncertainty     (float): The lower uncertainty associated with the length of the semi-major axis of the exoplanet in AU
            orbital_period                        (float): Orbital period in days
            orbital_period_upper_uncertainty      (float): The upper uncertainty associated with the orbital period in days
             
          
        Returns: 
            tc                                    (float): Upper limit of the transit Duration, in hours, if the orbit were circular
            
        """

        tc = (1+(planet_radius + planet_radius_upper_uncertainty)/(star_radius + star_radius_upper_uncertainty) * (self.EARTH_RADIUS_OVER_SOLAR_RADIUS + self.EARTH_RADIUS_OVER_SOLAR_RADIUS_UPPER_UNCERTAINTY))        
        b = max((impact_parameter + impact_parameter_lower_uncertainty),0)
        tc = max(tc*tc - b*b,0)
        tc = math.sqrt(tc)        
        tc = tc * (star_radius + star_radius_upper_uncertainty)/3.1416/(semi_major_axis + semi_major_axis_lower_uncertainty) * (orbital_period + orbital_period_upper_uncertainty) * (self.SOLAR_RADIUS_OVER_AU + self.SOLAR_RADIUS_OVER_AU_UPPER_UNCERTAINTY) * 24
        #last 24 is for converting 1 day to 24 hours
        
        return tc #in hours

    def find_transit_duration_lower_limit_if_circular_orbit(self,
                                                            star_radius,                                                            
                                                            star_radius_lower_uncertainty,
                                                            planet_radius,                                                            
                                                            planet_radius_lower_uncertainty,
                                                            impact_parameter,
                                                            impact_parameter_upper_uncertainty,                                                            
                                                            semi_major_axis,
                                                            semi_major_axis_upper_uncertainty,
                                                            orbital_period,
                                                            orbital_period_lower_uncertainty):
        """ 
        The function to work out the lower uncertainty limit of the transit duration if the orbit were circular
        (transit_duration_if_circular_orbit) in the unit of hours. 
  
        Parameters: 
            star_radius                           (float): Radius of the star in the unit of the Sun's radius, SOLAR_RADIUS
            
            star_radius_lower_uncertainty         (float): The lower uncertainty associated with the radius of the star in the unit of the Sun's radius, SOLAR_RADIUS  
            planet_radius                         (float): Radius of the planet in the unit of the Earth's radius, EARTH_RADIUS
            
            planet_radius_lower_uncertainty       (float): The lower uncertainty associated with the radius of the planet in the unit of the Earth's radius, EARTH_RADIUS
            impact_parameter                      (float): Impact parameter in the unit of the parent star's radius
            impact_parameter_upper_uncertainty    (float): The upper uncertainty associated with the impact parameter in the unit of the parent star's radius
            
            semi_major_axis                       (float): The length of the semi-major axis of the exoplanet in AU
            semi_major_axis_upper_uncertainty     (float): The upper uncertainty associated with the length of the semi-major axis of the exoplanet in AU
            orbital_period                        (float): Orbital period in days
            orbital_period_lower_uncertainty      (float): The lower uncertainty associated with the orbital period in days 
          
        Returns: 
            tc                                    (float): Upper limit of the transit Duration, in hours, if the orbit were circular
            
        """

        tc = (1+(planet_radius + planet_radius_lower_uncertainty)/(star_radius + star_radius_lower_uncertainty)
              * (self.EARTH_RADIUS_OVER_SOLAR_RADIUS + self.EARTH_RADIUS_OVER_SOLAR_RADIUS_LOWER_UNCERTAINTY))
        b = min(impact_parameter + impact_parameter_upper_uncertainty, 1)
        tc = max(tc*tc - b*b,0)
        tc = math.sqrt(tc)        
        tc = tc * (star_radius + star_radius_lower_uncertainty)/3.1416/(semi_major_axis + semi_major_axis_upper_uncertainty) * (orbital_period + orbital_period_lower_uncertainty) * (self.SOLAR_RADIUS_OVER_AU + self.SOLAR_RADIUS_OVER_AU_LOWER_UNCERTAINTY) * 24
        #last 24 is for converting 1 day to 24 hours
        
        return tc #in hours
        
    

    

    def find_delta(self, transit_duration, transit_duration_if_circular_orbit):
        """ 
        The function to calculate delta factor, i.e. the ratio of the transit duration
        to the transit duration if the orbit were circular.
  
        Parameters: 
            transit_duration                    (float): The transit duration of the exoplanet
            transit_duration_if_circular_orbit  (float): The transit duration if the orbit were circular
          
        Returns: 
            transit_duration/transit_duration_if_circular_orbit (float): delta 
        """
        return transit_duration/transit_duration_if_circular_orbit

    def find_delta_upper_limit(self, transit_duration, transit_duration_if_circular_orbit, transit_duration_upper_uncertainty, transit_duration_lower_uncertainty_if_circular_orbit ):
        """ 
        The function to calculate the upper uncertainty limit of the delta factor, i.e. the ratio of the transit duration
        to the transit duration if the orbit were circular.
  
        Parameters: 
            transit_duration                    (float): The transit duration of the exoplanet
            transit_duration_if_circular_orbit  (float): The transit duration if the orbit were circular
            transit_duration_upper_uncertainty  (float): The upper uncertainty of the transit duration
            transit_duration_lower_uncertainty_if_circular_orbit  (float): The lower uncertainty of the transit duration if the circuit were circular
        Returns: 
            (transit_duration + transit_duration_upper_uncertainty)/(transit_duration_if_circular_orbit - transit_duration_lower_uncertainty_if_circular_orbit) (float): delta 
        """

        return (transit_duration + transit_duration_upper_uncertainty)/(transit_duration_if_circular_orbit + transit_duration_lower_uncertainty_if_circular_orbit)

    def find_delta_lower_limit(self, transit_duration, transit_duration_if_circular_orbit, transit_duration_lower_uncertainty, transit_duration_upper_uncertainty_if_circular_orbit ):
        """ 
        The function to calculate the lower uncertainty limit of the delta factor, i.e. the ratio of the transit duration
        to the transit duration if the orbit were circular.
  
        Parameters: 
            transit_duration                    (float): The transit duration of the exoplanet
            transit_duration_if_circular_orbit  (float): The transit duration if the orbit were circular
            transit_duration_lower_uncertainty  (float): The lower uncertainty of the transit duration
            transit_duration_upper_uncertainty_if_circular_orbit  (float): The upper uncertainty of the transit duration if the circuit were circular
        Returns: 
            (transit_duration - transit_duration_lower_uncertainty)/(transit_duration_if_circular_orbit + transit_duration_upper_uncertainty_if_circular_orbit) (float): delta 
        """
        return (transit_duration + transit_duration_lower_uncertainty)/(transit_duration_if_circular_orbit + transit_duration_upper_uncertainty_if_circular_orbit)
    
        
    def find_emin(self, delta):
        """ 
        The function to calculate minimum eccentricity of the exoplanet's orbit
  
        Parameters: 
            delta (float): Delta factor, i.e. the ratio of the transit time to the transit time for circular orbit.
                      
        Returns: 
            minimum eccentricity of the exoplanet's orbit (float)
        """
        print('emin = ',abs((delta * delta - 1)/(delta * delta+1)))
        return abs((delta * delta - 1)/(delta * delta+1))

    def find_emin_upper_limit(self, delta, delta_upper_uncertainty, delta_lower_uncertainty):
        """ 
        The function to calculate the upper uncertainty limit of the minimum eccentricity of the exoplanet's orbit
  
        Parameters: 
            delta (float): Delta factor, i.e. the ratio of the transit time to the transit time for circular orbit.
            delta_lower_uncertainty (float): The lower uncertainty of the delta factor          
        Returns: 
            minimum eccentricity of the exoplanet's orbit (float)
        """
        if abs(delta)<1:
            d = delta + delta_lower_uncertainty
            
        else:
            d = delta + delta_upper_uncertainty

        print('emin upper = ', abs((d*d-1)/(d*d+1)))

        return abs((d*d-1)/(d*d+1))
        
        
    def find_emin_lower_limit(self, delta, delta_upper_uncertainty, delta_lower_uncertainty):
        """ 
        The function to calculate the lower uncertainty limit of the minimum eccentricity of the exoplanet's orbit
  
        Parameters: 
            delta (float): Delta factor, i.e. the ratio of the transit time to the transit time for circular orbit.
            delta_upper_uncertainty (float): The upper uncertainty of the delta factor          
        Returns: 
            minimum eccentricity of the exoplanet's orbit (float)
        """

        if abs(delta)<1:
            d = delta + delta_upper_uncertainty
            
        else:
            d = delta + delta_lower_uncertainty
        print('emin lower = ', abs((d*d-1)/(d*d+1)))
        return abs((d*d-1)/(d*d+1))

        
    def print_attributes(self):
        """ 
        The function to print attributes of exoplanet
          
        """
        print('Planet ID: ',self.planet_id,
              '\n Transit duration if the orbit were circular: ', self.transit_duration_if_circular_orbit,
              '\n Upper uncertainty of transit duration if the orbit were circular: ', self.transit_duration_upper_uncertainty_if_circular_orbit,
              '\n Lower uncertainty of transit duration if the orbit were circular: ', self.transit_duration_lower_uncertainty_if_circular_orbit, 
              '\n Delta: ', self.delta,
              '\n Upper uncertainty of Delta: ', self.delta_upper_uncertainty,
              '\n Lower uncertainty of Delta: ', self.delta_lower_uncertainty,
              '\n Minimum Eccentricity: ', self.minimum_eccentricity,
              '\n Upper uncertainty of minimum eccentricity: ', self.minimum_eccentricity_upper_uncertainty,
              '\n Lower uncertainty of minimum eccentricity: ', self.minimum_eccentricity_lower_uncertainty,'\n\n')

             
