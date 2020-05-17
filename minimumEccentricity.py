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
        planet_mass                           (float): The mass of the planet in the unit of the Earth's mass
        planet_mass_upper_uncertainty         (float): The upper uncertainty associated with the planet's mass
        planet_mass_lower_uncertainty         (float): The lower uncertainty associated with the planet's mass
        star_mass                             (float): The mass of the star in the unit of the solar mass
        star_mass_upper_uncertainty           (float): The upper uncertainty associated with the star's mass
        star_mass_lower_uncertainty           (float): The lower uncertainty associated with the star's mass
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

    SOLAR_MASS = 1.98847 * math.pow(10,30)
    SOLAR_MASS_UPPER_UNCERTAINTY = 0.00007 * math.pow(10,30)
    SOLAR_MASS_LOWER_UNCERTAINTY = -0.00007 * math.pow(10,30)
    
    EARTH_RADIUS = 6378000
    EARTH_RADIUS_UPPER_UNCERTAINTY = 0
    EARTH_RADIUS_LOWER_UNCERTAINTY = 0

    EARTH_MASS = 5.9722 * pow(10,24)
    EARTH_MASS_UPPER_UNCERTAINTY = 6 * pow(10,20)
    EARTH_MASS_LOWER_UNCERTAINTY = -6 * pow(10,20)

    G = 6.67259 * math.pow(10,-11)


    
    AU = 149600000000
    SOLAR_RADIUS_OVER_AU = SOLAR_RADIUS / AU #is the Sun's radius divided by 1AU
    SOLAR_RADIUS_OVER_AU_UPPER_UNCERTAINTY = SOLAR_RADIUS_UPPER_UNCERTAINTY / AU
    SOLAR_RADIUS_OVER_AU_LOWER_UNCERTAINTY = - SOLAR_RADIUS_LOWER_UNCERTAINTY / AU
    
    EARTH_RADIUS_OVER_SOLAR_RADIUS = EARTH_RADIUS / SOLAR_RADIUS #is the Earth's radius divided by the Sun's radius
    EARTH_RADIUS_OVER_SOLAR_RADIUS_UPPER_UNCERTAINTY = (EARTH_RADIUS + EARTH_RADIUS_UPPER_UNCERTAINTY)/(SOLAR_RADIUS + SOLAR_RADIUS_LOWER_UNCERTAINTY)-EARTH_RADIUS_OVER_SOLAR_RADIUS
    EARTH_RADIUS_OVER_SOLAR_RADIUS_LOWER_UNCERTAINTY = -(EARTH_RADIUS_OVER_SOLAR_RADIUS - (EARTH_RADIUS + EARTH_RADIUS_LOWER_UNCERTAINTY)/(SOLAR_RADIUS + SOLAR_RADIUS_UPPER_UNCERTAINTY))

    
    
    def __init__(self,
                 planet_id,
                 planet_mass,
                 planet_mass_upper_uncertainty,
                 planet_mass_lower_uncertainty,
                 star_mass,
                 star_mass_upper_uncertainty,
                 star_mass_lower_uncertainty,
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
           planet_mass                           (float): The mass of the planet in the unit of the Earth's mass
           planet_mass_upper_uncertainty         (float): The upper uncertainty associated with the planet's mass
           planet_mass_lower_uncertainty         (float): The lower uncertainty associated with the planet's mass
           star_mass                             (float): The mass of the star in the unit of the solar mass
           star_mass_upper_uncertainty           (float): The upper uncertainty associated with the star's mass
           star_mass_lower_uncertainty           (float): The lower uncertainty associated with the star's mass
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
        self.planet_mass = planet_mass        
        self.planet_mass_upper_uncertainty = planet_mass_upper_uncertainty        
        self.planet_mass_lower_uncertainty = planet_mass_lower_uncertainty        
        self.star_mass = star_mass        
        self.star_mass_upper_uncertainty = star_mass_upper_uncertainty        
        self.star_mass_lower_uncertainty = star_mass_lower_uncertainty        
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
        self.transit_duration_if_circular_orbit_knowing_a = self.find_transit_duration_if_circular_orbit_knowing_a(self.star_radius,
                                                                                               self.planet_radius,
                                                                                               self.impact_parameter,
                                                                                               self.semi_major_axis,
                                                                                               self.orbital_period)#Here, a is for semi-major axis of the planet's orbit
        
        self.simple_estimate_of_transit_duration_upper_uncertainty_if_circular_orbit = -self.transit_duration_if_circular_orbit_knowing_a + self.find_simple_estimate_of_transit_duration_upper_limit_if_circular_orbit(
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
        self.simple_estimate_of_transit_duration_lower_uncertainty_if_circular_orbit = -self.transit_duration_if_circular_orbit_knowing_a + self.find_simple_estimate_of_transit_duration_lower_limit_if_circular_orbit(
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
        self.delta_a = self.find_delta(self.transit_duration, self.transit_duration_if_circular_orbit_knowing_a) #Here, a is for semi-major axis of the planet's orbit
        self.simple_estimate_of_delta_upper_uncertainty = - self.delta_a + self.find_simple_estimate_of_delta_upper_limit(self.transit_duration,
                                                                                  self.transit_duration_if_circular_orbit_knowing_a,
                                                                                  self.transit_duration_upper_uncertainty,
                                                                                  self.simple_estimate_of_transit_duration_lower_uncertainty_if_circular_orbit)
        self.simple_estimate_of_delta_lower_uncertainty = -self.delta_a + self.find_simple_estimate_of_delta_lower_limit(self.transit_duration,
                                                                                 self.transit_duration_if_circular_orbit_knowing_a,
                                                                                 self.transit_duration_lower_uncertainty,
                                                                                 self.simple_estimate_of_transit_duration_upper_uncertainty_if_circular_orbit) 
        self.minimum_eccentricity_a = self.find_emin(self.delta_a) #Here, a is for semi-major axis of the planet's orbit
        self.simple_estimate_of_minimum_eccentricity_upper_uncertainty = self.find_simple_estimate_of_emin_upper_limit(self.delta_a, self.simple_estimate_of_delta_upper_uncertainty, self.simple_estimate_of_delta_lower_uncertainty) - self.minimum_eccentricity_a
        self.simple_estimate_of_minimum_eccentricity_lower_uncertainty = self.find_simple_estimate_of_emin_lower_limit(self.delta_a, self.simple_estimate_of_delta_upper_uncertainty, self.simple_estimate_of_delta_lower_uncertainty) - self.minimum_eccentricity_a
    
        self.transit_duration_if_circular_orbit_knowing_m = self.find_transit_duration_if_circular_orbit_knowing_m(self.star_radius,
                                                                                               self.planet_radius,
                                                                                               self.impact_parameter,
                                                                                               self.planet_mass,
                                                                                               self.star_mass,
                                                                                               self.orbital_period)#Here, m is for the mass of the planets and the star
        self.transit_duration_max_uncertainty_if_circular_orbit = self.find_transit_duration_uncertainty_if_circular_orbit(self.star_radius,
                                                            max(self.star_radius_upper_uncertainty,-self.star_radius_lower_uncertainty),
                                                            self.planet_radius,
                                                            max(self.planet_radius_upper_uncertainty, -self.planet_radius_lower_uncertainty),
                                                            self.impact_parameter,
                                                            max(self.impact_parameter_upper_uncertainty, -self.impact_parameter_lower_uncertainty),
                                                            self.planet_mass,
                                                            max(self.planet_mass_upper_uncertainty, -self.planet_mass_lower_uncertainty),
                                                            self.star_mass,
                                                            max(self.star_mass_upper_uncertainty, -self.star_mass_lower_uncertainty),
                                                            self.orbital_period,
                                                            max(self.orbital_period_upper_uncertainty, -self.orbital_period_lower_uncertainty))
        self.transit_duration_min_uncertainty_if_circular_orbit = self.find_transit_duration_uncertainty_if_circular_orbit(self.star_radius,
                                                            min(self.star_radius_upper_uncertainty,-self.star_radius_lower_uncertainty),
                                                            self.planet_radius,
                                                            min(self.planet_radius_upper_uncertainty, -self.planet_radius_lower_uncertainty),
                                                            self.impact_parameter,
                                                            min(self.impact_parameter_upper_uncertainty, -self.impact_parameter_lower_uncertainty),
                                                            self.planet_mass,
                                                            min(self.planet_mass_upper_uncertainty, -self.planet_mass_lower_uncertainty),
                                                            self.star_mass,
                                                            min(self.star_mass_upper_uncertainty, -self.star_mass_lower_uncertainty),
                                                            self.orbital_period,
                                                            min(self.orbital_period_upper_uncertainty, -self.orbital_period_lower_uncertainty))
        self.delta_m = self.find_delta(self.transit_duration, self.transit_duration_if_circular_orbit_knowing_m) #Here, m is for mass of the star and the planets
        self.delta_max_uncertainty = self.find_delta_uncertainty(self.transit_duration,
                               self.transit_duration_if_circular_orbit_knowing_m,
                               max(self.transit_duration_upper_uncertainty, -self.transit_duration_lower_uncertainty),
                               self.transit_duration_max_uncertainty_if_circular_orbit)
        self.delta_min_uncertainty = self.find_delta_uncertainty(self.transit_duration,
                               self.transit_duration_if_circular_orbit_knowing_m,
                               min(self.transit_duration_upper_uncertainty,-self.transit_duration_lower_uncertainty),
                               self.transit_duration_min_uncertainty_if_circular_orbit)
        self.minimum_eccentricity_m = self.find_emin(self.delta_m)#Here, m is for mass of the star and the planets
        self.minimum_eccentricity_max_uncertainty = self.find_emin_uncertainty(self.delta_m, self.delta_max_uncertainty)
        self.minimum_eccentricity_min_uncertainty = self.find_emin_uncertainty(self.delta_m, self.delta_min_uncertainty)

    def find_transit_duration_if_circular_orbit_knowing_a(self,
                                                star_radius,
                                                planet_radius,
                                                impact_parameter,
                                                semi_major_axis,
                                                orbital_period):

        """ 
        The function to work out the transit duration if the orbit were circular
        (transit_duration_if_circular_orbit) in the unit of hours by knowing the
        semimajor axis of the planet's orbit, a and other attributes. 
  
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
        j = star_radius/semi_major_axis * orbital_period * self.SOLAR_RADIUS/self.AU * 24
        j = star_radius/semi_major_axis * orbital_period * self.SOLAR_RADIUS_OVER_AU * 24
        return tc #in hours

    

        

    def find_simple_estimate_of_transit_duration_upper_limit_if_circular_orbit(self,
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
        The function to estimate the upper uncertainty limit of the transit duration if the orbit were circular
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

    def find_simple_estimate_of_transit_duration_lower_limit_if_circular_orbit(self,
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
        The function to estimate the lower uncertainty limit of the transit duration if the orbit were circular
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
        
    
    def find_transit_duration_if_circular_orbit_knowing_m(self,
                                                          star_radius,
                                                          planet_radius,
                                                          impact_parameter,
                                                          planet_mass,
                                                          star_mass,
                                                          orbital_period):

        """ 
        The function to work out the transit duration if the orbit were circular
        (transit_duration_if_circular_orbit) in the unit of hours. 
  
        Parameters: 
            star_radius      (float): Radius of the parent star in the unit of the Sun's radius, SOLAR_RADIUS
            planet_radius    (float): Radius of the exoplanet in the unit of the Earth's radius, EARTH_RADIUS
            impact_parameter (float): Impact parameter in the unit of the parent star's radius
            planet_mass      (float): The planet's mass in the unit of the Earth's mass
            star_mass        (float): The star's mass in the unit of the solar mass
            orbital_period   (float): Orbital period in days
          
        Returns: 
            tc               (float): Transit Duration, in hours, if the orbit were circular
            
        """

        
        tc = (1+planet_radius/star_radius * self.EARTH_RADIUS_OVER_SOLAR_RADIUS)        
        tc = tc*tc - impact_parameter*impact_parameter
        tc = max(tc, 0)
        tc = math.sqrt(tc)
        const = pow(4 * 3.14159 * 3.14159/self.G, 1.0/3.0)
        smass = star_mass * self.SOLAR_MASS
        pmass = planet_mass * self.EARTH_MASS
        starRadius = star_radius * self.SOLAR_RADIUS
        temp = orbital_period * 24*60*60 / (smass + pmass) #At first we have to convert P from days to seconds
        temp = pow(temp,1.0/3.0)
        tc = tc * starRadius /3.1416 * const * temp/60/60 #We divide tc twice by 60 to convert it from seconds to hours
        
        return tc #in hours

    def find_transit_duration_uncertainty_if_circular_orbit(self,
                                                            star_radius,
                                                            star_radius_uncertainty,
                                                            planet_radius,
                                                            planet_radius_uncertainty,
                                                            impact_parameter,
                                                            impact_parameter_uncertainty,
                                                            planet_mass,
                                                            planet_mass_uncertainty,
                                                            star_mass,
                                                            star_mass_uncertainty,
                                                            orbital_period,
                                                            orbital_period_uncertainty):
        """ 
        The function to workout the uncertainty of the transit duration if the orbit were circular
        (transit_duration_if_circular_orbit) in the unit of hours in the method of the summation of
        squares of uncertainties. 
  
        Parameters: 
            star_radius                     (float): Radius of the star in the unit of the Sun's radius, SOLAR_RADIUS
            star_radius_uncertainty         (float): The uncertainty associated with the radius of the star in the unit of the Sun's radius, SOLAR_RADIUS
            planet_radius                   (float): Radius of the planet in the unit of the Earth's radius, EARTH_RADIUS
            planet_radius_uncertainty       (float): The uncertainty associated with the radius of the planet in the unit of the Earth's radius, EARTH_RADIUS
            impact_parameter                (float): Impact parameter in the unit of the parent star's radius
            impact_parameter_uncertainty    (float): The uncertainty associated with the impact parameter in the unit of the parent star's radius
            planet_mass                     (float): The mass of the planet
            planet_mass_uncertainty         (float): The uncertainty associated with the planet's mass
            star_mass                       (float): The star's mass
            star_mass_uncertainty           (float): The uncertainty associated with the star's mass
            orbital_period                  (float): Orbital period in days
            orbital_period_uncertainty      (float): The uncertainty associated with the orbital period in days
            
        Returns: 
            pow(var_tc, 0.5)                (float): The uncertainty of the transit Duration, in hours, if the orbit were circular
            
        """


        var_rp = pow(planet_radius_uncertainty, 2)
        var_rs = pow(star_radius_uncertainty, 2)
        var_b = pow(impact_parameter_uncertainty, 2)
        var_P = pow(orbital_period_uncertainty * 24 * 60 * 60, 2)
        var_ms = pow(star_mass_uncertainty * self.SOLAR_MASS, 2)
        var_mp = pow(planet_mass_uncertainty * self.EARTH_MASS, 2)


        rs = star_radius
        rp = planet_radius
        b = impact_parameter
        P = orbital_period * 24 * 60 * 60 #days to seconds coversion
        ms = star_mass * self.SOLAR_MASS
        mp = planet_mass * self.EARTH_MASS

        
        r = planet_radius/star_radius * self.EARTH_RADIUS_OVER_SOLAR_RADIUS
        var_r = (var_rp + r*r * var_rs)/(rs*rs)

        f = pow((1+r)*(1+r)-b*b,0.5)
        var_f = ((1+r)*(1+r)*var_r + b*b * var_b)/(f*f)

        m = ms + mp
        var_m = var_ms + var_mp

        k = pow(P/m, 1/3)
        var_k = (var_P + pow(P/m,2) * var_m)/(3*k*k*m*m)

        tc = r * rs * k * pow(4 /self.G /3.141592654, 1/3)
        var_tc = tc*tc * (var_r/r/r + var_rs/rs/rs + var_k/k/k)

        return pow(var_tc, 0.5)/60/60 #converting from second to hours

    
    

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

    def find_simple_estimate_of_delta_upper_limit(self,
                                                  transit_duration,
                                                  transit_duration_if_circular_orbit,
                                                  transit_duration_upper_uncertainty,
                                                  transit_duration_lower_uncertainty_if_circular_orbit):
        """ 
        The function to calculate the upper uncertainty limit of the delta factor, i.e. the ratio of the transit duration
        to the transit duration if the orbit were circular.
  
        Parameters: 
            transit_duration                    (float): The transit duration of the exoplanet
            transit_duration_if_circular_orbit  (float): The transit duration if the orbit were circular
            transit_duration_upper_uncertainty  (float): The upper uncertainty of the transit duration
            transit_duration_lower_uncertainty_if_circular_orbit  (float): The lower uncertainty of the transit duration if the circuit were circular
        Returns: 
            (transit_duration + transit_duration_upper_uncertainty)/(transit_duration_if_circular_orbit - transit_duration_lower_uncertainty_if_circular_orbit) (float): Upper limit of the value of delta 
        """

        return (transit_duration + transit_duration_upper_uncertainty)/(transit_duration_if_circular_orbit + transit_duration_lower_uncertainty_if_circular_orbit)

    def find_simple_estimate_of_delta_lower_limit(self,
                                                  transit_duration,
                                                  transit_duration_if_circular_orbit,
                                                  transit_duration_lower_uncertainty,
                                                  transit_duration_upper_uncertainty_if_circular_orbit):
        """ 
        The function to calculate the lower uncertainty limit of the delta factor, i.e. the ratio of the transit duration
        to the transit duration if the orbit were circular.
  
        Parameters: 
            transit_duration                    (float): The transit duration of the exoplanet
            transit_duration_if_circular_orbit  (float): The transit duration if the orbit were circular
            transit_duration_lower_uncertainty  (float): The lower uncertainty of the transit duration
            transit_duration_upper_uncertainty_if_circular_orbit  (float): The upper uncertainty of the transit duration if the circuit were circular
        Returns: 
            (transit_duration - transit_duration_lower_uncertainty)/(transit_duration_if_circular_orbit + transit_duration_upper_uncertainty_if_circular_orbit) (float): lower limit of the value of delta 
        """
        return (transit_duration + transit_duration_lower_uncertainty)/(transit_duration_if_circular_orbit + transit_duration_upper_uncertainty_if_circular_orbit)
    
    def find_delta_uncertainty(self,
                               transit_duration,
                               transit_duration_if_circular_orbit,
                               transit_duration_uncertainty,
                               transit_duration_uncertainty_if_circular_orbit):
        """ 
        The function to calculate the uncertainty of the delta factor, i.e. the ratio of the transit duration
        to the transit duration if the orbit were circular, by the method of the square of summations of uncertainties
        associated with the independent variables.
  
        Parameters: 
            transit_duration                                (float): The transit duration of the exoplanet
            transit_duration_if_circular_orbit              (float): The transit duration if the orbit were circular
            transit_duration_uncertainty                    (float): The uncertainty of the transit duration
            transit_duration_uncertainty_if_circular_orbit  (float): The uncertainty of the transit duration if the circuit were circular
        Returns: 
            pow(var_delta, 0.5)                             (float): The uncertainty associated with the delta factor
        """


        t = transit_duration
        tc = transit_duration_if_circular_orbit
        delta = t/tc

        var_t = transit_duration_uncertainty*transit_duration_uncertainty
        var_tc = transit_duration_uncertainty_if_circular_orbit*transit_duration_uncertainty_if_circular_orbit

        var_delta = (var_t + delta*delta* var_tc)/tc/tc

        return pow(var_delta,0.5)

        





        
    def find_emin(self, delta):
        """ 
        The function to calculate minimum eccentricity of the exoplanet's orbit
  
        Parameters: 
            delta (float): Delta factor, i.e. the ratio of the transit time to the transit time for circular orbit.
                      
        Returns: 
            minimum eccentricity of the exoplanet's orbit (float)
        """
        
        return abs((delta * delta - 1)/(delta * delta+1))

    def find_simple_estimate_of_emin_upper_limit(self, delta, delta_upper_uncertainty, delta_lower_uncertainty):
        """ 
        The function to calculate the upper uncertainty limit of the minimum eccentricity of the exoplanet's orbit
  
        Parameters: 
            delta (float): Delta factor, i.e. the ratio of the transit time to the transit time for circular orbit.
            delta_lower_uncertainty (float): The lower uncertainty of the delta factor          
        Returns: 
            upper limit of the minimum eccentricity of the exoplanet's orbit (float)
        """
        
        a = delta + delta_lower_uncertainty
        b = delta + delta_upper_uncertainty
            
        return max(abs((a*a-1)/(a*a+1)), abs((b*b-1)/(b*b+1)))
            
            
        
        
        
    def find_simple_estimate_of_emin_lower_limit(self, delta, delta_upper_uncertainty, delta_lower_uncertainty):
        """ 
        The function to calculate the lower uncertainty limit of the minimum eccentricity of the exoplanet's orbit
  
        Parameters: 
            delta (float): Delta factor, i.e. the ratio of the transit time to the transit time for circular orbit.
            delta_upper_uncertainty (float): The upper uncertainty of the delta factor          
        Returns: 
            lower limit of the minimum eccentricity of the exoplanet's orbit (float)
        """

        if abs(delta)<1:
              
             b = delta + delta_upper_uncertainty
             if abs(b)<1:
                 return abs((b*b-1)/(b*b+1))
             else:
                 return 0
        else:
             a = delta + delta_lower_uncertainty
             if abs(a)>1:
                 return abs((a*a-1)/(a*a+1))
             else:
                 return 0
       
    def find_emin_uncertainty(self, delta, delta_uncertainty):
        """ 
        The function to calculate the uncertainty of the minimum eccentricity of the exoplanet's orbit
        by using the method of summation of the squares of the uncertainties associated with the independent variables.
  
        Parameters: 
            delta             (float): Delta factor, i.e. the ratio of the transit time to the transit time for circular orbit.
            delta_uncertainty (float): The uncertainty of the delta factor          
        Returns: 
            uncertainty with the minimum eccentricity of the exoplanet's orbit (float)
        """

        return 4*delta*delta_uncertainty/(delta*delta+1)


        
    def print_attributes(self):
        """ 
        The function to print attributes of exoplanet with their simple estimates of uncertainties 
          
        """
        print('Planet ID: ',self.planet_id,
              '\n Transit duration if the orbit were circular knowing semi major axis of the planetary orbit and other staffs: ', self.transit_duration_if_circular_orbit_knowing_a,
              '\n Transit duration if the orbit were circular knowing mass of the star and the exoplanet and other staffs: ', self.transit_duration_if_circular_orbit_knowing_m,
              '\n A simple estimate of the upper uncertainty of transit duration if the orbit were circular: ', self.simple_estimate_of_transit_duration_upper_uncertainty_if_circular_orbit,
              '\n A simple estimate of the lower uncertainty of transit duration if the orbit were circular: ', self.simple_estimate_of_transit_duration_lower_uncertainty_if_circular_orbit,
              '\n The maximum value of |uncertainty| of transit duration if the orbit were circular: ', self.transit_duration_max_uncertainty_if_circular_orbit,
              '\n The minimum value of |uncertainty| of transit duration if the orbit were circular: ', self.transit_duration_min_uncertainty_if_circular_orbit,
              '\n Delta knowing semi major axis of the planetary orbit and other staffs: ', self.delta_a,
              '\n Delta knowing mass of the star and the exoplanet and other staffs: ', self.delta_m,
              '\n A simple estimate of the upper uncertainty of Delta: ', self.simple_estimate_of_delta_upper_uncertainty,
              '\n A simple estimate of the lower uncertainty of Delta: ', self.simple_estimate_of_delta_lower_uncertainty,
              '\n The maximum |uncertainty| of Delta: ', self.delta_max_uncertainty,              
              '\n The minimum |uncertainty| of Delta: ', self.delta_min_uncertainty,
              '\n Minimum Eccentricity by knowing semi major axis of the planetary orbit and other staffs: ', self.minimum_eccentricity_a,
              '\n Minimum Eccentricity by knowing mass of the star and the exoplanet and other staffs: ', self.minimum_eccentricity_m,
              '\n A simple estimate of the upper uncertainty of minimum eccentricity: ', self.simple_estimate_of_minimum_eccentricity_upper_uncertainty,
              '\n A simple estiamte of the lower uncertainty of minimum eccentricity: ', self.simple_estimate_of_minimum_eccentricity_lower_uncertainty,
              '\n The maximum |uncertainty| of minimum eccentricity: ', self.minimum_eccentricity_max_uncertainty,              
              '\n The minimum |uncertainty| of minimum eccentricity: ', self.minimum_eccentricity_min_uncertainty,
              '\n\n')

             
