import math
import planetary_system_io
import astropy

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
        SOLAR_MASS                            (float): The mass of the Sun in the unit of kilograms. This is a constant quantity
        EARTH_RADIUS                          (float): The radius of the Earth in the unit of meters. This is a constant quantity
        AU                                    (float): Astronomical Unit in meters.This is a constant quantity
        G                                     (float): The universal gravitational constant in Newton meter-square over kilogram-square 
        

    Methods:
        find_transit_duration_if_circular_orbit   : Works out the transit duration, in hours, if the orbit were circular 
        find_delta                                : Works out the delta factor
        find_emin                                 : Works out the minimum eccentricity of the orbit of the exoplanet
        print_attributes                          : Prints out the attributes of the exoplanet  
        
    """
    
      
    SOLAR_RADIUS = 696342000   
    SOLAR_MASS = 1.98847 * math.pow(10,30) 
    EARTH_RADIUS = 6378000
    EARTH_MASS = 5.9722 * pow(10,24)
    G = 6.67259 * math.pow(10,-11)    
    AU = 149600000000
        
    
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
       
        self.delta = self.find_delta(self.transit_duration, self.transit_duration_if_circular_orbit) 
        
        self.minimum_eccentricity = self.find_emin(self.delta)
        
        self.transit_duration_max_uncertainty_if_circular_orbit = self.find_transit_duration_uncertainty_if_circular_orbit(self.star_radius,
                                                            max(self.star_radius_upper_uncertainty,-self.star_radius_lower_uncertainty),
                                                            self.planet_radius,
                                                            max(self.planet_radius_upper_uncertainty, -self.planet_radius_lower_uncertainty),
                                                            self.impact_parameter,
                                                            max(self.impact_parameter_upper_uncertainty, -self.impact_parameter_lower_uncertainty),
                                                            self.semi_major_axis,
                                                            max(self.semi_major_axis_upper_uncertainty, -self.semi_major_axis_lower_uncertainty),
                                                            self.orbital_period,
                                                            max(self.orbital_period_upper_uncertainty, -self.orbital_period_lower_uncertainty))
        self.transit_duration_min_uncertainty_if_circular_orbit = self.find_transit_duration_uncertainty_if_circular_orbit(self.star_radius,
                                                            min(self.star_radius_upper_uncertainty, -self.star_radius_lower_uncertainty),
                                                            self.planet_radius,
                                                            min(self.planet_radius_upper_uncertainty, -self.planet_radius_lower_uncertainty),
                                                            self.impact_parameter,
                                                            min(self.impact_parameter_upper_uncertainty, -self.impact_parameter_lower_uncertainty),
                                                            self.semi_major_axis,
                                                            min(self.semi_major_axis_upper_uncertainty, -self.semi_major_axis_lower_uncertainty),
                                                            self.orbital_period,
                                                            min(self.orbital_period_upper_uncertainty, -self.orbital_period_lower_uncertainty))



        
        self.delta_max_uncertainty = self.find_delta_uncertainty(self.transit_duration,
                               self.transit_duration_if_circular_orbit,
                               max(self.transit_duration_upper_uncertainty, -self.transit_duration_lower_uncertainty),
                               self.transit_duration_max_uncertainty_if_circular_orbit)
        self.delta_min_uncertainty = self.find_delta_uncertainty(self.transit_duration,
                               self.transit_duration_if_circular_orbit,
                               min(self.transit_duration_upper_uncertainty,-self.transit_duration_lower_uncertainty),
                               self.transit_duration_min_uncertainty_if_circular_orbit)
        
        self.minimum_eccentricity_max_uncertainty = self.find_emin_uncertainty(self.delta, self.delta_max_uncertainty)
        self.minimum_eccentricity_min_uncertainty = self.find_emin_uncertainty(self.delta, self.delta_min_uncertainty)

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

        
        rs = star_radius * self.SOLAR_RADIUS
        rp = planet_radius * self.EARTH_RADIUS
        b = impact_parameter
        a = semi_major_axis * self.AU
        P = orbital_period * 24

        tc = math.sqrt(max(pow(1 + rp/rs,2) - b*b, 0)) * rs/a /3.141592654 * P
        
        return tc   
    
    
    def find_transit_duration_uncertainty_if_circular_orbit(self,
                                                            star_radius,
                                                            star_radius_uncertainty,
                                                            planet_radius,
                                                            planet_radius_uncertainty,
                                                            impact_parameter,
                                                            impact_parameter_uncertainty,
                                                            semi_major_axis,
                                                            semi_major_axis_uncertainty,                                                                      
                                                            orbital_period,
                                                            orbital_period_uncertainty):
        """ 
        The function to workout the uncertainty of the transit duration if the orbit were circular
        in the unit of hours. 
  
        Parameters: 
            star_radius                     (float): Radius of the star in the unit of the Sun's radius, SOLAR_RADIUS
            star_radius_uncertainty         (float): The uncertainty associated with the radius of the star in the unit of the Sun's radius, SOLAR_RADIUS
            planet_radius                   (float): Radius of the planet in the unit of the Earth's radius, EARTH_RADIUS
            planet_radius_uncertainty       (float): The uncertainty associated with the radius of the planet in the unit of the Earth's radius, EARTH_RADIUS
            impact_parameter                (float): Impact parameter in the unit of the parent star's radius
            impact_parameter_uncertainty    (float): The uncertainty associated with the impact parameter in the unit of the parent star's radius
            semi_major_axis                 (float): The semi major axis of the orbit of the planet in AU
            semi_major_axis_uncertainty     (float): The uncertainty associated with the semi major axis of the orbit of the planet in AU
            orbital_period                  (float): Orbital period in days
            orbital_period_uncertainty      (float): The uncertainty associated with the orbital period in days
            
        Returns: 
            unc_tc                          (float): The uncertainty of the transit Duration, in hours, if the orbit were circular
            
        """


        var_rp = pow(planet_radius_uncertainty * self.EARTH_RADIUS, 2)
        var_rs = pow(star_radius_uncertainty * self.SOLAR_RADIUS, 2)
        var_b = pow(impact_parameter_uncertainty, 2)
        var_P = pow(orbital_period_uncertainty * 24 * 60 * 60, 2)
        var_a = pow(semi_major_axis_uncertainty * self.AU,2)
        

        rs = star_radius * self.SOLAR_RADIUS
        rp = planet_radius * self.EARTH_RADIUS
        b = impact_parameter
        P = orbital_period * 24 * 60 * 60 #days to seconds coversion
        a = semi_major_axis * self.AU
                
        r = rp/rs
        var_r = (var_rp + r*r * var_rs)/(rs*rs)

        f = pow((1+r)*(1+r)-b*b,0.5)
        var_f = ((1+r)*(1+r)*var_r + b*b * var_b)/(f*f)

        k = rs/a 
        var_k = (var_rs + k*k * var_a)/(a*a)

        tc = f * k * P /3.141592654
        var_tc = tc*tc * (var_f/f/f + var_k/k/k + var_P/P/P)
        unc_tc = math.sqrt(var_tc)/60/60 #converting from seconds to hours square root of var_tc is divided by 60 for two times

        return unc_tc 

    def find_delta(self, transit_duration, transit_duration_if_circular_orbit):
        """ 
        The function to calculate delta factor, i.e. the ratio of the transit duration
        to the transit duration if the orbit were circular.
  
        Parameters: 
            transit_duration                    (float): The transit duration of the exoplanet in hours
            transit_duration_if_circular_orbit  (float): The transit duration, in hours, if the orbit were circular
          
        Returns: 
            delta                               (float): delta 
        """
        delta = transit_duration/transit_duration_if_circular_orbit
        return delta
    
    
    def find_delta_uncertainty(self,
                               transit_duration,
                               transit_duration_if_circular_orbit,
                               transit_duration_uncertainty,
                               transit_duration_uncertainty_if_circular_orbit):
        """ 
        The function to calculate the uncertainty of the delta factor, i.e. the ratio of the transit duration
        to the transit duration if the orbit were circular.
  
        Parameters: 
            transit_duration                                (float): The transit duration of the exoplanet in hours
            transit_duration_if_circular_orbit              (float): The transit duration, in hours, if the orbit were circular
            transit_duration_uncertainty                    (float): The uncertainty of the transit duration in hours
            transit_duration_uncertainty_if_circular_orbit  (float): The uncertainty of the transit duration, in hours, if the circuit were circular
        Returns: 
            unc_delta                                       (float): The uncertainty associated with the delta factor
        """


        t = transit_duration
        tc = transit_duration_if_circular_orbit
        delta = t/tc

        var_t = transit_duration_uncertainty ** 2
        var_tc = transit_duration_uncertainty_if_circular_orbit ** 2

        var_delta = (var_t + delta*delta* var_tc)/tc/tc

        unc_delta = math.sqrt(var_delta)


        return unc_delta

        
    def find_emin(self, delta):
        """ 
        The function to calculate minimum eccentricity of the exoplanet's orbit
  
        Parameters: 
            delta (float): Delta factor, i.e. the ratio of the transit time to the transit time for circular orbit.
                      
        Returns: 
            emin  (float): Minimum eccentricity of the exoplanet's orbit
        """

        emin = abs((delta * delta - 1)/(delta * delta+1))

        
        return emin        
        
        
    
    def find_emin_uncertainty(self, delta, delta_uncertainty):
        """ 
        The function to calculate the uncertainty of the minimum eccentricity of the exoplanet's orbit.
  
        Parameters: 
            delta             (float): Delta factor, i.e. the ratio of the transit time to the transit time for circular orbit.
            delta_uncertainty (float): The uncertainty of the delta factor          
        Returns: 
            unc_emin          (float): Uncertainty with the minimum eccentricity of the exoplanet's orbit (float)
        """

        unc_emin = 4*delta*delta_uncertainty/(delta*delta+1)

        return unc_emin


        
    def print_attributes(self):
        """ 
        The function to print attributes of exoplanet with their simple estimates of uncertainties 
          
        """
        print('Planet ID: ',self.planet_id,
              '\n Transit duration if the orbit were circular: ', self.transit_duration_if_circular_orbit,              
              '\n The maximum value of |uncertainty| of transit duration if the orbit were circular: ', self.transit_duration_max_uncertainty_if_circular_orbit,
              '\n The minimum value of |uncertainty| of transit duration if the orbit were circular: ', self.transit_duration_min_uncertainty_if_circular_orbit,
              '\n Delta: ', self.delta,              
              '\n The maximum |uncertainty| of Delta: ', self.delta_max_uncertainty,              
              '\n The minimum |uncertainty| of Delta: ', self.delta_min_uncertainty,
              '\n Minimum Eccentricity: ', self.minimum_eccentricity,
              '\n The maximum |uncertainty| of minimum eccentricity: ', self.minimum_eccentricity_max_uncertainty,              
              '\n The minimum |uncertainty| of minimum eccentricity: ', self.minimum_eccentricity_min_uncertainty,
              '\n\n')

             
