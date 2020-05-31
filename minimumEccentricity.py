import math
import planetary_system_io
import astropy

class TransitingExoplanet:
    """ 
    Class for calculating the minimum eccentricity of the orbit of the transiting exoplanets.

    Attributes: 
        planet_id (str):    The identification label of the exoplanet
        
        semi_major_axis_over_star_radius (float):    The ratio of the semi-major axis to the
            stellar radius

        semi_major_axis_over_star_radius_upper_uncertainty (float):    The upper uncertainty
            associated with the ratio of the semi-major axis to the stellar radius

        semi_major_axis_over_star_radius_lower_uncertainty (float):     The lower uncertainty
            associated with the ratio of the semi-major axis to the stellar radius

        orbital_period (float):    Orbital period in days

        orbital_period_upper_uncertainty (float):    The upper uncertainty associated
            with the orbital period in days

        orbital_period_lower_uncertainty (float):    The lower uncertainty associated
            with the orbital period in days        

        transit_duration (float):    Transit duration in hours

        transit_duration_upper_uncertainty (float):    The upper uncertainty associated
            with the transit duration in hours

        transit_duration_lower_uncertainty (float):    The lower uncertainty associated
            with the transit duration in hours

        planet_radius_over_star_radius (float):    Ratio of the radius of the planet
            to the stellar radius

        planet_radius_over_star_radius_upper_uncertainty (float):    The upper uncertainty
            associated with the the radius of the planet to the stellar radius

        planet_radius_over_star_radius_lower_uncertainty (float):    The lower uncertainty
            associated with the radius of the planet to the stellar radius       

        impact_parameter (float):    Impact parameter in the unit of the parent star's
            radius

        impact_parameter_upper_uncertainty (float):    The upper uncertainty associated
            with the impact parameter in the unit of the parent star's radius

        impact_parameter_lower_uncertainty (float):    The lower uncertainty associated
            with the impact parameter in the unit of the parent star's radius        

        transit_duration_if_circular_orbit (float):    Transit duration in hours if the
            orbit is circular 

        transit_duration_if_circular_orbit_upper_uncertainty (float):    The upper uncertainty
            associated with the transit duration in hours if the orbit is circular

        transit_duration_if_circular_orbit_lower_uncertainty (float):    The lower uncertainty
            associated with the transit duration in hours if the orbit is circular        

        minimum_eccentricity (float):    Minimum eccentricity of the exoplanet

        minimum_eccentricity_upper_uncertainty (float):    The upper uncertainty associated
            with the minimum eccentricity of the exoplanet

        minimum_eccentricity_lower_uncertainty (float):    The lower uncertainty associated
            with the minimum eccentricity of the exoplanet        

        delta (float):    Delta factor

        delta_upper_uncertainty (float):    The upper uncertainty associated with
            the delta factor

        delta_lower_uncertainty (float):    The lower uncertainty associated with
            the delta factor         
    """
    
      
        
    def __init__(self,
                 planet_id,
                 semi_major_axis_over_star_radius,
                 semi_major_axis_over_star_radius_upper_uncertainty,
                 semi_major_axis_over_star_radius_lower_uncertainty,
                 orbital_period,
                 orbital_period_upper_uncertainty,
                 orbital_period_lower_uncertainty,
                 transit_duration,
                 transit_duration_upper_uncertainty,
                 transit_duration_lower_uncertainty,
                 planet_radius_over_star_radius,
                 planet_radius_over_star_radius_upper_uncertainty,
                 planet_radius_over_star_radius_lower_uncertainty,                 
                 impact_parameter,
                 impact_parameter_upper_uncertainty,
                 impact_parameter_lower_uncertainty):

        """ 
        The constructor for TransitingExoplanet class.
        
        Parameters: 
            planet_id (str):    The identification label of the exoplanet
        
            semi_major_axis_over_star_radius (float):    The ratio of the semi-major axis to the
                stellar radius

            semi_major_axis_over_star_radius_upper_uncertainty (float):    The upper uncertainty
                associated with the ratio of the semi-major axis to the stellar radius

            semi_major_axis_over_star_radius_lower_uncertainty (float):     The lower uncertainty
                associated with the ratio of the semi-major axis to the stellar radius

            orbital_period (float):    Orbital period in days

            orbital_period_upper_uncertainty (float):    The upper uncertainty associated
                with the orbital period in days

            orbital_period_lower_uncertainty (float):    The lower uncertainty associated
                with the orbital period in days        

            transit_duration (float):    Transit duration in hours

            transit_duration_upper_uncertainty (float):    The upper uncertainty associated
                with the transit duration in hours

            transit_duration_lower_uncertainty (float):    The lower uncertainty associated
                with the transit duration in hours

            planet_radius_over_star_radius (float):    Ratio of the radius of the planet
                to the stellar radius

            planet_radius_over_star_radius_upper_uncertainty (float):    The upper uncertainty
                associated with the the radius of the planet to the stellar radius

            planet_radius_over_star_radius_lower_uncertainty (float):    The lower uncertainty
                associated with the radius of the planet to the stellar radius       

            impact_parameter (float):    Impact parameter in the unit of the parent star's
                radius

            impact_parameter_upper_uncertainty (float):    The upper uncertainty associated
                with the impact parameter in the unit of the parent star's radius

            impact_parameter_lower_uncertainty (float):    The lower uncertainty associated
                with the impact parameter in the unit of the parent star's radius
           
        """

        self.planet_id = planet_id
        self.semi_major_axis_over_star_radius = semi_major_axis_over_star_radius
        self.semi_major_axis_over_star_radius_upper_uncertainty = semi_major_axis_over_star_radius_upper_uncertainty
        self.semi_major_axis_over_star_radius_lower_uncertainty = semi_major_axis_over_star_radius_lower_uncertainty
        self.orbital_period = orbital_period
        self.orbital_period_upper_uncertainty = orbital_period_upper_uncertainty
        self.orbital_period_lower_uncertainty = orbital_period_lower_uncertainty
        self.transit_duration =  transit_duration
        self.transit_duration_upper_uncertainty = transit_duration_upper_uncertainty
        self.transit_duration_lower_uncertainty = transit_duration_lower_uncertainty
        self.planet_radius_over_star_radius = planet_radius_over_star_radius
        self.planet_radius_over_star_radius_upper_uncertainty = planet_radius_over_star_radius_upper_uncertainty
        self.planet_radius_over_star_radius_lower_uncertainty = planet_radius_over_star_radius_lower_uncertainty
        self.impact_parameter = impact_parameter
        self.impact_parameter_upper_uncertainty = impact_parameter_upper_uncertainty
        self.impact_parameter_lower_uncertainty = impact_parameter_lower_uncertainty
        self.transit_duration_if_circular_orbit = self.find_transit_duration_if_circular_orbit(self.semi_major_axis_over_star_radius,
                                                                                               self.planet_radius_over_star_radius,
                                                                                               self.impact_parameter,
                                                                                               self.orbital_period)      
       
        self.delta = self.find_delta(self.transit_duration, self.transit_duration_if_circular_orbit) 
        
        self.minimum_eccentricity = self.find_emin(self.delta)
        
        self.transit_duration_max_uncertainty_if_circular_orbit = self.find_transit_duration_uncertainty_if_circular_orbit(self.semi_major_axis_over_star_radius,
                                                            max(self.semi_major_axis_over_star_radius_upper_uncertainty,-self.semi_major_axis_over_star_radius_lower_uncertainty),
                                                            self.planet_radius_over_star_radius,
                                                            max(self.planet_radius_over_star_radius_upper_uncertainty, -self.planet_radius_over_star_radius_lower_uncertainty),
                                                            self.impact_parameter,
                                                            max(self.impact_parameter_upper_uncertainty, -self.impact_parameter_lower_uncertainty),
                                                            self.orbital_period,
                                                            max(self.orbital_period_upper_uncertainty, -self.orbital_period_lower_uncertainty))
        self.transit_duration_min_uncertainty_if_circular_orbit = self.find_transit_duration_uncertainty_if_circular_orbit(self.semi_major_axis_over_star_radius,
                                                            min(self.semi_major_axis_over_star_radius_upper_uncertainty,-self.semi_major_axis_over_star_radius_lower_uncertainty),
                                                            self.planet_radius_over_star_radius,
                                                            min(self.planet_radius_over_star_radius_upper_uncertainty, -self.planet_radius_over_star_radius_lower_uncertainty),
                                                            self.impact_parameter,
                                                            min(self.impact_parameter_upper_uncertainty, -self.impact_parameter_lower_uncertainty),
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
                                                semi_major_axis_over_star_radius,
                                                planet_radius_over_star_radius,
                                                impact_parameter,
                                                orbital_period):

        """ 
        The function to work out the transit duration if the orbit were circular in hours
         
  
        Parameters: 
            semi_major_axis_over_star_radius (float):    Ratio of the semi major axis to the stellar radius
            
            planet_radius_over_star_radius (float):    Ratio fo the radius of the exoplanet to that of the parent star
            
            impact_parameter (float):     Impact parameter in the unit of the parent star's radius

            orbital_period (float):    Orbital period in days
          
        Returns: 
            transit_duration_if_circular_orbit (float):    Transit Duration, in hours, if the orbit were circular
            
        """

        
        P = orbital_period * 24

        transit_duration_if_circular_orbit = (math.sqrt(max(pow(1 + planet_radius_over_star_radius,2)
              - impact_parameter*impact_parameter,0))/semi_major_axis_over_star_radius
              /3.141592654*P)
        
        return transit_duration_if_circular_orbit   
    
    
    def find_transit_duration_uncertainty_if_circular_orbit(self,
                                                            semi_major_axis_over_star_radius,
                                                            semi_major_axis_over_star_radius_uncertainty,
                                                            planet_radius_over_star_radius,
                                                            planet_radius_over_star_radius_uncertainty,                                                                      
                                                            impact_parameter,
                                                            impact_parameter_uncertainty,
                                                            orbital_period,
                                                            orbital_period_uncertainty):
        """ 
        The function to workout the uncertainty of the transit duration if the orbit were circular
        in the unit of hours. 
  
        Parameters: 
            semi_major_axis_over_star_radius (float):    Ratio of the semi major axis to the stellar radius
            
            semi_major_axis_over_star_radius_uncertainty (float): The uncertainty associated with the ratio
                of the semi major axis to the stellar radius
               
            planet_radius_over_star_radius (float):    Ratio fo the radius of the exoplanet to that of the
                parent star
            
            planet_radius_over_star_radius_uncertainty (float):    The uncertainty associated with the
                ratio of the radius of the exoplanet to that of the parent star
                
            impact_parameter (float):    Impact parameter in the unit of the parent star's radius
            
            impact_parameter_uncertainty (float):    The uncertainty associated with the impact parameter
                in the unit of the parent star's radius
                
            orbital_period (float):    Orbital period in days
            
            orbital_period_uncertainty (float):    The uncertainty associated with the orbital period in days
            
        Returns: 
            orbital_period_uncertainty_if_circular_orbit (float):    The uncertainty of the transit duration,
                in hours, if the orbit were circular
            
        """


        var_r = planet_radius_over_star_radius_uncertainty**2
        var_b = impact_parameter_uncertainty**2
        var_P = (orbital_period_uncertainty * 24)**2
        var_f = semi_major_axis_over_star_radius_uncertainty**2
        
        

        r = planet_radius_over_star_radius_uncertainty
        b = impact_parameter
        P = orbital_period * 24 #days to hours coversion
        f = semi_major_axis_over_star_radius
                
        var_tc = (((P/f/3.141592654)**2)/((1+r)**2-b**2)*((1+r)**2 * var_r + (b**2) * var_b) + ((1+r)**2-b**2)/((f*3.141592654)**2)*(var_P + (P**2)/(f**2) * var_f))
        orbital_period_uncertainty_if_circular_orbit = math.sqrt(var_tc)
        
        return orbital_period_uncertainty_if_circular_orbit 

    def find_delta(self, transit_duration, transit_duration_if_circular_orbit):
        """ 
        The function to calculate delta factor

        The delta factor is the ratio of the transit duration
        to the transit duration if the orbit were circular.
  
        Parameters: 
            transit_duration (float):    The transit duration of the exoplanet in hours
            
            transit_duration_if_circular_orbit (float):    The transit duration, in hours, if
                the orbit were circular
          
        Returns: 
            delta (float):    delta 
        """
        delta = transit_duration/transit_duration_if_circular_orbit
        return delta
    
    
    def find_delta_uncertainty(self,
                               transit_duration,
                               transit_duration_if_circular_orbit,
                               transit_duration_uncertainty,
                               transit_duration_uncertainty_if_circular_orbit):
        """ 
        The function to calculate the uncertainty of the delta factor
  
        Parameters: 
            transit_duration (float):    The transit duration of the exoplanet in hours
            
            transit_duration_if_circular_orbit (float):    The transit duration, in hours,
                if the orbit were circular
            
            transit_duration_uncertainty (float):    The uncertainty of the transit duration
                in hours
            
            transit_duration_uncertainty_if_circular_orbit (float):    The uncertainty of the
                transit duration, in hours, if the circuit were circular
            
        Returns: 
            uncertainty_delta (float):    The uncertainty associated with the delta factor
        """


        t = transit_duration
        tc = transit_duration_if_circular_orbit
        delta = t/tc

        var_t = transit_duration_uncertainty ** 2
        var_tc = transit_duration_uncertainty_if_circular_orbit ** 2

        var_delta = (var_t + delta*delta* var_tc)/tc/tc

        uncertainty_delta = math.sqrt(var_delta)


        return uncertainty_delta

        
    def find_emin(self, delta):
        """ 
        The function to calculate minimum eccentricity of the exoplanet's orbit
  
        Parameters: 
            delta (float):    Delta factor, i.e. the ratio of the transit time to
                the transit time for circular orbit.
                      
        Returns: 
            emin (float):    Minimum eccentricity of the exoplanet's orbit
        """

        emin = abs((delta * delta - 1)/(delta * delta+1))

        
        return emin        
        
        
    
    def find_emin_uncertainty(self, delta, delta_uncertainty):
        """ 
        The function to calculate the uncertainty of the minimum eccentricity of
            the exoplanet's orbit.
  
        Parameters: 
            delta (float):    Delta factor, i.e. the ratio of the transit time to
                the transit time for circular orbit.
            
            delta_uncertainty (float):    The uncertainty of the delta factor
            
        Returns: 
            uncertainty_emin (float):    Uncertainty with the minimum eccentricity
                of the exoplanet's orbit (float)
        """

        uncertainty_emin = 4*delta*delta_uncertainty/(delta*delta+1)

        return uncertainty_emin


        
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

             
