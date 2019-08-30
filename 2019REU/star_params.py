#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 16:42:27 2019

@author: annamtetz
"""
import numpy

def find_star_params(system, interpolator, param, i):
    """
        Solves for the mass and age of the star using the star's metallicity,
        effective temperature, and a third parameter specified by the user.
        Returns a dictionary containing the found mass and age of the star.

        Args:
            system:    The parameters of the system we are trying to reproduce.
            
            interpolator:   The stellar evolution interpolator to use to find
                the mass and age of the star.
                
            param: The third stellar parameter to pass to change_variables to
                find the mass and age of the star.
                
            i:  The index number for the system

        Returns:
            star_params:
                A dictionary of the star's found mass and age with the keys 
                ['mass'] and ['age'].
        """
    
    if param == 'logg':
        mass_age = interpolator.change_variables(feh=system.st_metfe[i],
                                                 teff=system.st_teff[i],
                                                 logg=system.st_logg[i])
    elif param == 'rho':
        mass_age = interpolator.change_variables(feh=system.st_metfe[i],
                                                 teff=system.st_teff[i],
                                                 rho=system.st_dens[i])
    elif param == 'lum':
        mass_age = interpolator.change_variables(feh=system.st_metfe[i],
                                                 teff=system.st_teff[i],
                                                 lum=10**(system.st_lum[i]))
    
    choice = 0
    
    if len(mass_age) == 0:
        array = numpy.zeros(2)
        mass_age.append(array)
#    elif len(mass_age) > 0:
#        print(mass_age)
#        choice = int(input("Enter the number of the mass and age set you would" +
#                           " like to use (0 for the first, 1 for the second, etc.)"))
    
    split = numpy.split(mass_age[choice],2)
    star_params = {'mass': split[0],
                   'age': split[1]}
        
    return star_params
