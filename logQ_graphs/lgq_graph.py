#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 23:26:08 2019

@author: kartof1

Sorry this script is a mess -- Anna
"""

from matplotlib import pyplot
import matplotlib
import matplotlib.style
import matplotlib as mpl
mpl.style.use('default')
from astropy import constants
from general_purpose.planetary_system_io import read_nasa_planets

#Set the font and morker sizes for the figures
pyplot.rcParams.update({'axes.titlesize': 24, 'figure.titlesize': 24, 'axes.labelsize': 20, 'lines.linewidth': 3,
                        'lines.markersize': 10, 'xtick.labelsize': 16, 'ytick.labelsize': 16})

#Read in the file
file_name = 'system_data.csv'
system = read_nasa_planets(file_name, need_ages=False)

size = int(system.pl_pnum.size)

#A buncha lists for solar system planets, I didn't include all the planets for
#every graph as some of them would greatly throw off the scale of the graph.
#You can see which solar system planets are included in which list by looking
#at the corresponding figure legend at the bottom of the script
mass_list = [0.0553, 0.815, 1, 0.107, 14.5, 17.1, float('NaN')]
dens_list = [5.427,5.243,5.515,3.933,1.326,0.687,1.270]
rad_list = [0.383,0.949,1,0.532,9.45,4.01,3.88]
eccen_list = [0.2056,0.0068,0.0167,0.0934,0.0484,0.0542,0.0472,0.0086]
porb_list = [87.97, 224.70, 365.26, 686.98, 4332.82, 10755.7, 30687.15]
rad2 = [0.383,0.949,1,0.532,11.21,9.45,4.01]
smax_list = [0.387, 0.723, 1.0, 1.524, 5.20, 9.54, 19.19]

#logQ values for solar system planets, the different lists are to match
#different planets in the lists above
ss_lgq = [2.28, 1.24, 1.12, 1.42, 4.86, 4.86, float('NaN')] #, 6.0, 4.81
ss1 = [2.28, 1.24, 1.12, 1.42, 6.0, 4.81, 4.86]
ss2 = [2.28, 1.24, 1.12, 1.42, 4.81, 4.86, 4.86]

#Create all the figures and subplots to make graphs with error bars
mass_fig = pyplot.figure()
dens_fig = pyplot.figure()
rad_fig = pyplot.figure()
eccen_fig = pyplot.figure()

#Plots the exoplanets (ex) and solar system (ss) planets on the same figure
porb_fig, (exp, ssp) = pyplot.subplots(2)
rati_fig, (exr, ssr) = pyplot.subplots(2)

mass = mass_fig.add_subplot(111)
dens = dens_fig.add_subplot(111)
rad = rad_fig.add_subplot(111)
eccen = eccen_fig.add_subplot(111)

#Create a list containing the log(Q) values for the exoplanets found using find_lgq.py
lgq = [3.150521211268739, 4.920054497355928, 3.7459543641008506, 
       3.0792752810879858, 2.6359809088036923, 2.8485790643772644, 2.1432737833937483]

#Plots the exoplanets
for i in range(0, size):
    #Convert the mass and its error into earth masses
    con_mass = system.pl_massj[i] * constants.M_jup / constants.M_earth
    up_mass = system.pl_massjerr1[i] * constants.M_jup / constants.M_earth
    low_mass = system.pl_massjerr2[i] * constants.M_jup / constants.M_earth
    
    #Create the plots including error bars
    mass.errorbar(con_mass, lgq[i], xerr=[[-(low_mass)],[up_mass]], fmt='o')
    
    dens.errorbar(system.pl_dens[i], lgq[i], xerr=[[-(system.pl_denserr2[i])],[system.pl_denserr1[i]]], fmt='o')
    
    rad.errorbar(system.pl_rade[i], lgq[i], xerr=[[-(system.pl_radeerr2[i])],[system.pl_radeerr1[i]]], fmt='o')
    
    eccen.errorbar(system.pl_orbeccen[i], lgq[i], xerr=[[-(system.pl_orbeccenerr2[i])],[system.pl_orbeccenerr1[i]]], fmt='o')
    
    exp.errorbar(system.pl_orbper[i], lgq[i], xerr=[[-(system.pl_orbpererr2[i])],[system.pl_orbpererr1[i]]], markersize=10, fmt='o')
    
    #Convert the semimajor axis and planet radius into the same units
    con_rati = system.pl_orbsmax[i] * 23454.8 / system.pl_rade[i]
    exr.errorbar(con_rati, lgq[i], fmt='o')

#Plots the solar system planets
for i in range(0,size):

    mass.errorbar(mass_list[i], ss_lgq[i], fmt='P')
    dens.errorbar(dens_list[i], ss1[i], fmt='P')
    rad.errorbar(rad_list[i], ss2[i], fmt='P')
    eccen.errorbar(eccen_list[i], ss1[i], fmt='P')
    ssp.errorbar(porb_list[i], ss1[i], fmt='P')
    
    con_rati = smax_list[i] * 23454.8 / rad2[i]
    ssr.errorbar(con_rati, ss1[i], fmt='P')
    
#Set the title, labels, and legends for the plots   
mass.set_title("log(Q) vs Mass for Exoplanets\nwith semi-major axis <0.12 AU")
mass.set_ylabel("log(Q)")
mass.set_xlabel("Planet Mass (Earth masses)")
mass_fig.legend(('HD 15337 b', 'K2-265 b', 'Kepler-20 b', 'Kepler-20 c', 'Kepler-11 b', 'Kepler-11 c', 'Kepler-79 b',
             'Mercury', 'Venus', 'Earth', 'Mars', 'Uranus', 'Neptune'))
mass_fig.legend(bbox_to_anchor=(1.1, 1), loc=2, ncol=1, borderaxespad=0.5)

dens.set_title("log(Q) vs Density for Exoplanets\nwith semi-major axis <0.12 AU")
dens.set_ylabel("log(Q)")
dens.set_xlabel("Planet Density (g/cm^3)")
dens_fig.legend(('HD 15337 b', 'K2-265 b', 'Kepler-20 b', 'Kepler-20 c', 'Kepler-11 b', 'Kepler-11 c', 'Kepler-79 b',
             'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus'))
dens_fig.legend(bbox_to_anchor=(1.1, 1), loc=2, ncol=1, borderaxespad=0.5)

rad.set_title("log(Q) vs Radius for Exoplanets\nwith semi-major axis <0.12 AU")
rad.set_ylabel("log(Q)")
rad.set_xlabel("Planet Radius (Earth radii)")
rad_fig.legend(('HD 15337 b', 'K2-265 b', 'Kepler-20 b', 'Kepler-20 c', 'Kepler-11 b', 'Kepler-11 c', 'Kepler-79 b',
             'Mercury', 'Venus', 'Earth', 'Mars', 'Saturn', 'Uranus', 'Neptune'))
rad_fig.legend(bbox_to_anchor=(1.1, 1), loc=2, ncol=1, borderaxespad=0.5)

eccen.set_title("log(Q) vs Orbital Eccentricity for Exoplanets\nwith semi-major axis <0.12 AU")
eccen.set_ylabel("log(Q)")
eccen.set_xlabel("Orbital Eccentricity")
eccen_fig.legend(('HD 15337 b', 'K2-265 b', 'Kepler-20 b', 'Kepler-20 c', 'Kepler-11 b', 'Kepler-11 c', 'Kepler-79 b',
             'Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus'))
eccen_fig.legend(bbox_to_anchor=(1.1, 1), loc=2, ncol=1, borderaxespad=0.5)

porb_fig.suptitle("log(Q) vs Orbital Period for Exoplanets\nwith semi-major axis <0.12 AU")
exp.set(ylabel="log(Q)")
ssp.set(ylabel="log(Q)")
ssp.set(xlabel="Orbital Period (days)")
exp.legend(('HD 15337 b', 'K2-265 b', 'Kepler-20 b', 'Kepler-20 c', 'Kepler-11 b', 'Kepler-11 c', 'Kepler-79 b'))
ssp.legend(('Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus'))
porb_fig.legend(bbox_to_anchor=(1.1, 1), loc=2, borderaxespad=0.)

rati_fig.suptitle("log(Q) vs Semi-Major Axis/Planet Radius for Exoplanets\nwith semi-major axis <0.12 AU")
exr.set(ylabel="log(Q)")
ssr.set(ylabel="log(Q)")
ssr.set(xlabel="Semi-Major Axis/Planet Radius")
exr.legend(('HD 15337 b', 'K2-265 b', 'Kepler-20 b', 'Kepler-20 c', 'Kepler-11 b', 'Kepler-11 c', 'Kepler-79 b'))
ssr.legend(('Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus'))
rati_fig.legend(bbox_to_anchor=(1.1, 1), loc=2, borderaxespad=0.)

pyplot.show()
