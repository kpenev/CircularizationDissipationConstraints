import math
class TransitingExoplanet:
    #A transiting exoplanet is defined by following
    #attributes: KOI, length of semimajor axis (a),
    #orbital period (p), planet's radius (R_p),
    #parent star's radius (R_s), impact parameter (b)
    #actual transit period (T), transit period if its
    #orbit were circular (T_c)

    KOI = 'a'
    a = 1
    p = 1
    T = 1
    R_p = 1
    R_s = 1
    b = 1    
    T_c = 1
    e_min = 0
    delta = 1
    R_sun = 6.96 * pow(10,8)
    R_earth = 6378*1000
    AU = 1.496* pow(10,11)
    R_sun_over_AU = 0.00465 #is the Sun's radius divided by 1AU
    R_earth_over_R_sun = 0.00916 #is the Earth's radius divided by the Sun's radius

    
    #Define constructor:

    def _init_(self, KOI, semiMajorAxis, orbitalPeriod, transitTime, radiusOfPlanet, radiusOfParentStar, impactParameter):
        self.KOI = KOI
        self.a = semiMajorAxis #in AU
        self.p = orbitalPeriod #in days
        self.T =  transitTime  #in hours 
        self.R_p = radiusOfPlanet #in the unit of the Earth's radius, R_earth
        self.R_s = radiusOfParentStar #in the unit of the Sun's radius, R_sun
        self.b = impactParameter #in the unit of parent star's radius
        self.T_c = self.durationForCircularOrbit(self.R_s, self.R_p, self.b, self.a, self.p) #in the unit of days
        self.delta = self.delta(self.T, self.T_c) #is dimensionless
        self.e_min = self.emin(self.delta) #is dimensionless

    #Define a function for calculating
    #the duration of circular orbit, T_c
   
    def durationForCircularOrbit(self, radiusOfParentStar, radiusOfPlanet, impactParameter, semiMajorAxis, orbitalPeriod):        
        temp = (1+radiusOfPlanet/radiusOfParentStar * self.R_earth_over_R_sun)        
        temp = temp*temp - impactParameter*impactParameter
        temp = math.sqrt(temp)        
        temp = temp * radiusOfParentStar/3.1416/semiMajorAxis * orbitalPeriod * self.R_sun_over_AU * 24 #last 24 is for converting 1 day to 24 hours       
        return temp #in hours

    #Define a function for calculating delta:

    def delta(self, transitTime, transitTimeForCircularOrbit):
        return transitTime/transitTimeForCircularOrbit
    
    #Define minimum eccentricity:
    #e_min is the minimum eccentricity
    
    def emin(self, delta):         
        return abs((delta * delta - 1)/(delta * delta+1))

    #Define a method which prints all attributes

    def printAttributes(self):
        print(self.KOI, '   ', self.a, '    ', self.p, '    ', self.T, '    ', self.R_p, '  ', self.R_s, '  ', self.b, '    ', self.T_c, '  ', self.delta, '    ', self.e_min)

#______________________________________________________________________________________________________________________________________________________________________________

#Testing the class TransitingExoplanet

#Data on exoplanets found in Barnes' paper's table 1:
KOI = [01.01, 02.01, 03.01, 04.01, 05.01, 05.02, 07.01, 10.01, 17.01, 18.01, 20.01]
a = [0.036, 0.039, 0.052, 0.056, 0.058, 0.075, 0.044, 0.047, 0.045, 0.052, 0.056]
P = [2.471, 2.205, 4.888, 3.849, 4.780, 7.052, 3.214, 3.522, 3.235, 3.548, 4.438]
T = [1.732, 3.877, 2.368, 2.928, 2.012, 3.688, 4.111, 3.198, 3.602, 4.081, 4.671] 
Rp = [14.42, 22.29, 04.67, 11.79, 05.65, 00.66, 03.72, 15.88, 11.06, 17.37, 17.58]
Rs = [1.06, 2.71, 0.74, 2.60, 1.42, 1.42, 1.27, 1.56, 1.08, 2.02, 1.38]*56378*1000 
b = [0.822, 0.128, 0.029, 0.946, 0.951, 0.750, 0.640, 0.029, 0.006, 0.018]
Tc = [1.984, 5.810, 2.612, 2.764, 1.716, 3.169, 2.431, 3.682, 3.015, 5.282, 4.338]

#We are going to reproduce the table 1 of Barnes' paper.
#Following command is to print the attributes of the table
print('KOI', '    ', '  a', '    ', '    P', '    ', '    T', '    ', '    Rp', '   ', ' Rs', ' ', '    b', '    ', '         Tc', '    ','             delta','   ','              emin')

#Let's define an array called planet which will store
#11 instances of class TransitingExoplanet
#The first element of this array is initialized in the following
#way:
planet = [TransitingExoplanet()]
planet[0]._init_(KOI[0], a[0], P[0], T[0], Rp[0], Rs[0], b[0])
planet[0].printAttributes()

#Now we are running a loop to insert records of rest of the planets
#in the table:

for i in range(1, 10):
   planet = planet + [TransitingExoplanet()]
   planet[i]._init_(KOI[i], a[i], P[i], T[i], Rp[i], Rs[i], b[i])
   planet[i].printAttributes()

#_______________________________________________________________________________________________________________________________________________________________________________

#Data from planets_2019.09.18_13.19.49.csv found in the Data folder

import planetary_system_io
readPlanet = planetary_system_io.read_nasa_planets('C:/Users/moham/OneDrive/Documents/GitHub/CircularizationDissipationConstraints/data/planets_2019.09.18_13.19.49.csv',
                     eliminate=('SWEEPS-11',
                                'HD 41004 B',
                                'PSR J1719-1438',
                                'K2-22'),
                     need_ages=False,
                     )

#Now we are taking data on planets from the file
KOI = readPlanet.pl_name
a = readPlanet.pl_orbsmax
P = readPlanet.pl_orbper
T = readPlanet.pl_trandur
Rp = readPlanet.pl_rade
Rs = readPlanet.st_rad
b = readPlanet.pl_imppar

#Following command is to print the attributes of the table
print('Planet Name', '    ', '  a', '    ', '    P', '    ', '    T', '    ', '    Rp', '   ', ' Rs', ' ', '    b', '    ', '         Tc', '    ','             delta','   ','              emin')

planet = [TransitingExoplanet()]
j = -1
for i in range(0, (len(KOI)-1)):
    if not(math.isnan(a[i]) or math.isnan(P[i])or math.isnan(T[i])or math.isnan(Rp[i]) or math.isnan(Rs[i]) or math.isnan(b[i])):
        j = j + 1
        planet[j]._init_(KOI[i], a[i], P[i], T[i], Rp[i], Rs[i], b[i])
        planet[j].printAttributes()
        planet = planet + [TransitingExoplanet()]
        
#____________________________________________________________________________________________________________________________________________________________________________    
#Data from q1_q8_koi_2020.03.16_21.25.09.csv found in the Data folder
print('Now Printing results from q1_q8_koi_2020.03.16_21.25.09.csv')
import planetary_system_io
readPlanet = planetary_system_io.read_nasa_planets('C:/Users/moham/OneDrive/Documents/GitHub/CircularizationDissipationConstraints/data/new_q1_q8_koi_2020.03.16_21.25.09.csv',
                     eliminate=('SWEEPS-11',
                                'HD 41004 B',
                                'PSR J1719-1438',
                                'K2-22'),
                     need_ages=False,
                     )


#Now we are taking data on planets from the file
KOI = readPlanet.kepoi_name
Kepler = readPlanet.kepler_name
a = readPlanet.koi_sma
P = readPlanet.koi_period
T = readPlanet.koi_duration
Rp = readPlanet.koi_prad
Rs = readPlanet.koi_srad
b = readPlanet.koi_impact

#Following command is to print the attributes of the table
print('Planet Name', '    ', '  a', '    ', '    P', '    ', '    T', '    ', '    Rp', '   ', ' Rs', ' ', '    b', '    ', '         Tc', '    ','             delta','   ','              emin')

planet = [TransitingExoplanet()]
j = -1
for i in range(0, (len(KOI)-1)):
    if not(math.isnan(a[i]) or math.isnan(P[i])or math.isnan(T[i])or math.isnan(Rp[i]) or math.isnan(Rs[i]) or math.isnan(b[i])):
        j = j + 1
        planet[j]._init_(KOI[i], a[i], P[i], T[i], Rp[i], Rs[i], b[i])
        planet[j].printAttributes()
        planet = planet + [TransitingExoplanet()]
