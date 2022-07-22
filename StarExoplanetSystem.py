from astropy import units as un
import argparse
class System:
    def __init__(self,
                 primary_mass,
                 secondary_mass,
                 secondary_radius,
                 feh,
                 orbital_period,
                 obliquity,
                 age):
        self.primary_mass = primary_mass
        self.Mprimary = primary_mass
        self.secondary_mass = secondary_mass
        self.Msecondary = secondary_mass
        self.secondary_radius = secondary_radius
        self.Rsecondary = secondary_radius
        self.feh = feh
        self.Porb = orbital_period
        self.orbital_period = orbital_period
        self.obliquity = obliquity
        self.age = age

    def printing(self):
        print('Primary mass = ', self.primary_mass, '=', self.primary_mass.to(un.kg))
        print('Secondary mass = ', self.secondary_mass, '=', self.secondary_mass.to(un.kg))
        print('Secondary radius = ', self.secondary_radius, '=', self.secondary_radius.to(un.m))
        print('Stellar metallicity = ', self.feh.to(un.dimensionless_unscaled))
        print('Orbital period = ', self.orbital_period, '=', self.orbital_period.to(un.s))
        print('Obliquity = ', self.obliquity.to(un.deg))
        print('Age = ', self.age, '=', self.age.to(un.s))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-Mp', '--Primary_mass',
                        help = 'Store the mass of the primary body in the unit of solar mass',
                        type = float)
    parser.add_argument('-Ms', '--Secondary_mass',
                        help='Store the mass of the secondary body in the unit of earth mass',
                        type=float)
    parser.add_argument('-Rs', '--Secondary_radius',
                        help='Store the radius of the secondary body in the unit of earth radius',
                        type=float)
    parser.add_argument('-feh', '--Stellar_metallicity',
                        help='Store the metallicity of the star',
                        type=float)
    parser.add_argument('-Porb', '--Orbital_period',
                        help='Store the orbital period in the unit of days',
                        type=float)
    parser.add_argument('-Ob', '--Obliquity',
                        help='Store the obliquity in the unit of degrees',
                        type=float)
    parser.add_argument('-age', '--age',
                        help='Store the age of the star-exoplanet system in the unit of Gyear',
                        type=float)
    args = parser.parse_args()
    if args.Primary_mass != None \
            and args.Secondary_mass != None \
            and args.Secondary_radius != None \
            and args.Stellar_metallicity != None \
            and args.Orbital_period != None \
            and args.Obliquity != None \
            and args.age != None:
        starexoplanet_system = System(args.Primary_mass * un.solMass,
                                      args.Secondary_mass * un.earthMass,
                                      args.Secondary_radius * un.earthRad,
                                      args.Stellar_metallicity * un.dimensionless_unscaled,
                                      args.Orbital_period * un.d,
                                      args.Obliquity * un.degree,
                                      args.age * un.Gyr)
        starexoplanet_system.printing()

