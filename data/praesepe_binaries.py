"""Data for binary stars in/near the Praesepe open cluster."""

import os.path

from matplotlib import pyplot
import scipy
from astropy import units as u

_data_dir = os.path.dirname(__file__)

_non_members = ['KW55', 'KW258', 'KW425', 'KW530', 'KW548', 'KW553', 'KW557']

def read_systems(
        orbits_fname=os.path.join(_data_dir,
                                  'praesepe_binary_orbits.txt'),
        orbit_errors_fname=os.path.join(_data_dir,
                                       'praesepe_binary_orbit_errors.txt'),
        mass_fits_fname=os.path.join(_data_dir,
                                     'praesepe_mass_fits.txt')
):
    """Read the systems from the files containing the data to list of dicts."""

    orbits = scipy.genfromtxt(orbits_fname,
                              dtype=None,
                              names=True,
                              delimiter=',')
    orbit_errors = scipy.genfromtxt(orbit_errors_fname,
                                    dtype=None,
                                    names=True,
                                    delimiter=',')
    mass_fits = scipy.genfromtxt(mass_fits_fname,
                                 dtype=None,
                                 names=True)

    print('Orbits: ' + repr(orbits))
    print('Errors: ' + repr(orbit_errors))
    print('Mass: ' + repr(mass_fits))

if __name__ == '__main__':
    read_systems()
