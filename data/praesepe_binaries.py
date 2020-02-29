"""Data for binary stars in/near the Praesepe open cluster."""

import os.path

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
    result = []
    for sys_orbit, sys_error in zip(orbits, orbit_errors):
        assert sys_orbit['KW'] == sys_error['KW']
        id_str = (('' if sys_orbit['KW'].startswith(b'VL') else 'KW')
                  +
                  sys_orbit['KW'].decode())
        result.append(
            dict(
                ID=id_str,
                type=(
                    ('Double' if scipy.isfinite(sys_orbit['K2']) else 'Single')
                    +
                    ' Lined'
                ),
                #False positive
                #pylint: disable=no-member
                Porb=sys_orbit['Porb'] * u.day,
                errPorb=sys_error['Porb'] * u.day,
                Gamma=sys_orbit['Gamma'] * u.km / u.s,
                errGamma=sys_error['Gamma'] * u.km / u.s,
                K1=sys_orbit['K1'] * u.km / u.s,
                errK1=sys_error['K1'] * u.km / u.s,
                Ecc=sys_orbit['Ecc'],
                errEcc=sys_error['Ecc'],
                Omega=sys_orbit['Omega'] * u.deg,
                errOmega=sys_error['Omega'] * u.deg,
                MassFunc=sys_orbit['MassFunc'] * u.M_sun,
                errMassFunc=sys_error['MassFunc'] * u.M_sun,
                ProjSemimajor1=sys_orbit['ProjSemimajor'] * u.Gm,
                errProjSemimajor1=sys_error['ProjSemimajor'] * u.Gm,
                member=not id_str in _non_members
                #pylint: enable=no-member
            )
        )
        if scipy.isfinite(sys_orbit['K2']):
            result[-1]['K2'] = sys_orbit['K2']
            result[-1]['errK2'] = sys_error['K2']
        mass_index = scipy.where(mass_fits['KW'] == sys_orbit['KW'])[0]
        if mass_index:
            sys_mass_fit = mass_fits[mass_index]
            result[-1]['ModelM1'] = sys_mass_fit['Ma']
            if scipy.isfinite(sys_mass_fit['Mb']):
                result[-1]['ModelM2'] = sys_mass_fit['Mb']
            elif sys_mass_fit['q'] == b'0.99':
                result[-1]['ModelM2'] = sys_mass_fit['Ma']
            else:
                result[-1]['ModelM2'] = (sys_mass_fit['Mbmin'],
                                         sys_mass_fit['Mbmax'])
        else:
            result[-1]['ModelM1'] = result[-1]['ModelM2'] = scipy.nan
        #False positive
        #pylint: disable=no-member
        result[-1]['ModelM1'] *= u.M_sun
        result[-1]['ModelM2'] *= u.M_sun
        #pylint: enable=no-member
    return result

if __name__ == '__main__':
    systems = read_systems()
    keys = list(systems[0].keys())
    keys.remove('ID')
    for sys in systems:
        print(sys['ID'] + ':')
        for k in keys:
            print('\t%s: %s' % (repr(k), repr(sys[k])))
