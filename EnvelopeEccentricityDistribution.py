import math
import numpy as np
import Star_Exoplanet_system_io
import Constraints_for_selecting_systems
import matplotlib.pyplot as plt
import logging
import argparse
from astropy import units as unit
from astropy import constants as const
import os
import json

def ensure_directory(fname):
        """Make sure the directory containing the given name exists."""
        dirname = os.path.dirname(fname)
        if not os.path.exists(dirname):
            os.makedirs(dirname)

def getPathOfExoplanetSystemsData():
    return '/home1/08529/mmmahmud/CircularizationDissipationConstraints/data/PS_2022.10.19_20.26.52.csv'

def envelope_eccentricity_function(x,
                                   threshold_value_of_envelope_eccentricity = 0.01,
                                   largest_acceptable_value_of_envelope_eccentricity = 0.5,
                                   logger = None):

    """
        Workout the envelope eccentricity given the log(semi major axis / planetary radius)
         from a scatter plot of eccentricity vs. log(semi major axis/planetary radius) graph

        Args:
            x:                                                            semi major axis/ secondary radius
            threshold_value_of_envelope_eccentricity:                     threshold value of envelope eccentricity
            largest_acceptable_value_of_envelope_eccentricity:            largest acceptable value of envelope eccentricity

        Returns:
            envelope eccentricity
    """

    logx = math.log(x, 10)
    crit1 = -2.7
    crit2 = -2.32
    if logx <= crit1:
        return threshold_value_of_envelope_eccentricity
    if logx > crit2:
        if logger is not None:
                 logging.warning('Tidal effect on this system is negligible')
        return largest_acceptable_value_of_envelope_eccentricity
    if logx > crit1 and logx <= crit2:
        return threshold_value_of_envelope_eccentricity + (logx - crit1) * (
                largest_acceptable_value_of_envelope_eccentricity - threshold_value_of_envelope_eccentricity) / (
                       crit2 - crit1)
    return


class EnvelopeEccentricityDistribution:
    path_name_of_exoplanet_systems_database = getPathOfExoplanetSystemsData()

    @classmethod
    def get_path_name_of_exoplanet_systems_database(cls):
        return cls.path_name_of_exoplanet_systems_database

    @classmethod
    def set_path_name_of_exoplanet_systems_database(cls, path_name):
        cls.path_name_of_exoplanet_systems_database = path_name

    def __init__(self,
                 maximum_number_of_data_points=math.inf,
                 threshold_value_of_envelope_eccentricity=0.001,
                 constraints=Constraints_for_selecting_systems.constraints_for_eccentricity_envelope(),
                 largest_acceptable_value_of_envelope_eccentricity=0.5
                 ):
        readPlanet = Star_Exoplanet_system_io.read_nasa_planets(self.path_name_of_exoplanet_systems_database,
                                       eliminate=('SWEEPS-11', 'HD 41004 B', 'PSR J1719-1438', 'K2-22', 'HATS-67 b'),
                                       need_ages=False, )
        self.output_directory = "/work/08529/mmmahmud/scratch/circularization_exoplanet_system"
        ensure_directory(self.output_directory)

        self.planet_name = readPlanet.pl_name
        self.orbital_period = readPlanet.pl_orbper  # days
        self.orbital_period_upper_uncertainty = readPlanet.pl_orbpererr1
        self.orbital_period_lower_uncertainty = readPlanet.pl_orbpererr2
        self.orbital_period_limit_flag = readPlanet.pl_orbperlim
        self.primary_mass = readPlanet.st_mass  # solar mass
        self.primary_mass_upper_uncertainty = readPlanet.st_masserr1
        self.primary_mass_lower_uncertainty = readPlanet.st_masserr2
        self.primary_mass_limit_flag = readPlanet.st_masslim
        self.secondary_mass = readPlanet.pl_masse  # Earth mass
        self.secondary_mass_upper_uncertainty = readPlanet.pl_masseerr1
        self.secondary_mass_lower_uncertainty = readPlanet.pl_masseerr2
        self.secondary_mass_limit_flag = readPlanet.pl_masselim
        self.stellar_metallicity = readPlanet.st_met  # or readPlanet.st_metfe
        self.stellar_metallicity_upper_uncertainty = readPlanet.st_meterr1  # or readPlanet.st_metfeerr1
        self.stellar_metallicity_lower_uncertainty = readPlanet.st_meterr2  # or readPlanet.st_metfeerr2
        self.stellar_metallicity_limit_flag = readPlanet.st_metlim
        self.semi_major_axis = readPlanet.pl_orbsmax
        self.semi_major_axis_upper_uncertainty = readPlanet.pl_orbsmaxerr1
        self.semi_major_axis_lower_uncertainty = readPlanet.pl_orbsmaxerr2
        self.semi_major_axis_flag_limit = readPlanet.pl_orbsmaxlim
        self.primary_radius = readPlanet.st_rad  # solar radius
        self.primary_radius_upper_uncertainty = readPlanet.st_raderr1
        self.primary_radius_lower_uncertainty = readPlanet.st_raderr2
        self.primary_radius_limit_flag = readPlanet.st_radlim
        self.secondary_radius = readPlanet.pl_rade  # earth radius
        self.secondary_radius_upper_uncertainty = readPlanet.pl_radeerr1
        self.secondary_radius_lower_uncertainty = readPlanet.pl_radeerr2
        self.secondary_radius_limit_flag = readPlanet.pl_radelim
        self.stellar_age = readPlanet.st_age  # GYear
        self.stellar_age_upper_uncertainty = readPlanet.st_ageerr1
        self.stellar_age_lower_uncertainty = readPlanet.st_ageerr2
        self.stellar_age_limit_flag = readPlanet.st_agelim
        self.eccentricity_now = readPlanet.pl_orbeccen
        self.eccentricity_now_upper_uncertainty = readPlanet.pl_orbeccenerr1
        self.eccentricity_now_lower_uncertainty = readPlanet.pl_orbeccenerr2
        self.eccentricity_now_limit_flag = readPlanet.pl_orbeccenlim
        self.obliquity = readPlanet.pl_trueobliq  # degrees readPlanet.pl_orbincl
        self.obliquity_upper_uncertainty = readPlanet.pl_trueobliqerr1 # readPlanet.pl_orbinclerr1
        self.obliquity_lower_uncertainty = readPlanet.pl_trueobliqerr2 # readPlanet.pl_orbinclerr2
        self.obliquity_limit_flag = readPlanet.pl_trueobliqlim # readPlanet.pl_orbincllim
        self.stellar_log_g = readPlanet.st_logg  # log10(cm/s**2)
        self.stellar_log_g_upper_uncertainty = readPlanet.st_loggerr1
        self.stellar_log_g_lower_uncertainty = readPlanet.st_loggerr2
        self.stellar_log_g_limit_flag = readPlanet.st_logglim
        self.stellar_effective_temperature = readPlanet.st_teff  # Kelvin
        self.stellar_effective_temperature_upper_uncertainty = readPlanet.st_tefferr1
        self.stellar_effective_temperature_lower_uncertainty = readPlanet.st_tefferr2
        self.stellar_effective_temperature_limit_flag = readPlanet.st_tefflim
        self.stellar_density = readPlanet.st_dens  # gm/cm**3
        self.stellar_density_upper_uncertainty = readPlanet.st_denserr1
        self.stellar_density_lower_uncertainty = readPlanet.st_denserr2
        self.stellar_density_limit_flag = readPlanet.st_denslim
        self.ratio_of_planet_to_stellar_radius = readPlanet.pl_ratror
        self.ratio_of_planet_to_stellar_radius_upper_uncertainty = readPlanet.pl_ratrorerr1
        self.ratio_of_planet_to_stellar_radius_lower_uncertainty = readPlanet.pl_ratrorerr2
        self.transit_depth = readPlanet.pl_trandep
        self.transit_depth_upper_uncertainty = readPlanet.pl_trandeperr1
        self.transit_depth_lower_uncertainty = readPlanet.pl_trandeperr2
        self.transit_depth_limit_flag = readPlanet.pl_trandeplim
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
        self.envelope_eccentricity_function = self.create_envelope_eccentricity_function(maximum_number_of_data_points,
                                                                                         threshold_value_of_envelope_eccentricity,
                                                                                         largest_acceptable_value_of_envelope_eccentricity,
                                                                                         constraints)


    def create_envelope_eccentricity_function(self,
                                              maximum_number_of_data_points,
                                              threshold_value_of_envelope_eccentricity,
                                              largest_acceptable_value_of_envelope_eccentricity,
                                              constraints=Constraints_for_selecting_systems.constraints_for_eccentricity_envelope(),
                                              x_attribute='log of semi major axis over planetary radius',
                                              # or 'log of orbital period'
                                              manual_envelope=True,
                                              eliminate=('HATS-67 b', 'HATS-69 b', 'HATS-62 b')):

        def find_records_of_the_points_on_envelope(records_of_certain_attribute_and_eccentricity_now, attribute):
            records_of_the_points_on_envelope = []
            maximum_eccentricity_now = threshold_value_of_envelope_eccentricity
            starting_point_of_the_significant_tidal_dissipation_region_found = False
            end_point_of_the_significant_tidal_dissipation_region_found = False
            for i in range(0, len(records_of_certain_attribute_and_eccentricity_now)):
                present_eccentricity_of_ith_element = records_of_certain_attribute_and_eccentricity_now[i][
                    'present eccentricity']
                if present_eccentricity_of_ith_element > maximum_eccentricity_now:
                    if (not starting_point_of_the_significant_tidal_dissipation_region_found) and i > 0:
                        records_of_the_points_on_envelope = records_of_the_points_on_envelope + [
                            {attribute: records_of_certain_attribute_and_eccentricity_now[i - 1][attribute],
                             'envelope eccentricity': threshold_value_of_envelope_eccentricity}]
                        logging.debug('Name of the planet belongs to the binary system on the envelope %(planet)s %(att)s = %(val)f envelope eccentricity = %(eenv)f orbital eccentricity flag limit = %(flag)f = ' % dict(planet=records_of_certain_attribute_and_eccentricity_now[i - 1]['planet name'], att=attribute, val=records_of_certain_attribute_and_eccentricity_now[i - 1][attribute], eenv=threshold_value_of_envelope_eccentricity, flag=records_of_certain_attribute_and_eccentricity_now[i - 1]['orbital eccentricity limit flag']))
                    if present_eccentricity_of_ith_element <= largest_acceptable_value_of_envelope_eccentricity:
                        maximum_eccentricity_now = present_eccentricity_of_ith_element
                    if present_eccentricity_of_ith_element > largest_acceptable_value_of_envelope_eccentricity:
                        maximum_eccentricity_now = largest_acceptable_value_of_envelope_eccentricity
                        end_point_of_the_significant_tidal_dissipation_region_found = True
                    starting_point_of_the_significant_tidal_dissipation_region_found = True
                    records_of_the_points_on_envelope = records_of_the_points_on_envelope + [{attribute:
                                                                                                  records_of_certain_attribute_and_eccentricity_now[
                                                                                                      i][
                                                                                                      attribute],
                                                                                              'envelope eccentricity': maximum_eccentricity_now}]
                    logging.debug('Name of the planet belongs to the binary system on the envelope %(name)s ' % dict(name=records_of_certain_attribute_and_eccentricity_now[i]['planet name']))
                    logging.debug('%(att)s = %(val)f' % dict(att=attribute, val=records_of_certain_attribute_and_eccentricity_now[i][attribute]))
                    logging.debug('envelope eccentricity = %(eenv)f' % dict(eenv=maximum_eccentricity_now))
                    logging.debug('orbital eccentricity limit flag = %(flag)f' % dict(flag=records_of_certain_attribute_and_eccentricity_now[i]['orbital eccentricity limit flag']))
                    if end_point_of_the_significant_tidal_dissipation_region_found:
                        break
            return records_of_the_points_on_envelope

        if x_attribute == 'log of orbital period':
            # For present eccentricity vs. log of orbital period plot:
            records_of_log_orbital_period_and_eccentricity_now = []
            j = -1
            for i in range(0, len(self.orbital_period)):
                if not (math.isnan(self.orbital_period[i])
                        or math.isnan(self.eccentricity_now[i])
                ) and self.eccentricity_now_limit_flag[i] == 0 and not (self.planet_name[i] in eliminate):
                    if (Constraints_for_selecting_systems.constraints_for_eccentricity_envelope_are_satisfied(
                            secondary_radius=self.secondary_radius[i],
                            planet_mass=self.secondary_mass[i],
                            constraints=constraints)):
                        records_of_log_orbital_period_and_eccentricity_now = (
                                records_of_log_orbital_period_and_eccentricity_now
                                + [{'log of orbital period': math.log(self.orbital_period[i], 10),
                                    'present eccentricity': self.eccentricity_now[i],
                                    'planet name': self.planet_name[i],
                                    'primary mass': self.primary_mass[i],
                                    'orbital eccentricity limit flag': self.eccentricity_now_limit_flag[i]}])
                        j = j + 1
                if j >= maximum_number_of_data_points - 1:
                    break
            # Sorting
            records_of_log_orbital_period_and_eccentricity_now = sorted(
                records_of_log_orbital_period_and_eccentricity_now,
                key=lambda key_attribute_for_sorting: key_attribute_for_sorting['log of orbital period'])
            # Finding points on the envelope
            records_of_the_points_on_envelope = find_records_of_the_points_on_envelope(
                records_of_log_orbital_period_and_eccentricity_now, x_attribute)
        if x_attribute == 'log of semi major axis over planetary radius':
            # For present eccentricity vs. log of semi-major axis over planetary radius plot:
            records_of_log_semi_major_axis_over_planetary_radius_and_eccentricity_now = []
            j = -1
            for i in range(0, len(self.semi_major_axis)):
                if not (math.isnan(self.semi_major_axis[i])
                        or math.isnan(self.secondary_radius[i])
                        or math.isnan(self.eccentricity_now[i])
                ) and self.eccentricity_now_limit_flag[i] == 0 and self.semi_major_axis_flag_limit[i] == 0 and not (
                        self.planet_name[i] in eliminate):
                    if (Constraints_for_selecting_systems.constraints_for_eccentricity_envelope_are_satisfied(
                            secondary_radius=self.secondary_radius[i],
                            planet_mass=self.secondary_mass[i],
                            constraints=constraints)):
                        records_of_log_semi_major_axis_over_planetary_radius_and_eccentricity_now = (
                                records_of_log_semi_major_axis_over_planetary_radius_and_eccentricity_now
                                + [{'log of semi major axis over planetary radius': (
                            math.log(self.semi_major_axis[i] / self.secondary_radius[i], 10)),
                            'present eccentricity': self.eccentricity_now[i],
                            'planet name': self.planet_name[i],
                            'primary mass': self.primary_mass[i],
                            'orbital eccentricity limit flag': self.eccentricity_now_limit_flag[i],
                            'present eccentricity upper uncertainty': self.eccentricity_now_upper_uncertainty[i],
                            'present eccentricity lower uncertainty': self.eccentricity_now_lower_uncertainty[i]}])
                        j = j + 1

                if j >= maximum_number_of_data_points - 1:
                    break
            # Sorting
            records_of_log_semi_major_axis_over_planetary_radius_and_eccentricity_now = sorted(
                records_of_log_semi_major_axis_over_planetary_radius_and_eccentricity_now,
                key=lambda key_attribute_for_sorting:
                key_attribute_for_sorting['log of semi major axis over planetary radius'])
            # Finding points on the envelope
            records_of_the_points_on_envelope = find_records_of_the_points_on_envelope(
                records_of_log_semi_major_axis_over_planetary_radius_and_eccentricity_now, x_attribute)

        def envelope_eccentricity_function(x):
            log_of_x = math.log(x, 10)
            min_log_of_x = records_of_the_points_on_envelope[0][x_attribute]
            max_log_of_x = records_of_the_points_on_envelope[-1][x_attribute]
            if log_of_x <= min_log_of_x:
                return threshold_value_of_envelope_eccentricity
            if log_of_x >= max_log_of_x:
                return largest_acceptable_value_of_envelope_eccentricity
            for i in range(1, len(records_of_the_points_on_envelope)):
                if log_of_x == records_of_the_points_on_envelope[i][x_attribute]:
                    return records_of_the_points_on_envelope[i]['envelope eccentricity']
                if ((log_of_x < records_of_the_points_on_envelope[i][x_attribute])
                        and (log_of_x > records_of_the_points_on_envelope[i - 1][x_attribute])):
                    return (records_of_the_points_on_envelope[i]['envelope eccentricity']
                            + (records_of_the_points_on_envelope[i]['envelope eccentricity']
                               - records_of_the_points_on_envelope[i - 1]['envelope eccentricity'])
                            / (records_of_the_points_on_envelope[i][x_attribute]
                               - records_of_the_points_on_envelope[i - 1][x_attribute])
                            * (log_of_x - records_of_the_points_on_envelope[i][x_attribute]))
            return

        def envelope_eccentricity_function_manual(x):
            logx = math.log(x, 10)
            if x_attribute == 'log of orbital period':
                crit1 = -0.11
                crit2 = 0.7
            if x_attribute == 'log of semi major axis over planetary radius':
                crit1 = -2.7
                crit2 = -2.32
            if logx <= crit1:
                return threshold_value_of_envelope_eccentricity
            if logx > crit2:
                return largest_acceptable_value_of_envelope_eccentricity
            if logx > crit1 and logx <= crit2:
                return threshold_value_of_envelope_eccentricity + (logx - crit1) * (
                            largest_acceptable_value_of_envelope_eccentricity - threshold_value_of_envelope_eccentricity) / (
                                   crit2 - crit1)
            return

        if x_attribute == 'log of orbital period':
            self.plot_present_eccentricity_vs_x_attribute(records_of_log_orbital_period_and_eccentricity_now,
                                                          records_of_the_points_on_envelope,
                                                          envelope_eccentricity_function_manual if manual_envelope else envelope_eccentricity_function,
                                                          x_attribute)
        if x_attribute == 'log of semi major axis over planetary radius':
            self.plot_present_eccentricity_vs_x_attribute(
                records_of_log_semi_major_axis_over_planetary_radius_and_eccentricity_now,
                records_of_the_points_on_envelope,
                envelope_eccentricity_function_manual if manual_envelope else envelope_eccentricity_function,
                x_attribute)
        if manual_envelope:
            return envelope_eccentricity_function_manual
        else:
            return envelope_eccentricity_function

    def plot_present_eccentricity_vs_x_attribute(self,
                                                 records_of_x_attribute_and_eccentricity_now,
                                                 records_of_the_points_on_envelope,
                                                 envelope_eccentricity_function,
                                                 x_attribute):
        x = [element[x_attribute] for element in records_of_x_attribute_and_eccentricity_now]
        eccentricity_now = [element['present eccentricity'] for element in records_of_x_attribute_and_eccentricity_now]
        eccentricity_now_errormax = [math.fabs(element['present eccentricity upper uncertainty']) for element in records_of_x_attribute_and_eccentricity_now]
        eccentricity_now_errormin = [math.fabs(element['present eccentricity lower uncertainty']) for element in records_of_x_attribute_and_eccentricity_now]
        primary_mass = [element['primary mass'] for element in records_of_x_attribute_and_eccentricity_now]
        x_on_envelope = [element[x_attribute] for element in records_of_the_points_on_envelope]
        eccentricity_now_on_envelope = [element['envelope eccentricity'] for element in
                                        records_of_the_points_on_envelope]
        xdata = np.linspace(records_of_x_attribute_and_eccentricity_now[0][x_attribute],
                            records_of_x_attribute_and_eccentricity_now[-1][x_attribute],
                            50)
        ydata = []
        xdata_ = []
        for i in range(0, len(xdata)):
            ydata = ydata + [envelope_eccentricity_function(10 ** xdata[i])]
        for i in range(0, len(xdata)):
            xdata_ = xdata_ + [10**(xdata[i] + 4.37023)]

        x_1 = []
        eccentricity_now_1 = []
        eccentricity_now_errormax_1 = []
        eccentricity_now_errormin_1 = []
        x_2 = []
        eccentricity_now_2 = []
        eccentricity_now_errormax_2 = []
        eccentricity_now_errormin_2 = []
        x_3 = []
        eccentricity_now_3 = []
        eccentricity_now_errormax_3 = []
        eccentricity_now_errormin_3 = []
        for i in range(0, len(records_of_x_attribute_and_eccentricity_now)):
            if primary_mass[i] < 1.2:
                x_1 = x_1 + [10**(x[i] + 4.37023)]
                eccentricity_now_1 = eccentricity_now_1 + [eccentricity_now[i]]
                eccentricity_now_errormax_1 = eccentricity_now_errormax_1 + [eccentricity_now_errormax[i]]
                eccentricity_now_errormin_1 = eccentricity_now_errormin_1 + [eccentricity_now_errormin[i]]
            else:
                x_2 = x_2 + [10**(x[i]+4.37023)]
                eccentricity_now_2 = eccentricity_now_2 + [eccentricity_now[i]]
                eccentricity_now_errormax_2 = eccentricity_now_errormax_2 + [eccentricity_now_errormax[i]]
                eccentricity_now_errormin_2 = eccentricity_now_errormin_2 + [eccentricity_now_errormin[i]]
            x_3 = x_3 + [10**(x[i]+4.37023)]
            eccentricity_now_3 = eccentricity_now_3 + [eccentricity_now[i]]
            eccentricity_now_errormax_3 = eccentricity_now_errormax_3 + [eccentricity_now_errormax[i]]
            eccentricity_now_errormin_3 = eccentricity_now_errormin_3 + [eccentricity_now_errormin[i]]

        #plt.plot(x_1, eccentricity_now_1, 'x')
        y_error = [eccentricity_now_errormax_1, eccentricity_now_errormin_1]
        plt.errorbar(x_1, eccentricity_now_1, yerr = y_error, fmt = 'x')

        #plt.plot(x_2, eccentricity_now_2, 'o')
        y_error = [eccentricity_now_errormax_2, eccentricity_now_errormin_2]
        plt.errorbar(x_2, eccentricity_now_2, yerr = y_error, fmt = 'o')

        #plt.plot(x_3, eccentricity_now_3, 'x')
        plt.plot(xdata_, ydata)
        #y_error = [eccentricity_now_errormax_3, eccentricity_now_errormin_3]
        #plt.errorbar(x_3, eccentricity_now_3, yerr = y_error, fmt = 'o')

        #plt.plot(x_on_envelope, eccentricity_now_on_envelope, '.')
        # naming the x axis
        if x_attribute == 'log of semi major axis over planetary radius':
            x_attribute_ = 'Semi major axis over planetary radius'
        else:
            x_attribute_ = x_attribute
        plt.xlabel(x_attribute_)
        # naming the y axis
        plt.ylabel('Present Eccentricity')
        # giving a title to my graph
        #string = 'Present Eccentricity vs. ' + x_attribute_
        #plt.title(string)
        plt.xscale("log")
        fig_file_name = "%(outdir)s/Envelope Eccentricity Distribution.pdf" % dict(outdir=self.output_directory)
        plt.savefig(fig_file_name)
        plt.clf()
        return

    def properties_of_ith_binary_system_if_satisfies_constraints(self,
                                                                 i,
                                                                 constraints=Constraints_for_selecting_systems.constraints()):
        if i > len(self.planet_name) - 1 or i < 0: return None, None, None
        a = not (math.isnan(self.orbital_period[i])
                or math.isnan(self.orbital_period_upper_uncertainty[i])
                or math.isnan(self.orbital_period_lower_uncertainty[i])
                or math.isnan(self.primary_mass[i])
                or math.isnan(self.primary_mass_upper_uncertainty[i])
                or math.isnan(self.primary_mass_lower_uncertainty[i])
                or math.isnan(self.secondary_mass[i])
                or math.isnan(self.secondary_mass_upper_uncertainty[i])
                or math.isnan(self.secondary_mass_lower_uncertainty[i])
                or math.isnan(self.stellar_metallicity[i])
                or math.isnan(self.stellar_metallicity_upper_uncertainty[i])
                or math.isnan(self.stellar_metallicity_lower_uncertainty[i])
                or math.isnan(self.stellar_age[i])
                or math.isnan(self.stellar_age_upper_uncertainty[i])
                or math.isnan(self.stellar_age_lower_uncertainty[i])
                or math.isnan(self.eccentricity_now[i])
                or math.isnan(self.eccentricity_now_limit_flag[i]))
        b = not (math.isnan(self.ratio_of_planet_to_stellar_radius[i])
                or math.isnan(self.ratio_of_planet_to_stellar_radius_upper_uncertainty[i])
                or math.isnan(self.ratio_of_planet_to_stellar_radius_lower_uncertainty[i]))
        bb= not (math.isnan(self.transit_depth[i])
                or math.isnan(self.transit_depth_upper_uncertainty[i])
                or math.isnan(self.transit_depth_lower_uncertainty[i]))
        c = not (math.isnan(self.secondary_radius[i])
                or math.isnan(self.secondary_radius_upper_uncertainty[i])
                or math.isnan(self.secondary_radius_lower_uncertainty[i]))
        d = b or bb or c
        e = not (math.isnan(self.stellar_log_g[i])
                or math.isnan(self.stellar_log_g_upper_uncertainty[i])
                or math.isnan(self.stellar_log_g_lower_uncertainty[i])
                or math.isnan(self.stellar_density[i])
                or math.isnan(self.stellar_density_upper_uncertainty[i])
                or math.isnan(self.stellar_density_lower_uncertainty[i]))
        f = not (math.isnan(self.stellar_density[i])
                or math.isnan(self.stellar_density_upper_uncertainty[i])
                or math.isnan(self.stellar_density_lower_uncertainty[i])
                or math.isnan(self.stellar_effective_temperature[i])
                or math.isnan(self.stellar_effective_temperature_upper_uncertainty[i])
                or math.isnan(self.stellar_effective_temperature_lower_uncertainty[i]))
        g = not (math.isnan(self.stellar_effective_temperature[i])
                or math.isnan(self.stellar_effective_temperature_upper_uncertainty[i])
                or math.isnan(self.stellar_effective_temperature_lower_uncertainty[i])
                or math.isnan(self.stellar_log_g[i])
                or math.isnan(self.stellar_log_g_upper_uncertainty[i])
                or math.isnan(self.stellar_log_g_lower_uncertainty[i]))
        h = e or f or g
        j = True
        if self.eccentricity_now_limit_flag[i] == 0:
            if math.isnan(self.eccentricity_now_upper_uncertainty[i]) or math.isnan(self.eccentricity_now_lower_uncertainty[i]):
                j = False
        if not math.isnan(self.eccentricity_now_upper_uncertainty[i]):
            if (self.eccentricity_now_upper_uncertainty[i] == 0.0):
                j = False
        if a and d and h and j:
            if (Constraints_for_selecting_systems.constraints_are_satisfied(orbital_period=self.orbital_period[i],
                                          primary_mass=self.primary_mass[i],
                                          secondary_mass=self.secondary_mass[i],
                                          stellar_metallicity=self.stellar_metallicity[i],
                                          eccentricity_now=self.eccentricity_now[i],
                                          stellar_age=self.stellar_age[i],
                                          constraints=constraints)
               ):
                means = {'primary mass': self.primary_mass[i],
                         'secondary mass': self.secondary_mass[i],
                         'stellar metallicity': self.stellar_metallicity[i],
                         'orbital period': self.orbital_period[i],
                         'stellar age': self.stellar_age[i]
                         }
                standard_deviations = {'primary_mass_upper_uncertainty': self.primary_mass_upper_uncertainty[i],
                                       'primary_mass_lower_uncertainty': self.primary_mass_lower_uncertainty[i],
                                       'secondary_mass_upper_uncertainty': self.secondary_mass_upper_uncertainty[i],
                                       'secondary_mass_lower_uncertainty': self.secondary_mass_lower_uncertainty[i],
                                       'stellar_metallicity_upper_uncertainty':
                                           self.stellar_metallicity_upper_uncertainty[i],
                                       'stellar_metallicity_lower_uncertainty':
                                           self.stellar_metallicity_lower_uncertainty[i],
                                       'orbital_period_upper_uncertainty': self.orbital_period_upper_uncertainty[i],
                                       'orbital_period_lower_uncertainty': self.orbital_period_lower_uncertainty[i],
                                       'stellar_age_upper_uncertainty': self.stellar_age_upper_uncertainty[i],
                                       'stellar_age_lower_uncertainty': self.stellar_age_lower_uncertainty[i]
                                       }
                if self.eccentricity_now_limit_flag[i] == 0:
                      means['present eccentricity'] = self.eccentricity_now[i]
                      means['eccentricity now limit flag'] = 0
                      standard_deviations['eccentricity_now_upper_uncertainty'] = self.eccentricity_now_upper_uncertainty[i]
                      standard_deviations['eccentricity_now_lower_uncertainty'] = self.eccentricity_now_lower_uncertainty[i]
                else:
                      means['present eccentricity'] = 0
                      means['eccentricity now limit flag'] = self.eccentricity_now_limit_flag[i]
                      standard_deviations['eccentricity_now_upper_uncertainty'] = self.eccentricity_now[i]
                      standard_deviations['eccentricity_now_lower_uncertainty'] = 0

                if not math.isnan(self.primary_radius[i]):
                      means['primary radius']=self.primary_radius[i]
                      standard_deviations['primary_radius_upper_uncertainty'] = self.primary_radius_upper_uncertainty[i]
                      standard_deviations['primary_radius_lower_uncertainty'] = self.primary_radius_lower_uncertainty[i]
                if not math.isnan(self.secondary_radius[i]):
                      means['secondary radius']=self.secondary_radius[i]
                      standard_deviations['secondary_radius_upper_uncertainty'] = self.secondary_radius_upper_uncertainty[i]
                      standard_deviations['secondary_radius_lower_uncertainty'] = self.secondary_radius_lower_uncertainty[i]
                if not math.isnan(self.obliquity[i]):
                      means['obliquity']=self.obliquity[i]
                      standard_deviations['obliquity_upper_uncertainty'] = self.obliquity_upper_uncertainty[i]
                      standard_deviations['obliquity_lower_uncertainty'] = self.obliquity_lower_uncertainty[i]
                if not math.isnan(self.semi_major_axis[i]):
                      means['semi major axis']=self.semi_major_axis[i]
                      standard_deviations['semi_major_axis_upper_uncertainty'] = self.semi_major_axis_upper_uncertainty[i]
                      standard_deviations['semi_major_axis_lower_uncertainty'] = self.semi_major_axis_lower_uncertainty[i]
                else:
                      mass_star = self.primary_mass[i] * unit.solMass
                      mass_planet = self.secondary_mass[i] * unit.earthMass
                      period_orbital = self.orbital_period[i] * unit.d
                      gravitational_constant = const.G
                      a = (gravitational_constant * (mass_star + mass_planet)/4/(math.pi**2)*(period_orbital**2))**(1/3.0)
                      means['semi major axis'] = a.to(unit.AU).value
                      mass_star_u = (self.primary_mass[i] + self.primary_mass_upper_uncertainty[i]) * unit.solMass
                      mass_planet_u = (self.secondary_mass[i] + self.secondary_mass_upper_uncertainty[i]) * unit.earthMass
                      period_orbital_u = (self.orbital_period[i] + self.orbital_period_upper_uncertainty[i]) * unit.d
                      mass_star_l = (self.primary_mass[i] + self.primary_mass_lower_uncertainty[i]) * unit.solMass
                      mass_planet_l = (self.secondary_mass[i] + self.secondary_mass_lower_uncertainty[i]) * unit.earthMass
                      period_orbital_l = (self.orbital_period[i] + self.orbital_period_lower_uncertainty[i]) * unit.d
                      a_u = (gravitational_constant * (mass_star_u + mass_planet_u)/4/(math.pi**2)*(period_orbital_u**2))**(1/3.0)
                      a_l = (gravitational_constant * (mass_star_l + mass_planet_l)/4/(math.pi**2)*(period_orbital_l**2))**(1/3.0)
                      a_upper_uncertainty = a_u - a
                      a_lower_uncertainty = a_l - a
                      standard_deviations['semi_major_axis_upper_uncertainty'] = a_upper_uncertainty.to(unit.AU).value
                      standard_deviations['semi_major_axis_lower_uncertainty'] = a_lower_uncertainty.to(unit.AU).value
                if not math.isnan(self.stellar_log_g[i]):
                      means['stellar log g'] = self.stellar_log_g[i]
                      standard_deviations['stellar_log_g_upper_uncertainty'] = self.stellar_log_g_upper_uncertainty[i]
                      standard_deviations['stellar_log_g_lower_uncertainty'] = self.stellar_log_g_lower_uncertainty[i]
                if not math.isnan(self.stellar_density[i]):
                      means['stellar density'] = self.stellar_density[i]
                      standard_deviations['stellar_density_upper_uncertainty'] = self.stellar_density_upper_uncertainty[i]
                      standard_deviations['stellar_density_lower_uncertainty'] = self.stellar_density_lower_uncertainty[i]
                if not math.isnan(self.stellar_effective_temperature[i]):
                      means['stellar effective temperature'] = self.stellar_effective_temperature[i]
                      standard_deviations['stellar_effective_temperature_upper_uncertainty'] = self.stellar_effective_temperature_upper_uncertainty[i]
                      standard_deviations['stellar_effective_temperature_lower_uncertainty'] = self.stellar_effective_temperature_lower_uncertainty[i]
                if not math.isnan(self.ratio_of_planet_to_stellar_radius[i]):
                      means['ratio of planet to stellar radius'] = self.ratio_of_planet_to_stellar_radius[i]
                      standard_deviations['ratio_of_planet_to_stellar_radius_upper_uncertainty'] = self.ratio_of_planet_to_stellar_radius_upper_uncertainty[i]
                      standard_deviations['ratio_of_planet_to_stellar_radius_lower_uncertainty'] = self.ratio_of_planet_to_stellar_radius_lower_uncertainty[i]
                if not math.isnan(self.transit_depth[i]):
                      means['transit depth'] = self.transit_depth[i]
                      standard_deviations['transit_depth_upper_uncertainty'] = self.transit_depth_upper_uncertainty[i]
                      standard_deviations['transit_depth_lower_uncertainty'] = self.transit_depth_lower_uncertainty[i]
                if standard_deviations['eccentricity_now_upper_uncertainty'] == 0:
                      return None, None, None
                return means, standard_deviations, self.planet_name[i]
        return None, None, None

    def print_properties_of_binary_systems_satisfying_constraints(self,
                                                                  constraints=Constraints_for_selecting_systems.constraints()):
        index_of_binary_system_with_constrained_properties = []
        k = -1
        for i in range(0, len(self.planet_name)):
            means, standard_deviations, planet_name = self.properties_of_ith_binary_system_if_satisfies_constraints(i,
                                                                                                                    constraints=constraints)
            if not (means == None or standard_deviations == None or planet_name == None):
                logging.debug('______________________________')
                k = k + 1
                logging.debug('k = %(k)s'% dict(k=str(k)))
                string_means = json.dumps(means)
                string_standard_deviations = json.dumps(standard_deviations)
                logging.debug('means = %(means)s' % dict(means=string_means))
                logging.debug('standard deviations = %(stdev)s' % dict(stdev=string_standard_deviations))
                logging.debug('planet name = %(name)s' % dict(name=planet_name))
                logging.debug('e flag limit = %(eflag)s' % dict(eflag=str(means['eccentricity now limit flag'])))
                logging.debug('e_mean = %(e)s' % dict(e=str(means['present eccentricity'])))
                logging.debug('e_now_upper_uncertainty %(val)s' % dict(val=str(standard_deviations['eccentricity_now_upper_uncertainty'])))
                logging.debug('e_now_lower_uncertainty %(val)s' % dict(val=str(standard_deviations['eccentricity_now_lower_uncertainty'])))
                if 'secondary radius' in means:
                      a_over_Rpl = means['semi major axis'] / means['secondary radius']
                      log_a_over_Rpl = math.log(a_over_Rpl, 10)
                      logging.debug('log of semi major axis over secondary radius = %(val)s' % dict(val=str(log_a_over_Rpl)))
                      envelope_eccentricity = self.envelope_eccentricity_function(a_over_Rpl)
                      logging.debug('Envelope eccentricity for semi major axis over secondary radius = %(va)s is = %(ba)s' % dict(va=str(a_over_Rpl), ba=str(envelope_eccentricity)))
                index_of_binary_system_with_constrained_properties = index_of_binary_system_with_constrained_properties + [i]
        return index_of_binary_system_with_constrained_properties

if __name__ == '__main__':
    test = EnvelopeEccentricityDistribution()
    test.print_properties_of_binary_systems_satisfying_constraints()
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',
                        help='Store the path of the file name where the database of star-exoplanet systems is saved'
                        )
    args = parser.parse_args()

    if args.path:
        EnvelopeEccentricityDistribution.set_path_name_of_exoplanet_systems_database(args.path)
