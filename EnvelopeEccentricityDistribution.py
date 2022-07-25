import math
import numpy as np
import Star_Exoplanet_system_io
import Constraints_for_selecting_systems
import matplotlib.pyplot as plt
import logging
import argparse



def getPathOfExoplanetSystemsData():
    return '/home/mmmahmud/CircularizationDissipationConstraints/data/PS_2021.07.13_00.12.38.csv'


def envelope_eccentricity_function(x,
                                   threshold_value_of_envelope_eccentricity = 0.01,
                                   largest_acceptable_value_of_envelope_eccentricity = 0.5):

    """
        Workout the envelope eccentricity given the log(semi major axis / planetary radius)
         from a scatter plot of eccentricity vs. log(semi major axis/planetary radius) graph

        Args:
            x:                                                            log(semi major axis/ secondary radius)
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
        self.obliquity = readPlanet.pl_orbincl  # degrees
        self.obliquity_upper_uncertainty = readPlanet.pl_orbinclerr1
        self.obliquity_lower_uncertainty = readPlanet.pl_orbinclerr2
        self.obliquity_limit_flag = readPlanet.pl_orbincllim
        self.vsini = readPlanet.st_vsin  # or readPlanet.st_vsini in Km/s
        self.vsini_upper_uncertainty = readPlanet.st_vsinerr1  # or readPlanet.st_vsinierr1
        self.vsini_lower_uncertainty = readPlanet.st_vsinerr2  # or readPlanet.st_vsinierr2
        self.vsini_limit_flag = readPlanet.st_vsinlim
        self.planet_mass_sin_i = readPlanet.pl_msinie  # Earth mass
        self.planet_mass_sin_i_upper_uncertainty = readPlanet.pl_msinieerr1
        self.planet_mass_sin_i_lower_uncertainty = readPlanet.pl_msinieerr2
        self.planet_mass_sin_i_limit_flag = readPlanet.pl_msinielim
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
        self.stellar_luminosity = readPlanet.st_lum  # log(solar)
        self.stellar_luminosity_upper_uncertainty = readPlanet.st_lumerr1
        self.stellar_luminosity_lower_uncertainty = readPlanet.st_lumerr2
        self.stellar_luminosity_limit_flag = readPlanet.st_lumlim
        self.ratio_of_planet_to_stellar_radius = readPlanet.pl_ratror
        self.ratio_of_planet_to_stellar_radius_upper_uncertainty = readPlanet.pl_ratrorerr1
        self.ratio_of_planet_to_stellar_radius_lower_uncertainty = readPlanet.pl_ratrorerr2
        self.planet_density = readPlanet.pl_dens
        self.planet_density_upper_uncertainty = readPlanet.pl_denserr1
        self.planet_density_lower_uncertainty = readPlanet.pl_denserr2

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
                        print('Name of the planet belongs to the binary system on the envelope ',
                              records_of_certain_attribute_and_eccentricity_now[i - 1]['planet name'],
                              attribute, ' = ',
                              records_of_certain_attribute_and_eccentricity_now[i - 1][attribute],
                              'envelope eccentricity = ', threshold_value_of_envelope_eccentricity,
                              'orbital eccentricity limit flag = ',
                              records_of_certain_attribute_and_eccentricity_now[i - 1][
                                  'orbital eccentricity limit flag'])
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
                    print('Name of the planet belongs to the binary system on the envelope ',
                          records_of_certain_attribute_and_eccentricity_now[i]['planet name'],
                          attribute, ' = ',
                          records_of_certain_attribute_and_eccentricity_now[i][attribute],
                          'envelope eccentricity = ', maximum_eccentricity_now,
                          'orbital eccentricity limit flag = ',
                          records_of_certain_attribute_and_eccentricity_now[i]['orbital eccentricity limit flag'])
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
                            planet_mass_sin_i=self.planet_mass_sin_i[i],
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
                            planet_mass_sin_i=self.planet_mass_sin_i[i],
                            constraints=constraints)):
                        records_of_log_semi_major_axis_over_planetary_radius_and_eccentricity_now = (
                                records_of_log_semi_major_axis_over_planetary_radius_and_eccentricity_now
                                + [{'log of semi major axis over planetary radius': (
                            math.log(self.semi_major_axis[i] / self.secondary_radius[i], 10)),
                            'present eccentricity': self.eccentricity_now[i],
                            'planet name': self.planet_name[i],
                            'primary mass': self.primary_mass[i],
                            'orbital eccentricity limit flag': self.eccentricity_now_limit_flag[i]}])
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
        x_2 = []
        eccentricity_now_2 = []
        for i in range(0, len(records_of_x_attribute_and_eccentricity_now)):
            if primary_mass[i] < 1.2:
                x_1 = x_1 + [10**(x[i] + 4.37023)]
                eccentricity_now_1 = eccentricity_now_1 + [eccentricity_now[i]]

            else:
                x_2 = x_2 + [10**(x[i]+4.37023)]
                eccentricity_now_2 = eccentricity_now_2 + [eccentricity_now[i]]
        plt.plot(x_1, eccentricity_now_1, 'x')
        plt.plot(x_2, eccentricity_now_2, 'o')
        plt.plot(xdata_, ydata)
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
        plt.savefig("Envelope Eccentricity Distribution.pdf")
        plt.show()
        return

    def properties_of_ith_binary_system_if_satisfies_constraints(self,
                                                                 i,
                                                                 constraints=Constraints_for_selecting_systems.constraints()):
        if i > len(self.planet_name) - 1 or i < 0: return None, None, None
        if not (math.isnan(self.orbital_period[i])
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
                or math.isnan(self.secondary_radius[i])
                or math.isnan(self.secondary_radius_upper_uncertainty[i])
                or math.isnan(self.secondary_radius_lower_uncertainty[i])
                or math.isnan(self.stellar_age[i])
                or math.isnan(self.stellar_age_upper_uncertainty[i])
                or math.isnan(self.stellar_age_lower_uncertainty[i])
                or math.isnan(self.eccentricity_now[i])
                or math.isnan(self.eccentricity_now_lower_uncertainty[i])
                or math.isnan(self.eccentricity_now_limit_flag[i])
                or math.isnan(self.eccentricity_now_upper_uncertainty[i])
                or math.isnan(self.semi_major_axis[i])
                or math.isnan(self.semi_major_axis_lower_uncertainty[i])
                or math.isnan(self.semi_major_axis_upper_uncertainty[i])
                or math.isnan(self.stellar_density[i])
                or math.isnan(self.stellar_density_lower_uncertainty[i])
                or math.isnan(self.stellar_density_upper_uncertainty[i])
                or math.isnan(self.stellar_log_g[i])
                or math.isnan(self.stellar_log_g_lower_uncertainty[i])
                or math.isnan(self.stellar_log_g_upper_uncertainty[i])
                or math.isnan(self.stellar_effective_temperature[i])
                or math.isnan(self.stellar_effective_temperature_lower_uncertainty[i])
                or math.isnan(self.stellar_effective_temperature_upper_uncertainty[i])
        ):
            if (Constraints_for_selecting_systems.constraints_are_satisfied(orbital_period=self.orbital_period[i],
                                          primary_mass=self.primary_mass[i],
                                          secondary_mass=self.secondary_mass[i],
                                          stellar_metallicity=self.stellar_metallicity[i],
                                          eccentricity_now=self.eccentricity_now[i],
                                          stellar_age=self.stellar_age[i],
                                          constraints=constraints)
                    and self.eccentricity_now_limit_flag[i] == 0
            ):
                means = {'primary mass': self.primary_mass[i],
                         'secondary mass': self.secondary_mass[i],
                         'primary radius': self.primary_radius[i],
                         'secondary radius': self.secondary_radius[i],
                         'stellar metallicity': self.stellar_metallicity[i],
                         'orbital period': self.orbital_period[i],
                         'obliquity': self.obliquity[i],
                         'stellar age': self.stellar_age[i],
                         'present eccentricity': self.eccentricity_now[i],
                         'semi major axis': self.semi_major_axis[i],
                         'stellar log g': self.stellar_log_g[i],
                         'stellar density': self.stellar_density[i],
                         'stellar effective temperature': self.stellar_effective_temperature[i],
                         'ratio of planet to stellar radius': self.ratio_of_planet_to_stellar_radius[i]
                         }
                standard_deviations = {'primary_mass_upper_uncertainty': self.primary_mass_upper_uncertainty[i],
                                       'primary_mass_lower_uncertainty': self.primary_mass_lower_uncertainty[i],
                                       'secondary_mass_upper_uncertainty': self.secondary_mass_upper_uncertainty[i],
                                       'secondary_mass_lower_uncertainty': self.secondary_mass_lower_uncertainty[i],
                                       'primary_radius_upper_uncertainty': self.primary_radius_upper_uncertainty[i],
                                       'primary_radius_lower_uncertainty': self.primary_radius_lower_uncertainty[i],
                                       'secondary_radius_upper_uncertainty': self.secondary_radius_upper_uncertainty[i],
                                       'secondary_radius_lower_uncertainty': self.secondary_radius_lower_uncertainty[i],
                                       'stellar_metallicity_upper_uncertainty':
                                           self.stellar_metallicity_upper_uncertainty[i],
                                       'stellar_metallicity_lower_uncertainty':
                                           self.stellar_metallicity_lower_uncertainty[i],
                                       'orbital_period_upper_uncertainty': self.orbital_period_upper_uncertainty[i],
                                       'orbital_period_lower_uncertainty': self.orbital_period_lower_uncertainty[i],
                                       'obliquity_upper_uncertainty': self.obliquity_upper_uncertainty[i],
                                       'obliquity_lower_uncertainty': self.obliquity_lower_uncertainty[i],
                                       'stellar_age_upper_uncertainty': self.stellar_age_upper_uncertainty[i],
                                       'stellar_age_lower_uncertainty': self.stellar_age_lower_uncertainty[i],
                                       'eccentricity_now_upper_uncertainty': self.eccentricity_now_upper_uncertainty[i],
                                       'eccentricity_now_lower_uncertainty': self.eccentricity_now_lower_uncertainty[i],
                                       'semi_major_axis_upper_uncertainty': self.semi_major_axis_upper_uncertainty[i],
                                       'semi_major_axis_lower_uncertainty': self.semi_major_axis_lower_uncertainty[i],
                                       'stellar_log_g_upper_uncertainty': self.stellar_log_g_upper_uncertainty[i],
                                       'stellar_log_g_lower_uncertainty': self.stellar_log_g_lower_uncertainty[i],
                                       'stellar_density_upper_uncertainty': self.stellar_density_upper_uncertainty[i],
                                       'stellar_density_lower_uncertainty': self.stellar_density_lower_uncertainty[i],
                                       'stellar_effective_temperature_upper_uncertainty':
                                           self.stellar_effective_temperature_upper_uncertainty[i],
                                       'stellar_effective_temperature_lower_uncertainty':
                                           self.stellar_effective_temperature_lower_uncertainty[i],
                                       'ratio_of_planet_to_stellar_radius_upper_uncertainty':
                                           self.ratio_of_planet_to_stellar_radius_upper_uncertainty[i],
                                       'ratio_of_planet_to_stellar_radius_lower_uncertainty':
                                           self.ratio_of_planet_to_stellar_radius_lower_uncertainty[i]
                                       }

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
                print('______________________________')
                k = k + 1
                print('k = ', k)
                print('means = ', means, 'standard deviations = ', standard_deviations, ' planet name = ', planet_name)
                print('e_mean ', means['present eccentricity'])
                print('e_now_upper_uncertainty ', standard_deviations['eccentricity_now_upper_uncertainty'])
                print('e_now_lower_uncertainty ', standard_deviations['eccentricity_now_lower_uncertainty'])
                a_over_Rpl = means['semi major axis'] / means['secondary radius']
                print('log of semi major axis over secondary radius = ', math.log(a_over_Rpl, 10))
                print('Envelope eccentricity for semi major axis over secondary radius = ', a_over_Rpl, ' is = ',
                      self.envelope_eccentricity_function(a_over_Rpl))
                index_of_binary_system_with_constrained_properties = index_of_binary_system_with_constrained_properties + [
                    i]
        return index_of_binary_system_with_constrained_properties

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path',
                        help='Store the path of the file name where the database of star-exoplanet systems is saved'
                        )
    args = parser.parse_args()

    if args.path:
        EnvelopeEccentricityDistribution.set_path_name_of_exoplanet_systems_database(args.path)
