
import EnvelopeEccentricityDistribution
import argparse
import os
import json

class Inputting:
    envelope_eccentricity_distribution_instance = None
    index = None
    @classmethod
    def choosing_systems(cls):
        cls.envelope_eccentricity_distribution_instance = EnvelopeEccentricityDistribution.EnvelopeEccentricityDistribution()
        print('Binary systems whose probability density of eccentricity can be figured out:')
        cls.index = cls.envelope_eccentricity_distribution_instance.print_properties_of_binary_systems_satisfying_constraints()
        return

    def __init__(self):
        if self.index is None:
            self.choosing_systems()

if __name__ == '__main__':
    inputting_instance = Inputting()
    parser = argparse.ArgumentParser()
    parser.add_argument('--i',
                        help='Store the index of the system',
                        type = int
                        )
    parser.add_argument('--r',
                        help='Store the ratio of the planetary radius over stellar radius',
                        type=float
                        )
    parser.add_argument('--upper_uncertainty_r',
                        help='Store the upper uncertainty associated with r',
                        type=float
                        )
    parser.add_argument('--lower_uncertainty_r',
                        help='Store the lower uncertainty associated with r',
                        type=float
                        )
    parser.add_argument('--power_of_the_ratio_of_planetary_and_stellar_radius',
                        help='Store the power of the ratio of planetary and stellar radius'
                        )
    args = parser.parse_args()
    if args.i:
        measured_values, standard_deviations, system_name = inputting_instance.envelope_eccentricity_distribution_instance.properties_of_ith_binary_system_if_satisfies_constraints(
            inputting_instance.index[args.i])
        if args.r:
            measured_values['ratio of planet to stellar radius'] = args.r
            standard_deviations['ratio_of_planet_to_stellar_radius_upper_uncertainty'] = args.upper_uncertainty_r
            standard_deviations['ratio_of_planet_to_stellar_radius_lower_uncertainty'] = args.lower_uncertainty_r
        string_measured_values = json.dumps(measured_values)
        string_standard_deviations = json.dumps(standard_deviations)


        if args.power_of_the_ratio_of_planetary_and_stellar_radius:
            string = 'python3 SampleStarExoplanetSystemsProperties.py --measured_values \'' + string_measured_values + '\' --standard_deviations \'' + string_standard_deviations + '\' --system \'' + system_name + '\' --power_of_the_ratio_of_planetary_and_stellar_radius ' + args.power_of_the_ratio_of_planetary_and_stellar_radius
        else:
            string = 'python3 SampleStarExoplanetSystemsProperties.py --measured_values \'' + string_measured_values + '\' --standard_deviations \'' + string_standard_deviations + '\' --system \'' + system_name + '\''


        print(string)
        os.system(string)

    #print('rp over rs', means['ratio of planet to stellar radius'])
    #means['ratio of planet to stellar radius'] = 0.0149
    #standard_deviations['ratio_of_planet_to_stellar_radius_upper_uncertainty'] = 0.0002
    #standard_deviations['ratio_of_planet_to_stellar_radius_lower_uncertainty'] = -0.0002
    #print('Print properties of the chosen binary system: means = ', means, ' standard deviations = ',
          #standard_deviations, ' Star-Exoplanet system name = ', system_name)

    print('*********************************************************')
