
import EnvelopeEccentricityDistribution

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
    
    means, standard_deviations, system_name = inputting_instance.envelope_eccentricity_distribution_instance.properties_of_ith_binary_system_if_satisfies_constraints(
        inputting_instance.index[15])
    print('rp over rs', means['ratio of planet to stellar radius'])
    means['ratio of planet to stellar radius'] = 0.0149
    standard_deviations['ratio_of_planet_to_stellar_radius_upper_uncertainty'] = 0.0002
    standard_deviations['ratio_of_planet_to_stellar_radius_lower_uncertainty'] = -0.0002
    print('Print properties of the chosen binary system: means = ', means, ' standard deviations = ',
          standard_deviations, ' Star-Exoplanet system name = ', system_name)

    print('*********************************************************')
