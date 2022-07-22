import math

def constraints(smallest_acceptable_value_of_orbital_period=0,
                largest_acceptable_value_of_orbital_period=10,
                smallest_acceptable_value_of_primary_mass=0.4,
                largest_acceptable_value_of_primary_mass=1.2,
                smallest_acceptable_value_of_secondary_mass=0,
                largest_acceptable_value_of_secondary_mass=25000,  # mass of a brown dwarf
                smallest_acceptable_value_of_stellar_metallicity=-1.014,
                largest_acceptable_value_of_stellar_metallicity=0.537,
                smallest_acceptable_value_of_stellar_age=0,
                largest_acceptable_value_of_stellar_age=10,
                smallest_acceptable_value_of_eccentricity_now=0,
                largest_acceptable_value_of_eccentricity_now=0.45):
    smallest = {'orbital period': smallest_acceptable_value_of_orbital_period,
                'primary mass': smallest_acceptable_value_of_primary_mass,
                'secondary mass': smallest_acceptable_value_of_secondary_mass,
                'stellar metallicity': smallest_acceptable_value_of_stellar_metallicity,
                'stellar age': smallest_acceptable_value_of_stellar_age,
                'present eccentricity': smallest_acceptable_value_of_eccentricity_now}
    largest = {'orbital period': largest_acceptable_value_of_orbital_period,
               'primary mass': largest_acceptable_value_of_primary_mass,
               'secondary mass': largest_acceptable_value_of_secondary_mass,
               'stellar metallicity': largest_acceptable_value_of_stellar_metallicity,
               'stellar age': largest_acceptable_value_of_stellar_age,
               'present eccentricity': largest_acceptable_value_of_eccentricity_now}
    return smallest, largest

def constraints_for_eccentricity_envelope(smallest_acceptable_value_of_secondary_radius=8,
                                          largest_acceptable_value_of_secondary_radius=math.inf,
                                          smallest_acceptable_value_of_planet_mass_sin_i=50,
                                          largest_acceptable_value_of_planet_mass_sin_i=math.inf):
    smallest = {'secondary radius': smallest_acceptable_value_of_secondary_radius,
                'planet mass times sin i': smallest_acceptable_value_of_planet_mass_sin_i}
    largest = {'secondary radius': largest_acceptable_value_of_secondary_radius,
               'planet mass times sin i': largest_acceptable_value_of_planet_mass_sin_i}
    return smallest, largest
def constraints_are_satisfied(orbital_period,
                              primary_mass,
                              secondary_mass,
                              stellar_metallicity,
                              eccentricity_now,
                              stellar_age,
                              constraints=constraints()):
    smallest = constraints[0]
    largest = constraints[1]
    if ((orbital_period <= largest['orbital period'] and orbital_period > smallest['orbital period'])
            and (primary_mass > smallest['primary mass'] and primary_mass < largest['primary mass'])
            and (secondary_mass > smallest['secondary mass'] and secondary_mass < largest['secondary mass'])
            and (stellar_metallicity > smallest['stellar metallicity'] and stellar_metallicity < largest[
                'stellar metallicity'])
            and (eccentricity_now >= smallest['present eccentricity'] and eccentricity_now <= largest[
                'present eccentricity'])
            and (stellar_age >= smallest['stellar age'] and stellar_age <= largest['stellar age'])):
        return True
    return False


def constraints_for_eccentricity_envelope_are_satisfied(secondary_radius,
                                                        planet_mass_sin_i,
                                                        constraints=constraints_for_eccentricity_envelope()):
    smallest = constraints[0]
    largest = constraints[1]
    if secondary_radius <= largest['secondary radius'] and secondary_radius > smallest['secondary radius']:
        return True
    if planet_mass_sin_i <= largest['planet mass times sin i'] and planet_mass_sin_i > smallest[
        'planet mass times sin i']:
        return True

    return False