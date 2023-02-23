import logging
import EnvelopeEccentricityDistribution
import argparse
import os
import json
from SampleStarExoplanetSystemsPropertiesTesting import *


class Inputting:
    envelope_eccentricity_distribution_instance = None
    index = None
    @classmethod
    def creating_envelope_eccentricity_distribution_instance(cls):
        cls.envelope_eccentricity_distribution_instance = EnvelopeEccentricityDistribution.EnvelopeEccentricityDistribution()
        return
    @classmethod
    def choosing_systems(cls):
        logging.debug('Binary systems whose probability density of eccentricity can be figured out:')
        cls.index = cls.envelope_eccentricity_distribution_instance.print_properties_of_binary_systems_satisfying_constraints()
        return

    def __init__(self):
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
        if self.envelope_eccentricity_distribution_instance is None:
            self.creating_envelope_eccentricity_distribution_instance()
        if self.index is None:
            self.choosing_systems()

if __name__ == '__main__':
    inputting_instance = Inputting()
    parser = argparse.ArgumentParser()
    parser.add_argument('--i',
                        help='Store the index of the system',
                        type = int
                        )
    args = parser.parse_args()
    if args.i or args.i==0:
        logging.debug('* i = %(i)f' % dict(i=args.i))
        logging.debug('* index = %(x)s' % dict(x = repr(inputting_instance.index)))
        measured_values, standard_deviations, system_name = inputting_instance.envelope_eccentricity_distribution_instance.properties_of_ith_binary_system_if_satisfies_constraints(
            inputting_instance.index[args.i])
        logging.debug("** measured_values are %(x)s" % dict(x=repr(measured_values)))
        logging.debug("** standard deviations are %(x)s" % dict(x=repr(standard_deviations)))
        logging.debug("** system name is %(x)s" % dict(x=system_name))
        logging.debug('i = %(i)s' % dict(i=repr(args.i)))
        logging.debug('index = %(x)s' % dict(x = repr(inputting_instance.index)))
        logging.debug('Inputting: measured values are %(x)s' % dict(x=repr(measured_values)))
        logging.debug('Inputting: standard deviations are %(x)s' % dict(x=repr(standard_deviations)))
        logging.debug("Initialization is going to take place")
        InitializationOfSamplingPropertiesOfSystemTesting()
        logging.debug("Initialization is done.")
        test3 = SamplingPropertiesOfSystemTesting(measured_values,
                                                  standard_deviations,
                                                  system_name=system_name,
                                                  envelope_eccentricity_function=EnvelopeEccentricityDistribution.envelope_eccentricity_function
                                                  )
        #string_measured_values = json.dumps(measured_values)
        #string_standard_deviations = json.dumps(standard_deviations)
        #string = 'python3 SampleStarExoplanetSystemsProperties.py --measured_values \'' + string_measured_values + '\' --standard_deviations \'' + string_standard_deviations + '\' --system \'' + system_name + '\''

        #logging.debug(string)
        #os.system(string)
    logging.debug('This is the end of InputtingTesting.py ***************')

