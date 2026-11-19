#!/usr/bin/env python3

"""Prepare the Bayesian sampling all SB1 binary systems in a cluster."""

import logging
import traceback

from bayesian.sample_sb1 import prepare_sampling
from bayesian.sampling import setup_process
from bayesian.parse_command_line import \
    get_cluster_binary_ids, \
    parse_command_line

def main(config):
    """Avoid polluting global namespace."""

    for binary_id in get_cluster_binary_ids()[config.cluster]:
        config.system = config.cluster + '_' + str(binary_id)
        setup_process(config, 'manage')
        logging.info('Preparing sampling of %s', config.system)
        try:
            prepare_sampling(config)
        except ValueError:
            logging.error('Failed to prepare %s:\n%s',
                          config.system,
                          traceback.format_exc())
        logging.info('Successfully Prepared sampling of %s', config.system)


if __name__ == '__main__':
    try:
        main(
            parse_command_line(
                __doc__,
                'sb1_sampling.cfg',
                dissipation=True,
                cluster=True,
                primary_properties=('feh', 'logg', 'Teff', 'rho'),
                spindown=2
            )
        )
    except SystemExit:
        pass
    #Meant to simply report exception to log
    #pylint: disable=bare-except
    except:
        logging.critical(traceback.format_exc())
    #pylint: enable=bare-except
    else:
        logging.info('SB1 sampling completed successfully.')
