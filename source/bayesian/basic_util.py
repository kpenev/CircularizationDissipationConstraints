"""Collection of basic utilities."""

import logging

_logger = logging.getLogger(__name__)

def compare_frozen_distributions(distro1, distro2):
    """Check if the given frozen stipy.stats are identical."""

    if not isinstance(distro1.dist, type(distro2.dist)):
        _logger.debug('Distribution type mismatch: %s vs %s',
                      type(distro1.dist).__name__,
                      type(distro2).__name__)
        return False
    for attr_name in ['args', 'kwds']:
        if(
            getattr(distro1, attr_name)
            !=
            getattr(distro2, attr_name)
        ):
            _logger.debug('%s mismatch: %s vs %s',
                          attr_name,
                          getattr(distro1, attr_name),
                          getattr(distro2, attr_name))
            return False

    return True
