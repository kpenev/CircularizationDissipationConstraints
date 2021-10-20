"""Base class implementing common interface for pickling/unpickling."""

from abc import ABC, abstractmethod
import os.path
from pickle import Pickler, Unpickler
import logging

class Picklable(ABC):
    """Implement commont interface for objects that can be pickled/unpickled."""

    _logger = logging.getLogger(__name__)

    @abstractmethod
    def _check_pickle(self):
        """
        Return None if given pickle does not match otherwise unpickle.

        Child classes should use self._load_pickle_object() to read components
        of the pickle.
        """

    def _load_pickle_object(self):
        """Get the next object in the pickle, handling incomplete unpickling."""

        if self._skip_nobjects == 0:
            raise IOError('Attempt to read past the end of pickle section!')
        self._skip_nobjects -= 1
        return self._unpickler.load()

    def check_for_pickled(self, pickle_fname):
        """Check if given file contains a re-usable pickle for current setup."""

        if not os.path.exists(pickle_fname):
            open(pickle_fname, 'wb').close()
            return None
        try:
            with open(pickle_fname, 'rb') as pickle_file:
                self._unpickler = Unpickler(pickle_file)
                result = None
                while result is None:
                    section, self._skip_nobjects = self._unpickler.load()
                    assert isinstance(section, str)
                    if (
                            section == self.__class__.__name__
                            and
                            self._skip_nobjects == self._expect_nobjects
                    ):
                        result = self._check_pickle()
                        if result is None:
                            self._logger.debug('Pickled %s does not match',
                                               self.__class__.__name__)
                        else:
                            self._logger.info('Found matching %s pickle',
                                              self.__class__.__name__)
                            return result
                    for _ in range(self._skip_nobjects):
                        self._unpickler.load()

        except EOFError:
            self._logger.info(
                'None of the pickled %s matches.',
                self.__class__.__name__
            )
            return None
        finally:
            self._unpickler = None
            self._skip_nobjects = None

    def add_to_pickle_file(self, pickle_fname, *items_to_pickle):
        """Pickle a fully set-up instance to the given file for fast re-use."""

        assert len(items_to_pickle) == self._expect_nobjects
        with open(pickle_fname, 'ab') as pickle_file:
            pickler = Pickler(pickle_file)
            pickler.dump((self.__class__.__name__, self._expect_nobjects))
            for item in items_to_pickle:
                pickler.dump(item)
        self._logger.info('Added pickle of %s to %s',
                          self.__class__.__name__,
                          pickle_fname)


    def __init__(self, expected_nobjects):
        """Prepare to look for pickles with specified number of entries."""

        self._expect_nobjects = expected_nobjects
        self._unpickler = None
        self._skip_nobjects = None
