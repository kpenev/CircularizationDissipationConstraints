"""Define a custom logging class that keeps static context info."""

import logging

class CustomLogger(logging.Logger):
    """
    Logger tracking context information (with history) to add to log records.


    Attrs:
        _context
    """

    _context = dict()

    _context_history = []

    @classmethod
    def update_context(cls, **context):
        """Adds more context to log records."""

        cls._context_history.append(cls._context.copy())
        cls._context.update(context)

    @classmethod
    def revert_context(cls):
        """Revert to the previous context."""

        cls._context = cls._context_history.pop()

    def handle(self, record):
        """Add context and handle."""

        record.__dict__.update(self._context)
        super().handle(record)
