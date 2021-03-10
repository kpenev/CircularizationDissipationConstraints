"""Define a custom logging class that keeps static context info."""

import logging

class CustomLoggerAdapter(logging.LoggerAdapter):
    """Add context info to logging, updating rather than overwriting extra."""

    def process(self, msg, kwargs):
        """Update or set `'extra'` keyword."""

        if 'extra' in kwargs:
            kwargs['extra'].update(self.extra)
        else:
            kwargs['extra'] = self.extra

        return msg, kwargs


