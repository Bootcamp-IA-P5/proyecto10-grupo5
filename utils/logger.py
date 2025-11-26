import logging

class Logger:
    """
    A logger mixin class that configures and provides a logging instance for views.
    """
    @property
    def log(self):
        # Create the logger only when accessed (lazy loading)
        if not hasattr(self, '_log'):
            # Use the class name as the logger name
            self._log = logging.getLogger(self.__class__.__name__)
        return self._log
        