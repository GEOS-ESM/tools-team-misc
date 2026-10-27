"""
Registry for observation readers
"""

OBS_READERS = {}


def register(name):
    """
    Decorator to register a data reader class.

    Usage:
        @register("aeronet")
        class AeronetDataReader:
            ...
    """

    def decorator(cls):
        OBS_READERS[name] = cls
        return cls

    return decorator
