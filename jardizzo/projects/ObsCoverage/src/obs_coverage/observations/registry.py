"""
Registry for observation readers
"""

OBS_TYPES = {}


def register(name):
    """
    Decorator to register a data reader class.

    Usage:
        @register("aeronet")
        class AeronetDataReader:
            ...
    """

    def decorator(cls):
        OBS_TYPES[name] = cls
        return cls

    return decorator
