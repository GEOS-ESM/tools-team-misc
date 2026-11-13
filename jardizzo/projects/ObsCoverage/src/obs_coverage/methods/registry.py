"""
Registry for observation methods
"""

OBS_METHODS = {}


def register(name):
    def decorator(func):
        OBS_METHODS[name] = func
        return func

    return decorator
