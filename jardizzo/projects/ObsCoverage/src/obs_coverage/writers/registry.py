"""
Registry for observation writers
"""


OBS_WRITERS = {}

def register(name):
    def decorator(func):
        OBS_WRITERS[name] = func
        return func
    return decorator
