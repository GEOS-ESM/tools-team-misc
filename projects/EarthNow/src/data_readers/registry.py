"""
Registry for data readers
"""

DATA_READERS = {}

def register(name):
    """
    Decorator to register a data reader class.
    
    Usage:
        @register("cycled_replays")
        class GEOSDataReader:
            ...
    """
    def decorator(cls):
        DATA_READERS[name] = cls
        return cls
    return decorator

