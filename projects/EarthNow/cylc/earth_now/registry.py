"""
Registry for data readers
"""

PRODUCTS = {}


def register(name):
    """
    Decorator to register a Product class.

    Usage:
        @register("sea_level_pressure")
        class Product:
            ...
    """

    def decorator(cls):
        PRODUCTS[name] = cls
        return cls

    return decorator
