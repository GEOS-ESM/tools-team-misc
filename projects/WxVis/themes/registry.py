"""
Registry for themes
"""

THEMES = {}

def register(name):
    """
    Decorator to register a theme class.

    Usage:
        @register("theme_name")
        class Theme:
            ...
    """
    def decorator(cls):
        THEMES[name] = cls
        return cls
    return decorator
