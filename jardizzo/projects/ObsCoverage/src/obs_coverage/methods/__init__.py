"""
Observation Data Methods
"""

from .registry import OBS_METHODS

# Import modules so they self-register (import the modules, not functions from them)
from . import obs_methods

__all__ = ["OBS_METHODS"]
